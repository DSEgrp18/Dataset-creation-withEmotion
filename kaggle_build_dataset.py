#!/usr/bin/env python3
"""
kaggle_build_dataset.py -- build the full Sinhala sentence dataset on Kaggle.

Self-contained: imports nothing from this repo, so it can be dropped straight
into a Kaggle notebook. Downloads episodes from archive.org, transcribes them
with Gemini, and writes sentence-level clips + a manifest.

WHY KAGGLE
    Only for the network. Downloading a 25-minute episode takes ~1 minute there
    versus ~50 on a home connection, and mp3 decode is faster too.

WHAT KAGGLE DOES *NOT* FIX
    The binding limit is the Gemini free tier: ~20 generateContent calls per day
    PER MODEL. That quota belongs to your API KEY, not the machine, so running on
    Kaggle does not raise it. With ~5 models usable and 5 requests per episode,
    expect roughly 20 episodes/day. The script fails over between models, stops
    cleanly when they are all spent, and resumes where it left off.

SETUP (three things, all in the Kaggle UI)
    1. Settings -> Internet -> ON              (downloads + API calls need it)
    2. Add-ons -> Secrets -> add GEMINI_API_KEY
    3. Accelerator: None. This is network- and API-bound; a GPU buys nothing.

OUTPUT (under /kaggle/working/dataset)
    <episode>/clips/*.wav     one sentence per file, 24 kHz mono
    <episode>/manifest.csv    clip_id, start_sec, end_sec, duration, text
    all_manifests.csv         every episode combined
    Save Version when done -- /kaggle/working is wiped between sessions.

COPYRIGHT
    These archive.org items carry no license field, so treat them as
    copyrighted. The manifest (timestamps + text) is the shareable artifact;
    the wavs are for your own training and review. Do not redistribute them.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

# =========================================================================== #
# CONFIG -- edit these
# =========================================================================== #

# "muwan-palassa" is not one episode -- it is the whole serial: 201 files,
# Episode 001-201, 82 hours. The other identifiers are separate uploads of
# individual episodes (several duplicate what is already in the collection).
IDENTIFIERS = [
    "muwan-palassa",
]

# Optional: pull more items automatically. e.g. 'title:("Muwan Palassa")'
# finds 17 items. Leave "" to use IDENTIFIERS only.
SEARCH_QUERY = ""
SEARCH_MAX = 50

OUT_ROOT = Path("/kaggle/working/dataset")
RAW_ROOT = Path("/kaggle/working/raw")

# ---- running the full 82 hours across several days ----------------------- #
# 201 episodes x ~5 requests is ~1000 requests against a ~100/day allowance,
# so the build spans roughly ten days. Two mechanisms carry it:
#
#   WITHIN a session : when every model is spent, sleep and retry. Free-tier
#                      quota resets daily, and a Kaggle session lives ~12 h, so
#                      one session can sit through a reset and carry on.
#   ACROSS sessions  : /kaggle/working is wiped, so Save Version at the end and
#                      add that output as a Data source next run. Anything found
#                      under /kaggle/input is adopted before starting, and
#                      finished episodes are skipped for free.
WAIT_FOR_QUOTA = True      # False -> stop immediately when quota runs out
QUOTA_POLL_MIN = 20.0      # minutes between retries while waiting
SESSION_HOURS = 11.0       # stop before Kaggle kills the session (~12 h limit)
DELETE_RAW_AFTER = True    # drop each mp3 once processed (saves ~600 MB)
MAX_EPISODES = 0           # 0 = no cap; else stop after this many this run

# Quota is per model, so a later one still works when an earlier is exhausted.
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

CHUNK_SEC = 300.0     # 5 min per request: a 25-min episode costs 5 requests
RPM = 6.0             # request pacing; 429s start well below the documented 15
KEEP_CLIPS = True     # False -> manifest only (smaller output, still shareable)

MODEL_SR = 16_000     # what the VAD and the API see
TARGET_SR = 24_000    # what the clips are written at (TTS-ready)
MIN_CLIP, MAX_CLIP = 1.0, 15.0
PAD = 0.10
SNAP_TOL = 1.0

AUDIO_EXTS = (".mp3", ".ogg", ".flac", ".wav", ".m4a")
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/{model}:generateContent")

PROMPT = """\
This is a clip from a Sinhala radio drama. Transcribe it and split it into
SENTENCES (or, for a long sentence, natural clause-level parts).

For each sentence return:
  start_sec : when it starts, in seconds from the START OF THIS CLIP
  end_sec   : when it ends, in seconds from the START OF THIS CLIP
  text      : the sentence, in Sinhala script

Rules:
- Sinhala script only. Do NOT romanise. Do NOT translate.
- Transcribe verbatim, including false starts and repeated words.
- One entry per sentence. Do not merge two sentences into one entry.
- Do not include music, sound effects, speaker names, or descriptions.
- Timestamps must increase and must not overlap.
- If a stretch has no speech, simply produce no entry for it.
- If there is no speech at all, return an empty list.
"""

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start_sec": {"type": "NUMBER"},
            "end_sec": {"type": "NUMBER"},
            "text": {"type": "STRING"},
        },
        "required": ["start_sec", "end_sec", "text"],
    },
}


def log(stage, msg):
    print(f"[{stage}] {msg}", flush=True)


def warn(stage, msg):
    print(f"[{stage}] WARNING: {msg}", flush=True)


class QuotaExhausted(RuntimeError):
    """A model's per-DAY allowance is gone; waiting will not help."""


@dataclass
class Sentence:
    start_sec: float
    end_sec: float
    text: str
    snapped: bool = False

    @property
    def duration(self):
        return self.end_sec - self.start_sec


# =========================================================================== #
# Setup
# =========================================================================== #

def ensure_deps():
    """Kaggle ships most of this; install only what is missing."""
    need = []
    try:
        import soundfile  # noqa: F401
    except ImportError:
        need.append("soundfile")
    try:
        import librosa  # noqa: F401
    except ImportError:
        need.append("librosa")
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        need.append("silero-vad")
    if need:
        log("setup", f"installing {' '.join(need)}")
        os.system(f"{sys.executable} -m pip install -q " + " ".join(need))


def get_api_key() -> str:
    """Kaggle Secrets first, then env, then a local file."""
    try:
        from kaggle_secrets import UserSecretsClient
        k = UserSecretsClient().get_secret("GEMINI_API_KEY")
        if k:
            return k.strip()
    except Exception:
        pass
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var].strip()
    f = Path("api_key.txt")
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "No API key.\n"
        "  Kaggle: Add-ons -> Secrets -> add GEMINI_API_KEY, then re-run.\n"
        "  Free key (no card): https://aistudio.google.com/apikey")


# =========================================================================== #
# archive.org -- plain HTTP, no extra dependency
# =========================================================================== #

def ia_search(query: str, rows: int) -> list[str]:
    r = requests.get("https://archive.org/advancedsearch.php",
                     params={"q": query, "fl[]": "identifier", "rows": rows,
                             "page": 1, "output": "json"}, timeout=120)
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    return [d["identifier"] for d in docs if d.get("identifier")]


def ia_pick_audio(ident: str) -> tuple[list[dict], str]:
    """Return (audio files worth downloading, license string).

    archive.org keeps the uploaded file (source == 'original') alongside
    auto-generated lower-bitrate copies (source == 'derivative'). Taking
    originals when they exist avoids downloading the same episode twice.
    """
    r = requests.get(f"https://archive.org/metadata/{ident}", timeout=120)
    r.raise_for_status()
    meta = r.json()
    lic = str(meta.get("metadata", {}).get("licenseurl", "ABSENT"))
    audio = [f for f in meta.get("files", [])
             if str(f.get("name", "")).lower().endswith(AUDIO_EXTS)]
    originals = [f for f in audio
                 if str(f.get("source", "")).lower() == "original"]
    pool = originals or audio
    best: dict[str, dict] = {}
    for f in pool:
        stem = Path(f["name"]).stem
        cur = best.get(stem)
        if cur is None or float(f.get("size", 0) or 0) > float(cur.get("size", 0) or 0):
            best[stem] = f
    return list(best.values()), lic


def ia_download(ident: str, fname: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log("download", f"  cached: {dest.name}")
        return dest
    url = f"https://archive.org/download/{ident}/{requests.utils.quote(fname)}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as fh:
            for block in r.iter_content(1 << 20):
                fh.write(block)
        tmp.replace(dest)
    log("download", f"  got {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


# =========================================================================== #
# Audio
# =========================================================================== #

def load_audio(path, sr=None):
    import librosa
    y, out_sr = librosa.load(str(path), sr=sr, mono=True)
    return y.astype("float32"), out_sr


def resample(y, src, dst):
    if src == dst:
        return y
    import librosa
    return librosa.resample(y, orig_sr=src, target_sr=dst).astype("float32")


def save_wav(path, y, sr):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")


def flac_bytes(y, sr) -> bytes:
    """FLAC keeps a 5-minute chunk inside the ~20 MB inline-data ceiling."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(y, dtype="float32"), -1, 1), sr,
             format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def speech_regions(y16) -> list[tuple[float, float]]:
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(y16), model,
                               sampling_rate=MODEL_SR,
                               min_speech_duration_ms=200)
    return [(t["start"] / MODEL_SR, t["end"] / MODEL_SR) for t in ts]


def make_chunks(regions, target):
    """Group regions into ~target-second chunks, cutting only in silence."""
    if not regions:
        return []
    out, cur_start, cur_end = [], regions[0][0], regions[0][1]
    for s, e in regions[1:]:
        if e - cur_start > target:
            out.append((cur_start, cur_end))
            cur_start = s
        cur_end = e
    out.append((cur_start, cur_end))
    return out


def snap_chunk(raw, regions, c0, c1) -> list[Sentence]:
    """Award each speech region to exactly ONE sentence, in time order.

    Letting every sentence independently grab the regions it overlaps makes a
    two-word line swallow its neighbours' audio, so clips overlap and the same
    speech ships repeatedly under different transcripts. Single ownership makes
    that impossible. The walk is monotonic because over a 5-minute chunk the
    model's absolute timestamps drift by seconds, while the ORDER of sentences
    is still exactly right -- so order, not the numbers, is what we trust.
    """
    inside = [r for r in regions if r[1] > c0 and r[0] < c1]
    owner: dict[int, list] = {i: [] for i in range(len(raw))}
    cur = 0
    for r in inside:
        best_i, best_ov = None, 0.0
        for i in range(cur, len(raw)):
            s, e, _ = raw[i]
            if s > r[1]:
                break
            ov = min(r[1], e) - max(r[0], s)
            if ov > best_ov:
                best_ov, best_i = ov, i
        if best_i is not None:
            owner[best_i].append(r)
            cur = best_i
    out = []
    for i, (s, e, text) in enumerate(raw):
        mine = owner[i]
        out.append(Sentence(mine[0][0], mine[-1][1], text, True) if mine
                   else Sentence(s, e, text, False))
    return out


def enforce_no_overlap(sents) -> int:
    sents.sort(key=lambda x: (x.start_sec, x.end_sec))
    n = 0
    for a, b in zip(sents, sents[1:]):
        if a.end_sec > b.start_sec:
            a.end_sec = b.start_sec
            n += 1
    return n


# =========================================================================== #
# Gemini
# =========================================================================== #

def call_gemini(y16_chunk, api_key, model, min_interval, last):
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "audio/flac",
                             "data": base64.b64encode(
                                 flac_bytes(y16_chunk, MODEL_SR)).decode()}},
        ]}],
        "generationConfig": {"temperature": 0.0,
                             "responseMimeType": "application/json",
                             "responseSchema": RESPONSE_SCHEMA},
    }
    url = GEMINI_URL.format(model=model)
    for attempt in range(8):
        gap = time.time() - last[0]
        if gap < min_interval:
            time.sleep(min_interval - gap)
        last[0] = time.time()
        r = requests.post(url, json=body, timeout=600,
                          headers={"x-goog-api-key": api_key})
        if r.status_code == 200:
            cands = r.json().get("candidates") or []
            if not cands:
                return []
            raw = "".join(p.get("text", "") for p in
                          cands[0].get("content", {}).get("parts") or []).strip()
            if not raw:
                return []
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\[.*\]", raw, re.S)
                if not m:
                    return []
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return []
            return data if isinstance(data, list) else []
        if r.status_code in (429, 500, 503):
            wait = 0.0
            try:
                for d in r.json().get("error", {}).get("details", []):
                    for v in d.get("violations", []):
                        # Per-DAY caps do not clear by waiting -- switch model.
                        if "PerDay" in str(v.get("quotaId", "")):
                            raise QuotaExhausted(
                                f"{model}: daily limit ({v.get('quotaValue')}) reached")
                    if d.get("retryDelay"):
                        wait = float(str(d["retryDelay"]).rstrip("s"))
            except QuotaExhausted:
                raise
            except Exception:
                pass
            wait = max(wait, 15.0 * (attempt + 1))
            warn("gemini", f"HTTP {r.status_code}; waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("rate-limited after 8 attempts")


def transcribe_chunk_waiting(seg, api_key, state):
    """Transcribe one chunk, failing over between models and WAITING for quota.

    Returns the parsed items, or None if the run should stop (session budget
    spent, or waiting is disabled).

    The daily allowance is per model, so exhausting one only retires that one.
    When all are retired the function sleeps and clears the graveyard, because
    a per-day cap does clear -- just not soon. Rather than predict Google's
    reset moment (which the API never states), it simply retries every
    QUOTA_POLL_MIN and lets a successful call be the proof that quota returned.
    """
    while True:
        for model in MODELS:
            if model in state["dead"]:
                continue
            try:
                items = call_gemini(seg, api_key, model,
                                    60.0 / RPM if RPM else 0.0, state["last"])
                state["requests"] += 1
                return items
            except QuotaExhausted as ex:
                warn("gemini", f"{ex}")
                state["dead"].add(model)
            except Exception as ex:
                warn("gemini", f"  {type(ex).__name__}: {ex}")
                return None

        # Every model is spent.
        if not WAIT_FOR_QUOTA:
            state["stop"] = "quota"
            return None
        left = state["deadline"] - time.time()
        if left < QUOTA_POLL_MIN * 60:
            state["stop"] = "session"
            warn("gemini", "quota spent and not enough session time left to "
                           "wait it out -- stopping cleanly")
            return None
        state["waits"] += 1
        resume = time.strftime("%H:%M:%S",
                               time.localtime(time.time() + QUOTA_POLL_MIN * 60))
        log("quota", f"all {len(MODELS)} models spent. Sleeping "
                     f"{QUOTA_POLL_MIN:.0f} min (retry ~{resume}); "
                     f"{left/3600:.1f} h of session left.")
        time.sleep(QUOTA_POLL_MIN * 60)
        state["dead"].clear()      # let the retry decide whether quota is back


def adopt_previous_output():
    """Seed /kaggle/working from a previous run attached under /kaggle/input.

    Kaggle wipes /kaggle/working between sessions, so a multi-day build would
    otherwise restart from zero every time. Attaching the last run's output as
    a Data source and copying its caches in makes finished episodes free.
    """
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return 0
    adopted = 0
    for cand in inp.glob("*/dataset"):
        if not cand.is_dir():
            continue
        for ep in cand.iterdir():
            if not ep.is_dir():
                continue
            dst = OUT_ROOT / ep.name
            if dst.exists():
                continue
            # Only the cheap-to-copy evidence of past work: the chunk cache
            # (so no request is repeated) and the manifest (so the episode
            # counts as done). Clips are regenerable and would be GBs.
            src_chunks, src_man = ep / ".chunks", ep / "manifest.csv"
            if src_man.exists() or src_chunks.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                if src_chunks.is_dir():
                    shutil.copytree(src_chunks, dst / ".chunks",
                                    dirs_exist_ok=True)
                if src_man.exists():
                    shutil.copy2(src_man, dst / "manifest.csv")
                adopted += 1
    if adopted:
        log("resume", f"adopted {adopted} episode(s) from a previous run")
    return adopted


# =========================================================================== #
# One episode
# =========================================================================== #

def build_episode(src: Path, ident: str, api_key: str, state: dict) -> dict:
    stem = re.sub(r"\W+", "_", src.stem).strip("_")
    out_dir = OUT_ROOT / stem
    cache_dir = out_dir / ".chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)

    y_raw, sr = load_audio(src)
    y16 = resample(y_raw, sr, MODEL_SR)
    dur = len(y_raw) / sr
    log("audio", f"  {dur/60:.1f} min @ {sr} Hz")

    regions = speech_regions(y16)
    chunks = make_chunks(regions, CHUNK_SEC)
    log("vad", f"  {len(regions)} speech regions -> {len(chunks)} chunk(s)")

    sents: list[Sentence] = []
    failed = 0
    for i, (c0, c1) in enumerate(chunks):
        cf = cache_dir / f"chunk_{i:04d}.json"
        items = None
        if cf.exists():
            try:
                items = json.loads(cf.read_text(encoding="utf-8"))
                log("gemini", f"  chunk {i+1}/{len(chunks)} cached")
            except Exception:
                cf.unlink(missing_ok=True)
        if items is None:
            seg = y16[int(c0 * MODEL_SR):int(c1 * MODEL_SR)]
            log("gemini", f"  chunk {i+1}/{len(chunks)} ({c1-c0:.0f}s)")
            items = transcribe_chunk_waiting(seg, api_key, state)
            if items is None:
                failed += 1
                if state["stop"]:
                    break
                continue
            cf.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

        raw = []
        for it in items or []:
            try:
                s = max(float(it["start_sec"]) + c0, c0)
                e = min(float(it["end_sec"]) + c0, c1)
                t = str(it["text"]).strip()
            except Exception:
                continue
            if t and e > s:
                raw.append((s, e, t))
        raw.sort()
        sents.extend(snap_chunk(raw, regions, c0, c1))

    if not sents:
        return {"identifier": ident, "clips": 0, "minutes": 0.0,
                "failed_chunks": failed, "complete": failed == 0}

    enforce_no_overlap(sents)
    final, seen = [], set()
    for s in sents:
        k = (round(s.start_sec, 2), round(s.end_sec, 2))
        if k in seen or not (MIN_CLIP <= s.duration <= MAX_CLIP):
            continue
        seen.add(k)
        final.append(s)

    y_hi = resample(y_raw, sr, TARGET_SR)
    total = len(y_hi) / TARGET_SR
    rows, prev_b = [], 0.0
    for n, s in enumerate(final):
        nxt = final[n + 1].start_sec if n + 1 < len(final) else total
        a = max(0.0, prev_b, s.start_sec - PAD)
        b = min(total, nxt, s.end_sec + PAD)
        if b - a < MIN_CLIP:
            continue
        prev_b = b
        cid = f"{stem}__{n:04d}"
        if KEEP_CLIPS:
            save_wav(out_dir / "clips" / f"{cid}.wav",
                     y_hi[int(a * TARGET_SR):int(b * TARGET_SR)], TARGET_SR)
        rows.append({"clip_id": cid, "identifier": ident,
                     "source_file": src.name,
                     "start_sec": round(a, 3), "end_sec": round(b, 3),
                     "duration": round(b - a, 3),
                     "text": s.text, "snapped": s.snapped})

    if rows:
        with (out_dir / "manifest.csv").open("w", newline="",
                                             encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    mins = sum(r["duration"] for r in rows) / 60.0
    ov = sum(1 for x, y in zip(rows, rows[1:])
             if y["start_sec"] < x["end_sec"] - 1e-6)
    if ov:
        warn("verify", f"  {ov} overlapping pair(s) -- audio duplicated")
    log("episode", f"  {len(rows)} clips, {mins:.1f} min "
                   f"({100*mins/(total/60):.0f}% coverage)")
    return {"identifier": ident, "clips": len(rows), "minutes": round(mins, 2),
            "failed_chunks": failed, "complete": failed == 0, "rows": rows}


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    ensure_deps()
    api_key = get_api_key()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    adopt_previous_output()

    idents = list(IDENTIFIERS)
    if SEARCH_QUERY:
        log("search", SEARCH_QUERY)
        found = ia_search(SEARCH_QUERY, SEARCH_MAX)
        log("search", f"matched {len(found)} item(s)")
        idents = list(dict.fromkeys(idents + found))

    # Flatten to (identifier, file) pairs. One archive.org ITEM can hold many
    # episodes -- "muwan-palassa" holds all 201 -- so the unit of work is a
    # FILE, not an item.
    work: list[tuple[str, dict]] = []
    for ident in idents:
        try:
            files, lic = ia_pick_audio(ident)
        except Exception as ex:
            warn("plan", f"{ident}: metadata failed ({ex}); skipping")
            continue
        if lic != "ABSENT":
            log("plan", f"{ident}: license {lic}")
        for f in sorted(files, key=lambda x: x["name"]):
            work.append((ident, f))

    def already_done(ident, fname) -> bool:
        stem = re.sub(r"\W+", "_", Path(fname).stem).strip("_")
        return (OUT_ROOT / stem / "manifest.csv").exists()

    todo = [(i, f) for i, f in work if not already_done(i, f["name"])]
    skipped = len(work) - len(todo)
    hours = sum(float(f.get("length", 0) or 0) for _, f in todo) / 3600
    log("plan", f"{len(work)} episode(s) total; {skipped} already done; "
                f"{len(todo)} to do (~{hours:.1f} h audio)")
    est = sum(max(1, round(float(f.get('length', 0) or 0) / CHUNK_SEC))
              for _, f in todo)
    log("plan", f"~{est} API requests needed; free tier gives roughly "
                f"{len(MODELS)*20}/day")
    if MAX_EPISODES:
        todo = todo[:MAX_EPISODES]
        log("plan", f"capped to {len(todo)} episode(s) this run")

    state = {"dead": set(), "last": [0.0], "requests": 0, "stop": "",
             "waits": 0, "deadline": time.time() + SESSION_HOURS * 3600}
    summary, all_rows = [], []

    for n, (ident, f) in enumerate(todo, 1):
        if state["stop"]:
            break
        if time.time() > state["deadline"]:
            state["stop"] = "session"
            log("plan", "session time budget reached; stopping cleanly")
            break
        log("episode", f"[{n}/{len(todo)}] {ident} :: {f['name']}")
        try:
            src = ia_download(ident, f["name"], RAW_ROOT / ident / f["name"])
        except Exception as ex:
            warn("episode", f"  download failed ({ex}); skipping")
            continue
        try:
            res = build_episode(src, ident, api_key, state)
        except Exception as ex:
            warn("episode", f"  build failed ({type(ex).__name__}: {ex})")
            continue
        all_rows.extend(res.pop("rows", []))
        summary.append(res)
        if DELETE_RAW_AFTER:
            src.unlink(missing_ok=True)

    # Re-read every manifest so the combined CSV covers previous runs too, not
    # just what this session happened to produce.
    all_rows = []
    for man in sorted(OUT_ROOT.glob("*/manifest.csv")):
        try:
            all_rows.extend(csv.DictReader(man.open(encoding="utf-8-sig")))
        except Exception:
            pass

    if all_rows:
        with (OUT_ROOT / "all_manifests.csv").open("w", newline="",
                                                   encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps({"episodes": summary, "requests_used": state["requests"],
                    "models_exhausted": sorted(state["dead"])},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    total_eps = len(list(OUT_ROOT.glob("*/manifest.csv")))
    total_min = sum(float(r["duration"]) for r in all_rows) / 60.0
    print("\n" + "=" * 64)
    print(f"  THIS RUN   episodes {len(summary)} | requests {state['requests']}"
          f" | quota waits {state['waits']}")
    print(f"  CUMULATIVE episodes {total_eps}/{len(work)} | clips "
          f"{len(all_rows)} | audio {total_min/60:.1f} h")
    if state["dead"]:
        print(f"  models spent : {', '.join(sorted(state['dead']))}")
    print(f"  output       : {OUT_ROOT}")
    print("=" * 64)

    remaining = len(work) - total_eps
    if remaining > 0:
        print(f"\n  {remaining} episode(s) still to do.")
        if state["stop"] == "session":
            print("  Stopped on the session time budget, not on quota.")
        elif state["stop"] == "quota":
            print("  Stopped on quota (WAIT_FOR_QUOTA is off).")
        print("\n  TO CONTINUE TOMORROW:")
        print("   1. Save Version now (File -> Save Version -> Save & Run All)")
        print("   2. Next run: Add Data -> Your Work -> this notebook's output")
        print("   3. Run again. Finished episodes are adopted and skipped free.")
    else:
        print("\n  ALL EPISODES COMPLETE.")
    print("\n  /kaggle/working is wiped between sessions -- Save Version to keep it.")


if __name__ == "__main__":
    main()
