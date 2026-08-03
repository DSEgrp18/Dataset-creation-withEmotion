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

# TWO QUALITY TIERS, and the order here matters -- hi-fi is built FIRST so the
# valuable audio lands before quota or session time runs out.
#
#   hi-fi : 16 single-episode uploads, 44.1 kHz / 64-128 kbps, 6.8 h total.
#           Real bandwidth; suitable for TTS voice quality.
#   lo-fi : "muwan-palassa" -- the whole serial, 201 files, 82 h, but every one
#           is 11025 Hz / 16 kbps. Nyquist caps it at 5.5 kHz, so fricatives and
#           sibilants are largely gone. Useful for prosody and emotional range,
#           marginal for final voice quality.
#
# Both are built and every clip is tagged (`quality`, `sample_rate`) so training
# can filter. NOTE: some hi-fi uploads duplicate episodes inside the collection,
# so filter to ONE tier before training or the same speech appears twice.
IDENTIFIERS = [
    # --- hi-fi first ---
    "muwan-palassa-70113", "muwan-palassa-140113", "muwan-palassa-210113",
    "muwan-palassa-280113", "muwan-palassa-040213", "muwan-palassa-110213",
    "muwan-palassa-180213", "MuwanPalassa29816", "MuwanPalassa22816",
    "MuwanPalassa12916", "MuwanPalassa19916", "MuwanPalassa101016",
    "MuwanPalassa51216", "muwanpalassa_03613", "muwanpalassa_20513",
    "muwanpalassa_27513",
    # --- then the full lo-fi serial ---
    "muwan-palassa",
]

# A clip is "hifi" at or above this source rate. 22050 Hz puts the ceiling at
# 11 kHz, which covers the sibilant energy 11025 Hz audio loses.
HIFI_MIN_SR = 22_050

# Optional: pull more items automatically. e.g. 'title:("Muwan Palassa")'
# finds 17 items. Leave "" to use IDENTIFIERS only.
SEARCH_QUERY = ""
SEARCH_MAX = 50

OUT_ROOT = Path("/kaggle/working/dataset")
RAW_ROOT = Path("/kaggle/working/raw")

# ---- running the full corpus across several days ------------------------- #
# ~217 episodes x ~5 requests is ~1090 requests against a ~100/day allowance,
# so the build spans roughly a fortnight of short runs. Resuming is what carries
# it: /kaggle/working is wiped between sessions, so Save Version at the end and
# attach that output as a Data source next run. Anything found under
# /kaggle/input is adopted before starting and skipped for free.
#
# STOP as soon as every model reports its daily cap, write the output, and let
# the run finish. Waiting was tried and measured: the full ~100-request daily
# allowance is spent in about 90 minutes, and the free-tier cap resets at a
# fixed time rather than trickling back, so retrying just burns session hours
# for nothing. One short commit run per day is the real cadence.
WAIT_FOR_QUOTA = False
QUOTA_POLL_MIN = 20.0      # only used if WAIT_FOR_QUOTA is turned back on

SESSION_HOURS = 11.0       # hard stop before Kaggle kills the session (~12 h)
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

# Target clip length. Raising the floor to 3 s would THROW AWAY roughly a third
# of sentences (measured median 4.3 s, minimum 1.1 s), so short neighbours are
# merged up to MAX_CLIP instead of discarded -- text is joined too.
MIN_CLIP, MAX_CLIP = 3.0, 30.0

# Two sentences are only merged if the silence between them is under this.
# CAUTION: nothing here diarizes speakers, so merging across a short pause in
# dialogue can put TWO VOICES in one clip, which is poor as a TTS reference.
# Lower this toward 0.3 for safer (but shorter) clips; raise it for longer ones.
MERGE_GAP = 0.6

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


class ProjectDenied(RuntimeError):
    """HTTP 403 -- the whole Google Cloud project is blocked, not rate limited.

    Google's automated system flags free-tier projects, often wrongly. Every
    model fails identically and no amount of retrying or model-switching helps,
    so the run must stop rather than download 200 more episodes to fail on.
    """


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


def get_api_keys() -> list[str]:
    """Every key we can find, in priority order.

    MULTIPLE KEYS ARE WORTH HAVING. Quota is per PROJECT, so a key from a second
    project brings its own daily allowance -- and if one project gets hit by
    Google's automated 403 block, the run continues on the other instead of
    stopping dead.

    Sources, all merged:
      * Kaggle Secrets  GEMINI_API_KEY, GEMINI_API_KEY_2 .. _5
      * env             GEMINI_API_KEY / GOOGLE_API_KEY
      * api_key.txt     ONE KEY PER LINE (blank lines and #comments ignored)
    """
    keys: list[str] = []

    def add(k):
        k = (k or "").strip()
        if k and not k.startswith("#") and k not in keys:
            keys.append(k)

    try:
        from kaggle_secrets import UserSecretsClient
        us = UserSecretsClient()
        for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                     "GEMINI_API_KEY_4", "GEMINI_API_KEY_5"):
            try:
                add(us.get_secret(name))
            except Exception:
                pass          # not every slot will exist; that is fine
    except Exception:
        pass

    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        add(os.environ.get(var))

    f = Path("api_key.txt")
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            add(line)

    if not keys:
        raise SystemExit(
            "No API key.\n"
            "  Kaggle: Add-ons -> Secrets -> add GEMINI_API_KEY, then re-run.\n"
            "  Free key (no card): https://aistudio.google.com/apikey")
    return keys


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


def merge_short(sents: list[Sentence]) -> list[Sentence]:
    """Join adjacent short sentences so clips land in the MIN_CLIP..MAX_CLIP band.

    Simply dropping everything under MIN_CLIP would discard about a third of the
    corpus, and the discarded part is the emotionally dense part -- exclamations,
    interjections, one-word replies are exactly the short ones. Merging keeps
    them, with the texts concatenated in order.

    Guards: only merges across a gap under MERGE_GAP, and never past MAX_CLIP.
    A merged clip is `snapped` only if every part was, so an unverified
    timestamp taints the whole clip rather than hiding inside it.
    """
    out: list[Sentence] = []
    for s in sents:
        if out:
            prev = out[-1]
            gap = s.start_sec - prev.end_sec
            too_short = prev.duration < MIN_CLIP or s.duration < MIN_CLIP
            fits = (s.end_sec - prev.start_sec) <= MAX_CLIP
            if too_short and fits and 0 <= gap <= MERGE_GAP:
                prev.end_sec = s.end_sec
                prev.text = (prev.text + " " + s.text).strip()
                prev.snapped = prev.snapped and s.snapped
                continue
        out.append(Sentence(s.start_sec, s.end_sec, s.text, s.snapped))
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
        if r.status_code == 403:
            raise ProjectDenied(r.text[:200])
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("rate-limited after 8 attempts")


def transcribe_chunk_waiting(seg, state):
    """Transcribe one chunk, failing over between models and WAITING for quota.

    Returns the parsed items, or None if the run should stop (session budget
    spent, or waiting is disabled).

    The daily allowance is per model, so exhausting one only retires that one.
    When all are retired the function sleeps and clears the graveyard, because
    a per-day cap does clear -- just not soon. Rather than predict Google's
    reset moment (which the API never states), it simply retries every
    QUOTA_POLL_MIN and lets a successful call be the proof that quota returned.
    """
    keys = state["keys"]
    while True:
        # Walk every (key, model) pair. Quota is per project, so a second key
        # is a second allowance -- and a 403 on one project does not touch the
        # other. Keys are the outer loop so a healthy project is reached even
        # when the first one is entirely blocked.
        for ki, api_key in enumerate(keys):
            if ki in state["dead_keys"]:
                continue
            for model in MODELS:
                if (ki, model) in state["dead"]:
                    continue
                try:
                    items = call_gemini(seg, api_key, model,
                                        60.0 / RPM if RPM else 0.0,
                                        state["last"])
                    state["requests"] += 1
                    state["used_key"] = ki
                    return items
                except QuotaExhausted as ex:
                    warn("gemini", f"key{ki+1}: {ex}")
                    state["dead"].add((ki, model))
                except ProjectDenied:
                    # Blocks the whole project, so every model on this key
                    # fails identically. Retire the key, try the next one.
                    warn("gemini", f"key{ki+1}: project DENIED (403) -- "
                                   f"retiring this key for the run")
                    state["dead_keys"].add(ki)
                    break
                except Exception as ex:
                    warn("gemini", f"  {type(ex).__name__}: {ex}")
                    return None

        if len(state["dead_keys"]) >= len(keys):
            state["stop"] = "denied"
            return None
        # Every key/model pair is spent.
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


def write_aggregate() -> list[dict]:
    """Rebuild dataset/all_manifests.csv from every per-episode manifest.

    Reads from disk rather than from memory so the combined file covers
    PREVIOUS runs adopted from /kaggle/input too, not just this session.
    Returns the rows so callers can report on them.
    """
    rows: list[dict] = []
    for man in sorted(OUT_ROOT.glob("*/manifest.csv")):
        try:
            rows.extend(csv.DictReader(man.open(encoding="utf-8-sig")))
        except Exception:
            pass
    if rows:
        with (OUT_ROOT / "all_manifests.csv").open(
                "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def adopt_previous_output() -> int:
    """Copy whatever a previous run left under /kaggle/input into working.

    Kaggle wipes /kaggle/working between sessions, so without this a multi-day
    build restarts from zero. Everything cheap is taken verbatim -- the chunk
    cache (so no request is ever repeated), the manifest, and the .done marker.
    Clips are skipped: they are regenerable and would be gigabytes.

    Nothing is decided here. Whether an adopted episode still needs work is
    worked out later by resolve_done(), which can see how long the episode
    actually is.
    """
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return 0

    # Find episode folders by SEARCHING for their contents at any depth, rather
    # than assuming a layout. Kaggle mounts a notebook's output, an uploaded
    # dataset, and a zip-derived dataset at different depths, and a fixed glob
    # silently finds nothing -- which looks identical to "no previous run" and
    # quietly re-spends a whole day of quota.
    eps: dict[str, Path] = {}
    for marker in list(inp.rglob("manifest.csv")) + list(inp.rglob(".chunks")):
        ep = marker.parent
        if ep.is_dir() and ep.name not in eps:
            eps[ep.name] = ep

    adopted = 0
    for name, ep in sorted(eps.items()):
        dst = OUT_ROOT / name
        if dst.exists():
            continue
        src_chunks, src_man = ep / ".chunks", ep / "manifest.csv"
        dst.mkdir(parents=True, exist_ok=True)
        if src_chunks.is_dir():
            shutil.copytree(src_chunks, dst / ".chunks", dirs_exist_ok=True)
        if src_man.exists():
            shutil.copy2(src_man, dst / "manifest.csv")
        if (ep / ".done").exists():
            shutil.copy2(ep / ".done", dst / ".done")
        adopted += 1
    if adopted:
        log("resume", f"adopted {adopted} episode(s) from {inp}")
    elif eps:
        log("resume", f"{len(eps)} episode(s) already present; nothing to adopt")
    return adopted


def resolve_done(fname: str, length_sec: float) -> bool:
    """Is this episode already finished? Decided from what is on disk.

    Three cases, no configuration required:

      * a .done marker           -> finished (written by a completed run)
      * a manifest but no marker -> came from a run predating the marker, OR was
                                    cut off mid-episode by quota. Tell them apart
                                    by counting cached chunks against how many
                                    the episode's duration implies. A complete
                                    episode has them all, so backfill .done and
                                    skip it. A truncated one does not, so redo it
                                    -- its cached chunks make that nearly free.
      * neither                  -> not started.
    """
    stem = re.sub(r"\W+", "_", Path(fname).stem).strip("_")
    d = OUT_ROOT / stem
    if (d / ".done").exists():
        return True
    if not (d / "manifest.csv").exists():
        return False
    cached = len(list((d / ".chunks").glob("*.json"))) if (d / ".chunks").is_dir() else 0
    expected = max(1, round(length_sec / CHUNK_SEC))
    if cached >= expected:
        d.joinpath(".done").write_text("adopted", encoding="utf-8")
        return True
    log("resume", f"  {stem}: only {cached}/{expected} chunks cached "
                  f"-- finishing it")
    return False


# =========================================================================== #
# One episode
# =========================================================================== #

def build_episode(src: Path, ident: str, state: dict) -> dict:
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
            items = transcribe_chunk_waiting(seg, state)
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
    uniq, seen = [], set()
    for s in sents:
        k = (round(s.start_sec, 2), round(s.end_sec, 2))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    merged = merge_short(uniq)
    final = [s for s in merged if MIN_CLIP <= s.duration <= MAX_CLIP]
    log("filter", f"  {len(uniq)} sentences -> {len(merged)} after merge -> "
                  f"{len(final)} in {MIN_CLIP:.0f}-{MAX_CLIP:.0f}s "
                  f"({len(merged)-len(final)} outside the band)")

    # NEVER upsample. Writing 24 kHz from an 11025 Hz source doubles the file
    # size while adding exactly no information, and makes the header claim a
    # bandwidth the audio does not have. Clips keep the source rate when it is
    # below TARGET_SR, and `sample_rate` records the truth for the loader.
    out_sr = min(TARGET_SR, sr)
    quality = "hifi" if sr >= HIFI_MIN_SR else "lofi"
    y_out = resample(y_raw, sr, out_sr)
    total = len(y_out) / out_sr
    log("audio", f"  source {sr} Hz -> clips {out_sr} Hz [{quality}]")

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
                     y_out[int(a * out_sr):int(b * out_sr)], out_sr)
        rows.append({"clip_id": cid, "identifier": ident,
                     "source_file": src.name,
                     "start_sec": round(a, 3), "end_sec": round(b, 3),
                     "duration": round(b - a, 3),
                     "text": s.text, "snapped": s.snapped,
                     "sample_rate": out_sr, "quality": quality})

    if rows:
        with (out_dir / "manifest.csv").open("w", newline="",
                                             encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # A manifest alone does NOT mean the episode is finished. When quota runs
    # out mid-episode the transcribed chunks still produce clips, and treating
    # that partial manifest as "done" would skip the rest of the episode on
    # every future run -- silently losing most of it. Completion is recorded
    # separately, and only when every chunk actually came back.
    done_marker = out_dir / ".done"
    if failed == 0:
        done_marker.write_text(f"{len(rows)} clips", encoding="utf-8")
    else:
        done_marker.unlink(missing_ok=True)
        warn("episode", f"  INCOMPLETE ({failed} chunk(s) missing) -- will be "
                        f"finished on a later run")

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
    keys = get_api_keys()
    log("setup", f"{len(keys)} API key(s) available")
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

    todo = [(i, f) for i, f in work
            if not resolve_done(f["name"], float(f.get("length", 0) or 0))]
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

    state = {"dead": set(), "dead_keys": set(), "keys": keys,
             "last": [0.0], "requests": 0, "stop": "", "waits": 0,
             "used_key": 0,
             "deadline": time.time() + SESSION_HOURS * 3600}
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
            res = build_episode(src, ident, state)
        except Exception as ex:
            warn("episode", f"  build failed ({type(ex).__name__}: {ex})")
            continue
        all_rows.extend(res.pop("rows", []))
        summary.append(res)
        if DELETE_RAW_AFTER:
            src.unlink(missing_ok=True)
        # Rewrite the aggregate after EVERY episode. Writing it only at the end
        # meant an interrupted run left no combined manifest at all, even though
        # every per-episode file was already on disk.
        write_aggregate()

    all_rows = write_aggregate()
    (OUT_ROOT / "summary.json").write_text(
        json.dumps({"episodes": summary, "requests_used": state["requests"],
                    "models_exhausted": sorted(state["dead"])},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # Count FINISHED episodes, not merely started ones.
    total_eps = len(list(OUT_ROOT.glob("*/.done")))
    partial = len(list(OUT_ROOT.glob("*/manifest.csv"))) - total_eps
    total_min = sum(float(r["duration"]) for r in all_rows) / 60.0
    hifi = [r for r in all_rows if r.get("quality") == "hifi"]
    lofi = [r for r in all_rows if r.get("quality") == "lofi"]
    print("\n" + "=" * 64)
    print(f"  THIS RUN   episodes {len(summary)} | requests {state['requests']}"
          f" | quota waits {state['waits']}")
    print(f"  CUMULATIVE episodes {total_eps}/{len(work)} complete"
          + (f" (+{partial} partial, will resume)" if partial else "")
          + f" | clips {len(all_rows)} | audio {total_min/60:.1f} h")
    if hifi or lofi:
        hh = sum(float(r["duration"]) for r in hifi) / 3600
        lh = sum(float(r["duration"]) for r in lofi) / 3600
        print(f"  BY QUALITY hifi {len(hifi)} clips / {hh:.2f} h"
              f"  |  lofi {len(lofi)} clips / {lh:.2f} h")
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
