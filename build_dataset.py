#!/usr/bin/env python3
"""
build_dataset.py -- sentence-level Sinhala clips + timestamps, from raw audio.

WHAT THIS PRODUCES
------------------
For one episode:
    dataset/<episode>/clips/<clip_id>.wav   one SENTENCE per file, 24 kHz mono
    dataset/<episode>/manifest.csv          clip_id, start_sec, end_sec, duration, text
    dataset/<episode>/review.html           play each clip next to its sentence

That manifest is the deliverable -- it describes clips by timestamp so anyone can
regenerate the audio from the archive.org source. The wavs are for YOUR review and
training; do not redistribute them.

HOW THE TIMESTAMPS ARE MADE ACCURATE
------------------------------------
Asking an LLM for timestamps and trusting them produces clips that start
mid-syllable: the text is right but the numbers drift by hundreds of ms. Asking a
VAD for sentences does not work either -- a VAD hears pauses, not grammar, so one
speech region can hold three sentences or half of one.

So each source is used for what it is actually reliable at:

    Gemini  ->  WHAT was said, and roughly WHERE  (sentence text + rough times)
    VAD     ->  EXACTLY where speech starts and stops (acoustic evidence)

Every proposed sentence span is then SNAPPED onto the speech regions it overlaps:
the clip begins at the start of the first region it touches and ends at the end of
the last. The text comes from the model; the cut points come from the audio. A
sentence whose span touches no speech region at all is kept but flagged
`snapped=False` in the manifest, because that is the signature of a hallucinated
timestamp and those clips should be reviewed before use.

USAGE
    python build_dataset.py                      # first 5 min, quick check
    python build_dataset.py --duration-sec 0     # the whole episode
    python build_dataset.py --audio path/to.mp3 --out dataset

Needs a free Gemini key in api_key.txt next to this script (or --api-key).
Get one at https://aistudio.google.com/apikey -- no credit card.

For many episodes at once, use kaggle_build_dataset.py instead: it adds
archive.org downloading and runs where the network is fast.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MODEL_SR = 16_000         # what the VAD and the API see

TARGET_SR = 24_000        # TTS-ready output rate
MIN_CLIP = 1.0            # drop anything shorter; too little for a TTS reference
MAX_CLIP = 15.0           # drop anything longer; likely a merge failure
PAD = 0.10                # seconds of breathing room each side
SNAP_TOL = 1.0            # how far a proposed boundary may be from real speech
HIFI_MIN_SR = 22_050      # at/above this the source keeps sibilant energy

# The binding constraint is REQUESTS PER DAY, not audio minutes. The free tier
# allows only ~20 generateContent calls per day PER MODEL (the API reports
# `GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20`). Widely-quoted
# "1500 requests/day" figures do not apply to this model.
#
# So chunks are big: at 300 s a 25-minute episode costs 6 requests instead of 25.
# Audio is sent as FLAC, which roughly halves the payload and keeps a 5-minute
# chunk well inside the ~20 MB inline-data ceiling.
CHUNK_TARGET = 300.0

# Quota is per model, so exhausting one does not exhaust the next. Tried in
# order; a model that reports its daily limit is retired for the rest of the run.
DEFAULT_MODELS = ("gemini-3.6-flash,gemini-3.5-flash,gemini-flash-latest,"
                  "gemini-3-flash-preview,gemini-2.5-flash")

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


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def warn(stage: str, msg: str) -> None:
    print(f"[{stage}] WARNING: {msg}", file=sys.stderr, flush=True)


def resolve_api_key(cli_key: str = "") -> str:
    """Find the Gemini key. Order: --api-key, api_key.txt, environment."""
    import os
    if cli_key:
        return cli_key.strip()
    keyfile = Path(__file__).with_name("api_key.txt")
    if keyfile.exists():
        txt = keyfile.read_text(encoding="utf-8").strip()
        if txt:
            return txt
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY") or "").strip()


def find_default_audio() -> Path | None:
    """Locate the downloaded episode so the script runs with no arguments."""
    root = Path(__file__).with_name("work") / "raw"
    if not root.is_dir():
        return None
    for ext in ("*.mp3", "*.ogg", "*.flac", "*.wav", "*.m4a"):
        found = sorted(root.rglob(ext))
        if found:
            return found[0]
    return None


def _load_audio(path, sr=None, mono=True, duration=None, offset=0.0):
    import librosa
    y, out_sr = librosa.load(str(path), sr=sr, mono=mono,
                             duration=duration, offset=offset)
    return y.astype("float32"), out_sr


def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def _wav_bytes(y, sr) -> bytes:
    """In-memory PCM16 wav, for embedding players in the review page."""
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    y = np.clip(np.asarray(y, dtype="float32"), -1.0, 1.0)
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _save_wav(path: Path, y, sr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_wav_bytes(y, sr))


def load_source_cached(src: Path, cache_dir: Path, offset: float,
                       duration: float | None):
    """Decode the source once and cache it as wav; reuse it on later runs.

    mp3 decode runs well below realtime on a cold start, and this script is
    meant to be re-run while adjusting chunking, so the cost is paid once.
    Cached at the SOURCE sample rate so clip export stays full quality.
    """
    key = f"{src.stem}__off{offset:g}__dur{duration if duration else 'all'}.wav"
    cached = cache_dir / key
    if cached.exists():
        log("audio", f"cache hit: {cached.name}")
        return _load_audio(cached, sr=None, mono=True)
    log("audio", f"decoding {src.name} (cache miss)")
    t0 = time.time()
    y, sr = _load_audio(src, sr=None, mono=True, duration=duration, offset=offset)
    log("audio", f"decoded {len(y)/sr:.1f}s @ {sr} Hz in {time.time()-t0:.1f}s")
    cached.parent.mkdir(parents=True, exist_ok=True)
    _save_wav(cached, y, sr)
    return y, sr


class QuotaExhausted(RuntimeError):
    """A model's per-DAY free-tier allowance is gone; waiting will not help."""


@dataclass
class Sentence:
    start_sec: float
    end_sec: float
    text: str
    snapped: bool = False
    chunk: int = 0

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# --------------------------------------------------------------------------- #
# VAD -- the acoustic ground truth for boundaries
# --------------------------------------------------------------------------- #

def speech_regions(y16) -> list[tuple[float, float]]:
    """Absolute (start, end) seconds of every speech region Silero finds."""
    import torch
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        model = load_silero_vad()
    except Exception:
        model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                      trust_repo=True)
        get_speech_timestamps = utils[0]
    ts = get_speech_timestamps(torch.from_numpy(y16), model,
                               sampling_rate=MODEL_SR,
                               min_speech_duration_ms=200)
    return [(t["start"] / MODEL_SR, t["end"] / MODEL_SR) for t in ts]


def make_chunks(regions, target=CHUNK_TARGET) -> list[tuple[float, float]]:
    """Group speech regions into ~`target`-second chunks, cutting only in silence.

    Cutting mid-word would make the model transcribe half a syllable at each
    seam, so chunk boundaries always fall in the gap BETWEEN speech regions.
    """
    if not regions:
        return []
    chunks = []
    cur_start, cur_end = regions[0]
    for start, end in regions[1:]:
        if end - cur_start > target:
            chunks.append((cur_start, cur_end))
            cur_start = start
        cur_end = end
    chunks.append((cur_start, cur_end))
    return chunks


def snap_chunk(raw: list[tuple[float, float, str]], regions, c0: float, c1: float
               ) -> list[Sentence]:
    """Snap a chunk's sentences onto speech regions, WITHOUT overlapping.

    The obvious implementation -- expand each sentence to the first and last
    region it touches -- is wrong: a two-word line like "sit down, sit down"
    sits inside a busy stretch, touches its neighbours' regions too, and
    balloons into an 8-second clip containing three speakers. Sentences then
    overlap and the same audio ships several times under different transcripts.

    So instead of each sentence claiming regions independently, every speech
    region is awarded to EXACTLY ONE sentence -- whichever the model's rough
    span overlaps most. A sentence's clip is then the span of the regions it
    actually won. Because regions are disjoint and singly owned, clips cannot
    duplicate audio. A region no sentence overlaps is left out entirely: that
    is untranscribed speech, not part of any sentence.

    A sentence that wins no region keeps its model span and is flagged
    `snapped=False` -- unverified cut points, for review rather than training.
    """
    inside = [r for r in regions if r[1] > c0 and r[0] < c1]
    owner: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(raw))}
    # Assignment is MONOTONIC: regions are walked in time order and the sentence
    # pointer never moves backwards. Over a 5-minute chunk the model's absolute
    # timestamps drift by seconds, and without this a drifted sentence can claim
    # a region from far away, interleaving two speakers. Speech and sentences
    # are both strictly ordered in time, so order is the reliable signal even
    # when the numbers are not.
    cur = 0
    for r in inside:
        best_i, best_ov = None, 0.0
        for i in range(cur, len(raw)):
            s, e, _ = raw[i]
            if s > r[1]:                # sentences are sorted; no later one fits
                break
            ov = min(r[1], e) - max(r[0], s)
            if ov > best_ov:
                best_ov, best_i = ov, i
        if best_i is not None:          # no overlap with any sentence -> drop
            owner[best_i].append(r)
            cur = best_i

    out: list[Sentence] = []
    for i, (s, e, text) in enumerate(raw):
        mine = owner[i]
        if mine:
            out.append(Sentence(mine[0][0], mine[-1][1], text, True))
        else:
            out.append(Sentence(s, e, text, False))
    return out


def _absorb(items, c0: float, c1: float, regions,
            sents: list[Sentence], chunk_i: int) -> None:
    """Turn one chunk's raw API items into snapped Sentences, appended to `sents`.

    Shared by the fresh-request and cache-hit paths so a resumed run produces
    byte-identical output to an uninterrupted one.
    """
    raw: list[tuple[float, float, str]] = []
    for it in items or []:
        try:
            s = float(it["start_sec"]) + c0
            e = float(it["end_sec"]) + c0
            text = str(it["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not text or e <= s:
            continue
        # Clamp into the chunk before snapping: a model that overshoots the
        # clip length would otherwise pull in audio it never heard.
        s, e = max(s, c0), min(e, c1)
        if e - s > 0:
            raw.append((s, e, text))
    raw.sort()
    chunk_sents = snap_chunk(raw, regions, c0, c1)
    for cs in chunk_sents:
        cs.chunk = chunk_i
    sents.extend(chunk_sents)
    n_unsnapped = sum(1 for c in chunk_sents if not c.snapped)
    log("gemini", f"  -> {len(chunk_sents)} sentence(s)"
                  + (f" ({n_unsnapped} unsnapped)" if n_unsnapped else ""))


def enforce_no_overlap(sents: list[Sentence]) -> int:
    """Final guarantee: trim any residual overlap. Returns how many were trimmed.

    Region ownership already prevents overlap in the normal case, but a
    sentence that won a non-contiguous set of regions (or an unsnapped one
    keeping raw model times) can still straddle its neighbour. Trimming here
    means the invariant holds for every row in the manifest, not just most.
    """
    sents.sort(key=lambda x: (x.start_sec, x.end_sec))
    trimmed = 0
    for a, b in zip(sents, sents[1:]):
        if a.end_sec > b.start_sec:
            a.end_sec = b.start_sec
            trimmed += 1
    return trimmed


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

def _flac_bytes(y, sr) -> bytes:
    """FLAC-encode a chunk. Lossless, ~half the size of WAV, accepted by Gemini.

    Matters because the payload ceiling is what caps chunk length, and chunk
    length is what determines how many of the day's ~20 requests an episode
    costs.
    """
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    y = np.clip(np.asarray(y, dtype="float32"), -1.0, 1.0)
    sf.write(buf, y, sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def transcribe_chunk(y16_chunk, api_key: str, model: str,
                     min_interval: float, last: list) -> list[dict]:
    """One chunk -> list of {start_sec, end_sec, text} (times relative to chunk).

    Raises QuotaExhausted if `model` has spent its daily allowance, so the
    caller can move to the next model rather than burning retries.
    """
    import requests
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "audio/flac",
                             "data": base64.b64encode(
                                 _flac_bytes(y16_chunk, MODEL_SR)).decode("ascii")}},
        ]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    url = GEMINI_URL.format(model=model)
    for attempt in range(8):
        gap = time.time() - last[0]
        if gap < min_interval:
            time.sleep(min_interval - gap)
        last[0] = time.time()
        r = requests.post(url, json=body, timeout=300,
                          headers={"x-goog-api-key": api_key})
        if r.status_code == 200:
            cands = r.json().get("candidates") or []
            if not cands:
                return []
            parts = cands[0].get("content", {}).get("parts") or []
            raw = "".join(p.get("text", "") for p in parts).strip()
            if not raw:
                return []
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                # Schema mode makes this rare, but a truncated response is
                # still possible; salvage the array if we can find one.
                m = re.search(r"\[.*\]", raw, re.S)
                if not m:
                    warn("gemini", f"unparseable response: {raw[:120]}")
                    return []
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    warn("gemini", "response was truncated JSON; skipping chunk")
                    return []
            return data if isinstance(data, list) else []
        if r.status_code in (429, 500, 503):
            # A 429 body carries the server's own RetryInfo. Honouring it beats
            # guessing: the free tier's real limit is lower than the documented
            # RPM suggests, and blind exponential backoff just burns attempts.
            wait = 0.0
            try:
                for d in r.json().get("error", {}).get("details", []):
                    for v in d.get("violations", []):
                        # A PER-DAY violation will not clear by waiting, so stop
                        # retrying this model and let the caller switch.
                        if "PerDay" in str(v.get("quotaId", "")):
                            raise QuotaExhausted(
                                f"{model}: daily free-tier limit "
                                f"({v.get('quotaValue')}) reached")
                    rd = d.get("retryDelay")
                    if rd:
                        wait = float(str(rd).rstrip("s"))
            except QuotaExhausted:
                raise
            except Exception:
                pass
            wait = max(wait, 15.0 * (attempt + 1))
            warn("gemini", f"HTTP {r.status_code}; waiting {wait:.0f}s "
                           f"(attempt {attempt+1}/8)")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("rate-limited after 8 attempts")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui,-apple-system,"Segoe UI",sans-serif; margin:0;
       padding:24px; line-height:1.5; }
h1 { font-size:20px; margin:0 0 4px; }
.sub { opacity:.7; font-size:13px; margin-bottom:20px; }
.warnbox { border:1px solid #d98324; border-radius:8px; padding:10px 14px;
           margin-bottom:18px; font-size:13px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { border:1px solid rgba(128,128,128,.35); padding:8px 10px;
        vertical-align:top; text-align:left; }
th { position:sticky; top:0; background:Canvas; }
td.si { font-family:"Iskoola Pota","Noto Sans Sinhala",sans-serif; font-size:16px; }
td.meta { white-space:nowrap; font-variant-numeric:tabular-nums; }
audio { width:210px; }
tr.unsnapped td { background:rgba(217,131,36,.14); }
tr:nth-child(even) td { background:rgba(128,128,128,.06); }
"""


def write_review(path: Path, sents: list[Sentence], clip_b64: dict, title: str):
    e = html.escape
    bad = sum(1 for s in sents if not s.snapped)
    p = [f"<title>{e(title)} &mdash; sentence clips</title>", f"<style>{CSS}</style>",
         f"<h1>{e(title)}</h1>",
         f"<div class='sub'>{len(sents)} sentence clips &middot; "
         f"{sum(s.duration for s in sents)/60:.1f} min total</div>"]
    if bad:
        p.append(f"<div class='warnbox'><b>{bad} clip(s) highlighted below could not "
                 "be snapped to detected speech.</b> Their timestamps came only from "
                 "the model, so the cut points are unverified &mdash; check these "
                 "before using them.</div>")
    p.append("<table><thead><tr><th>clip</th><th>time</th><th>dur</th>"
             "<th>audio</th><th>sentence</th></tr></thead><tbody>")
    for s, cid in zip(sents, clip_b64):
        cls = "" if s.snapped else " class='unsnapped'"
        p.append(f"<tr{cls}>")
        p.append(f"<td class='meta'>{e(cid)}</td>")
        p.append(f"<td class='meta'>{s.start_sec:.2f}&ndash;{s.end_sec:.2f}s</td>")
        p.append(f"<td class='meta'>{s.duration:.2f}s</td>")
        p.append("<td><audio controls preload='none' "
                 f"src='data:audio/wav;base64,{clip_b64[cid]}'></audio></td>")
        p.append(f"<td class='si'>{e(s.text)}</td></tr>")
    p.append("</tbody></table>")
    path.write_text("\n".join(p), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", help="source audio; default: auto-detect under work/raw/")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="comma list tried in order; quota is per model, so a "
                         "later one still works when an earlier is exhausted")
    ap.add_argument("--offset-sec", type=float, default=60.0)
    ap.add_argument("--duration-sec", type=float, default=300.0,
                    help="0 = the whole episode")
    ap.add_argument("--chunk-sec", type=float, default=CHUNK_TARGET)
    # 15 RPM is the documented free-tier ceiling, but audio requests hit 429
    # well below it in practice, so pace conservatively by default. Failed
    # chunks are cached-around and retried on the next run either way.
    ap.add_argument("--rpm", type=float, default=6.0, help="requests per minute cap")
    ap.add_argument("--no-embed-audio", action="store_true")
    args = ap.parse_args(argv)

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print("no Gemini API key. Put it in api_key.txt next to this script, "
              "or pass --api-key. Free: https://aistudio.google.com/apikey",
              file=sys.stderr)
        return 2

    src = Path(args.audio) if args.audio else find_default_audio()
    if not src or not src.exists():
        print("no audio found; pass --audio", file=sys.stderr)
        return 2
    duration = args.duration_sec or None

    out_dir = Path(args.out) / src.stem.replace(" ", "_")
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    log("dataset", f"source: {src.name}")
    y_raw, sr_raw = load_source_cached(src, out_dir / ".cache",
                                       args.offset_sec, duration)
    y16 = _resample(y_raw, sr_raw, MODEL_SR)
    log("dataset", f"{len(y_raw)/sr_raw:.1f}s @ {sr_raw} Hz")

    regions = speech_regions(y16)
    log("vad", f"{len(regions)} speech regions")
    chunks = make_chunks(regions, args.chunk_sec)
    log("vad", f"grouped into {len(chunks)} chunk(s) of ~{args.chunk_sec:.0f}s")
    if not chunks:
        warn("dataset", "no speech found")
        return 1

    min_interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last = [0.0]
    sents: list[Sentence] = []

    # Transcriptions are cached per chunk so a run killed by rate limits can be
    # resumed by simply running the command again: completed chunks cost no
    # further quota, and only the gaps are retried.
    cache_dir = out_dir / ".chunks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed: list[int] = []
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    dead: set[str] = set()
    log("gemini", f"models (in order): {', '.join(models)}")

    for i, (c0, c1) in enumerate(chunks):
        cache_f = cache_dir / f"chunk_{i:04d}_{c0:.1f}-{c1:.1f}.json"
        if cache_f.exists():
            try:
                items = json.loads(cache_f.read_text(encoding="utf-8"))
                log("gemini", f"chunk {i+1}/{len(chunks)}  cached "
                              f"({len(items)} sentence(s))")
            except Exception:
                cache_f.unlink(missing_ok=True)
                items = None
            if items is not None:
                _absorb(items, c0, c1, regions, sents, i)
                continue

        seg = y16[int(c0 * MODEL_SR):int(c1 * MODEL_SR)]
        log("gemini", f"chunk {i+1}/{len(chunks)}  {c0:.1f}-{c1:.1f}s "
                      f"({c1-c0:.1f}s)")
        items = None
        for model in models:
            if model in dead:
                continue
            try:
                items = transcribe_chunk(seg, api_key, model, min_interval, last)
                if model != models[0]:
                    log("gemini", f"  (via {model})")
                break
            except QuotaExhausted as ex:
                warn("gemini", f"{ex}; switching model")
                dead.add(model)
            except Exception as ex:
                warn("gemini", f"chunk {i+1} on {model} failed "
                               f"({type(ex).__name__}: {ex})")
                break
        if items is None:
            if len(dead) >= len(models):
                warn("gemini", "every model is out of daily quota -- stopping. "
                               "Re-run tomorrow; finished chunks are cached.")
                failed.extend(range(i, len(chunks)))
                break
            failed.append(i)
            continue
        cache_f.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        _absorb(items, c0, c1, regions, sents, i)

    if not sents:
        warn("dataset", "no sentences produced")
        return 1

    trimmed = enforce_no_overlap(sents)
    if trimmed:
        log("filter", f"trimmed {trimmed} residual overlap(s)")

    # Drop duplicates created when two model sentences snap onto the same
    # region span, and enforce the length window.
    sents.sort(key=lambda x: (x.start_sec, x.end_sec))
    final: list[Sentence] = []
    seen: set[tuple[float, float]] = set()
    dropped = {"dupe": 0, "short": 0, "long": 0}
    for s in sents:
        key = (round(s.start_sec, 2), round(s.end_sec, 2))
        if key in seen:
            dropped["dupe"] += 1
            continue
        if s.duration < MIN_CLIP:
            dropped["short"] += 1
            continue
        if s.duration > MAX_CLIP:
            dropped["long"] += 1
            continue
        seen.add(key)
        final.append(s)

    log("filter", f"kept {len(final)}; dropped "
                  f"{dropped['dupe']} duplicate, {dropped['short']} <{MIN_CLIP}s, "
                  f"{dropped['long']} >{MAX_CLIP}s")

    # ---- export --------------------------------------------------------- #
    # Never upsample -- see kaggle_build_dataset.py. A 24 kHz header on 11 kHz
    # content is a lie that costs double the disk.
    out_sr = min(TARGET_SR, sr_raw)
    quality = "hifi" if sr_raw >= HIFI_MIN_SR else "lofi"
    y_hi = _resample(y_raw, sr_raw, out_sr)
    total = len(y_hi) / out_sr
    log("audio", f"source {sr_raw} Hz -> clips {out_sr} Hz [{quality}]")
    stem = src.stem.replace(" ", "_")
    rows, clip_b64 = [], {}
    prev_b = 0.0    # end of the previously EXPORTED clip, padding included
    for n, s in enumerate(final):
        # Padding must not eat into a neighbour, or the no-overlap guarantee
        # that region ownership just bought would be given straight back.
        # The lower bound has to be the previous clip's PADDED end: bounding
        # against its unpadded end lets the two pads meet in the middle and
        # overlap by up to 2*PAD.
        next_start = final[n + 1].start_sec if n + 1 < len(final) else total
        a = max(0.0, prev_b, s.start_sec - PAD)
        b = min(total, next_start, s.end_sec + PAD)
        if b - a < MIN_CLIP:
            continue
        prev_b = b
        clip = y_hi[int(a * out_sr):int(b * out_sr)]
        cid = f"{stem}__{n:04d}"
        _save_wav(clips_dir / f"{cid}.wav", clip, out_sr)
        if not args.no_embed_audio:
            clip_b64[cid] = base64.b64encode(
                _wav_bytes(_resample(clip, out_sr, MODEL_SR), MODEL_SR)
            ).decode("ascii")
        else:
            clip_b64[cid] = ""
        rows.append({
            "clip_id": cid,
            "source_file": src.name,
            # Absolute seconds in the ORIGINAL file, so the manifest stands
            # alone: offset is added back in.
            "start_sec": round(a + args.offset_sec, 3),
            "end_sec": round(b + args.offset_sec, 3),
            "duration": round(b - a, 3),
            "text": s.text,
            "snapped": s.snapped,
            "sample_rate": out_sr, "quality": quality,
        })

    man = out_dir / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    write_review(out_dir / "review.html", final, clip_b64, src.stem)

    # Self-check: clip minutes must not exceed source minutes. If they do, the
    # clips overlap and the same audio is shipping more than once.
    mins = sum(r["duration"] for r in rows) / 60.0
    overlaps = sum(1 for x, y in zip(rows, rows[1:])
                   if y["start_sec"] < x["end_sec"] - 0.001)
    if overlaps:
        warn("verify", f"{overlaps} overlapping clip pair(s) -- audio is duplicated")
    else:
        log("verify", f"no overlaps; {mins:.1f} clip-min from "
                      f"{total/60:.1f} source-min "
                      f"({100*mins/(total/60):.0f}% speech coverage)")

    unsnapped = sum(1 for r in rows if not r["snapped"])
    if failed:
        warn("dataset", f"{len(failed)} chunk(s) never transcribed "
                        f"({', '.join(str(f+1) for f in failed)}). "
                        f"Re-run the SAME command to fill only those gaps -- "
                        f"finished chunks are cached and cost no quota.")
    log("dataset", f"DONE. {len(rows)} clips, {mins:.1f} min -> {out_dir}")
    log("dataset", f"  manifest : {man}")
    log("dataset", f"  review   : {out_dir/'review.html'}")
    if unsnapped:
        warn("dataset", f"{unsnapped} clip(s) have unverified timestamps "
                        f"(snapped=False) -- check them in review.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
