#!/usr/bin/env python3
"""
sinhala_stt_bench.py -- side-by-side Sinhala STT bench for radio-drama audio.

WHY THIS EXISTS
---------------
The main pipeline transcribes with faster-whisper large-v3. Whisper's Sinhala
training data is thin, so on expressive radio-drama speech the output is often
unusable. Before rebuilding the pipeline around a different ASR, we need
evidence about (a) which engine is least bad and (b) whether the AUDIO reaching
the ASR is itself damaged.

Two things are compared at once:

  * AUDIO VARIANT  -- what front-end the clip went through before ASR.
      raw     : the source audio, untouched (music/SFX still present)
      demucs  : htdemucs "vocals" stem -- what the pipeline does TODAY. htdemucs
                is a MUSIC source-separation model; running it on speech is a
                plausible cause of mangled clips, so it is on trial here.
      denoise : DeepFilterNet speech enhancement (trained on speech, not music)
  * ENGINE         -- which ASR model produced the text.

Segmentation runs ONCE on the raw audio and the SAME time windows are cut from
every variant, so a difference in the output is attributable to the front-end
and not to different clip boundaries.

There is no Sinhala ground truth here, so this tool does NOT score anything. It
emits a bench.html where each row is one clip: an audio player plus one column
per engine. A Sinhala speaker listens and reads. That is the measurement.

VERIFIED MODEL AVAILABILITY (checked against the model files, not model cards):
  * facebook/seamless-m4t-v2-large  -- NO Sinhala. Its 98 text output languages
    include npi/urd/tam/mal but not 'sin'.
  * facebook/mms-1b-all / -l1107 / -fl102 -- NO Sinhala adapter ('sin' absent
    from all three vocab.json files).
  So the open-weight field is Whisper (which does list 'si') plus community
  Sinhala fine-tunes, which are small read-speech models. Expectations low.

RUNS LOCALLY ON CPU. No GPU, no ffmpeg required: mp3 is decoded through
libsndfile via soundfile. Budget a few minutes per engine for ~15 short clips;
large-v3 in int8 is the slow one. Everything also works unchanged on a GPU box.

USAGE
    1. Paste a free Gemini key into GEMINI_API_KEY below
       (https://aistudio.google.com/apikey -- no credit card)
    2. python sinhala_stt_bench.py
    3. Open work/bench/bench.html, play each clip, read across the columns.

That is the whole flow: with no arguments it finds the episode under work/raw/,
takes 6 clips from a 5-minute slice, and transcribes them with Gemini.

To compare against the open-weight models instead (slow: multi-GB downloads):
    python sinhala_stt_bench.py --engines whisper-large-v3,whisper-small-si-hlasith
To run the whole episode:
    python sinhala_stt_bench.py --duration-sec 0 --num-clips 40
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# =========================================================================== #
#  PASTE YOUR GEMINI API KEY HERE, THEN RUN:   python sinhala_stt_bench.py
#
#  Get one free (no credit card) at: https://aistudio.google.com/apikey
# =========================================================================== #

GEMINI_API_KEY = ""        # <-- paste between the quotes

#  SECURITY: this file is tracked by git. A key pasted above WILL be committed
#  and pushed if you commit this file. Safer alternative: put the key in a file
#  called  api_key.txt  next to this script (already gitignored) and leave the
#  line above empty. Either works; the file is checked if the constant is blank.
#
#  If you do paste it here and later push, treat that key as burned and revoke
#  it at the link above.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODEL_SR = 16_000          # every ASR here expects 16 kHz mono
MIN_DUR = 2.0              # seconds; matches the main pipeline
MAX_DUR = 12.0

VARIANTS = ("raw", "demucs", "denoise")

# Set from --api-key so the engine factory can see it without threading the
# value through every call site.
_CLI_API_KEY = ""

# Run with no arguments and the bench picks these up.
DEFAULT_ENGINE_SET = "gemini-flash"
DEFAULT_NUM_CLIPS = 6
DEFAULT_OFFSET_SEC = 60.0     # skip the theme music at the head
DEFAULT_DURATION_SEC = 300.0  # a 5-minute slice is enough to judge quality


def log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)


def warn(stage: str, msg: str) -> None:
    print(f"[{stage}] WARNING: {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Engine registry
# --------------------------------------------------------------------------- #
#
# Each engine is a lazy factory returning a callable  fn(y16: np.ndarray) -> str.
# Loading is wrapped by the caller, so one broken repo cannot abort the bench.

@dataclass
class Engine:
    key: str
    repo: str
    kind: str                 # "faster-whisper" | "hf-whisper" | "hf-ctc" | "gemini"
    note: str = ""
    # Fallback for checkpoints that ship no usable preprocessor_config.json.
    # The feature extractor is identical across all fine-tunes of a given
    # Whisper size, so borrowing the base model's is safe; the fine-tune's own
    # tokenizer and weights are still used.
    base_repo: str = ""


ENGINES: dict[str, Engine] = {
    # --- free hosted API ----------------------------------------------------
    # The open-weight field for Sinhala is all small READ-SPEECH fine-tunes,
    # which is the wrong distribution for expressive multi-speaker radio drama.
    # A large multimodal model is the only free option with a real shot.
    # Free tier: 1500 requests/day, 15/min, no card, no expiry -- and one clip
    # is one request, so a whole episode fits inside a single day's quota.
    "gemini-flash": Engine(
        "gemini-flash", "gemini-2.5-flash", "gemini",
        "Gemini 2.5 Flash via the free AI Studio tier. Needs GEMINI_API_KEY. "
        "NOTE: free-tier data may be used to improve Google's products."),
    "gemini-flash-lite": Engine(
        "gemini-flash-lite", "gemini-2.5-flash-lite", "gemini",
        "Cheaper/faster Flash variant, same free quota. Try if Flash rate-limits."),

    # --- baseline: what the pipeline uses today -----------------------------
    "whisper-large-v3": Engine(
        "whisper-large-v3", "large-v3", "faster-whisper",
        "Current pipeline baseline. Sinhala is a low-resource language for Whisper."),
    "whisper-large-v3-turbo": Engine(
        "whisper-large-v3-turbo", "large-v3-turbo", "faster-whisper",
        "Distilled decoder; ~4x faster, usually a small accuracy loss."),

    # --- community Sinhala fine-tunes ---------------------------------------
    # All are student/hobby fine-tunes on READ speech. The best documented WER
    # is ~46% on the author's own clean test set, so treat these as candidates
    # to disprove rather than as expected winners.
    "xlsr-300m-si": Engine(
        "xlsr-300m-si",
        "SpideyDLK/wav2vec2-large-xls-r-300m-sinhala-low-LR-part1", "hf-ctc",
        "wav2vec2 XLS-R 300m CTC fine-tuned on Sinhala; no LM."),
    "w2v-bert-si": Engine(
        "w2v-bert-si",
        "janiduchamika/wav2vec2-bert-sinhala-general-185k", "hf-ctc",
        "w2v-BERT 2.0 CTC. Architecture that hit 1.79% WER in the literature, "
        "but this is not that checkpoint."),
    "whisper-small-si": Engine(
        "whisper-small-si",
        "Lingalingeswaran/whisper-small-sinhala_v3", "hf-whisper",
        "whisper-small fine-tune; author reports 46.5% WER on own read-speech test set."),
    "whisper-small-si-hlasith": Engine(
        "whisper-small-si-hlasith",
        "hlasith/whisper-sinhala-small", "hf-whisper",
        "whisper-small fine-tune. Card claims WER 130.9->61.7%, CER 244.4->14.9%, but "
        "on only 10 validation samples -- too few to mean anything. The CER/WER gap "
        "says characters land while word boundaries do not. Repo ships a misnamed "
        "'processor_config.json' and no feature extractor, so the base model's is used.",
        base_repo="openai/whisper-small"),
    "whisper-medium-si": Engine(
        "whisper-medium-si",
        "SPEAK-ASR/whisper-medium-si-merged", "hf-whisper",
        "whisper-medium Sinhala merge; undocumented."),
    "whisper-large-v2-si": Engine(
        "whisper-large-v2-si",
        "RRashmini/whisper-large-v2-sinhala", "hf-whisper",
        "whisper-large-v2 Sinhala fine-tune; undocumented."),
}

# Handy preset for comparing the open-weight field against each other. Pass it
# with:  --engines "$(...)"  -- or just copy the string.
OPEN_WEIGHT_SET = ("whisper-large-v3,xlsr-300m-si,"
                   "whisper-small-si,whisper-small-si-hlasith")


# The prompt does real work here. Without the constraints, a chat model
# happily returns "Here is the transcription:", romanises Sinhala, translates
# to English, or narrates that the audio is unclear -- all of which are useless
# as TTS ground truth and would silently pollute the dataset.
GEMINI_PROMPT = (
    "Transcribe this Sinhala audio clip verbatim.\n"
    "Rules:\n"
    "- Output ONLY the transcription, nothing else. No preamble, no notes, "
    "no explanation, no quotation marks.\n"
    "- Write in Sinhala script. Do NOT romanise and do NOT translate.\n"
    "- Transcribe exactly what is said, including false starts and repeats.\n"
    "- Do not describe sounds, music or speakers.\n"
    "- If no speech is audible, output nothing at all."
)

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/{model}:generateContent")


def resolve_api_key(cli_key: str = "") -> str:
    """Find the Gemini key. Order: --api-key, pasted constant, file, env."""
    import os
    if cli_key:
        return cli_key.strip()
    if GEMINI_API_KEY.strip():
        return GEMINI_API_KEY.strip()
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


def _load_gemini(eng: Engine):
    """Return fn(y16)->str backed by the Gemini API.

    Uses plain REST so the bench needs no extra SDK. Clips are a few seconds
    of 16 kHz mono, i.e. well under the ~20 MB inline-data ceiling, so no file
    upload step is required.
    """
    import requests

    api_key = resolve_api_key(_CLI_API_KEY)
    if not api_key:
        raise RuntimeError(
            "no Gemini API key found. Open this script and paste your key into "
            "GEMINI_API_KEY near the top (free, no card: "
            "https://aistudio.google.com/apikey)")

    url = GEMINI_URL.format(model=eng.repo)
    # Free tier is ~15 requests/min, so pace at 4s. Going faster just earns 429s.
    min_interval = 4.0
    last = [0.0]

    def run(y16):
        wav_b64 = base64.b64encode(_wav_bytes(y16, MODEL_SR)).decode("ascii")
        body = {
            "contents": [{"parts": [
                {"text": GEMINI_PROMPT},
                {"inline_data": {"mime_type": "audio/wav", "data": wav_b64}},
            ]}],
            # Deterministic: this is transcription, not creative writing.
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
        }
        for attempt in range(5):
            gap = time.time() - last[0]
            if gap < min_interval:
                time.sleep(min_interval - gap)
            last[0] = time.time()
            r = requests.post(url, json=body, timeout=120,
                              headers={"x-goog-api-key": api_key})
            if r.status_code == 200:
                data = r.json()
                cands = data.get("candidates") or []
                if not cands:
                    # Usually a safety block or an empty clip.
                    return ""
                parts = cands[0].get("content", {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts).strip()
            if r.status_code in (429, 500, 503):
                wait = min(8 * (attempt + 1), 40)
                warn("gemini", f"HTTP {r.status_code}; backing off {wait}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        raise RuntimeError("rate-limited after 5 attempts")

    return run


def _load_engine(eng: Engine, device: str, whisper_lang: str = "si"):
    """Return fn(y16)->str for the engine, or raise.

    `whisper_lang` controls how Whisper models are conditioned:
      "si"    -- force the Sinhala language token (the obvious choice)
      "en"    -- force English. Not as silly as it sounds: some Sinhala
                 fine-tunes are trained without setting the language, so the
                 model learns to emit Sinhala script under whatever token the
                 training script defaulted to. Which one a checkpoint expects
                 CANNOT be read off its config, so it is worth testing.
      "as-is" -- touch nothing; use whatever the checkpoint ships.
    """
    if eng.kind == "faster-whisper":
        from faster_whisper import WhisperModel
        compute = "float16" if device == "cuda" else "int8"
        model = WhisperModel(eng.repo, device=device, compute_type=compute)
        lang = None if whisper_lang == "as-is" else whisper_lang

        def run(y16):
            segs, _info = model.transcribe(
                y16, language=lang, beam_size=5, vad_filter=False)
            return "".join(s.text for s in segs).strip()
        return run

    if eng.kind == "gemini":
        return _load_gemini(eng)

    if eng.kind == "hf-whisper":
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        try:
            proc = AutoProcessor.from_pretrained(eng.repo)
        except Exception as ex:
            if not eng.base_repo:
                raise
            # Some community fine-tunes are broken as published. hlasith's, for
            # example, ships a misnamed 'processor_config.json' (so there is no
            # preprocessor_config.json to read) AND a tokenizer_config.json whose
            # extra_special_tokens is a LIST where transformers calls .keys() on
            # a dict. Both files are unusable, so take the whole processor from
            # the base model. That is only safe because these fine-tunes leave
            # the vocab untouched -- assert it rather than assume it.
            warn("engine", f"{eng.key}: processor unusable ({type(ex).__name__}); "
                           f"falling back to {eng.base_repo}")
            proc = AutoProcessor.from_pretrained(eng.base_repo)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            eng.repo,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

        # If the processor came from the base model, a vocab mismatch would
        # silently decode to the wrong characters. Fail loudly instead.
        tok_n = len(proc.tokenizer) if hasattr(proc, "tokenizer") else None
        cfg_n = getattr(model.config, "vocab_size", None)
        if tok_n and cfg_n and tok_n < cfg_n:
            raise RuntimeError(
                f"{eng.key}: tokenizer ({tok_n}) smaller than model vocab "
                f"({cfg_n}); the borrowed processor does not match this "
                f"checkpoint, so decoded text would be wrong")

        # NOTE: every stock whisper config.json carries a boilerplate
        # forced_decoder_ids of [[1,50259(<|en|>)],[2,50359],[3,50363]]. That is
        # NOT evidence the checkpoint was fine-tuned in English -- generate()
        # reads generation_config.json, where the language slot is normally left
        # open. Clear it anyway so it cannot fight an explicit language kwarg.
        gen_kwargs = {"max_new_tokens": 200}
        if whisper_lang != "as-is":
            model.generation_config.forced_decoder_ids = None
            try:
                model.generation_config.language = whisper_lang
                model.generation_config.task = "transcribe"
            except Exception:
                pass

        def run(y16):
            feats = proc(y16, sampling_rate=MODEL_SR,
                         return_tensors="pt").input_features
            feats = feats.to(device, model.dtype)
            with torch.no_grad():
                ids = model.generate(feats, **gen_kwargs)
            return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return run

    if eng.kind == "hf-ctc":
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        proc = AutoProcessor.from_pretrained(eng.repo)
        model = AutoModelForCTC.from_pretrained(eng.repo).to(device).eval()

        def run(y16):
            inputs = proc(y16, sampling_rate=MODEL_SR, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            return proc.batch_decode(ids)[0].strip()
        return run

    raise ValueError(f"unknown engine kind {eng.kind!r}")


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

def _load_audio(path, sr=None, mono=True, duration=None, offset=0.0):
    import librosa
    y, out_sr = librosa.load(str(path), sr=sr, mono=mono,
                             duration=duration, offset=offset)
    return y.astype("float32"), out_sr


def load_source_cached(src: Path, cache_dir: Path, offset: float,
                       duration: float | None):
    """Decode the source once and cache it as wav; reuse it on later runs.

    mp3 decode through libsndfile runs at only ~4x realtime here, so a 25-minute
    episode costs minutes on EVERY invocation. The bench is meant to be re-run
    while swapping engines, so that cost is paid once instead.

    Cached at the SOURCE sample rate (not 16 kHz) so the demucs variant, which
    wants the original rate, stays available without forcing a re-decode.
    """
    key = f"{src.stem}__off{offset:g}__dur{duration if duration else 'all'}.wav"
    cached = cache_dir / key
    if cached.exists():
        log("audio", f"cache hit: {cached.name}")
        return _load_audio(cached, sr=None, mono=True)
    log("audio", f"decoding {src.name} (cache miss -- this is the slow part)")
    t0 = time.time()
    y, sr = _load_audio(src, sr=None, mono=True, duration=duration, offset=offset)
    log("audio", f"decoded {len(y)/sr:.1f}s @ {sr} Hz in {time.time()-t0:.1f}s")
    _save_wav(cached, y, sr)
    log("audio", f"cached -> {cached}")
    return y, sr


def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def _wav_bytes(y, sr) -> bytes:
    """In-memory PCM16 wav, for embedding in the HTML report."""
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    y = np.clip(np.asarray(y, dtype="float32"), -1.0, 1.0)
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _save_wav(path: Path, y, sr) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_wav_bytes(y, sr))


# --------------------------------------------------------------------------- #
# Audio variants (front-ends on trial)
# --------------------------------------------------------------------------- #

def build_variant(name: str, y_raw, sr_raw, device: str):
    """Return (y16, MODEL_SR) for the whole recording under front-end `name`."""
    if name == "raw":
        return _resample(y_raw, sr_raw, MODEL_SR), MODEL_SR

    if name == "demucs":
        import torch
        from demucs.api import Separator
        log("variant", "loading demucs htdemucs (MUSIC separation -- on trial)")
        sep = Separator(model="htdemucs", device=device)
        wav = torch.from_numpy(y_raw if y_raw.ndim == 2 else y_raw[None, :])
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)          # htdemucs expects stereo
        _origin, stems = sep.separate_tensor(wav, sr_raw)
        vocals = stems["vocals"].mean(dim=0).cpu().numpy()
        return _resample(vocals, sr_raw, MODEL_SR), MODEL_SR

    if name == "denoise":
        from df.enhance import enhance, init_df
        import torch
        log("variant", "loading DeepFilterNet (SPEECH enhancement)")
        model, df_state, _ = init_df()
        y48 = _resample(y_raw, sr_raw, 48_000)
        out = enhance(model, df_state,
                      torch.from_numpy(y48).unsqueeze(0)).squeeze(0).cpu().numpy()
        return _resample(out, 48_000, MODEL_SR), MODEL_SR

    raise ValueError(f"unknown variant {name!r}")


# --------------------------------------------------------------------------- #
# Segmentation (once, on raw -- shared by every variant)
# --------------------------------------------------------------------------- #

def pick_clips(y16, num_clips: int) -> list[tuple[float, float]]:
    """Silero VAD over the raw audio; return `num_clips` windows spread evenly.

    Spreading matters: radio dramas open with a music/announcer bed, so taking
    the first N speech regions would benchmark the theme tune.
    """
    import torch
    # Two ways in. torch.hub needs network + GitHub + torchaudio; the pip
    # package needs neither GitHub nor a working hub cache. Try the pip package
    # first because it is the one that fails least often.
    model = get_speech_timestamps = None
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps as _gst
        model, get_speech_timestamps = load_silero_vad(), _gst
        log("segment", "loaded Silero VAD (pip package)")
    except Exception as e:
        log("segment", f"silero-vad pip package unavailable ({e}); trying torch.hub")
        try:
            model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                          trust_repo=True)
            get_speech_timestamps = utils[0]
            log("segment", "loaded Silero VAD (torch.hub)")
        except Exception as e2:
            raise RuntimeError(
                f"could not load Silero VAD by either route ({e2}). "
                f"Fix with:  pip install silero-vad torchaudio") from e2

    speech = get_speech_timestamps(
        torch.from_numpy(y16), model, sampling_rate=MODEL_SR,
        min_speech_duration_ms=int(MIN_DUR * 1000),
        max_speech_duration_s=MAX_DUR,
    )
    windows = []
    for sp in speech:
        a, b = sp["start"] / MODEL_SR, sp["end"] / MODEL_SR
        if MIN_DUR <= (b - a) <= MAX_DUR:
            windows.append((round(a, 3), round(b, 3)))
    log("segment", f"{len(windows)} VAD windows in range [{MIN_DUR},{MAX_DUR}]s")
    if not windows:
        return []
    if len(windows) <= num_clips:
        return windows
    step = len(windows) / num_clips
    return [windows[int(i * step)] for i in range(num_clips)]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; line-height: 1.5; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { opacity: .7; font-size: 13px; margin-bottom: 20px; }
.legend { border: 1px solid rgba(128,128,128,.35); border-radius: 8px;
          padding: 12px 16px; margin-bottom: 24px; font-size: 13px; }
.legend code { font-size: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid rgba(128,128,128,.35); padding: 8px 10px;
         vertical-align: top; text-align: left; }
th { position: sticky; top: 0; background: Canvas; font-weight: 600; }
td.si { font-family: "Iskoola Pota", "Noto Sans Sinhala", sans-serif;
        font-size: 16px; min-width: 260px; }
td.meta { white-space: nowrap; font-variant-numeric: tabular-nums; }
audio { width: 220px; }
.empty { opacity: .45; font-style: italic; }
.err { color: #c0392b; font-size: 12px; }
tr:nth-child(even) td { background: rgba(128,128,128,.06); }
.vtag { display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 4px;
        border: 1px solid rgba(128,128,128,.5); }
"""


def write_html(out: Path, rows: list[dict], engine_keys: list[str],
               variants: list[str], meta: dict, embed: bool) -> None:
    e = html.escape
    p = []
    p.append(f"<title>Sinhala STT bench &mdash; {e(meta['source_name'])}</title>")
    p.append(f"<style>{CSS}</style>")
    p.append(f"<h1>Sinhala STT bench &mdash; {e(meta['source_name'])}</h1>")
    p.append(f"<div class='sub'>{len(rows)} clip-rows &middot; "
             f"variants: {e(', '.join(variants))} &middot; "
             f"engines: {e(', '.join(engine_keys))} &middot; "
             f"generated {e(meta['generated'])}</div>")

    p.append("<div class='legend'><b>How to read this.</b> Play the clip, then "
             "compare the engine columns. Two separate questions:<br>"
             "1. <b>Is the audio itself intelligible?</b> If you cannot make out "
             "the words, no ASR can &mdash; that is a segmentation/front-end bug, "
             "not a model choice. Compare the same <code>clip</code> across "
             "<code>variant</code> rows to see what the front-end did to it.<br>"
             "2. <b>Which engine is closest?</b> Compare columns within a row.")
    notes = [f"<code>{e(k)}</code> &mdash; {e(ENGINES[k].note)}"
             for k in engine_keys if k in ENGINES]
    if notes:
        p.append("<br><br>" + "<br>".join(notes))
    p.append("</div>")

    p.append("<div class='scroll'><table><thead><tr>")
    p.append("<th>clip</th><th>variant</th><th>time</th><th>audio</th>")
    for k in engine_keys:
        p.append(f"<th>{e(k)}</th>")
    p.append("</tr></thead><tbody>")

    for r in rows:
        p.append("<tr>")
        p.append(f"<td class='meta'>{e(r['clip'])}</td>")
        p.append(f"<td class='meta'><span class='vtag'>{e(r['variant'])}</span></td>")
        p.append(f"<td class='meta'>{r['start_sec']:.1f}&ndash;{r['end_sec']:.1f}s</td>")
        if embed and r.get("wav_b64"):
            p.append("<td><audio controls preload='none' "
                     f"src='data:audio/wav;base64,{r['wav_b64']}'></audio></td>")
        else:
            rel = e(r.get("wav_rel", ""))
            p.append(f"<td><audio controls preload='none' src='{rel}'></audio></td>")
        for k in engine_keys:
            txt = r["text"].get(k, "")
            if txt.startswith("<<ERROR"):
                p.append(f"<td class='err'>{e(txt)}</td>")
            elif txt:
                p.append(f"<td class='si'>{e(txt)}</td>")
            else:
                p.append("<td class='empty'>(empty)</td>")
        p.append("</tr>")
    p.append("</tbody></table></div>")

    out.write_text("\n".join(p), encoding="utf-8")


def write_csv(out: Path, rows: list[dict], engine_keys: list[str]) -> None:
    cols = ["clip", "variant", "start_sec", "end_sec", "duration"] + engine_keys
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["clip"], r["variant"], r["start_sec"], r["end_sec"],
                        round(r["end_sec"] - r["start_sec"], 3)]
                       + [r["text"].get(k, "") for k in engine_keys])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def resolve_device(force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Side-by-side Sinhala STT bench (audio variants x engines).")
    ap.add_argument("--audio", help="source audio file; default: auto-detect "
                                    "the episode under work/raw/")
    ap.add_argument("--api-key", default="",
                    help="Gemini API key (overrides the one pasted in the script)")
    ap.add_argument("--out", default="work/bench", help="output directory")
    ap.add_argument("--num-clips", type=int, default=DEFAULT_NUM_CLIPS)
    # Default is raw-only: on CPU, Demucs over a full episode takes far longer
    # than the ASR itself, and it is the front-end under suspicion anyway.
    # Add it back with --variants raw,demucs once the ASR question is settled.
    ap.add_argument("--variants", default="raw",
                    help=f"comma list from {','.join(VARIANTS)}")
    ap.add_argument("--engines", default=DEFAULT_ENGINE_SET,
                    help=f"comma list from {','.join(ENGINES)}")
    ap.add_argument("--whisper-lang", default="si", choices=["si", "en", "as-is"],
                    help="language token forced on Whisper models (see _load_engine)")
    ap.add_argument("--offset-sec", type=float, default=DEFAULT_OFFSET_SEC,
                    help="skip this much of the head (theme music)")
    ap.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC,
                    help="only analyse this many seconds (0 = whole episode)")
    ap.add_argument("--no-embed-audio", action="store_true",
                    help="reference clip wavs on disk instead of inlining base64")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--list-engines", action="store_true")
    ap.add_argument("--list-gemini-models", action="store_true",
                    help="ask the API which Gemini models your key can use "
                         "(names change; this avoids guessing)")
    args = ap.parse_args(argv)

    if args.list_engines:
        for k, eng in ENGINES.items():
            print(f"{k:24s} {eng.repo:58s} {eng.note}")
        return 0

    if args.list_gemini_models:
        import requests
        api_key = resolve_api_key(args.api_key)
        if not api_key:
            print("no API key. Paste one into GEMINI_API_KEY at the top of "
                  "this script -- free, no card: "
                  "https://aistudio.google.com/apikey", file=sys.stderr)
            return 2
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key}, timeout=60)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
            return 1
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                print(f"  {m['name'].removeprefix('models/')}")
        return 0

    global _CLI_API_KEY
    _CLI_API_KEY = args.api_key
    if not args.duration_sec:        # 0 (or 0.0) means "the whole episode"
        args.duration_sec = None

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    engine_keys = [k.strip() for k in args.engines.split(",") if k.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v!r}; choose from {VARIANTS}")
    for k in engine_keys:
        if k not in ENGINES:
            ap.error(f"unknown engine {k!r}; --list-engines to see options")

    # Fail before decoding audio or downloading weights, not 10 minutes in.
    if any(ENGINES[k].kind == "gemini" for k in engine_keys):
        if not resolve_api_key(args.api_key):
            print("\n  No Gemini API key found.\n\n"
                  "  1. Get one free (no credit card):"
                  " https://aistudio.google.com/apikey\n"
                  "  2. Open this script and paste it into the line near the top:\n"
                  '       GEMINI_API_KEY = "your-key-here"\n'
                  "  3. Run again:  python sinhala_stt_bench.py\n\n"
                  "  (Or keep it out of git: save the key in api_key.txt "
                  "next to this script.)\n", file=sys.stderr)
            return 2

    if args.audio:
        src = Path(args.audio)
    else:
        found = find_default_audio()
        if not found:
            ap.error("no audio found under work/raw/; pass --audio explicitly")
        src = found
        log("bench", f"auto-detected audio: {src}")
    if not src.exists():
        ap.error(f"audio not found: {src}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.cpu)
    embed = not args.no_embed_audio
    log("bench", f"device={device}  source={src.name}")

    # ---- load source once, at native rate (demucs wants the original sr) ----
    log("audio", f"loading {src.name} "
                 f"(offset={args.offset_sec}s dur={args.duration_sec})")
    y_raw, sr_raw = load_source_cached(src, out_dir / "cache",
                                       args.offset_sec, args.duration_sec)
    log("audio", f"{len(y_raw)/sr_raw:.1f}s @ {sr_raw} Hz")

    # ---- segment ONCE on raw; identical windows applied to every variant ----
    y16_raw = _resample(y_raw, sr_raw, MODEL_SR)
    windows = pick_clips(y16_raw, args.num_clips)
    if not windows:
        warn("bench", "VAD found no speech windows; nothing to benchmark")
        return 1
    log("segment", f"selected {len(windows)} clips spread across the recording")

    # ---- build each audio variant over the whole recording -----------------
    variant_audio: dict[str, "object"] = {}
    for v in variants:
        try:
            if v == "raw":
                variant_audio[v] = y16_raw
            else:
                y16, _ = build_variant(v, y_raw, sr_raw, device)
                variant_audio[v] = y16
            log("variant", f"{v}: ready")
        except Exception as ex:
            warn("variant", f"{v} unavailable ({ex}); skipping this variant")
    if not variant_audio:
        warn("bench", "no audio variant could be built")
        return 1

    # ---- cut clips ---------------------------------------------------------
    rows: list[dict] = []
    clips_dir = out_dir / "clips"
    for v, y16 in variant_audio.items():
        for i, (a, b) in enumerate(windows):
            seg = y16[int(a * MODEL_SR): int(b * MODEL_SR)]
            if len(seg) < int(MIN_DUR * MODEL_SR):
                continue
            name = f"clip{i:03d}__{v}.wav"
            row = {"clip": f"clip{i:03d}", "variant": v,
                   "start_sec": a + args.offset_sec,
                   "end_sec": b + args.offset_sec,
                   "audio": seg, "text": {},
                   "wav_rel": f"clips/{name}"}
            _save_wav(clips_dir / name, seg, MODEL_SR)
            if embed:
                row["wav_b64"] = base64.b64encode(
                    _wav_bytes(seg, MODEL_SR)).decode("ascii")
            rows.append(row)
    log("clips", f"cut {len(rows)} clip-rows into {clips_dir}")

    # ---- run engines -------------------------------------------------------
    # Engines are loaded one at a time and released before the next, so a T4
    # with 15 GB can bench large-v3 and a large-v2 fine-tune in the same run.
    ok_engines: list[str] = []
    for k in engine_keys:
        eng = ENGINES[k]
        log("engine", f"loading {k} ({eng.repo})")
        t0 = time.time()
        try:
            run = _load_engine(eng, device, args.whisper_lang)
        except Exception as ex:
            warn("engine", f"{k} failed to load: {ex}")
            traceback.print_exc()
            for r in rows:
                r["text"][k] = f"<<ERROR load: {type(ex).__name__}>>"
            ok_engines.append(k)
            continue
        log("engine", f"{k} loaded in {time.time()-t0:.1f}s; transcribing "
                      f"{len(rows)} clips")
        t0 = time.time()
        for n, r in enumerate(rows, 1):
            try:
                r["text"][k] = run(r["audio"])
            except Exception as ex:
                r["text"][k] = f"<<ERROR {type(ex).__name__}: {ex}>>"
            if n % 10 == 0:
                log("engine", f"  {k}: {n}/{len(rows)}")
        log("engine", f"{k} done in {time.time()-t0:.1f}s")
        ok_engines.append(k)
        del run
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- report ------------------------------------------------------------
    for r in rows:
        r.pop("audio", None)
    meta = {"source_name": src.name,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device, "variants": list(variant_audio),
            "engines": ok_engines, "num_clips": len(windows)}
    write_csv(out_dir / "bench.csv", rows, ok_engines)
    write_html(out_dir / "bench.html", rows, ok_engines,
               list(variant_audio), meta, embed)
    (out_dir / "bench_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    log("bench", f"DONE -> {out_dir/'bench.html'}  (and bench.csv)")
    log("bench", "Listen to each clip FIRST. If the audio is unintelligible, "
                 "the problem is the front-end, not the engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
