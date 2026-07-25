#!/usr/bin/env python3
"""build_emotional_sinhala_dataset.py

Staged, reproducible pipeline that turns Sri Lanka Broadcasting Corporation (SLBC)
radio dramas hosted on the Internet Archive into a cleaned, emotion-labeled,
TTS-ready Sinhala speech dataset -- described by a MANIFEST, not by redistributed
audio.

The source items are audio-only with NO transcripts, so text is produced by ASR.
Most items carry no license field, so the deliverable is MANIFEST-FIRST: we ship
identifiers + timestamps + this code, and each user regenerates the audio locally.

Stages (run one, several, or `all`):
    download   fetch mp3/ogg for identifiers/queries; log identifier + license
    separate   Demucs (htdemucs) -> keep the vocals stem (optional DeepFilterNet)
    diarize    pyannote speaker diarization -> per-speaker turns
    segment    Silero VAD -> 2-12 s single-speaker utterances on silence
    transcribe Sinhala-capable Whisper (faster-whisper large-v3) -> text + conf
    align      word timestamps + an alignment/word confidence score
    filter     drop clipped / noisy / low-confidence / overlapping clips
    emotion    audEERING wav2vec2 dimensional model -> arousal / valence (default)
    manifest   write manifest.csv + a yield & emotion-distribution report.json

Design notes:
  * Top-level imports are STDLIB ONLY. Every heavy dependency (torch, demucs,
    pyannote, whisper, ...) is imported lazily inside the stage that needs it, so
    `--stage download` and `--help` run on a bare Python with no ML stack.
  * Idempotent / resumable: every stage records progress in a per-item sidecar
    (work/meta/<id>.json) and skips work whose outputs already exist. This
    survives the /kaggle/working wipe -- rerun and it re-derives what is missing.
  * Kaggle-aware: pins CUDA_VISIBLE_DEVICES=0, reads the HF token from an env var
    or a Kaggle secret, and degrades gracefully when a gated model is unavailable.
  * Honest by construction: emotional speech degrades ASR, so high-arousal clips
    that fail ASR are routed to needs_manual_transcription.csv, never silently
    dropped. The report prints the funnel so the low yield is visible.

Usage:
    python build_emotional_sinhala_dataset.py --stage download --smoke \
        --identifiers muwan-palassa-140113
    python build_emotional_sinhala_dataset.py --stage all --smoke \
        --identifiers muwan-palassa-140113          # needs GPU + ML deps (Kaggle)

This is a research tool. Respect the copyright of the source recordings: keep the
output manifest-first and do not redistribute the audio.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants / tunables
# --------------------------------------------------------------------------- #

STAGES = [
    "download", "separate", "diarize", "segment",
    "transcribe", "align", "filter", "emotion", "manifest",
]

# Known Muwan Palassa (SLBC radio drama) items on archive.org, verified live.
DEFAULT_IDENTIFIERS = [
    "muwan-palassa-140113",
    "muwan-palassa-210113",
    "MuwanPalassa29816",
    "muwanpalassa_27513",
]

# archive.org file "format" strings we accept as source audio.
AUDIO_FORMAT_HINTS = ("mp3", "ogg vorbis", "ogg", "flac", "wav")

# Output audio spec (TTS-ready).
TARGET_SR = 24_000            # final clip sample rate (Hz)
MODEL_SR = 16_000             # sample rate fed to ASR / VAD / emotion / diarization
TARGET_LUFS = -24.0          # integrated loudness of final clips
CLIP_SUBTYPE = "PCM_16"       # 16-bit PCM wav

# Segmentation limits (seconds).
MIN_DUR = 2.0
MAX_DUR = 12.0

# Smoke mode caps.
SMOKE_AUDIO_SECONDS = 180.0   # only process the first N seconds of one episode
SMOKE_ASR_MODEL = "base"      # tiny/fast, quality irrelevant for a wiring test

# Filter thresholds (defaults; override via CLI).
DEFAULT_MIN_ALIGN_CONF = 0.50
DEFAULT_MIN_ASR_CONF = 0.55
DEFAULT_MIN_SNR_DB = 8.0
DEFAULT_MIN_DNSMOS = 2.6      # only applied if DNSMOS is available
DEFAULT_MAX_CLIP_FRAC = 0.01  # fraction of samples at full scale
# A clip that fails ASR but is emotionally salient is worth manual transcription.
MANUAL_AROUSAL = 0.60         # arousal >= this (0..1) => keep for manual review

# The speaker-diarization-3.1 pipeline pulls weights from SEVERAL gated repos
# (the pipeline config, the segmentation model, and -- in newer pyannote builds --
# the x-vector/PLDA embedding from community-1). Every one must be accepted with
# the SAME HuggingFace account, or diarization falls back to SPEAKER_UNK.
PYANNOTE_GATED_REPOS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/speaker-diarization-community-1",
]

REPORT_NOTES = [
    "Manifest-first: this report describes clips by identifier + timestamps. "
    "Source audio is NOT redistributed; regenerate it locally from archive.org.",
    "Radio dramas are multi-speaker with music/SFX; expect LOW yield (<10-20%) "
    "skewed toward calmer speech.",
    "Emotional speech (shouting/crying/whispering) degrades ASR. The most "
    "emotional clips often fail ASR and are routed to needs_manual_transcription.csv "
    "rather than discarded.",
    "WhisperX ships no Sinhala forced-alignment model; align_conf falls back to "
    "Whisper word probabilities for language 'si'. Treat align_conf accordingly.",
    "Pilot on ONE episode, review this report, THEN scale. Do not download the "
    "whole archive unprompted.",
]

MANIFEST_FIELDS = [
    "clip_id", "ia_identifier", "source_file", "start_sec", "end_sec", "duration",
    "speaker_id", "text", "asr_conf", "align_conf", "snr", "dnsmos",
    "arousal", "valence", "emotion_label",
]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

_T0 = time.time()


def log(stage: str, msg: str) -> None:
    """Timestamped, stage-tagged progress line (flushed for Kaggle logs)."""
    elapsed = time.time() - _T0
    print(f"[{elapsed:7.1f}s][{stage:9s}] {msg}", flush=True)


def warn(stage: str, msg: str) -> None:
    log(stage, "WARNING: " + msg)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    stage: str
    identifiers: list[str]
    search: str | None
    work_dir: Path
    smoke: bool
    device: str
    hf_token_env: str
    asr_model: str
    emotion_model: str        # "audeering" | "emotion2vec"
    diarization_model: str    # "auto" | explicit HF pipeline id
    denoise: bool
    aligner: str              # "auto" | "whisperx" | "whisper"
    max_minutes: float | None
    force: bool
    # filter thresholds
    min_align_conf: float
    min_asr_conf: float
    min_snr_db: float
    min_dnsmos: float
    max_clip_frac: float

    # derived paths -------------------------------------------------------- #
    @property
    def raw_dir(self) -> Path: return self.work_dir / "raw"
    @property
    def vocals_dir(self) -> Path: return self.work_dir / "vocals"
    @property
    def clips_dir(self) -> Path: return self.work_dir / "clips"
    @property
    def meta_dir(self) -> Path: return self.work_dir / "meta"
    @property
    def download_manifest(self) -> Path: return self.work_dir / "download_manifest.csv"
    @property
    def manifest_csv(self) -> Path: return self.work_dir / "manifest.csv"
    @property
    def needs_manual_csv(self) -> Path: return self.work_dir / "needs_manual_transcription.csv"
    @property
    def report_json(self) -> Path: return self.work_dir / "report.json"


def on_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def default_work_dir() -> Path:
    return Path("/kaggle/working/eesd") if on_kaggle() else Path("./work")


def build_config(args: argparse.Namespace) -> Config:
    idents = []
    if args.identifiers:
        idents = [x.strip() for x in args.identifiers.split(",") if x.strip()]
    if not idents and not args.search:
        idents = DEFAULT_IDENTIFIERS
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else default_work_dir().resolve()
    asr_model = args.asr_model or (SMOKE_ASR_MODEL if args.smoke else "large-v3")
    return Config(
        stage=args.stage,
        identifiers=idents,
        search=args.search,
        work_dir=work_dir,
        smoke=args.smoke,
        device=args.device,
        hf_token_env=args.hf_token_env,
        asr_model=asr_model,
        emotion_model=args.emotion_model,
        diarization_model=args.diarization_model,
        denoise=args.denoise,
        aligner=args.aligner,
        max_minutes=args.max_minutes,
        force=args.force,
        min_align_conf=args.min_align_conf,
        min_asr_conf=args.min_asr_conf,
        min_snr_db=args.min_snr_db,
        min_dnsmos=args.min_dnsmos,
        max_clip_frac=args.max_clip_frac,
    )


# --------------------------------------------------------------------------- #
# JSON / state helpers (atomic writes so a killed Kaggle session cannot corrupt)
# --------------------------------------------------------------------------- #

def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_meta(cfg: Config, ident: str) -> dict:
    p = cfg.meta_dir / f"{ident}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"ia_identifier": ident, "stages_done": [], "segments": []}


def save_meta(cfg: Config, meta: dict) -> None:
    _atomic_write_json(cfg.meta_dir / f"{meta['ia_identifier']}.json", meta)


def stage_done(meta: dict, stage: str) -> bool:
    return stage in meta.get("stages_done", [])


def mark_done(meta: dict, stage: str) -> None:
    if stage not in meta.setdefault("stages_done", []):
        meta["stages_done"].append(stage)


def all_meta(cfg: Config) -> list[dict]:
    if not cfg.meta_dir.exists():
        return []
    out = []
    for p in sorted(cfg.meta_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            warn("meta", f"skipping unreadable {p.name}")
    return out


def load_report(cfg: Config) -> dict:
    if cfg.report_json.exists():
        return json.loads(cfg.report_json.read_text(encoding="utf-8"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "smoke": cfg.smoke,
        "items": {},
        "funnel_minutes": {},
        "counts": {},
        "emotion_distribution": {},
        "licenses": {},
        "notes": REPORT_NOTES,
    }


def save_report(cfg: Config, report: dict) -> None:
    report["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _atomic_write_json(cfg.report_json, report)


# --------------------------------------------------------------------------- #
# HuggingFace token (env var or Kaggle secret); graceful when absent
# --------------------------------------------------------------------------- #

def get_hf_token(cfg: Config) -> str | None:
    for key in (cfg.hf_token_env, "HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = os.environ.get(key)
        if val:
            return val
    try:  # Kaggle secret
        from kaggle_secrets import UserSecretsClient  # type: ignore
        for name in (cfg.hf_token_env, "HF_TOKEN", "HUGGINGFACE_TOKEN"):
            try:
                val = UserSecretsClient().get_secret(name)
                if val:
                    return val
            except Exception:
                continue
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Audio helpers (lazy imports)
# --------------------------------------------------------------------------- #

def _load_audio(path, sr=None, mono=True, duration=None):
    """Return (samples float32, sr). Uses librosa (decodes mp3/ogg via ffmpeg)."""
    import librosa
    y, out_sr = librosa.load(str(path), sr=sr, mono=mono, duration=duration)
    return y.astype("float32"), out_sr


def _resample(y, orig_sr, target_sr):
    if orig_sr == target_sr:
        return y
    import librosa
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def _save_wav_pcm16(path, y, sr):
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(np.asarray(y, dtype="float32"), -1.0, 1.0)
    sf.write(str(path), y, sr, subtype=CLIP_SUBTYPE)


def _loudness_normalize(y, sr, target_lufs=TARGET_LUFS):
    """Loudness-normalize to target LUFS; returns peak-limited float32."""
    import numpy as np
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if loudness == float("-inf"):
            return y
        y = pyln.normalize.loudness(y, loudness, target_lufs)
    except Exception:
        pass
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype("float32")


def _clip_fraction(y):
    import numpy as np
    if len(y) == 0:
        return 1.0
    return float(np.mean(np.abs(y) >= 0.99))


def _estimate_snr_db(y, sr):
    """Rough frame-energy SNR: noise = low percentile, signal = high percentile."""
    import numpy as np
    if len(y) < sr // 10:
        return 0.0
    frame = max(1, int(0.025 * sr))
    hop = frame
    n = 1 + (len(y) - frame) // hop if len(y) >= frame else 1
    energies = np.array([
        float(np.mean(np.square(y[i * hop:i * hop + frame]))) for i in range(max(n, 1))
    ])
    energies = energies[energies > 0]
    if energies.size < 4:
        return 0.0
    noise = np.percentile(energies, 10)
    signal = np.percentile(energies, 90)
    if noise <= 0:
        return 40.0
    return float(10.0 * np.log10(max(signal / noise, 1e-9)))


def _dnsmos(y, sr):
    """Optional DNSMOS (overall MOS). Returns None if the package is missing."""
    try:
        from speechmos import dnsmos  # type: ignore
        import numpy as np
        y16 = _resample(y, sr, MODEL_SR) if sr != MODEL_SR else y
        res = dnsmos.run(np.asarray(y16, dtype="float32"), MODEL_SR)
        return float(res.get("ovrl_mos") or res.get("ovrl") or 0.0)
    except Exception:
        return None


def _audio_duration(path) -> float:
    """Duration in seconds, read from the header (no full decode)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def _parse_length(val) -> float:
    """archive.org 'length' is seconds ('1518.26') or 'MM:SS' / 'HH:MM:SS'."""
    if val is None:
        return 0.0
    s = str(val).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    try:
        return float(s)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# CUDA
# --------------------------------------------------------------------------- #

def pin_gpu0() -> None:
    """Kaggle gives T4 x2; pin to a single GPU for deterministic memory use."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def resolve_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ========================================================================== #
# STAGE 1 : download
# ========================================================================== #

def _select_audio_files(files: list[dict]) -> list[dict]:
    """Pick the source audio file(s), avoiding archive.org's derived duplicates.

    archive.org marks user uploads as source=='original' and auto-generated
    lower-bitrate copies as source=='derivative' (with a different filename, e.g.
    'ep_64kb.mp3'). We keep originals when present -- so we don't download both
    the original and its derivatives -- and only fall back to derivatives if an
    item has no originals. A final basename dedup guards against odd cases.
    """
    audio = []
    for f in files:
        fmt = str(f.get("format", "")).lower()
        name = str(f.get("name", ""))
        if any(h in fmt for h in AUDIO_FORMAT_HINTS) or name.lower().endswith((".mp3", ".ogg", ".flac", ".wav")):
            audio.append(f)
    originals = [f for f in audio if str(f.get("source", "")).lower() == "original"]
    pool = originals if originals else audio

    def rank(f):
        fmt = str(f.get("format", "")).lower()
        score = 0
        if "vbr" in fmt:
            score += 10
        if "flac" in fmt or "wav" in fmt:
            score += 5
        score += _parse_length(f.get("length")) / 10_000.0  # tie-break: longer
        return score

    groups: dict[str, dict] = {}
    for f in pool:
        base = Path(str(f.get("name", ""))).stem
        if base not in groups or rank(f) > rank(groups[base]):
            groups[base] = f
    return list(groups.values())


def stage_download(cfg: Config) -> None:
    import internetarchive as ia  # lazy: the only dep this stage needs

    identifiers = list(cfg.identifiers)
    if cfg.search:
        log("download", f"searching: {cfg.search}")
        found = [r["identifier"] for r in ia.search_items(cfg.search)]
        log("download", f"search matched {len(found)} item(s)")
        identifiers = list(dict.fromkeys(identifiers + found))
    if cfg.smoke:
        identifiers = identifiers[:1]
        log("download", f"SMOKE: limiting to 1 item -> {identifiers}")

    report = load_report(cfg)
    rows = []
    total_minutes = 0.0

    for ident in identifiers:
        log("download", f"item: {ident}")
        try:
            item = ia.get_item(ident)
        except Exception as e:
            warn("download", f"could not fetch metadata for {ident}: {e}")
            continue
        md = item.metadata or {}
        license_url = md.get("licenseurl", "ABSENT")
        rights = md.get("rights", "ABSENT")
        language = md.get("language", "?")
        files = list(item.files) if getattr(item, "files", None) else []
        audio_files = _select_audio_files(files)
        if cfg.smoke and audio_files:
            audio_files = [min(audio_files, key=lambda f: _parse_length(f.get("length")) or 1e9)]

        if not audio_files:
            warn("download", f"no audio files found in {ident}")
        log("download",
            f"  license={license_url} rights={rights} lang={language} "
            f"audio_files={len(audio_files)}")

        meta = load_meta(cfg, ident)
        meta.update({
            "license": license_url, "rights": rights, "language": language,
            "source_files": [],
        })
        item_minutes = 0.0
        dest = cfg.raw_dir / ident
        for f in audio_files:
            name = f["name"]
            length = _parse_length(f.get("length"))
            size = int(f.get("size", 0) or 0)
            local_path = dest / name
            already = local_path.exists() and local_path.stat().st_size > 0
            if already and not cfg.force:
                log("download", f"  skip (exists): {name}")
            else:
                log("download", f"  downloading: {name} ({size/1e6:.1f} MB, {length/60:.1f} min)")
                try:
                    ia.download(
                        ident, files=[name], destdir=str(cfg.raw_dir),
                        ignore_existing=True, verbose=False, retries=3,
                    )
                except Exception as e:
                    warn("download", f"  failed {name}: {e}")
                    continue
            item_minutes += length / 60.0
            meta["source_files"].append({
                "name": name, "path": str(local_path), "format": f.get("format"),
                "length_sec": length, "size": size,
            })
            rows.append({
                "ia_identifier": ident, "source_file": name, "format": f.get("format"),
                "duration_sec": round(length, 2), "size_bytes": size,
                "license": license_url, "rights": rights, "language": language,
                "local_path": str(local_path), "downloaded": local_path.exists(),
            })
        mark_done(meta, "download")
        save_meta(cfg, meta)
        total_minutes += item_minutes
        report["items"][ident] = {
            "license": license_url, "rights": rights, "language": language,
            "n_audio_files": len(audio_files),
            "minutes_downloaded": round(item_minutes, 2),
        }
        report["licenses"][ident] = license_url

    # write download_manifest.csv
    if rows:
        cfg.download_manifest.parent.mkdir(parents=True, exist_ok=True)
        with cfg.download_manifest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log("download", f"wrote {cfg.download_manifest} ({len(rows)} file rows)")

    report.setdefault("funnel_minutes", {})["downloaded"] = round(total_minutes, 2)
    save_report(cfg, report)
    log("download", f"DONE. {len(rows)} file(s), {total_minutes:.1f} minutes total.")
    _print_license_warning(report)


def _print_license_warning(report: dict) -> None:
    absent = [k for k, v in report.get("licenses", {}).items() if v in (None, "", "ABSENT")]
    if absent:
        log("download",
            f"NOTE: {len(absent)} item(s) have NO license field -> treat as "
            f"COPYRIGHTED. Keep the dataset manifest-first (do not redistribute "
            f"audio): {', '.join(absent)}")


# ========================================================================== #
# STAGE 2 : separate (Demucs -> vocals stem)
# ========================================================================== #

def _iter_items_with_stage(cfg: Config, prev_stage: str):
    """Yield (meta) for items that have completed prev_stage."""
    metas = all_meta(cfg)
    for meta in metas:
        if prev_stage and not stage_done(meta, prev_stage):
            warn("pipe", f"{meta['ia_identifier']} has no '{prev_stage}' output; skipping")
            continue
        yield meta


def stage_separate(cfg: Config) -> None:
    import torch
    from demucs.api import Separator

    device = resolve_device(cfg)
    log("separate", f"loading demucs htdemucs on {device}")
    sep = Separator(model="htdemucs", device=device)

    report = load_report(cfg)
    kept_minutes = 0.0
    for meta in _iter_items_with_stage(cfg, "download"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "separate") and not cfg.force:
            for sf in meta.get("source_files", []):
                kept_minutes += sf.get("length_sec", 0) / 60.0
            log("separate", f"skip (done): {ident}")
            continue
        for sf in meta.get("source_files", []):
            src = Path(sf["path"])
            if not src.exists():
                warn("separate", f"missing raw file {src}; run download first")
                continue
            out = cfg.vocals_dir / ident / (Path(sf["name"]).stem + ".wav")
            if out.exists() and not cfg.force:
                log("separate", f"  skip (exists): {out.name}")
                sf["vocals_path"] = str(out)
                kept_minutes += sf.get("length_sec", 0) / 60.0
                continue
            dur = SMOKE_AUDIO_SECONDS if cfg.smoke else None
            log("separate", f"  {ident}: loading {src.name} (dur={dur})")
            y, sr = _load_audio(src, sr=None, mono=False, duration=dur)
            wav = torch.from_numpy(y if y.ndim == 2 else y[None, :])
            if wav.shape[0] == 1:  # demucs expects stereo; duplicate mono
                wav = wav.repeat(2, 1)
            log("separate", f"  {ident}: separating ({wav.shape[1]/sr:.1f}s)...")
            _origin, stems = sep.separate_tensor(wav, sr)
            vocals = stems["vocals"].mean(dim=0).cpu().numpy()  # -> mono
            if cfg.denoise:
                vocals = _deepfilter(vocals, sr)
            _save_wav_pcm16(out, _resample(vocals, sr, TARGET_SR), TARGET_SR)
            sf["vocals_path"] = str(out)
            sf["vocals_sr"] = TARGET_SR
            kept_minutes += (len(vocals) / sr) / 60.0
            log("separate", f"  wrote {out}")
        mark_done(meta, "separate")
        save_meta(cfg, meta)
    report.setdefault("funnel_minutes", {})["after_separate"] = round(kept_minutes, 2)
    save_report(cfg, report)
    log("separate", f"DONE. ~{kept_minutes:.1f} vocal-minutes.")


def _deepfilter(y, sr):
    """Optional DeepFilterNet denoise; returns input unchanged if unavailable."""
    try:
        from df.enhance import enhance, init_df  # type: ignore
        import torch
        model, df_state, _ = init_df()
        y48 = _resample(y, sr, 48_000)
        t = torch.from_numpy(y48).unsqueeze(0)
        out = enhance(model, df_state, t).squeeze(0).cpu().numpy()
        return _resample(out, 48_000, sr)
    except Exception as e:
        warn("separate", f"DeepFilterNet unavailable ({e}); skipping denoise")
        return y


# ========================================================================== #
# STAGE 3 : diarize (pyannote)
# ========================================================================== #

def stage_diarize(cfg: Config) -> None:
    token = get_hf_token(cfg)
    if not token:
        warn("diarize",
             "no HF token (env or Kaggle secret) -> cannot load gated "
             "pyannote/speaker-diarization-3.1. Marking every clip SPEAKER_UNK "
             "and continuing (segmentation will not be speaker-pure).")
    # 'auto' walks the candidates until one loads: 3.1 first (it is what the
    # pipeline was designed around), then community-1, which newer pyannote
    # builds use and which is self-contained rather than pulling 3.1's deps.
    candidates = ([cfg.diarization_model] if cfg.diarization_model != "auto"
                  else ["pyannote/speaker-diarization-3.1",
                        "pyannote/speaker-diarization-community-1"])
    pipeline = None
    if token:
        try:
            from pyannote.audio import Pipeline
            import torch
        except Exception as e:
            warn("diarize", f"pyannote.audio not importable ({e}); using SPEAKER_UNK")
            Pipeline = None
        if Pipeline is not None:
            for model_id in candidates:
                try:
                    # pyannote renamed the auth kwarg (use_auth_token -> token);
                    # try the modern name first, fall back to the legacy one.
                    try:
                        pipeline = Pipeline.from_pretrained(model_id, token=token)
                    except TypeError:
                        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
                    if pipeline is None:
                        # pyannote returns None rather than raising when the gated
                        # terms have not been accepted.
                        raise RuntimeError("from_pretrained returned None (gated access)")
                    pipeline.to(torch.device(resolve_device(cfg)))
                    log("diarize", f"loaded {model_id}")
                    break
                except Exception as e:
                    warn("diarize", f"{model_id} failed: {e}")
                    pipeline = None
        if pipeline is None:
            warn("diarize", "no diarization pipeline loaded -> using SPEAKER_UNK "
                            "(clips will NOT be speaker-pure)")
            warn("diarize", "accept the gated terms for ALL of these, with the "
                            "SAME HF account that issued your token:")
            for repo in PYANNOTE_GATED_REPOS:
                warn("diarize", f"    https://huggingface.co/{repo}")

    for meta in _iter_items_with_stage(cfg, "separate"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "diarize") and not cfg.force:
            log("diarize", f"skip (done): {ident}")
            continue
        for sf in meta.get("source_files", []):
            vpath = sf.get("vocals_path")
            if not vpath or not Path(vpath).exists():
                warn("diarize", f"no vocals for {ident}; run separate first")
                continue
            turns = []
            if pipeline is not None:
                log("diarize", f"  {ident}: diarizing {Path(vpath).name}")
                try:
                    diar = pipeline(vpath)
                    # pyannote >= 4 returns a DiarizeOutput wrapper rather than a
                    # bare Annotation, so itertracks() may live on an attribute.
                    ann = diar
                    if not hasattr(ann, "itertracks"):
                        for attr in ("speaker_diarization", "diarization",
                                     "annotation", "output"):
                            cand = getattr(diar, attr, None)
                            if hasattr(cand, "itertracks"):
                                ann = cand
                                break
                    if not hasattr(ann, "itertracks"):
                        public = [a for a in dir(diar) if not a.startswith("_")]
                        raise RuntimeError(
                            f"no Annotation inside {type(diar).__name__}; "
                            f"attributes seen: {public[:12]}")
                    for turn, _, spk in ann.itertracks(yield_label=True):
                        turns.append({"start": float(turn.start),
                                      "end": float(turn.end), "speaker": str(spk)})
                except Exception as e:
                    warn("diarize", f"  diarization failed ({e}); SPEAKER_UNK")
            if not turns:
                # One turn spanning the ACTUAL separated audio -- not the full
                # source length. Using length_sec here made --smoke (which keeps
                # only the first few minutes) report a diarize duration longer
                # than the audio that exists, corrupting the yield funnel.
                dur = _audio_duration(vpath) or float(sf.get("length_sec") or 0.0)
                turns = [{"start": 0.0, "end": dur, "speaker": "SPEAKER_UNK"}]
            sf["diarization"] = turns
            log("diarize", f"  {ident}: {len(turns)} turn(s)")
        mark_done(meta, "diarize")
        save_meta(cfg, meta)
    log("diarize", "DONE.")


# ========================================================================== #
# STAGE 4 : segment (Silero VAD within speaker turns -> 2-12 s clips)
# ========================================================================== #

def _split_long(start, end, max_dur):
    """Split [start,end] into <=max_dur chunks (even split)."""
    dur = end - start
    if dur <= max_dur:
        return [(start, end)]
    n = int(dur // max_dur) + 1
    step = dur / n
    return [(start + i * step, start + (i + 1) * step) for i in range(n)]


def stage_segment(cfg: Config) -> None:
    import torch
    log("segment", "loading Silero VAD")
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    get_speech_timestamps = utils[0]

    report = load_report(cfg)
    diar_minutes = 0.0
    vad_minutes = 0.0
    for meta in _iter_items_with_stage(cfg, "diarize"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "segment") and not cfg.force:
            for s in meta.get("segments", []):
                vad_minutes += s["duration"] / 60.0
            log("segment", f"skip (done): {ident}")
            continue
        segments = []
        for sf in meta.get("source_files", []):
            vpath = sf.get("vocals_path")
            if not vpath or not Path(vpath).exists():
                continue
            turns = sf.get("diarization", [])
            diar_minutes += sum(t["end"] - t["start"] for t in turns) / 60.0
            y16, _ = _load_audio(vpath, sr=MODEL_SR, mono=True)
            speech = get_speech_timestamps(
                torch.from_numpy(y16), model, sampling_rate=MODEL_SR,
                min_speech_duration_ms=int(MIN_DUR * 1000),
                max_speech_duration_s=MAX_DUR,
            )
            idx = 0
            for sp in speech:
                s0 = sp["start"] / MODEL_SR
                s1 = sp["end"] / MODEL_SR
                # Intersect each VAD region with the speaker turns so that no
                # utterance straddles a speaker change. Sampling the speaker at
                # the segment midpoint (the previous approach) let a single clip
                # contain two voices, which is useless as a TTS reference. With
                # the SPEAKER_UNK fallback (one turn spanning the file) this
                # reduces exactly to the old behaviour.
                for tn in turns:
                    a0 = max(s0, tn["start"])
                    b0 = min(s1, tn["end"])
                    if b0 - a0 < MIN_DUR:
                        continue
                    for (a, b) in _split_long(a0, b0, MAX_DUR):
                        if b - a < MIN_DUR:
                            continue
                        spk = tn["speaker"]
                        clip_id = f"{ident}__{Path(sf['name']).stem}__seg{idx:04d}"
                        segments.append({
                            "clip_id": clip_id, "source_file": sf["name"],
                            "vocals_path": vpath, "start_sec": round(a, 3),
                            "end_sec": round(b, 3), "duration": round(b - a, 3),
                            "speaker_id": spk,
                            "text": "", "asr_conf": None, "align_conf": None,
                            "snr": None, "dnsmos": None,
                            "arousal": None, "valence": None,
                            "emotion_label": None,
                            "audio_ok": None, "align_ok": None,
                            "reject_reason": "", "needs_manual": False,
                            "clip_path": None,
                        })
                        idx += 1
                        vad_minutes += (b - a) / 60.0
        meta["segments"] = segments
        mark_done(meta, "segment")
        save_meta(cfg, meta)
        log("segment", f"  {ident}: {len(segments)} candidate utterances")
    fm = report.setdefault("funnel_minutes", {})
    fm["after_diarize"] = round(diar_minutes, 2)
    fm["after_vad"] = round(vad_minutes, 2)
    save_report(cfg, report)
    log("segment", f"DONE. ~{vad_minutes:.1f} minutes of candidate utterances.")


# ========================================================================== #
# STAGE 5 : transcribe (faster-whisper) -> text + asr_conf + word probs
# ========================================================================== #

def _extract_clip(cfg: Config, seg: dict):
    """Load the segment's audio slice at MODEL_SR (for models). Returns (y16, sr)."""
    y, sr = _load_audio(seg["vocals_path"], sr=None, mono=True)
    a = int(seg["start_sec"] * sr)
    b = int(seg["end_sec"] * sr)
    y = y[a:b]
    return _resample(y, sr, MODEL_SR), MODEL_SR


def stage_transcribe(cfg: Config) -> None:
    from faster_whisper import WhisperModel
    import math

    device = resolve_device(cfg)
    compute = "float16" if device == "cuda" else "int8"
    log("transcribe", f"loading faster-whisper '{cfg.asr_model}' ({device}/{compute})")
    model = WhisperModel(cfg.asr_model, device=device, compute_type=compute)

    for meta in _iter_items_with_stage(cfg, "segment"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "transcribe") and not cfg.force:
            log("transcribe", f"skip (done): {ident}")
            continue
        segs = meta.get("segments", [])
        for i, seg in enumerate(segs):
            if seg.get("text") and not cfg.force:
                continue
            y16, sr = _extract_clip(cfg, seg)
            segments, _info = model.transcribe(
                y16, language="si", word_timestamps=True, beam_size=5,
                vad_filter=False,
            )
            texts, words, logprobs = [], [], []
            for s in segments:
                texts.append(s.text)
                logprobs.append(s.avg_logprob)
                for w in (s.words or []):
                    words.append({"word": w.word, "start": float(w.start),
                                  "end": float(w.end), "prob": float(w.probability)})
            seg["text"] = "".join(texts).strip()
            # asr_conf: map avg logprob -> (0,1) via exp; average over sub-segments
            seg["asr_conf"] = round(
                float(sum(math.exp(lp) for lp in logprobs) / len(logprobs)) if logprobs else 0.0,
                4)
            seg["words"] = words
            if (i + 1) % 25 == 0:
                log("transcribe", f"  {ident}: {i+1}/{len(segs)}")
        mark_done(meta, "transcribe")
        save_meta(cfg, meta)
        log("transcribe", f"  {ident}: transcribed {len(segs)} clip(s)")
    log("transcribe", "DONE.")


# ========================================================================== #
# STAGE 6 : align -> word timestamps + align_conf
# ========================================================================== #

# WhisperX ships forced-alignment models only for a fixed language set; Sinhala
# ('si') is NOT among them. So default 'auto' uses Whisper's own word
# probabilities as align_conf and only calls WhisperX when a language model
# actually exists (e.g. if you transcribe a supported language).

def stage_align(cfg: Config) -> None:
    import numpy as np

    use_whisperx = cfg.aligner == "whisperx"
    align_model = None
    if cfg.aligner in ("whisperx", "auto"):
        try:
            import whisperx  # noqa: F401
            device = resolve_device(cfg)
            try:
                model_a, metadata = whisperx.load_align_model(language_code="si", device=device)
                align_model = (whisperx, model_a, metadata, device)
                log("align", "loaded WhisperX alignment model for 'si'")
            except Exception:
                if use_whisperx:
                    warn("align", "WhisperX has no 'si' align model; falling back "
                                  "to Whisper word probabilities")
                align_model = None
        except Exception:
            if use_whisperx:
                warn("align", "whisperx not installed; using Whisper word probs")

    for meta in _iter_items_with_stage(cfg, "transcribe"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "align") and not cfg.force:
            log("align", f"skip (done): {ident}")
            continue
        for seg in meta.get("segments", []):
            words = seg.get("words", [])
            if align_model is not None and seg.get("text"):
                try:
                    wx, model_a, metadata, device = align_model
                    y16, _ = _extract_clip(cfg, seg)
                    tr = [{"start": 0.0, "end": seg["duration"], "text": seg["text"]}]
                    res = wx.align(tr, model_a, metadata, y16, device,
                                   return_char_alignments=False)
                    scores = [w.get("score", 0.0) for s in res["segments"]
                              for w in s.get("words", []) if "score" in w]
                    seg["align_conf"] = round(float(np.mean(scores)) if scores else 0.0, 4)
                    continue
                except Exception as e:
                    warn("align", f"  whisperx align failed ({e}); word-prob fallback")
            # fallback: mean word probability from faster-whisper
            probs = [w["prob"] for w in words] if words else []
            seg["align_conf"] = round(float(np.mean(probs)) if probs else 0.0, 4)
        mark_done(meta, "align")
        save_meta(cfg, meta)
        log("align", f"  {ident}: aligned")
    log("align", "DONE.")


# ========================================================================== #
# STAGE 7 : filter (audio quality; ASR-confidence handled at manifest time)
# ========================================================================== #

def stage_filter(cfg: Config) -> None:
    report = load_report(cfg)
    kept_minutes = 0.0
    kept = 0
    total = 0
    for meta in _iter_items_with_stage(cfg, "align"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "filter") and not cfg.force:
            for seg in meta.get("segments", []):
                if seg.get("audio_ok"):
                    kept_minutes += seg["duration"] / 60.0
                    kept += 1
                total += 1
            continue
        for seg in meta.get("segments", []):
            total += 1
            y16, sr = _extract_clip(cfg, seg)
            clip_frac = _clip_fraction(y16)
            snr = _estimate_snr_db(y16, sr)
            dnsmos = _dnsmos(y16, sr)
            seg["snr"] = round(snr, 2)
            seg["dnsmos"] = round(dnsmos, 3) if dnsmos is not None else None
            reasons = []
            if not (MIN_DUR <= seg["duration"] <= MAX_DUR):
                reasons.append("duration")
            if clip_frac > cfg.max_clip_frac:
                reasons.append("clipping")
            if snr < cfg.min_snr_db:
                reasons.append("low_snr")
            if dnsmos is not None and dnsmos < cfg.min_dnsmos:
                reasons.append("low_dnsmos")
            seg["audio_ok"] = len(reasons) == 0
            ac = seg.get("align_conf") or 0.0
            seg["align_ok"] = ac >= cfg.min_align_conf
            seg["reject_reason"] = ",".join(reasons)
            if seg["audio_ok"]:
                kept_minutes += seg["duration"] / 60.0
                kept += 1
        mark_done(meta, "filter")
        save_meta(cfg, meta)
        log("filter", f"  {ident}: {kept} audio-ok so far")
    report.setdefault("funnel_minutes", {})["after_filter"] = round(kept_minutes, 2)
    report.setdefault("counts", {})["clips_total"] = total
    save_report(cfg, report)
    log("filter", f"DONE. {kept}/{total} clips passed audio-quality filter.")


# ========================================================================== #
# STAGE 8 : emotion (audEERING dimensional [default] or emotion2vec categorical)
# ========================================================================== #

def _av_to_label(arousal: float, valence: float) -> str:
    """Coarse categorical label from the arousal/valence quadrant (0..1)."""
    hi_a, hi_v = arousal >= 0.55, valence >= 0.55
    lo_a, lo_v = arousal <= 0.45, valence <= 0.45
    if hi_a and hi_v:
        return "happy_excited"
    if hi_a and lo_v:
        return "angry_fear"
    if lo_a and lo_v:
        return "sad"
    if lo_a and hi_v:
        return "calm_content"
    return "neutral"


def _load_audeering(device):
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model, Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        # transformers >= ~4.57 reads `all_tied_weights_keys` while finalizing
        # from_pretrained(). audEERING's published snippet predates that, so the
        # attribute is missing and loading crashes with an AttributeError. This
        # model ties no weights, so an empty mapping is the correct value -- and
        # declaring it keeps the class working on old and new transformers alike.
        all_tied_weights_keys: dict = {}

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            hidden = torch.mean(hidden, dim=1)
            return hidden, self.classifier(hidden)

    name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    proc = Wav2Vec2Processor.from_pretrained(name)
    model = EmotionModel.from_pretrained(name).to(device).eval()
    return proc, model


def stage_emotion(cfg: Config) -> None:
    import numpy as np
    device = resolve_device(cfg)

    proc = model = e2v = None
    if cfg.emotion_model == "audeering":
        log("emotion", "loading audEERING dimensional model (arousal/valence)")
        proc, model = _load_audeering(device)
    else:
        log("emotion", "loading emotion2vec_plus_large (categorical) via FunASR")
        try:
            from funasr import AutoModel
            e2v = AutoModel(model="iic/emotion2vec_plus_large")
        except Exception as e:
            warn("emotion", f"emotion2vec unavailable ({e}); falling back to audeering")
            proc, model = _load_audeering(device)

    for meta in _iter_items_with_stage(cfg, "filter"):
        ident = meta["ia_identifier"]
        if stage_done(meta, "emotion") and not cfg.force:
            log("emotion", f"skip (done): {ident}")
            continue
        for seg in meta.get("segments", []):
            if not seg.get("audio_ok"):
                continue  # label only clips that passed audio QC
            y16, sr = _extract_clip(cfg, seg)
            if proc is not None:  # audeering dimensional
                import torch
                y = proc(y16, sampling_rate=sr)["input_values"][0]
                y = torch.from_numpy(np.asarray(y)).reshape(1, -1).to(device)
                with torch.no_grad():
                    out = model(y)[1].cpu().numpy()[0]  # [arousal, dominance, valence]
                arousal, valence = float(out[0]), float(out[2])
                seg["arousal"] = round(arousal, 4)
                seg["valence"] = round(valence, 4)
                seg["emotion_label"] = _av_to_label(arousal, valence)
            else:  # emotion2vec categorical
                res = e2v.generate(np.asarray(y16, dtype="float32"),
                                   granularity="utterance", extract_embedding=False)
                if res:
                    labels = res[0].get("labels", [])
                    scores = res[0].get("scores", [])
                    if labels and scores:
                        top = int(np.argmax(scores))
                        seg["emotion_label"] = str(labels[top]).split("/")[-1]
                seg["arousal"] = None
                seg["valence"] = None
        mark_done(meta, "emotion")
        save_meta(cfg, meta)
        log("emotion", f"  {ident}: labeled")
    log("emotion", "DONE.")


# ========================================================================== #
# STAGE 9 : manifest (+ finalize report)
# ========================================================================== #

def _render_clip(cfg: Config, seg: dict) -> str:
    """Write the final 24 kHz mono PCM16 loudness-normalized clip; return path."""
    if seg.get("clip_path") and Path(seg["clip_path"]).exists() and not cfg.force:
        return seg["clip_path"]
    y, sr = _load_audio(seg["vocals_path"], sr=None, mono=True)
    a = int(seg["start_sec"] * sr)
    b = int(seg["end_sec"] * sr)
    y = _resample(y[a:b], sr, TARGET_SR)
    y = _loudness_normalize(y, TARGET_SR)
    out = cfg.clips_dir / (seg["clip_id"] + ".wav")
    _save_wav_pcm16(out, y, TARGET_SR)
    return str(out)


def stage_manifest(cfg: Config) -> None:
    import numpy as np
    report = load_report(cfg)

    usable_rows, manual_rows = [], []
    final_minutes = manual_minutes = 0.0
    arousals, valences, labels = [], [], []

    for meta in _iter_items_with_stage(cfg, "emotion"):
        ident = meta["ia_identifier"]
        for seg in meta.get("segments", []):
            if not seg.get("audio_ok"):
                continue
            asr_ok = (seg.get("asr_conf") or 0.0) >= cfg.min_asr_conf and seg.get("text")
            align_ok = seg.get("align_ok")
            arousal = seg.get("arousal")
            usable = bool(asr_ok and align_ok)
            emotional = arousal is not None and arousal >= MANUAL_AROUSAL
            row = {
                "clip_id": seg["clip_id"], "ia_identifier": ident,
                "source_file": seg["source_file"],
                "start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
                "duration": seg["duration"], "speaker_id": seg["speaker_id"],
                "text": seg.get("text", ""), "asr_conf": seg.get("asr_conf"),
                "align_conf": seg.get("align_conf"), "snr": seg.get("snr"),
                "dnsmos": seg.get("dnsmos"), "arousal": arousal,
                "valence": seg.get("valence"),
                "emotion_label": seg.get("emotion_label"),
            }
            if usable:
                seg["clip_path"] = _render_clip(cfg, seg)
                usable_rows.append(row)
                final_minutes += seg["duration"] / 60.0
                if arousal is not None:
                    arousals.append(arousal)
                    valences.append(seg.get("valence"))
                if seg.get("emotion_label"):
                    labels.append(seg["emotion_label"])
            elif emotional or (not asr_ok and align_ok):
                # emotionally salient or clean-audio-but-ASR-failed => manual
                seg["needs_manual"] = True
                seg["clip_path"] = _render_clip(cfg, seg)
                manual_rows.append({**row, "reason": "high_emotion_low_asr"
                                    if emotional else "low_asr"})
                manual_minutes += seg["duration"] / 60.0
        save_meta(cfg, meta)

    _write_csv(cfg.manifest_csv, usable_rows, MANIFEST_FIELDS)
    _write_csv(cfg.needs_manual_csv, manual_rows, MANIFEST_FIELDS + ["reason"])
    log("manifest", f"wrote {cfg.manifest_csv} ({len(usable_rows)} usable clips)")
    log("manifest", f"wrote {cfg.needs_manual_csv} ({len(manual_rows)} manual clips)")

    # ---- finalize report / funnel ---------------------------------------- #
    fm = report.setdefault("funnel_minutes", {})
    fm["final_usable"] = round(final_minutes, 2)
    fm["needs_manual"] = round(manual_minutes, 2)
    counts = report.setdefault("counts", {})
    counts["clips_usable"] = len(usable_rows)
    counts["clips_needs_manual"] = len(manual_rows)

    ed = report["emotion_distribution"] = {}
    if arousals:
        ed["arousal_hist"] = _hist(arousals)
        ed["valence_hist"] = _hist([v for v in valences if v is not None])
        ed["arousal_mean"] = round(float(np.mean(arousals)), 3)
    if labels:
        from collections import Counter
        ed["category_counts"] = dict(Counter(labels))

    downloaded = fm.get("downloaded", 0.0) or 0.0
    yield_pct = (final_minutes / downloaded * 100.0) if downloaded else 0.0
    report["yield_percent"] = round(yield_pct, 2)
    save_report(cfg, report)
    _print_final_report(cfg, report, yield_pct)


def _hist(vals, bins=10, lo=0.0, hi=1.0):
    import numpy as np
    if not vals:
        return {}
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    return {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(counts[i]) for i in range(len(counts))}


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_final_report(cfg: Config, report: dict, yield_pct: float) -> None:
    fm = report.get("funnel_minutes", {})
    log("manifest", "=" * 60)
    log("manifest", "YIELD FUNNEL (minutes)")
    for k in ["downloaded", "after_separate", "after_diarize", "after_vad",
              "after_filter", "final_usable", "needs_manual"]:
        if k in fm:
            log("manifest", f"  {k:16s}: {fm[k]:8.2f}")
    log("manifest", f"  USABLE YIELD    : {yield_pct:7.2f}% of downloaded audio")
    ed = report.get("emotion_distribution", {})
    if ed.get("category_counts"):
        log("manifest", "EMOTION DISTRIBUTION (usable clips)")
        for k, v in sorted(ed["category_counts"].items(), key=lambda x: -x[1]):
            log("manifest", f"  {k:16s}: {v}")
    if ed.get("arousal_hist"):
        log("manifest", f"  arousal_hist    : {ed['arousal_hist']}")
    log("manifest", f"  needs_manual    : {report.get('counts',{}).get('clips_needs_manual',0)} clips")
    log("manifest", "=" * 60)
    log("manifest", "REVIEW this report before scaling beyond the pilot episode.")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

STAGE_FUNCS = {
    "download": stage_download,
    "separate": stage_separate,
    "diarize": stage_diarize,
    "segment": stage_segment,
    "transcribe": stage_transcribe,
    "align": stage_align,
    "filter": stage_filter,
    "emotion": stage_emotion,
    "manifest": stage_manifest,
}


def run_stage(cfg: Config, stage: str) -> None:
    t0 = time.time()
    log("run", f"--- stage: {stage} ---")
    STAGE_FUNCS[stage](cfg)
    log("run", f"--- {stage} took {time.time()-t0:.1f}s ---")


def run_all(cfg: Config) -> None:
    for stage in STAGES:
        run_stage(cfg, stage)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build an emotion-labeled Sinhala speech dataset from SLBC "
                    "radio dramas on the Internet Archive (manifest-first).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", required=True, choices=STAGES + ["all"],
                   help="pipeline stage to run ('all' runs every stage in order)")
    p.add_argument("--identifiers", default="",
                   help="comma-separated archive.org identifiers "
                        "(default: known Muwan Palassa items)")
    p.add_argument("--search", default=None,
                   help="archive.org query, e.g. 'title:(\"Muwan Palassa\")'")
    p.add_argument("--work-dir", default=None,
                   help="output root (default: /kaggle/working/eesd or ./work)")
    p.add_argument("--smoke", action="store_true",
                   help="fast pilot: one episode, first %ds, tiny ASR model"
                        % int(SMOKE_AUDIO_SECONDS))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--hf-token-env", default="HF_TOKEN",
                   help="env var / Kaggle secret name holding the HuggingFace token")
    p.add_argument("--asr-model", default="",
                   help="faster-whisper model or HF id (default: large-v3, or "
                        "'base' in --smoke). Pass a Sinhala fine-tune to override.")
    p.add_argument("--emotion-model", default="audeering",
                   choices=["audeering", "emotion2vec"],
                   help="audeering=dimensional arousal/valence (default); "
                        "emotion2vec=categorical")
    p.add_argument("--diarization-model", default="auto",
                   help="'auto' tries speaker-diarization-3.1 then "
                        "speaker-diarization-community-1; or pass an explicit "
                        "HF pipeline id")
    p.add_argument("--denoise", action="store_true",
                   help="apply DeepFilterNet denoise after Demucs")
    p.add_argument("--aligner", default="auto", choices=["auto", "whisperx", "whisper"],
                   help="'auto' uses WhisperX when a language model exists, else "
                        "Whisper word probabilities (required for Sinhala)")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop after roughly this many audio-minutes (quota guard)")
    p.add_argument("--force", action="store_true",
                   help="recompute even if outputs already exist")
    p.add_argument("--min-align-conf", type=float, default=DEFAULT_MIN_ALIGN_CONF)
    p.add_argument("--min-asr-conf", type=float, default=DEFAULT_MIN_ASR_CONF)
    p.add_argument("--min-snr-db", type=float, default=DEFAULT_MIN_SNR_DB)
    p.add_argument("--min-dnsmos", type=float, default=DEFAULT_MIN_DNSMOS)
    p.add_argument("--max-clip-frac", type=float, default=DEFAULT_MAX_CLIP_FRAC)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    pin_gpu0()
    cfg = build_config(args)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    log("run", f"work_dir={cfg.work_dir}  smoke={cfg.smoke}  kaggle={on_kaggle()}")
    log("run", f"identifiers={cfg.identifiers or '(search)'}  search={cfg.search}")
    if cfg.stage == "all":
        run_all(cfg)
    else:
        run_stage(cfg, cfg.stage)
    log("run", "complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
