# Emotional Sinhala Speech Dataset — SLBC radio-drama pipeline

A staged, reproducible pipeline that turns **Sri Lanka Broadcasting Corporation
(SLBC) radio dramas** hosted on the [Internet Archive](https://archive.org) into a
cleaned, **emotion-labeled, TTS-ready Sinhala speech dataset** — described by a
**manifest**, not by redistributed audio.

Built for University of Moratuwa TTS research. The existing single-speaker corpus
(PathNirvana, ~13.6 h) and VITS / F5-TTS models cover *neutral* Sinhala; this pipeline
adds *expressive / emotional* speech so F5-TTS can be conditioned on emotional reference
audio. No public Sinhala emotional-speech dataset exists, so we build one.

> **Radio dramas, not teledramas:** acted (expressive) emotion, but audio-only and far
> less music/SFX than TV — a better yield/effort trade-off.

---

## ⚠️ Two honest caveats — read before you trust the output

1. **ASR fails on the most emotional speech.** Shouting, crying, and whispering degrade
   Whisper. Radio dramas are also multi-speaker with background music. **Expect a low
   yield (<10–20 %)** of clean, transcribable clips, **skewed toward calmer speech.**
   The pipeline prints this funnel honestly in `report.json` — it does not hide it.
2. **The clips you most want are the ones ASR drops.** So instead of discarding
   high-arousal / low-ASR-confidence clips, the pipeline routes them to
   **`needs_manual_transcription.csv`** for a human pass. Budget for manual transcription.

There is no free lunch here: a fully-automatic *emotional* Sinhala corpus is not
achievable at high quality today. This pipeline maximizes the automatic yield and makes
the rest a tractable manual list.

---

## Manifest-first licensing (important)

Most of these Internet Archive items **have no license field** (verified: `licenseurl`,
`rights`, and copyright-status are all absent). We therefore **treat them as copyrighted**
and **do not redistribute the audio**.

The deliverable is a **manifest**: `ia_identifier` + `start_sec`/`end_sec` timestamps +
this code. Anyone with the manifest re-downloads the source from archive.org and
regenerates the exact clips locally. `.gitignore` blocks committing any `*.mp3/wav/…`.

Please keep it that way: share the manifest + code, not the audio.

---

## Quickstart

### Local — stage 1 (`download`) only (no GPU / ML deps needed)

```bash
pip install "internetarchive>=3.5,<5"
python build_emotional_sinhala_dataset.py --stage download --smoke \
    --identifiers muwan-palassa-140113 --work-dir ./work
```

This fetches one episode and writes `work/download_manifest.csv` + `work/report.json`
logging each item's license (see the real pilot result below). The `download` stage has
**no ML dependencies** — the script's top-level imports are stdlib-only and every heavy
model is imported lazily inside the stage that uses it.

### Kaggle — the full pipeline (GPU)

Stages 2–9 need a GPU, `ffmpeg`, and gated models — i.e. **Kaggle (T4 ×2)**, not a laptop.
Open **`kaggle_pilot.ipynb`** and follow it (Accelerator = GPU T4 ×2, Internet = ON, add an
`HF_TOKEN` Kaggle secret, accept the `pyannote/speaker-diarization-3.1` terms). Or by hand:

```bash
pip install -r requirements-kaggle.txt          # torch is preinstalled on Kaggle — don't reinstall
export HF_TOKEN=hf_xxx                            # for gated pyannote
python build_emotional_sinhala_dataset.py --stage all --smoke \
    --identifiers muwan-palassa-140113 --work-dir /kaggle/working/eesd   # ~3-min wiring test
python build_emotional_sinhala_dataset.py --stage all \
    --identifiers muwan-palassa-140113 --work-dir /kaggle/working/eesd   # full-episode pilot
```

Then **review the yield + emotion distribution and stop** before scaling. Do **not**
download the whole archive unprompted.

---

## Pipeline stages

| # | stage | tool | output |
|---|-------|------|--------|
| 1 | `download`   | `internetarchive` | source MP3/OGG + `download_manifest.csv` (with license) |
| 2 | `separate`   | Demucs `htdemucs` (opt. DeepFilterNet) | vocals stem, music removed |
| 3 | `diarize`    | `pyannote/speaker-diarization-3.1` (gated) | per-speaker turns |
| 4 | `segment`    | Silero VAD | 2–12 s single-speaker utterances on silence |
| 5 | `transcribe` | faster-whisper `large-v3` (`lang=si`) | Sinhala text + `asr_conf` + word probs |
| 6 | `align`      | Whisper word probs (WhisperX if a lang model exists) | `align_conf` |
| 7 | `filter`     | SNR / clipping / DNSMOS* / duration | drop bad audio; flag low-ASR |
| 8 | `emotion`    | audEERING wav2vec2 (dimensional) | `arousal`, `valence`, coarse label |
| 9 | `manifest`   | — | `manifest.csv`, `needs_manual_transcription.csv`, `report.json` |

Run any single stage, a subset by rerunning, or `--stage all`. Every stage is
**idempotent**: it records progress per item in `work/meta/<id>.json` and skips work whose
outputs already exist — so a killed Kaggle session just resumes on rerun.

\* DNSMOS (`speechmos`) and DeepFilterNet are optional; if not installed, those steps are
skipped gracefully (`dnsmos = null`).

### Model choices (all verified live before wiring)

- **ASR — `whisper large-v3` by default.** The Sinhala-specific checkpoints on HF are all
  `whisper-small`/`tiny` fine-tuned on Common Voice *read* speech (best documented WER
  ≈ 46 %); they generalize *worse* to expressive multi-speaker drama than large-v3's native
  Sinhala. Override with `--asr-model <hf-id>` to try a Sinhala fine-tune.
- **Alignment — Whisper word probabilities for `si`.** WhisperX ships **no** Sinhala
  forced-alignment model, so `--aligner auto` uses faster-whisper's per-word probabilities
  as `align_conf` and only calls WhisperX when a language model actually exists.
- **Emotion — audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` (dimensional).**
  Outputs continuous **arousal / valence** (preferred over categorical for a first pass).
  `--emotion-model emotion2vec` switches to the categorical `emotion2vec_plus_large`
  (needs FunASR).
- **Diarization — `pyannote/speaker-diarization-3.1`** (gated; needs `HF_TOKEN`). Without a
  token the pipeline warns and continues with a single `SPEAKER_UNK` label.

---

## Manifest schema (`manifest.csv`, one row per clip)

```
clip_id | ia_identifier | source_file | start_sec | end_sec | duration |
speaker_id | text | asr_conf | align_conf | snr | dnsmos | arousal | valence | emotion_label
```

`needs_manual_transcription.csv` has the same columns plus a `reason`
(`high_emotion_low_asr` or `low_asr`). Final clips (regenerated locally, gitignored) are
**24 kHz mono 16-bit PCM**, loudness-normalized to **≈ −24 LUFS**.

`report.json` records the yield funnel (minutes at each stage), usable-yield %, emotion
distribution, per-item license, and the caveats above.

---

## Pilot status (this repo)

**Stage 1 (`download`) was run for real** on `muwan-palassa-140113`:

| field | value |
|-------|-------|
| file | `Muwan Palassa 140113.mp3` (VBR MP3, 24.3 MB) |
| duration | 1518.26 s (**25.3 min**) |
| language | `sin` |
| **license / rights** | **ABSENT → treated as copyrighted (manifest-first)** |

Stages 2–9 are **wired and syntax-verified but not yet executed** (they need a GPU + the ML
stack). Run the full smoke and full-episode pilot on Kaggle via `kaggle_pilot.ipynb`, then
review `report.json` before scaling. **The yield and emotion-distribution numbers must come
from that Kaggle run — they are not fabricated here.**

---

## Known source items (verified live)

- **Muwan Palassa** (uploader *Ceylon Waves*): `muwan-palassa-140113`,
  `muwan-palassa-210113`, `MuwanPalassa29816`, `muwanpalassa_27513`
- **Guwanviduli Rangamadala** (SLBC): `RadioDramas-SLBC` (31 audio files)
- Find more: `ia search 'title:("Muwan Palassa")' --itemlist`, or pass `--search`.

## Files

```
build_emotional_sinhala_dataset.py   # the pipeline (single file, staged CLI)
kaggle_pilot.ipynb                   # one-click Kaggle pilot notebook
kernel-metadata.json                 # `kaggle kernels push` stub (edit `id` first)
requirements-kaggle.txt              # additive deps (torch preinstalled on Kaggle)
README.md
.gitignore                           # blocks committing audio/work dir
```
