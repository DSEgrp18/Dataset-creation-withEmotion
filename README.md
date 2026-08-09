# Emotional Sinhala Speech Dataset

Builds a **sentence-level Sinhala speech corpus with transcripts** from SLBC radio
dramas on the Internet Archive, for training expressive / emotional TTS
(F5-TTS, VITS) alongside the existing neutral PathNirvana corpus (~13.6 h).

No public Sinhala emotional-speech dataset exists, so this makes one. Radio drama is
the source because it is naturally expressive — shouting, crying, whispering, arguing
— which is exactly what read-speech corpora lack.

---

## Status

A full episode is built and verified. The pipeline for the remaining 200 is ready.

| | |
|---|---|
| **Built** | 1 episode → **192 clips**, 16.2 min |
| **Available** | **201 episodes**, 82.0 h |
| **Projected yield** | ~52 h of clean speech at measured coverage |

---

## Results on the built episode

*Muwan Palassa 140113 — 25.3 min source*

| Metric | Value |
|---|---|
| Clips | **192** |
| Usable audio | **16.2 min** (**64.2 %** of source) |
| Clip length | median **4.3 s**, mean 5.1 s, range 1.1–15.0 s |
| Transcript | 1,691 words, mean 8.8 words/clip |
| Timestamps verified against audio | **93.2 %** (179/192) |
| **Overlapping clips** | **0** |
| Disk | 46 MB |

**64 % coverage is the headline number.** The original Whisper-based design predicted
under 10–20 % usable yield. The rest of the audio is music, sound effects, and silence.

**0 overlaps matters more than it looks.** Overlapping clips would ship the same audio
repeatedly under different transcripts and quietly poison a TTS training set. It is
asserted at the end of every run — clip-minutes are compared against source-minutes.

---

## Why Gemini, and not an open-weight model

Every Sinhala ASR checkpoint on HuggingFace was benchmarked on this audio. They all
failed, and **not because of model choice — because of domain mismatch.** They are
fine-tuned on clean single-speaker *read* speech; this is lossy 2013 broadcast with
music, overlap, and theatrical delivery.

Same clip, same audio:

| Engine | Output |
|---|---|
| `hlasith/whisper-sinhala-small` | එක්යන්මෙතාමතේ රස්නේ ඉවරවිලන් නෑ |
| **Gemini** | **ඒ කියන්නේ තාමත් ඒ ප්‍රශ්නය ඉවර වෙලා නෑ.** |

The open-weight output is word-salad; Gemini's is a grammatical sentence with correct
conjuncts and punctuation.

Also checked and ruled out, against the model files rather than the model cards:

- **SeamlessM4T-v2** — no Sinhala (`sin` absent from its 98 output languages)
- **MMS-1B-all** — no Sinhala adapter (`sin` absent from all vocab files)
- **Whisper large-v3** — supports `si`, but it is one of its thinnest languages

Paid services do support Sinhala (Google Chirp 2 and Azure both list `si-LK`, verified
in vendor docs), but Gemini's free tier removes the need.

---

## How accurate timestamps are produced

The central problem: **an LLM's timestamps drift**, so trusting them gives clips that
start mid-syllable. But a VAD cannot find *sentences* either — it hears pauses, not
grammar, so one speech region may hold three sentences or half of one.

Each source is used only for what it is reliable at:

```
Gemini  →  WHAT was said, and roughly where   (sentence text + rough times)
Silero  →  EXACTLY where speech starts/stops  (acoustic evidence)
```

Every speech region is then awarded to **exactly one** sentence — whichever the model's
rough span overlaps most — and a sentence's clip becomes the span of the regions it won.
Because regions are disjoint and singly owned, clips *cannot* duplicate audio.

Two refinements that turned out to be necessary:

- **Monotonic assignment.** Over a 5-minute chunk the model's absolute timestamps drift
  by seconds, but the *order* of sentences stays exactly right. Regions are walked in
  time order and the sentence pointer never moves backwards, so a drifted sentence
  cannot claim a distant region and splice two speakers together.
- **Neighbour-aware padding.** Clips get 0.1 s of breathing room, bounded by the
  previous clip's *padded* end — otherwise two pads meet in the middle and reintroduce
  the overlap that ownership just eliminated.

A sentence that wins no region keeps the model's raw times and is flagged
`snapped=False` (6.8 % here). Those are kept, not discarded, and highlighted in
`review.html` for a human to judge.

---

## Output format

```
dataset/<episode>/
├── manifest.csv          ← the deliverable
├── clips/<clip_id>.wav   ← 24 kHz · mono · PCM 16-bit
├── review.html           ← audio players beside text, for QC
└── .chunks/              ← resume cache (raw API output)
```

```csv
clip_id,source_file,start_sec,end_sec,duration,text,snapped
Muwan_Palassa_140113__0000,Muwan Palassa 140113.mp3,11.038,14.946,3.908,ගුවන් විදුලියට රචනා කරන්නේ කලාශූරී මුදලිනායක සෝමරත්න,True
```

| Column | Meaning |
|---|---|
| `clip_id` | filename stem — `clips/<clip_id>.wav` |
| `start_sec` / `end_sec` | **absolute seconds in the source**, so clips regenerate without the wavs |
| `text` | Sinhala sentence, UTF-8 |
| `snapped` | `True` = cut points confirmed against VAD; `False` = review before training |

CSVs are **UTF-8-BOM** so Excel renders Sinhala correctly — read with
`encoding="utf-8-sig"`. The Kaggle build adds an `identifier` column (archive.org ID).

Audio is deliberately **not loudness-normalized**: flattening a shouted line and a
whispered one to the same level would destroy the emotional dynamics this dataset
exists to capture.

---

## Usage

### One episode, locally

```bash
pip install librosa soundfile silero-vad requests
echo "YOUR_KEY" > api_key.txt          # https://aistudio.google.com/apikey (free)

python build_dataset.py                       # 5-min sample
python build_dataset.py --duration-sec 0      # whole episode
```

### All 201 episodes, on Kaggle

Kaggle is used **only for its network** — a 25-min episode downloads in ~1 min there
versus ~50 on a home connection.

1. Upload [`kaggle_build.ipynb`](kaggle_build.ipynb)
2. **Settings → Internet ON**, **Accelerator: None** (network-bound; a GPU buys nothing)
3. **Add-ons → Secrets →** add `GEMINI_API_KEY`
4. Run All

---

## The real constraint: API quota

**Not compute, not bandwidth — requests per day.** The API reports it plainly:

```
GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20
```

**20 requests per day, per model.** (Widely-quoted "1,500/day" figures do not apply to
these models — read the quota from a 429 body, never from a blog.)

| | |
|---|---|
| Requests for all 201 episodes | **991** |
| Free-tier allowance | ~**100/day** (5 models × 20) |
| **Sessions needed** | **~10 days** |

Three things make that survivable:

- **5-minute chunks + FLAC payloads** — an episode costs **5 requests, not 25**
- **Per-model failover** — quota is per model, so exhausting one moves to the next
- **Automatic waiting** — when all models are spent the script sleeps 20 min and
  retries. It does not try to predict Google's reset time; a successful call is the
  proof quota returned. An 11-hour session sits through a daily reset unattended.

### Continuing between sessions — required

Kaggle wipes `/kaggle/working`, so **each run must be handed to the next**:

1. When it stops → **File → Save Version → Save & Run All**
2. Next run → **Add Data → Your Work →** this notebook's output
3. Run again — finished episodes are adopted and skipped for free

Miss step 2 and it restarts from zero. The notebook's pre-flight cell reports whether a
previous run was found, so a missed step shows up immediately rather than after
re-spending a day's quota.

---

## Source

[archive.org/details/muwan-palassa](https://archive.org/details/muwan-palassa) — *Muwan
Palassa*, SLBC Sinhala radio drama, 201 episodes, 82.0 h, mean 24.5 min/episode.
Sixteen further single-episode uploads exist ([search
`title:("Muwan Palassa")`](https://archive.org/search?query=title%3A%28%22Muwan+Palassa%22%29),
17 items, 7.2 h), several duplicating the collection.

> **Copyright.** All 17 items were checked individually and **every one has no license
> field**, so they are treated as copyrighted. `manifest.csv` — timestamps and text — is
> the shareable artifact and regenerates the audio from the source. The wavs are for
> local training and review and are **git-ignored on purpose**. Do not redistribute them.

---

## Files

**Dataset build** (this directory)

| File | Purpose |
|---|---|
| [`kaggle_build.ipynb`](kaggle_build.ipynb) | Kaggle notebook — pre-flight checks + Run All |
| [`kaggle_build_dataset.py`](kaggle_build_dataset.py) | Kaggle builder — archive.org downloads + many episodes |
| [`build_dataset.py`](build_dataset.py) | Local builder — one episode |
| [`check_quota.py`](check_quota.py) | when the Gemini quota resets, and which keys/models are live |
| `api_key.txt` | Gemini key(s), one per line (git-ignored) |

**Model training** — [`xtts_baseline/`](xtts_baseline/)

| File | Purpose |
|---|---|
| [`xtts_baseline/kaggle_xtts_train.ipynb`](xtts_baseline/kaggle_xtts_train.ipynb) | fine-tune XTTS-v2 on the clips (needs a **GPU**) |
| [`xtts_baseline/prepare_for_xtts.py`](xtts_baseline/prepare_for_xtts.py) | filters the manifest into XTTS's input format |

The two halves are independent: the builder produces `manifest.csv` + `clips/`,
and anything downstream consumes them. Training code itself lives in
[DSEgrp18/XTTS_V2_Baseline](https://github.com/DSEgrp18/XTTS_V2_Baseline) and is
cloned by the notebook — only the dataset-specific filter lives here.

An engine-comparison bench (`sinhala_stt_bench.py`) produced the Gemini-vs-open-weight
evidence above and was removed once the question was settled. Recover it with
`git checkout f62400a -- sinhala_stt_bench.py` if the comparison needs reproducing for a
write-up.

---

## Known limitations

- **No speaker labels.** Radio drama is multi-speaker and nothing here diarizes it, so
  every clip is anonymous. Reference-conditioned expressive TTS will want speaker IDs —
  a separate pass, better decided before building all 201 episodes than after.
- **No emotion labels yet.** Clip extraction came first; arousal/valence or categorical
  labelling runs on top of these clips.
- **6.8 % of clips have unverified cut points** (`snapped=False`). Filter them out for a
  first training run.
- **Transcripts are unvalidated by a native speaker.** They read as fluent Sinhala and
  are far better than the open-weight alternatives, but no WER has been measured against
  ground truth — there is none for this audio.
