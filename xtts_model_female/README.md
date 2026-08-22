# XTTS-v2 on the VoiceMakers Sinhala female voices

Fine-tunes XTTS-v2 on **Dinithi (4.82 h) + Harini (2.14 h)** from
[SinhalaTTS Dataset Publication by VoiceMakers](https://www.kaggle.com/datasets/safnask/sinhalatts-dataset-publication-by-voicemakers)
— ~7 h across two genuinely distinct female speakers — and evaluates the result on
every metric the Sinhala TTS literature uses.

Upload [`kaggle_xtts_female.ipynb`](kaggle_xtts_female.ipynb), attach the dataset,
GPU on, internet on, Run All.

---

## Why this run should beat the earlier attempts

**The text becomes ASCII before the tokenizer sees it.** XTTS-v2's `vocab.json` is a
whitespace-pretokenised BPE with an `[UNK]` fallback, and it contains no Sinhala
codepoint — *nor the diacritics this corpus romanises with*. `ā ī ū ē ṭ ḍ ṇ ḷ ṁ` are
all absent. Because the pre-tokeniser splits on whitespace, one unknown character
turns the **entire word** into a single `[UNK]`, so either column fed raw trains the
model on "unknown unknown unknown": the loss falls convincingly and the audio is
noise. [`sinhala_text.py`](../xtts_sinhala/sinhala_text.py) folds the romanisation to
plain ASCII the pretrained vocabulary already covers, and the prepare step asserts
**0 `[UNK]`** before a GPU is touched.

**Two speakers, correctly labelled.** XTTS samples a conditioning clip from the *same
speaker* on every training step, so a correctly-labelled two-speaker set teaches "the
reference predicts the output voice" — the capability actually being fine-tuned.
Pooling speakers under one label teaches the opposite and no amount of data repairs it.

**More female data than the current state of the art had.** SPECOM 2025 trained VITS on
pathnirvana and its female voice was the weakest configuration by a wide margin —
MCD 20.56 dB against 13.27 for male — which the authors attribute to insufficient
female audio (~2 h). This corpus has ~7 h of it.

---

## Files

| File | Purpose |
|---|---|
| [`prepare_voicemakers.py`](prepare_voicemakers.py) | discover → filter → fold → `metadata_train/eval.csv`, with the `[UNK]` gate |
| [`train_xtts_female.py`](train_xtts_female.py) | the official Coqui recipe, deviations documented in the docstring |
| [`evaluate_xtts.py`](evaluate_xtts.py) | MCD, F0, speaker similarity, failure rate, RTF, optional UTMOS and ASR |
| [`listening_test.py`](listening_test.py) | builds the MOS + SUS panel as one self-contained HTML file |
| [`score_listening.py`](score_listening.py) | rater CSVs → MOS and SUS numbers, split by rater group |
| [`kaggle_xtts_female.ipynb`](kaggle_xtts_female.ipynb) | Run All |
| [`RESULTS.md`](RESULTS.md) | one row per finished run — the only way these metrics mean anything |
| [`calibrate_mcd.py`](calibrate_mcd.py) | what this script's MCD scale actually is, measured on known pairs |

Local equivalent:

```bash
pip install "coqui-tts>=0.25.1" "coqui-tts-trainer>=0.2.0" soundfile librosa
python prepare_voicemakers.py --src <dataset> --out ./female_dataset --vocab vocab.json
python train_xtts_female.py --dataset ./female_dataset --out ./run --smoke   # 2 min
python train_xtts_female.py --dataset ./female_dataset --out ./run --epochs 40
python evaluate_xtts.py --run ./run/training/GPT_XTTS_si_female-<stamp> \
    --base ./run/training/XTTS_v2.0_original_model_files --dataset ./female_dataset
```

---

## Nothing about the dataset layout is assumed

The published folders are inconsistent — `Isuru-44100Hz` beside `Yasindu-44100`, with at
least one speaker directory nested inside a duplicate of itself. So `prepare_voicemakers.py`
discovers rather than assumes, and prints every decision:

- **speaker directories** matched by name substring, recursively, shallowest first
- **`metadata.csv`** found by `rglob` under each speaker directory
- **delimiter** scored across `|` tab `;` `,` by whether the split actually *separates*
  script from romanisation, and only then by column-count consistency
- **column order** found by content — the Sinhala column by codepoint range, the
  romanised column by its diacritics — never by index
- **header row** detected and skipped, or absent; either works
- **missing romanised column** falls back to transliterating the script

The last two bullets are not hypothetical. The delimiter scoring originally used
consistency alone and **failed on the real corpus**: these files are pipe-delimited, but
splitting a pipe-delimited line on commas also yields a consistent column count whenever
each sentence contains one comma, which put Sinhala on both sides of the split. And
**Harini's file has no romanised column at all** — only Sinhala script — so
`sinhala_to_ascii()` transliterates it instead. Both paths verify at 0 `[UNK]`; which one
each speaker took is printed and recorded in `prepare_report.json` under `text_source`.

Verified end to end against synthetic fixtures reproducing all of the above, including the
nested directory, the comma pattern that broke the delimiter scoring, a script-only
metadata file, and rows that must be dropped.

### Rows that are dropped, and why

| Filter | Reason |
|---|---|
| duration > 11.6 s | `GPTArgs.max_wav_length` = 255995 @ 22050 Hz. The dataloader drops longer clips **silently** — they are dropped here where the count prints. |
| duration < 1.0 s | too short to condition on |
| contains a digit | the audio speaks a number the text does not contain, and XTTS's `en` cleaner would expand it to **English** words inside a Sinhala sentence |
| unmapped character | see below |
| > 250 characters | `VoiceBpeTokenizer.char_limits["en"]` |

**The unmapped-character check is the one that matters.** `fold()` ends by dropping
anything outside its keep-set, so an unmapped diacritic raises nothing — it silently
turns `vaṟdanak` into `vadanak`, a different word, and poisons training with no warning.
The check therefore runs on the **raw** romanisation against what the map actually
covers, and aborts with a list of offending codepoints rather than training on corrupted
text. This was caught by a deliberate negative test, not by reasoning.

---

## Deviations from the official recipe

Everything in `GPTArgs`, `XttsAudioConfig` and the optimizer block is copied verbatim
from `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`. Three things differ, and one is
deliberately left alone:

**`language="en"`** selects a tokenizer branch, not a claim about the audio.
`VoiceBpeTokenizer.preprocess_text` raises `NotImplementedError` outside its 17-language
set and `[si]` is not a token in `vocab.json`. The text is already ASCII by then, so the
`en` cleaner path is exactly right for it.

**Effective batch 64, not 252.** Upstream's note is correct advice for a datacentre. On
one T4 an effective batch of 252 is ~100 s per optimiser step, so a full session buys
about 400 steps — nowhere near enough to move the model onto a new sound inventory.
`4 × 16` trades gradient noise for roughly 4× the steps.

**`lr=1e-5`, not `5e-6`**, compensating for the smaller batch. Drop back if
`loss_mel_ce` gets noisy or rises.

**`mixed_precision=False`** — the upstream default, and the only workable one here. fp16
roughly doubles throughput but drives `loss_mel_ce` to `nan` on a Kaggle T4 on the
first step, and it never recovers: an 8-hour run that writes only NaN checkpoints and
exits cleanly. Turing has no bf16, so there is no stable mixed-precision option on that
card. `--mixed-precision` opts in if you have verified finite losses on your GPU.

One thing that *looks* like a deviation but is not: the scheduler milestones are
`[900000, 2700000, 5400000]` and `Trainer` defaults to `scheduler_after_epoch=True`, so
those are **epochs** and the learning rate never decays. That is upstream behaviour.

---

## GPU pinning

All three scripts set `CUDA_VISIBLE_DEVICES=0` at import, before torch loads. Without it
the coqui Trainer refuses to start on any two-GPU accelerator:

```
RuntimeError: [!] 2 active GPUs. Define the target GPU by `CUDA_VISIBLE_DEVICES`.
For multi-gpu training use `python -m trainer.distribute`.
```

Kaggle's "GPU T4 x2" therefore failed outright before this was pinned. `setdefault` is
used, so an explicit `CUDA_VISIBLE_DEVICES` in the environment still wins.

Real multi-GPU would roughly double throughput and is worth trying *after* a single-GPU
baseline succeeds: it needs `python -m trainer.distribute --script train_xtts_female.py …`,
`optimizer_wd_only_on_weights=False`, and `CUDA_VISIBLE_DEVICES` left unset. Every failed
attempt costs a whole Kaggle session, which is why it is not the default.

---

## Disk budget — why training does not run in `/kaggle/working`

A GPTTrainer checkpoint carries the model **and** the AdamW state, so each one is
~5.5 GB. `save_n_checkpoints=1` suggests only one survives at a time. Four do:

| On disk during a run | |
|---|---|
| base model files | ~2.1 GB |
| `best_model_<step>.pth` | ~5.5 GB |
| `best_model.pth` | ~5.5 GB |
| `checkpoint_<step>.pth` (periodic) | ~5.5 GB |
| **steady state** | **~18.6 GB** |
| + the next best model, written *before* the old one is deleted | **~24.1 GB peak** |

The second row of 5.5 GB is the one that is easy to miss: `trainer.io.save_best_model`
ends with `fs.copy(checkpoint_path, "best_model.pth")` — a **full second copy**, not a
symlink or a hardlink.

`/kaggle/working` is a 20 GB volume, so a run pointed there reaches
`OSError: [Errno 28] No space left on device` at the *second* best-model save — global
step 1760 at these settings, about 45 minutes in. The smoke test survives it only
because it writes two best models and no periodic checkpoint, peaking at ~18.6 GB.

**It also destroys the notebook.** papermill writes `__notebook__.ipynb` to the same
volume, so a full disk truncates the JSON mid-write and the session ends with
`NotJSONError: Notebook does not appear to be JSON` — every cell's output gone, not
just the training cell's.

So: training writes to `/kaggle/temp` (hundreds of GB), and the notebook mirrors
`config.json` + `best_model.pth` back to `/kaggle/working/run` when it stops. Those two
files are the whole resume kit — `best_model.pth` carries the optimizer state, so
`--continue-path` on that directory picks up where the session left off.

Two guards back that up, mirroring the two NaN guards:

- **`--min-free-gb` (default 25)** refuses to start when the target volume cannot hold
  the peak. It runs *before* the TTS imports and the base-model download, so a doomed
  run costs a second instead of forty-five minutes.
- **The notebook's disk guard** checks free space every 30 s — not on the 10-minute
  heartbeat, because one checkpoint save can fill a volume between reports — and stops
  training while there is still room to finish writing. Unlike the NaN guard it does
  **not** raise: a disk-stopped checkpoint is perfectly good, so evaluation and export
  still run on it.

## The smoke test asserts finite losses

`--smoke` runs a handful of steps on 64 clips. Its job is to catch the one failure a
zero exit code does **not** reveal: `loss_mel_ce: nan`. A diverged run trains happily for
eight hours, writes NaN checkpoints and exits 0, so the notebook parses the smoke log and
raises if any loss is non-finite. The long training loop repeats the check on its
heartbeat and aborts if the loss has been `nan` for three consecutive reports.

That guard exists because a real run burned an entire session at `nan` before it was
added — the exit code was 0 the whole way.

The parsing lives in [`train_log.py`](train_log.py), not in the notebook, because it has
now failed in both directions: once by missing a `nan` run, and once by aborting a
perfectly healthy one. The trainer prints its epoch averages in colour, so the raw bytes
read `avg_loss_mel_ce:[92m 3.66`, and a naive regex captures `[92m` as the
value — which is not a number, which looks exactly like divergence. `train_log.py` strips
ANSI first and carries a selftest built from both real logs; run `python train_log.py`
after touching it.

---

## Evaluation

### What runs unattended — `evaluate_xtts.py`

| Metric | What it catches |
|---|---|
| **MCD (dB)**, DTW-aligned | spectral distance to the real recording. The only objective metric any Sinhala TTS paper reports |
| **log-F0 RMSE (cents)** | pitch accuracy over aligned voiced frames |
| **F0 correlation** | whether the intonation *contour* tracks the reference — a flat monotone can score decent RMSE and still sound dead |
| **SECS** | speaker-encoder cosine similarity: does the output sound like the conditioning speaker. Measured against a **held-out** clip, never the conditioning clip, which would inflate it |
| **Duration ratio** + **failure rate** | XTTS is autoregressive; truncation and runaway looping show here before you hear them |
| **RTF** | synthesis speed |
| `--utmos` | learned MOS predictor, English-trained — a relative signal between checkpoints, not an absolute MOS |
| `--asr` | intelligibility proxy. Sinhala ASR is weak, so the script transcribes the **real** held-out audio through the same model and reports the **gap**, which controls for the ASR's own error rate |

The MCD implementation was unit-tested on synthetic signals: identical inputs give
exactly 0.00 dB, a small perturbation gives ~13.5 dB, and the value is monotone in
spectral distance. F0 RMSE recovers a known 200→260 Hz shift as 456.5 cents against a
true 454.2. **MCD is implementation-dependent — compare runs of this script against each
other, never against a published figure.**

### What needs humans — `listening_test.py`

MOS and SUS are the two metrics the literature actually compares on, and there is no
automatic substitute. Claiming one is how papers end up reporting "98% accuracy" for a
synthesiser, which is not a meaningful synthesis metric at all.

The script synthesises the stimuli and writes **one self-contained HTML file** — audio
embedded, no server, no internet. Raters open it in a browser, listen, rate, and
download a CSV. The protocol matches SPECOM 2025 so the numbers are comparable: 15 MOS
sentences (five short, five medium, five long) rated on separate 5-point scales for
intelligibility and naturalness, plus 10 SUS sentences transcribed by ear.

Two things to know:

- **SUS sentences are generated mechanically**, by interleaving the words of two real
  corpus sentences of equal length. That preserves inflection while destroying meaning,
  which is the standard construction — but **have a native speaker read
  `answer_key.json` and drop any that came out ungrammatical** before running the panel.
  An ungrammatical SUS item measures nothing.
- **Report blind and sighted raters separately.** Nanayakkara et al. found visually
  impaired listeners scored the same system 66% where sighted listeners scored ~70%, and
  argued the sighted group was simply less practised at synthetic speech. Pooling hides
  that, and for this project the blind listeners are the ones who matter.

---

## What to send back after the Kaggle run

To improve the next iteration:

1. `eval_out/report.md` — the objective table
2. the tail of `train.log`, especially `loss_mel_ce` and the eval lines
3. `prepare_report.json` — how many clips survived and what was dropped
4. two or three synthesised wavs, ideally alongside the real recording of the same sentence

The most useful single number is **`loss_mel_ce` at the end versus its minimum**. If the
minimum came well before the end, it is overfitting and the export should come from
`best_model.pth` rather than the last checkpoint.

---

## Expectations, stated honestly

Seven hours across two speakers is enough for intelligible, correctly-accented Sinhala
in both trained voices, and enough to judge whether the approach works. It is **not**
enough for reliable zero-shot cloning of unseen speakers — that capability comes from
the pretrained model and this fine-tune will erode it somewhat.

Expect artefacts on words whose letter sequences never appeared in training, and expect
the two voices to be distinguishable but not perfectly separated at this data scale.
"Perfect" is not on the table from 7 hours; clearly better than the previous attempt is.
