# XTTS-v2 baseline

Fine-tunes [DSEgrp18/XTTS_V2_Baseline](https://github.com/DSEgrp18/XTTS_V2_Baseline)
on the clips produced by the dataset builder one level up.

| File | Purpose |
|---|---|
| [`kaggle_xtts_train.ipynb`](kaggle_xtts_train.ipynb) | upload to Kaggle, Run All |
| [`prepare_for_xtts.py`](prepare_for_xtts.py) | filters our manifest into XTTS's input format |

The training code itself lives in the other repo and is cloned by the notebook.
Only the **filter** lives here, because that repo's `prepare_dataset.py` knows
nothing about our `quality` column or XTTS's length ceiling.

---

## Kaggle settings

| Setting | Value |
|---|---|
| **Accelerator** | **GPU T4 x2** (or P100) — *different from the dataset notebook* |
| Internet | ON |
| Data | your dataset output (the one with `dataset/all_manifests.csv`) |

Training is GPU-bound and will not run on CPU.

---

## Pipeline

```
prepare_for_xtts.py   filter to hi-fi, <=11.4 s, VAD-verified
        |
prepare_dataset.py    resample 22050, trim, normalise, train/eval split
        |
extend_vocab.py       add ~80 Sinhala chars to the BPE vocab   <-- the critical step
        |
train_xtts.py         fine-tune, eval every epoch
        |
monitor.py            overfitting verdict from the eval curves
        |
export_checkpoint.py  export at the eval minimum, not the last step
```

**`extend_vocab` is the one that matters.** XTTS-v2 supports 17 languages and
Sinhala is not among them — every Sinhala codepoint resolves to `[UNK]` against
the stock vocabulary. Fine-tuning without fixing that trains the model to map
"unknown unknown unknown" onto audio: the loss drops convincingly and the samples
are babble.

---

## Two limits worth knowing before you spend GPU hours

### XTTS silently discards clips over 11.6 s

`GPTArgs.max_wav_length = 255995` samples at 22050 Hz. Longer clips are dropped
by the dataloader without a warning. Our clips are built to a 3–30 s band, so on
a measured episode:

| | Clips | Audio |
|---|---|---|
| As built (3–30 s) | 141 | 16.0 min |
| Usable by XTTS | 129 | 13.4 min |

**~16 % of the audio is wasted.** Re-exporting the dataset with `MAX_CLIP = 11.4`
costs no API quota (the transcriptions are cached), so it is worth doing before
the corpus grows.

### Voice cloning will not work yet

XTTS is a voice-cloning model: during training it samples another clip from the
*same speaker* as the conditioning reference. Radio drama is multi-speaker and
nothing in the pipeline diarizes it, so every clip carries one speaker label. The
model therefore learns that the reference clip does **not** predict the output
voice — which is exactly the capability being fine-tuned.

Expect this baseline to teach **Sinhala phonetics and prosody**, not controllable
speaker identity. Speaker diarization is the fix and it belongs upstream in the
dataset build; retrofitting it across a finished corpus is far more expensive
than adding it now.

---

## Filtering

```bash
python prepare_for_xtts.py --src ../dataset --out xtts_metadata.csv
python prepare_for_xtts.py --src ../dataset --quality all      # include the 11 kHz serial
python prepare_for_xtts.py --src ../dataset --allow-unsnapped  # keep unverified cut points
```

Defaults keep only `hifi` clips (the 201-episode serial is 11 kHz / 16 kbps, so
its fricatives are largely gone), under 11.4 s, and VAD-verified. Every drop is
counted and printed.

---

## Reading the training curves

**Watch `loss_mel_ce`** — the acoustic reconstruction term. `loss_text_ce` carries
weight 0.01 in the total and mostly reflects the new Sinhala embeddings settling
in; a steep early drop there is expected and is not progress on audio quality.

`monitor.py` writes a verdict; when it says `overfitting`, export at `best_step`
rather than the last checkpoint.

---

## Expectations

At the time of writing the hi-fi tier is ~16 episodes ≈ 2.9 h raw, which filters
to well under 2 h. That is enough to **validate the pipeline end to end** and far
short of what an XTTS fine-tune needs to judge quality on. Run the smoke cell
first — it catches wiring faults in two minutes rather than two hours.
