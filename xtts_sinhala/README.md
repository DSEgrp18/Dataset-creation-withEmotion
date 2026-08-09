# XTTS-v2 on Sinhala, the version that works

A minimal fine-tune of XTTS-v2 on
[pathnirvana/sinhala-tts-dataset](https://github.com/pathnirvana/sinhala-tts-dataset),
following the official Coqui recipe
([`recipes/ljspeech/xtts_v2/train_gpt_xtts.py`](https://github.com/idiap/coqui-ai-TTS/blob/main/recipes/ljspeech/xtts_v2/train_gpt_xtts.py)
and [`TTS/demos/xtts_ft_demo`](https://github.com/idiap/coqui-ai-TTS/tree/main/TTS/demos/xtts_ft_demo))
with one change to the text pipeline and one to the batching.

Upload [`kaggle_xtts_sinhala.ipynb`](kaggle_xtts_sinhala.ipynb) to Kaggle, set the
accelerator to GPU, Run All. Nothing needs to be attached — the notebook downloads
the corpus and the base model itself.

---

## Why the previous attempt produced nothing usable

XTTS-v2's tokenizer is not what it looks like. Checked against the real
`vocab.json`:

```
pre_tokenizer : Whitespace
model         : BPE
unk_token     : "[UNK]"
vocab size    : 6681
```

It is a **whitespace-pretokenised BPE with an `[UNK]` fallback**, not a byte-level
BPE. When a word contains any character outside the vocabulary, the *entire word*
becomes one `[UNK]` token — not the offending character, the whole word. Sinhala
(U+0D80–U+0DFF) is absent from the vocabulary, so every word in the corpus becomes
`[UNK]`, and the model is trained to map "unknown unknown unknown" onto audio.
`loss_mel_ce` falls the whole way down and the samples are babble. That is not a
data problem and no amount of extra epochs fixes it.

`extend_vocab.py` in the old pipeline was aimed at exactly this, and the diagnosis
was right. But adding ~80 rows to the token embedding matrix means those rows start
from random initialisation, and the published new-language XTTS ports that take that
route (viXTTS and similar) use hundreds of hours and tens of thousands of optimiser
steps to train them. The old run had **~1.5 h of audio and roughly 100 optimiser
steps** — 12 epochs at an effective batch of 84. Even with a perfect vocabulary
extension there was nothing there to succeed.

VITS did better because VITS trains a character embedding table from scratch, so
Sinhala script costs it nothing.

---

## What this does instead

**Spell Sinhala with characters XTTS already knows.**

pathnirvana ships every line twice — an ISO-style romanisation and Sinhala script.
[`sinhala_text.py`](sinhala_text.py) folds either one to plain ASCII using Sri Lankan
"Singlish" conventions, chosen so the pretrained English tokens already carry roughly
the right sound:

| Sinhala | | ASCII | why |
|---|---|---|---|
| ට ඩ | retroflex | `t` `d` | English t/d are the nearest match |
| ත ද | dental | `th` `dh` | the h-digraph, as Sri Lankans write it |
| ඛ ඝ ඡ ඣ ඨ ඪ ථ ධ ඵ භ | aspirated | dropped | modern spoken Sinhala has no phonemic aspiration, so ථ and ත are the same sound |
| ා ී ූ ේ ෝ | long vowels | `aa` `ii` `uu` `ee` `oo` | length written as doubling |
| ණ ළ | | `n` `l` | homophones of න and ල |
| ං ඞ ඟ | | `ng` | |
| ශ ෂ | | `sh` | English "sh" *is* /ʃ/ |

Measured over all 6386 lines of the corpus:

```
output charset : " !',-.:;?abcdefghijklmnoprstuvy"     pure ASCII
[UNK] tokens   : 0  out of  380,730
tokens/line    : median 59   p95 96   max 117          (GPTArgs limit is 200)
```

Reproduce it yourself, against the real vocabulary:

```bash
python sinhala_text.py --selftest metadata.csv vocab.json
```

Nothing is randomly initialised, no embedding matrix is resized, and the task drops
from "learn a new script" to "learn a new accent" — which ~8 h of audio can actually
do.

`sinhala_to_ascii()` also converts Sinhala script directly, for inference. It agrees
with `fold(romanisation)` on **96.6%** of lines; the residue is quote style, the
`-පෙ-` ellipsis placeholder rows, and 175 `t`+`h` clusters the romanisation writes
ambiguously as `th`.

**And use a dataset whose speaker conditioning means something.** XTTS samples a
second clip *from the same speaker* as the conditioning reference during training.
Radio drama under one speaker label taught the model that the reference does not
predict the output voice — the exact capability being fine-tuned. pathnirvana's
`mettananda` split is one voice, studio-recorded, already 22050 Hz mono: ~5400 clips
/ ~11.8 h, of which everything under XTTS's 11.6 s ceiling survives.

---

## Files

| File | Purpose |
|---|---|
| [`sinhala_text.py`](sinhala_text.py) | the fold, both directions, with the selftest |
| [`prepare_pathnirvana.py`](prepare_pathnirvana.py) | download → filter → `metadata_train.csv` / `metadata_eval.csv` |
| [`train_xtts_si.py`](train_xtts_si.py) | the official recipe, deviations documented in the docstring |
| [`infer_xtts_si.py`](infer_xtts_si.py) | Sinhala script in, wav out; `--export` bundles a portable model dir |
| [`kaggle_xtts_sinhala.ipynb`](kaggle_xtts_sinhala.ipynb) | Run All |

Local equivalent of the notebook:

```bash
pip install "coqui-tts>=0.25.1" "coqui-tts-trainer>=0.2.0" soundfile
python prepare_pathnirvana.py --out ./si_dataset --speaker mettananda
python train_xtts_si.py --dataset ./si_dataset --out ./run --smoke   # 2 min
python train_xtts_si.py --dataset ./si_dataset --out ./run --epochs 40
python infer_xtts_si.py --run ./run/training/GPT_XTTS_si-<stamp> \
                        --base ./run/training/XTTS_v2.0_original_model_files \
                        --ref ./si_dataset/wavs/sinh_0042.wav
```

---

## Deviations from the official recipe

Everything in `GPTArgs`, `XttsAudioConfig` and the optimizer block is copied verbatim.
Four things differ:

**`language="en"`.** This selects a tokenizer branch, not a claim about the audio.
`VoiceBpeTokenizer.preprocess_text` raises `NotImplementedError` for anything outside
its 17-language set, and `[si]` is not a token in `vocab.json`. The text is already
ASCII by the time it gets there, so the `en` cleaner path — lowercase, number
expansion, whitespace collapse — is exactly the right one.

**Effective batch 64 instead of 252.** Upstream's note reads *"BATCH_SIZE \*
GRAD_ACUMM_STEPS need to be at least 252 for more efficient training"*, which is
correct advice for a datacentre. On one T4 that is roughly 100 s per optimiser step,
so a 12-hour session buys about 400 steps — nowhere near enough to move a model to a
new sound inventory. `4 × 16 = 64` trades gradient noise for ~4× the steps. Raise it
if you have the hours.

**`lr=1e-5` instead of `5e-6`.** Compensates for the smaller effective batch. Drop
back to `5e-6` if `loss_mel_ce` is noisy or rising.

**`mixed_precision=True`.** Off upstream, on here: roughly 2× throughput on a T4. The
smoke run will show `nan` immediately if fp16 destabilises; pass
`--no-mixed-precision` then.

One thing that *looks* like a deviation but is not: the scheduler milestones are
`[900000, 2700000, 5400000]`, and `Trainer` defaults to `scheduler_after_epoch=True`,
so those are **epochs** and the learning rate never actually decays. That is upstream
behaviour, kept as-is.

---

## What to expect

`loss_mel_ce` is the acoustic reconstruction term and the only one that tracks audio
quality. `loss_text_ce` carries weight 0.01 in the total.

~8 h of one speaker is enough for intelligible, correctly-accented Sinhala in the
trained speaker's voice, and it is the right amount to judge whether the approach is
working. It is not enough for reliable zero-shot cloning of *unseen* speakers — that
capability comes from the pretrained model and this fine-tune will erode it somewhat.
Expect accent artefacts on words whose letter sequences never appeared in training.

Two failure modes worth naming, because they look like model problems and are not:

- **Sinhala text passed straight to the model.** Always run it through
  `sinhala_text.to_ascii()` first. The fold is not optional at inference either —
  raw script produces `[UNK]` and therefore noise.
- **Clips over 11.6 s.** `GPTArgs.max_wav_length = 255995` samples at 22050 Hz. The
  dataloader drops longer clips without a warning. `prepare_pathnirvana.py` drops
  them where the count is printed instead.

## Adding your own emotional data later

The corpus from the parent directory can be appended once it is diarized: fold its
text with the same `sinhala_text.fold`, give each actor a real `speaker_name`, and
concatenate the metadata CSVs. Keeping pathnirvana in the mix is what will hold the
Sinhala phonetics steady while the smaller emotional set teaches prosody.
