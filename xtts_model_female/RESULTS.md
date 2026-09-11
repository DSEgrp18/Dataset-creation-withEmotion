# Run log — XTTS-v2 Sinhala female

Every metric in `evaluate_xtts.py` is implementation-dependent, MCD most of all. The
numbers here are only meaningful **against each other**, which is what this file is for:
one row per run, with enough configuration recorded that a difference between two rows
can be attributed to something.

Append a row when a run finishes. Do not edit an old row — if a run is re-scored under a
changed `evaluate_xtts.py`, that is a new row.

---

## Summary

| # | Date | Checkpoint | Steps | MCD | F0 corr | SECS | Fail % | UTMOS synth / real |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-08-19 | `GPT_XTTS_si_female-August-19-2026_12+15PM-b292719` | ~5 000 | 63.13 | 0.399 | 0.700 | 2.5 | 2.74 / 3.27 |
| 2 | 2026-08-22 | `GPT_XTTS_si_female-August-22-2026_11+42AM-3c817d0` | 5 850 | 63.34 | 0.384 | 0.701 | 1.2 | 2.73 / 3.27 |
| 3 | 2026-08-22 | `GPT_XTTS_si_female-August-22-2026_04+41PM-03c7fa2` | 5 850 | 63.04 | 0.429 | 0.705 | **0.0** | — |
| 4 | 2026-08-23 | first run past the OOM ceiling — best at step **15 800** | ~22 000 | 63.67 | 0.407 | 0.704 | 3.8 | 2.68 / 3.27 |
| 5 | 2026-09-10 | `GPT_XTTS_si_female-September-10-2026_04+52PM-9b1b514` — best at step **14 080** | ~22 950 | 62.99 | 0.448 | 0.710 | 2.5 | 2.70 / 3.27 |

---

## Reading MCD on this script's scale

**63.13 dB is not comparable to the 13.27 / 20.56 dB that SPECOM 2025 reports.** Different
implementation, different scale. To find out what this script's numbers mean, run
`mcd_and_f0()` over pairs whose relationship is already known — measured on real speech
from `dataset/Muwan_Palassa_140113/clips/`:

| Pair | This script's MCD |
|---|---|
| file vs itself | **0.00** |
| pitch-shifted +2 semitones | 45.9 |
| band-limited to 8 kHz | 56.6 |
| **run 1 vs held-out reference** | **63.1** |
| clean speech + white noise at 30 dB SNR | 65.2 |
| unrelated real speech, different sentences | **108 – 154** |
| speech vs white noise / silence / a 440 Hz tone | 192 – 215 |

The load-bearing reading: **63 sits nowhere near the 108–154 "unrelated speech" band.** The
model lands beside *the same utterance with a mild degradation applied*, so it is tracking
target content and voice, not generating plausible-sounding noise. That is the check that
distinguishes a working fine-tune from the `[UNK]` failure described in the README, where
`loss_mel_ce` falls convincingly and the audio is babble.

Caveat on the calibration itself: it was measured on radio-drama clips (lossy 2013
broadcast, multi-speaker) because that is the audio in this repo, not on VoiceMakers studio
recordings. The bands would shift on cleaner speech. The **ordering and order of magnitude**
are what to rely on, not the exact boundaries.

---

## Run 1 — 2026-08-19

`GPT_XTTS_si_female-August-19-2026_12+15PM-b292719`

### Data

| | Clips | Hours | Text source |
|---|---|---|---|
| dinithi | 2 462 | 4.69 | corpus romanisation |
| harini | 1 135 | 2.11 | **transliterated from Sinhala script** (no romanised column) |
| train / eval | 3 517 / 80 | 6.81 | |

Dropped: 56 `unreadable_wav`, 47 `longer_than_11.6s`.
Tokeniser gate: 225 558 tokens, **`[UNK]` = 0**, length median 63 / p95 89 / max 114.

### Configuration

Effective batch 64 (`4 × 16`), `lr=1e-5`, `save_step=1000`, fp16 off, 40 epochs requested.
Eval: 80 clips, temperature 0.75, seed 1234.

### Results

| Scope | MCD dB | log-F0 RMSE (cents) | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| best_model | 63.13 | 359.5 | 0.399 | 0.700 | 0.968 | 2.5 | 0.537 |
| dinithi | 60.76 | 317.4 | 0.550 | 0.679 | 1.001 | 0.0 | 0.535 |
| harini | 65.50 | 401.6 | 0.248 | 0.721 | 0.936 | 5.0 | 0.539 |

| Scope | UTMOS synth | UTMOS real recordings |
|---|---|---|
| best_model | 2.74 | 3.27 |
| dinithi | 2.78 | 3.37 |
| harini | 2.71 | 3.17 |

### What these say

- **Duration ratio 0.968 and failure rate 2.5 % are the strongest result.** XTTS is
  autoregressive; truncation and runaway looping are its characteristic failures and both
  land in these two columns. Well-formed utterances of about the right length.
- **UTMOS gap 0.53** (2.74 vs 3.27). The real recordings only score 3.27 because UTMOS is
  English-trained and under-rates Sinhala, so the *gap* is the signal, not the absolute.
- **SECS 0.700** — the voice is recognisably in the right region, not tightly matched.
  Measured against a held-out clip, never the conditioning clip.
- **F0 correlation 0.399 is the weak spot.** Intonation contours only loosely track the
  reference. Consistent with an undertrained model: prosody is the last thing to arrive.

### The dinithi / harini split — the most actionable finding

| | MCD | F0 corr | Fail % | Hours | Text path |
|---|---|---|---|---|---|
| dinithi | 60.8 | 0.550 | 0.0 | 4.69 | corpus romanisation |
| harini | 65.5 | 0.248 | 5.0 | 2.11 | transliterated from script |

Harini is worse on every axis except SECS. **Two causes are confounded**: less than half the
data, *and* the fallback text path. `sinhala_to_ascii(script)` agrees with
`fold(romanisation)` on only 96.6 % of lines (README), so ~3.4 % of harini's text carries
systematically different spellings from what dinithi's shares with the pretrained tokens.

That is separable, and cheaply: **run dinithi through the transliterator too** and re-score.
If her numbers fall, the text path is implicated; if they hold, it is a pure data-volume
effect and the fix is more of harini.

### Open question

The run directory is stamped `12+15PM` and `model.pth` was last written at `14:16` — about
**2 hours against an 8.5 h budget**, roughly 5 of 40 epochs at ~1.42 s/step. If that is
right, every number above is from a substantially undertrained model and there is real
headroom left. The training-cell tail would settle it: budget reached, disk guard, or an
unprompted exit. **Not yet confirmed.**

---

## Run 2 — 2026-08-22

`GPT_XTTS_si_female-August-22-2026_11+42AM-3c817d0`

Same dataset and configuration as Run 1. The pipeline produced `curves.png` and
`next_run.md` automatically for the first time.

### Training

| | |
|---|---|
| reached | global step **5 850**, epoch **6 of 40** |
| wall clock | **2.27 h** at **1.39 s/step** |
| train `loss_mel_ce` | 4.6359 → 2.5994 over 118 points |
| eval `loss_mel_ce` | best **2.8442 at step 5 250** — the last of 6 evals |
| curve verdict | **`improving`** — every eval was better than the one before |

The curve is textbook: train falling with noise, eval falling smoothly and
monotonically, no divergence, no gap opening. **Nothing about this run says the model
had stopped learning.** It stopped for an external reason.

### Results

| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| best_model | 63.34 | 357.8 | 0.384 | 0.701 | 0.994 | 1.2 | 0.554 |
| dinithi | 61.03 | 315.6 | 0.534 | 0.677 | 1.020 | 0.0 | 0.557 |
| harini | 65.64 | 400.0 | 0.233 | 0.724 | 0.968 | 2.5 | 0.550 |

| Scope | UTMOS synth | UTMOS real |
|---|---|---|
| best_model | 2.73 | 3.27 |
| dinithi | 2.71 | 3.37 |
| harini | 2.75 | 3.17 |

### Run 1 vs Run 2 — an accidental repeatability measurement

Both runs stopped at essentially the same amount of training (≈5 000 vs 5 850 steps,
both epoch 5–6 of 40), on the same data and configuration. So the difference between
them is **mostly evaluation noise, and that makes it useful**: it is the first estimate
of what counts as a real change in this pipeline.

| Metric | Run 1 | Run 2 | Δ |
|---|---|---|---|
| MCD dB | 63.13 | 63.34 | +0.21 |
| log-F0 RMSE | 359.5 | 357.8 | −1.7 |
| F0 corr | 0.399 | 0.384 | −0.015 |
| SECS | 0.700 | 0.701 | +0.001 |
| duration ratio | 0.968 | 0.994 | +0.026 |
| failure rate | 2.5 % | 1.2 % | −1.3 pt |
| UTMOS | 2.74 | 2.73 | −0.01 |

**Read: a change smaller than ~0.3 dB MCD or ~0.02 F0 corr is not evidence of anything.**
The `compare_quality.py` tolerances (2 % of MCD ≈ 1.3 dB, 0.02 on F0 corr) sit just
outside this spread, which is the right side to be on.

Two real improvements did land, both plausibly from the extra 850 steps: duration ratio
moved from 0.968 to 0.994, and the failure rate halved. Those are the metrics that track
autoregressive stability, and they are the ones that improve first.

### Why this run did not beat Run 1

**Because it trained the same amount.** Both stopped around epoch 6 of 40, at ~2.3 h of
an 8.5 h budget. Nothing was learned about the model between them; what was learned is
that *the run stops early for a reason neither report captured.*

That gap is now closed: the training cell records `run_status.json` (stop reason, exit
code, wall clock against budget, and the last 40 log lines), and `next_run_report.py`
puts it at the top of section 2 and raises a **do first** finding when a run uses less
than 60 % of its budget. Run 3 will say which of budget, NaN guard, disk guard or an
unexpected process exit ended it.

---

## Run 3 — 2026-08-22 — the one that found the ceiling

`GPT_XTTS_si_female-August-22-2026_04+41PM-03c7fa2`

Same data and configuration again. This run carried the new `run_status.json`
instrumentation, and it answered the question the first two could not.

### Why every run stopped at step 5850

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 14.81 MiB is free.
Of the allocated memory 13.31 GiB is allocated by PyTorch,
and 1.08 GiB is reserved by PyTorch but unallocated.
```

**This is fragmentation, not a batch that is too large**, and the traceback says so
itself. Three things establish it:

- the failing allocation is **14 MiB** — trivially small
- **1.08 GiB is reserved but unallocated** — the memory exists, it is not contiguous
- it ran **two healthy hours** first. A batch that does not fit fails in the first
  minute, not the third hour.

Clip lengths vary and `batch_group_size=48` re-sorts them, so block sizes keep changing
and the caching allocator's heap slowly fragments until no contiguous block is left.

**Runs 2 and 3 both died at exactly global step 5850, both with eval best at step 5250.**
Same data, same seed, same batch order — the fragmentation is deterministic, so it
recurs at precisely the same step. Every run so far has been capped at epoch 6 of 40 by
this and nothing else.

### Fixes applied

| Change | Where |
|---|---|
| `PYTORCH_ALLOC_CONF=expandable_segments:True` (and the older `PYTORCH_CUDA_ALLOC_CONF`), set before torch imports | `train_xtts_female.py` |
| OOM retry ladder `(4×16) → (3×21) → (2×32)`, resuming from the checkpoint the failed attempt reached, sharing the remaining budget | notebook training cell |
| OOM named explicitly as a finding, with its own fix, instead of the generic "short run" advice | `next_run_report.py` |

The ladder halves peak activation memory per rung while holding
`batch_size × grad_accum` near 64, so the effective batch — and the learning dynamics —
barely move. It only retries on a genuine OOM with more than ten minutes of budget left;
NaN, disk and budget stops break out immediately, because a smaller batch changes
nothing for those.

### Results

| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| best_model | 63.04 | 342.3 | 0.429 | 0.705 | 0.982 | **0.0** | 0.520 |
| dinithi | 60.22 | 308.1 | 0.556 | 0.691 | 1.007 | 0.0 | 0.519 |
| harini | 65.85 | 376.5 | 0.301 | 0.718 | 0.956 | 0.0 | 0.522 |

Training: step 5 850, epoch 6/40, **2.02 h at 1.24 s/step**, eval best **2.8503 at step
5 250**, verdict `improving` — every one of the 6 evals beat the one before it.

### Three runs at the same step count

| Metric | Run 1 | Run 2 | Run 3 | spread |
|---|---|---|---|---|
| MCD dB | 63.13 | 63.34 | 63.04 | 0.30 |
| log-F0 RMSE | 359.5 | 357.8 | 342.3 | 17.2 |
| F0 corr | 0.399 | 0.384 | 0.429 | 0.045 |
| SECS | 0.700 | 0.701 | 0.705 | 0.005 |
| failure rate | 2.5 % | 1.2 % | 0.0 % | 2.5 pt |
| harini F0 corr | 0.248 | 0.233 | 0.301 | 0.068 |

All three trained the same amount, so this spread is **the pipeline's noise floor, not
progress**. Run 3 is nominally best on almost every metric, and that is exactly the trap
this table exists to prevent: at ±0.3 dB MCD and ±0.045 F0 corr, none of the differences
between these runs mean anything. Do not read Run 3 as an improvement over Run 2.

The one movement that looks real is **failure rate 2.5 % → 1.2 % → 0.0 %**, monotone
across three runs. Worth watching, not yet worth believing.

**The honest summary of runs 1–3: no model progress at all.** Three sessions were spent
discovering that training stops at epoch 6, and the third one found out why.

---

## Run 4 — 2026-08-23 — converged, and the first real information

The OOM fix worked: **~22 000 steps (~25 epochs) against the 5 850 every earlier run was
capped at.** This is the first run whose result says anything about the model rather than
about the harness.

### Training

| | |
|---|---|
| steps | ~22 000, ~25 epochs |
| eval `loss_mel_ce` | best **2.7480 at step 15 800**, ending 2.7743 |
| verdict | **`plateau`** — flat within noise for 7 evals |
| train `loss_mel_ce` | ~2.28 at the end |

The shape is unambiguous. Eval falls steeply to ~10 000, flattens, bottoms at 15 800, and
**turns very slightly upward** while train keeps falling to 2.28. The train/eval gap opens
from ~0.1 early to **~0.49** at the end.

**Training longer is finished as a lever.** 2.850 → 2.748 is a real 3.6 % improvement in
held-out loss, and it is all the improvement there was; steps 15 800–22 000 bought nothing
and began to cost.

### Results

| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| best_model | 63.67 | 353.4 | 0.407 | 0.704 | 1.026 | 3.8 | 0.495 |
| dinithi | 61.88 | 326.1 | 0.528 | 0.681 | **1.057** | 5.0 | 0.495 |
| harini | 65.46 | 380.8 | 0.287 | 0.728 | 0.995 | 2.5 | 0.495 |

UTMOS 2.68 synth / 3.27 real (dinithi 2.70/3.37, harini 2.66/3.17).

### 3.8× the training did not improve the output

| Metric | Run 3 (5 850 steps) | Run 4 (best @ 15 800) | beyond noise? |
|---|---|---|---|
| **eval loss** | 2.8503 | **2.7480** | yes — the only clear win |
| MCD dB | 63.04 | 63.67 | see below |
| F0 corr | 0.429 | 0.407 | no (spread 0.045) |
| SECS | 0.705 | 0.704 | no |
| duration ratio | 0.982 | 1.026 | marginal |
| failure rate | 0.0 % | **3.8 %** | **yes — worse** |
| UTMOS | 2.73 | **2.68** | **yes — worse** |

**The MCD difference is probably not real.** Failed clips are included in the pooled MCD
mean, and a truncated or looping clip scores catastrophically — on the calibration scale
above, near the 108–154 "unrelated speech" band. Three such clips in 80 shift the mean by
several dB on their own, which is far more than the 0.63 dB separating these two runs.
Run 3 had zero failures; Run 4 has three. `evaluate_xtts.py` now reports a second table
over non-failed clips only, so future comparisons do not confuse "the model got worse"
with "three clips blew up" — those have different fixes.

**What is real: the model is over-generating.** Failure rate 0 → 3.8 %, duration ratio
0.982 → 1.026, and dinithi at **1.057** — 5.7 % longer than the reference. UTMOS down 0.05
against a run-to-run spread of 0.01. All four point the same way: the autoregressive
decoder is running past where it should stop.

That is the classic cost of training an AR TTS model to convergence on a small corpus.
The model grows confident on training-like sequences, and stop-token prediction degrades
on held-out text. The loss says it fits better; the generator says it is less stable.

### The state of things after four runs

**6.81 h of audio gets XTTS-v2 to roughly `loss_mel_ce` 2.75 and no further, and quality
tops out before the loss does.** Nothing in the training configuration is now the binding
constraint. The next gains have to come from data or from decoding, not from more steps.

---

---

## Run 5 — 2026-09-10 — a replication, and the deployment pipeline proved end to end

`GPT_XTTS_si_female-September-10-2026_04+52PM-9b1b514`

Same data and configuration as Run 4. The model is not the interesting part — it landed
in the same place — but this was the first run to carry the checkpoint sweep, the
strip/fp16 export and the benchmark through to numbers.

### Training

| | |
|---|---|
| reached | global step **22 950**, epoch **26 of 40** |
| wall clock | **8.48 h** at **1.33 s/step**, ended by the 8.5 h budget (exit −15) |
| train `loss_mel_ce` | 4.6359 → 2.5604 over 460 logged points |
| eval `loss_mel_ce` | best **2.7502 at step 14 050**, last 2.7750, over 26 evals |
| curve verdict | **`plateau`** — flat within noise for 10 evals |
| written as best | 8 800, 9 680, 10 560, 11 440, 13 200, **14 080** |

Run 4 bottomed at **2.7480 @ step 15 800**; this run at **2.7502 @ step 14 050**. That is
a **replication, not a difference** — 0.0022 of held-out loss, on the flat part of the
curve. Two independent runs now agree: **6.81 h of audio takes XTTS-v2 to `loss_mel_ce`
≈ 2.75 and no further.** Every guard behaved — no NaN, no OOM, no disk stop; the budget
ended it.

### Results — `best_model.pth`, which is the step-14 080 copy

| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| best_model | 62.99 | 346.9 | 0.448 | 0.710 | 1.007 | 2.5 | 0.521 |
| dinithi | 61.04 | 315.3 | 0.564 | 0.686 | 1.037 | 2.5 | 0.520 |
| harini | 64.93 | 378.4 | 0.331 | 0.735 | 0.977 | 2.5 | 0.522 |

Over the 78 of 80 clips that did not fail generation:

| Scope | Clips | MCD dB | log-F0 RMSE | F0 corr | SECS |
|---|---|---|---|---|---|
| best_model | 78/80 | 63.12 | 349.0 | 0.443 | 0.711 |
| dinithi | 39/40 | 61.13 | 318.4 | 0.558 | 0.687 |
| harini | 39/40 | 65.11 | 379.6 | 0.329 | 0.735 |

UTMOS 2.70 synth / 3.27 real (dinithi 2.71/3.37, harini 2.68/3.17).

**The harini gap is now five for five.** F0 corr 0.331 against dinithi's 0.564, and it has
appeared in every run this file records. Less audio and a different text path remain
confounded.

### Run 4 vs Run 5

| Metric | Run 4 | Run 5 | beyond noise? |
|---|---|---|---|
| eval loss | 2.7480 | 2.7502 | no — a replication |
| MCD dB | 63.67 | 62.99 | **see below** |
| F0 corr | 0.407 | 0.448 | no (spread 0.045) |
| SECS | 0.704 | 0.710 | marginal (spread 0.005) |
| duration ratio | 1.026 | 1.007 | marginal |
| failure rate | 3.8 % | 2.5 % | no (spread 2.5 pt) |
| UTMOS | 2.68 | 2.70 | marginal (spread 0.01) |

**Do not read the 0.68 dB MCD gain as progress.** Run 4 pooled three failed clips into its
mean and Run 5 pools two, and a failed clip scores in the 108–154 "unrelated speech" band
— enough to move a mean by more than 0.68 dB on its own. Run 4 predates the non-failed
table, so there is no like-for-like `_ok` comparison to be had. The honest reading is that
**Run 5 is consistent with Run 4 on every axis.**

### The checkpoint sweep — step 2, and it did not settle

Both surviving checkpoints, same decode config, same seed:

| Checkpoint | eval loss | Fail % | Dur. | MCD (ok) | F0 corr | SECS | UTMOS |
|---|---|---|---|---|---|---|---|
| `checkpoint_22000` | 2.7750 | **0.0** | 1.015 | **62.69** | 0.436 | 0.715 | 2.68 |
| `best_model_14080` | **2.7502** | 2.5 | **1.007** | 63.12 | **0.443** | 0.711 | **2.70** |

Per speaker, fail % / MCD (ok) / F0 corr:

| Checkpoint | dinithi | harini |
|---|---|---|
| `checkpoint_22000` | 0.0 / 60.56 / 0.586 | 0.0 / 64.81 / 0.286 |
| `best_model_14080` | 2.5 / 61.13 / 0.558 | 2.5 / 65.11 / 0.329 |

The checkpoint with the **worse** eval loss has zero failures and 0.43 dB better MCD,
which is the shape the Run 4 notes predicted: best-eval-loss may be the wrong export
criterion for an autoregressive model. But 0.43 dB sits barely outside a 0.30 dB noise
floor, the failure difference is **two clips in eighty**, and everything else is inside
noise. **One seed cannot settle this, and it is now unsettleable:** `save_n_checkpoints=1`
deleted the four earlier bests, and `checkpoint_22000.pth` lived on `/kaggle/temp` and died
with the session. Only `best_model.pth` was mirrored.

The fix for next time is to mirror **stripped** 1.9 GB checkpoints instead of 5.5 GB ones —
several fit in the 20 GB quota, where two do not.

### Deployment — 5.608 GB → 1.868 GB → 0.934 GB

`optimize_checkpoint.py --strip` on the real export dropped exactly what it claims:

```
optimizer state    : dropped -- optimizer
scheduler state    : dropped -- scheduler
gradient scaler    : dropped -- scaler
training metadata  : dropped -- step, epoch, model_loss, date, config
bit-identical      : 963
```

**963 inference tensors, every one bit-for-bit identical to the original.** `--verify`
proves that structurally; `compare_quality.py` then proved it independently by measurement
— `+0.0000` on MCD, F0 RMSE, F0 corr, SECS and UTMOS, reported as *Metrics are EXACTLY
equal*, at **67 % smaller**. The strip is free, and that is now demonstrated twice by
different means rather than argued from the format.

The fp16 gate passed with every metric inside tolerance:

| Metric | slim fp32 | fp16 | change |
|---|---|---|---|
| MCD dB | 62.9863 | 62.7917 | −0.1946 |
| log-F0 RMSE | 346.8558 | 347.5275 | +0.6717 |
| F0 corr | 0.4476 | 0.4428 | −0.0048 |
| SECS | 0.7104 | 0.7101 | −0.0003 |
| UTMOS | 2.6967 | 2.7172 | +0.0205 |
| failure rate | 2.5 % | 0.0 % | −2.5 pt |

**fp16 is not better.** It is nominally ahead on four of six, and every one of those deltas
is inside the runs 1–3 noise floor — the failure difference is two clips. The defensible
claim is that **at n=80 and one seed, fp16 storage is not distinguishable from fp32, at
half the size.**

### What it costs to run — Tesla T4

| Configuration | Size GB | Load s | Peak VRAM GB | RTF mean | First audio s |
|---|---|---|---|---|---|
| exported | 5.608 | 16.7 | **2.47** | 0.501 | 0.51 |
| slim fp32 | 1.868 | 14.0 | **2.47** | 0.494 | 0.48 |
| fp16 storage | 0.934 | **13.3** | **2.47** | 0.507 | 0.48 |

**Peak VRAM is identical across a 6× spread in file size.** `load_state_dict` casts each
tensor to the dtype of the parameter receiving it, so an fp16 *file* becomes an fp32
*model* in VRAM. fp16 storage buys disk and ~3.4 s of load time; it does not buy memory,
and it does not lower the GPU tier the model needs. At **2.47 GB it fits any 4 GB card.**

RTF came back 0.494–0.507 across identical weights, so **RTF differences under ~10 % in
this table mean nothing.** First audio at ~0.48 s is the number a listener notices, and it
is not RTF.

A fourth row, `fp16-compute`, is **missing and was never measured**: the notebook built its
`--half` argument and failed to pass it, so the row it produced was a duplicate of
slim-fp32 wearing another name. fp16 compute is the only lever here that could move RTF.

### What Run 5 settled

- **The deployment pipeline works and its central claim is proved.** 5.608 → 1.868 GB is
  lossless by construction *and* by measurement; 0.934 GB costs nothing detectable at this
  sample size. All three size targets met.
- **Training is finished as a lever**, confirmed by replication rather than asserted.
- **Best-eval-loss as export criterion is still open**, and this run's checkpoints can no
  longer answer it.

## Next experiments, in order of expected value

Revised after Run 5. **Training longer is off this list for good** — Runs 4 and 5 bottomed
at 2.7480 and 2.7502 independently, so the plateau is replicated, not a fluke of one run.
~~Get past step 5850~~ done. ~~Prove the deployment pipeline~~ done in Run 5: 5.608 →
1.868 GB lossless → 0.934 GB with nothing measurable lost.

Each has a command, and the register is where the answers go.

1. **Settle fp16 against fp32 with more seeds — the cheapest open question.** Run 5's
   comparison was one seed, and fp16 came out nominally ahead on four of six metrics with
   every delta inside the noise floor. Two more seeds say whether anything survives
   resampling. Until then the shipping claim is "not distinguishable", not "equal".
   → `sweep_eval.py --checkpoints model_slim.pth model_fp16.pth --seeds 1235,1236`
2. **Measure fp16 compute.** Never measured — Run 5's notebook built the `--half` argument
   and did not pass it, so that row duplicated slim-fp32. It is the only lever that can
   move RTF, and it changes arithmetic, so it needs its own gate before shipping.
   → `benchmark.py --checkpoint model_slim.pth --tag fp16-compute --half`
3. **Decode tuning — now lower value than Run 4 implied.** Run 5's deployment candidate
   already reports 0.0 % failures and duration ratio 1.005 at t=0.75, so the
   over-generation this was meant to fix has largely gone. Two temperatures, not ten.
   → `sweep_eval.py --checkpoints model_fp16.pth --temperature 0.65,0.7`
4. **Preserve checkpoints so the export criterion can be tested at all.** Run 5 could not
   settle whether best-eval-loss is the right export: `save_n_checkpoints=1` deleted four
   earlier bests and `checkpoint_22000.pth` died with the session, leaving two candidates
   that differ by less than noise. Mirror **stripped** 1.9 GB checkpoints — several fit the
   20 GB quota where two 5.5 GB ones do not.
   → `optimize_checkpoint.py --strip` inside the training cell, before mirroring.
5. **MOS / SUS panel.** Now genuinely worth the effort — the model has converged, and no
   objective metric here can say whether `loss_mel_ce` 2.75 *sounds* acceptable in Sinhala.
   `listening_test.html` is already built. Have a native speaker vet `answer_key.json` for
   ungrammatical SUS items first, then report blind and sighted raters separately.
6. **Transliterate dinithi's text too**, to decouple text path from data volume in the
   harini gap (F0 corr 0.331 vs 0.564 in Run 5, and it has persisted across all five).
   → inference side, free: `evaluate_xtts.py --text-from script`.
   → training side, one run: `prepare_voicemakers.py --text-path script`.
7. **Speaker-balanced sampling.** dinithi has 2 462 clips to harini's 1 135, so roughly
   68 % of gradient steps teach dinithi's voice — which is one of the two candidate causes
   of the harini gap, and the one that is separable from the text path by experiment 4.
   → one run: `prepare_voicemakers.py --balance-speakers oversample`.
8. **More audio, especially harini.** With training converged at 6.81 h, this is the only
   lever left that raises the ceiling rather than moving along it. The radio-drama corpus
   upstream is the obvious source once it is diarised.

---

## The experiment register

Runs get a section above. **Experiments** — same weights or same data, one thing varied —
get a row in `experiments.csv`, written by `sweep_eval.py` and `compare_quality.py`, with
the decode configuration and the per-speaker columns in the row itself. Two rows are only
comparable if their `temperature` / `repetition_penalty` / `top_k` / `top_p` / `text_from`
columns match, which is why those are columns and not prose.

The register is the input to this file, not a replacement for it: promote a result up here
once it has survived a second seed, and say which experiment id it came from.

Nothing in a register row is an improvement until it beats the run-to-run spread measured
across runs 1–3: **~0.3 dB MCD, ~0.045 F0 correlation, ~0.005 SECS, and 2.5 points of
failure rate.** That spread came from three runs that trained identically, so it is noise
by construction.

---

## Deployment size — measured, Run 5

Every number below is measured, not targeted. Method: `optimize_checkpoint.py --strip`
(and `--fp16`), verified with `--verify`, gated with `compare_quality.py`, timed with
`benchmark.py`. Full detail in the Run 5 entry above.

| Artifact | Target | **Measured** | Quality |
|---|---|---|---|
| training checkpoint | ~5.6 GB | **5.608 GB** | — |
| stripped fp32 | ~1.9 GB | **1.868 GB** | **exactly equal** — 963 tensors bit-identical |
| fp16 storage | ~0.95 GB | **0.934 GB** | no metric outside the noise floor |

The one result that changes a deployment decision: **peak VRAM is 2.47 GB for all three**,
because an fp16 file loads into an fp32 model. Shrinking the file does not shrink the card
you need — it buys download size and ~3.4 s of load time, which matters for a
scale-to-zero host paying for cold starts and not much otherwise.

Not measured: **fp16 compute** (`--half`), the only lever here that could move RTF. It
would need its own quality gate before shipping, because it changes arithmetic rather than
storage.
