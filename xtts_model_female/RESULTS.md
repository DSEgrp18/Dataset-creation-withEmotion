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

## Next experiments, in order of expected value

Revised after Run 4. **Training longer is no longer on this list** — eval loss bottomed at
step 15 800 and rose after, so the training configuration has stopped being the binding
constraint. ~~Get past step 5850~~ is done.

Each of these now has a command. **None of them has been run yet** — the rows below are
the plan, and the register is where the answers go.

1. **Fix the over-generation, at decode time — costs no GPU hours.** Failure rate 0 → 3.8 %
   and duration ratio 1.026 (dinithi 1.057) are the only clear regressions, and both are
   decoding behaviour, not weights. Sweep `--temperature` down from 0.75 (try 0.65, 0.6)
   and `repetition_penalty` up from 5.0 on the *existing* checkpoint, and read
   `failure_rate` and `duration_ratio` from the non-failed table. If failures go to zero
   without MCD moving, that is a free win over Run 4 and it is the cheapest thing here.
   → `sweep_eval.py --checkpoints <run>/best_model.pth --temperature 0.75,0.65,0.6`
2. **Evaluate an earlier checkpoint against step 15 800.** Run 3's model (~step 5 250) had
   0 % failures and a better UTMOS at a worse loss. If an intermediate checkpoint beats
   both, "best eval loss" is the wrong export criterion for this model and the export
   should follow UTMOS/failure rate instead.
   → `sweep_eval.py --all-checkpoints --utmos`
3. **MOS / SUS panel.** Now genuinely worth the effort — the model has converged, and no
   objective metric here can say whether `loss_mel_ce` 2.75 *sounds* acceptable in Sinhala.
   `listening_test.html` is already built. Have a native speaker vet `answer_key.json` for
   ungrammatical SUS items first, then report blind and sighted raters separately.
4. **Transliterate dinithi's text too**, to decouple text path from data volume in the
   harini gap (F0 corr 0.287 vs 0.528, and it has persisted across all four runs).
   → inference side, free: `evaluate_xtts.py --text-from script`.
   → training side, one run: `prepare_voicemakers.py --text-path script`.
5. **Speaker-balanced sampling.** dinithi has 2 462 clips to harini's 1 135, so roughly
   68 % of gradient steps teach dinithi's voice — which is one of the two candidate causes
   of the harini gap, and the one that is separable from the text path by experiment 4.
   → one run: `prepare_voicemakers.py --balance-speakers oversample`.
6. **More audio, especially harini.** With training converged at 6.81 h, this is the only
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

## Deployment size — targets, and what has been measured

The stripping pipeline is built and its offline claims are proved by `--verify`, which
checks every kept tensor for identical dtype, shape and bytes. **The on-GPU numbers below
are targets, not results.** No checkpoint has been through it yet.

| Artifact | Target size | How it is justified | Measured |
|---|---|---|---|
| training checkpoint | ~5.6 GB | what the trainer exports: weights + AdamW state + dvae | — |
| stripped fp32 | ~1.9 GB | weights only; bit-identical, so quality is not at risk | not yet |
| fp16 storage | ~0.95 GB | weights rounded to fp16, arithmetic unchanged | **not yet — must pass the gate first** |

Fill the "measured" column from `experiments/results.csv` (size, load, VRAM, RTF,
first-audio) and the gate output (quality). Until then, "0.95 GB with no degradation" is a
target this repo has tooling for and no evidence of.
