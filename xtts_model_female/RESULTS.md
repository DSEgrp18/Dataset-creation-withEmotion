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

## Next experiments, in order of expected value

1. **Train longer.** Nothing else is worth tuning until a run has used its full budget.
   Resume from the mirror with `--continue-path`.
2. **Transliterate dinithi's text too**, to decouple text path from data volume in the
   harini gap.
3. **MOS / SUS panel.** `listening_test.py` already wrote `listening_test.html` (8.7 MB,
   audio embedded). Have a native speaker vet `answer_key.json` for ungrammatical SUS items
   first, then report blind and sighted raters separately.
4. **More harini**, if (2) says the gap is data volume rather than spelling.
