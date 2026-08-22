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
| 1 | 2026-08-19 | `GPT_XTTS_si_female-August-19-2026_12+15PM-b292719` | ~5 k (unconfirmed) | 63.13 | 0.399 | 0.700 | 2.5 | 2.74 / 3.27 |

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

## Next experiments, in order of expected value

1. **Train longer.** Nothing else is worth tuning until a run has used its full budget.
   Resume from the mirror with `--continue-path`.
2. **Transliterate dinithi's text too**, to decouple text path from data volume in the
   harini gap.
3. **MOS / SUS panel.** `listening_test.py` already wrote `listening_test.html` (8.7 MB,
   audio embedded). Have a native speaker vet `answer_key.json` for ungrammatical SUS items
   first, then report blind and sighted raters separately.
4. **More harini**, if (2) says the gap is data volume rather than spelling.
