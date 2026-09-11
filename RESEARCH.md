# Writing this up — what exists, what it shows, and what is still missing

Everything in this repo that a paper could be built from, assembled in one place, with the
evidence status of each claim stated rather than implied. **Numbers here are measured. Where
something is not measured, it says so.**

Source of record for the raw numbers is
[`xtts_model_female/RESULTS.md`](xtts_model_female/RESULTS.md) (per-run detail) and the
experiment registers in each Kaggle run's `experiments/experiments.csv`. This file does not
introduce any number that is not in one of those.

---

## 1. The contribution, honestly assessed

### What is publishable

**Transliteration into a pretrained BPE's existing token space, instead of extending the
vocabulary.**

XTTS-v2's tokenizer is a whitespace-pretokenised BPE with an `[UNK]` fallback, containing no
Sinhala codepoint. Because the pre-tokeniser splits on whitespace and BPE falls back to
`[UNK]` for the *whole word* when any character is unknown, Sinhala script makes every word a
single `[UNK]`; the model learns to map "unknown unknown unknown" onto audio, the loss falls
convincingly, and the samples are noise.

The standard response is to extend the vocab and train new embedding rows from random init —
what published new-language XTTS ports do, with hundreds of hours and tens of thousands of
steps. **This work takes the other route:** spell Sinhala with characters the tokenizer
already knows, using Sri Lankan "Singlish" conventions, and fine-tune with `language="en"`.
Every token is then a pretrained token with a sensible acoustic prior, nothing is randomly
initialised, and the task collapses from *learn a new script* to *learn a new accent*.

Measured consequence: **`[UNK]` = 0 across 225 558 tokens**, verified before any GPU time,
and a working voice from **6.81 h on a single T4**.

The claim generalises beyond Sinhala to any language whose script is absent from a pretrained
TTS vocabulary, which is most of them.

### What is a secondary, methodological contribution

**A measured sampling-noise floor, and the discipline of refusing to claim anything inside
it.** Section 4.3. Autoregressive TTS samples; two evaluations of *identical weights* differ.
Much of the TTS literature reports single-run deltas without establishing what is
distinguishable from resampling. This work measures that floor and then discards several of
its own apparently-positive results because they fall inside it.

### What is NOT a contribution — do not claim it as one

**The compression pipeline.** Stripping optimizer state and storing weights as fp16 is
standard practice. The verification is unusually rigorous (§6) and the engineering is sound,
but "we removed AdamW state from a checkpoint" is not a research finding. It belongs as a
short deployment subsection. Overclaiming it costs reviewer trust that the rest of the paper
needs.

### Unknown, and it decides everything

**Whether Sinhala XTTS fine-tuning has already been published.** Novelty depends entirely on
that and nobody here has done the search. Do it first. If prior work exists, the angle shifts
to the data-efficiency comparison (§9), which is still defensible.

---

## 2. The pipeline, as it would be described in a Method section

```
VoiceMakers corpus ──► prepare_voicemakers.py ──► train_xtts_female.py ──► evaluate_xtts.py
   (Kaggle dataset)      discovery + filtering      XTTS-v2 fine-tune       objective metrics
                         + transliteration          (GPTTrainer, T4)        + sweep_eval.py
                         + [UNK] gate                                       (experiment grid)
                                                            │
                                                            ▼
                                              optimize_checkpoint.py ──► benchmark.py
                                                 strip / fp16 / verify     size, VRAM, RTF
```

### 2.1 Text front-end — `xtts_sinhala/sinhala_text.py`

Two entry points reaching one ASCII target:

- `fold(roman)` — ISO-style romanisation → ASCII, one left-to-right pass with
  longest-match-first alternation (so `ṭh` is consumed before `ṭ`, and `th` before `t`; a
  naive sequence of `str.replace` calls turns `t → th → thh`).
- `sinhala_to_ascii(script)` — Sinhala script → the same ASCII, handling consonants, the
  inherent vowel, vowel signs, virama, anusvara, visarga and ZWJ.

Design decisions, each with a stated phonological rationale:

| Rule | Mapping | Why |
|---|---|---|
| retroflex | ට ඩ → `t` `d` | English `t`/`d` are the closest match |
| dental | ත ද → `th` `dh` | h-digraph, as Sri Lankans write it |
| aspiration | ඛ ඝ ඡ ඣ ඨ ඪ ථ ධ ඵ භ → dropped | modern spoken Sinhala has no phonemic aspiration, so collapsing removes sparsity for free and frees the h-digraph for the dental series |
| long vowels | ා ී ූ ේ ෝ → `aa ii uu ee oo` | length written as doubling |
| homophones | ණ/න, ළ/ල → `n`, `l` | homophones in modern Sinhala |
| sibilants | ශ ෂ → `sh` | English `sh` is exactly /ʃ/ |

**These are hand-designed and unablated.** See §8, gap 4.

Agreement between the two paths on the pathnirvana paired corpus: **92.6 %** of lines
identical (residue: quote style, ellipsis placeholder rows, and 175 `t`+`h` clusters the
romanisation writes ambiguously as `th`). On the VoiceMakers corpus the two speakers take
different paths — see §7.

### 2.2 Data preparation — `xtts_model_female/prepare_voicemakers.py`

Discovers rather than assumes: speaker directories by name substring, `metadata.csv` by
rglob, delimiter by whether the split actually *separates* script from romanisation, column
order by codepoint range and diacritic content, header row by detection. Every decision is
printed.

Filters, each tied to a documented constraint:

| Filter | Threshold | Source of the number |
|---|---|---|
| duration | 1.0 – 11.6 s | `GPTArgs.max_wav_length = 255995` @ 22050 Hz; the dataloader drops longer clips **silently** |
| digits | rows dropped | a digit means the audio speaks a number the text does not contain, and the "en" cleaner would expand it to *English* words inside a Sinhala sentence |
| characters | ≤ 250 | `VoiceBpeTokenizer.char_limits["en"]` |

Then a **hard `[UNK]` gate**: the prepared text is tokenised with the real `vocab.json` and
the script exits non-zero if any `[UNK]` survives. This is the check that distinguishes a
working fine-tune from the failure mode above, and it costs no GPU time.

### 2.3 Training — `xtts_model_female/train_xtts_female.py`

The official Coqui recipe with three deliberate deviations:

| Setting | This work | Upstream | Why |
|---|---|---|---|
| `language` | `"en"` | per-language | selects a tokenizer branch, not a claim about the audio; `[si]` is not a token in `vocab.json` |
| effective batch | 64 (4 × 16) | ≥ 252 | upstream is right for a datacentre; on one T4 that is ~100 s/step, so a 12 h session buys ~400 steps. 64 trades gradient noise for ~4× the steps |
| lr | 1e-5 | 5e-6 | halving the batch four times over shrinks the per-step signal |
| mixed precision | off | off | fp16 drives `loss_mel_ce` to NaN on the first step on a T4 and never recovers; Turing has no bf16 |

Multi-speaker matters structurally: XTTS samples a conditioning clip from the **same speaker**
each step, so two correctly-labelled speakers teach "the reference predicts the output voice".
Pooling both under one label teaches the opposite.

### 2.4 Evaluation — `xtts_model_female/evaluate_xtts.py`

| Metric | What it measures | Caveat carried in the paper |
|---|---|---|
| MCD (dB) | DTW-aligned mel-cepstral distortion | **implementation-dependent**; see §8 gap 3 |
| log-F0 RMSE (cents) | pitch accuracy over DTW-aligned voiced frames | |
| F0 correlation | whether the intonation *contour* tracks the reference | a flat monotone can score decent RMSE |
| SECS | XTTS's own speaker encoder, cosine similarity | measured against a **held-out** clip, never the conditioning clip |
| duration ratio | synth length / reference length | truncation and looping show here before you hear them |
| failure rate | share outside a 0.7–1.4 duration band | |
| RTF | real-time factor | |
| UTMOS | learned MOS predictor | **English-trained**, under-rates Sinhala — the *gap* to real recordings is the signal, not the absolute |

Conditioning is always a *different* clip from the same speaker; using the target itself
would leak the answer into SECS and MCD alike.

Metrics are reported twice: pooled, and **over non-failed clips only**. A truncated or
looping clip still gets an MCD and it is a terrible one, so a handful of failures moves the
pooled mean by more than a real change in the model does.

---

## 3. Dataset

| | Clips | Hours | Text path |
|---|---|---|---|
| dinithi | 2 462 | 4.69 | `fold(romanisation)` |
| harini | 1 135 | 2.11 | `sinhala_to_ascii(script)` — no romanised column exists |
| **train / eval** | **3 517 / 80** | **6.81** | |

Dropped: 56 `unreadable_wav`, 47 `longer_than_11.6s`. Source 44.1 kHz (resampled to 22.05 kHz
on load), clip duration median 6.82 s, max 11.51 s.

Tokeniser gate: **225 558 tokens, `[UNK]` = 0**, length median 63 / p95 89 / max 114 against
a `GPTArgs.max_text_length` of 200.

**The two speakers reached ASCII by different routes**, because harini's metadata carries no
romanised column. That is a confound running through every per-speaker result (§7).

---

## 4. Results

### 4.1 Training

Five runs. Runs 1–3 were capped at step 5 850 by allocator fragmentation on a 16 GB T4 —
diagnosed, fixed with `expandable_segments:True` plus a batch-size retry ladder, and
documented in RESULTS.md. Runs 4 and 5 are the converged ones.

| | Run 4 | Run 5 |
|---|---|---|
| steps | ~22 000 | 22 950 (epoch 26/40) |
| wall clock | — | 8.48 h at 1.33 s/step |
| best eval `loss_mel_ce` | **2.7480 @ step 15 800** | **2.7502 @ step 14 050** |
| verdict | plateau | plateau, flat for 10 evals |

**Two independent runs agree to 0.0022 of held-out loss.** That is a replication, and it is
what licenses the claim that **6.81 h takes XTTS-v2 to `loss_mel_ce` ≈ 2.75 and no further** —
rather than an inference from one curve. Training longer is exhausted as a lever.

### 4.2 Objective quality — Run 5, step 14 080

| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |
|---|---|---|---|---|---|---|---|
| overall | 62.99 | 346.9 | 0.448 | 0.710 | 1.007 | 2.5 | 0.521 |
| dinithi | 61.04 | 315.3 | 0.564 | 0.686 | 1.037 | 2.5 | 0.520 |
| harini | 64.93 | 378.4 | 0.331 | 0.735 | 0.977 | 2.5 | 0.522 |

Non-failed clips only (78/80): MCD 63.12, F0 RMSE 349.0, F0 corr 0.443, SECS 0.711.

UTMOS 2.70 synth / **3.27 real** (dinithi 2.71/3.37, harini 2.68/3.17).

### 4.3 The sampling-noise floor — the methodological result

Three seeds of **identical weights**, so the only variable is XTTS's sampling:

| Seed | MCD (ok) | F0 corr | SECS | Dur. | Fail % | UTMOS |
|---|---|---|---|---|---|---|
| 1234 | 63.121 | 0.443 | 0.711 | 1.007 | 2.5 | 2.697 |
| 1235 | 62.913 | 0.435 | 0.707 | 1.018 | 1.2 | 2.668 |
| 1236 | 63.163 | 0.429 | 0.717 | 1.021 | 2.5 | 2.667 |
| **spread** | **0.25 dB** | **0.014** | **0.010** | **0.014** | **1.3 pt** | **0.030** |

An earlier estimate from three *training* runs gave ~0.30 dB MCD and ~0.045 F0 corr, but
mixed sampling noise with training variation. The two agree closely, which is itself
reassuring.

**Every comparison in this paper is judged against this row.**

### 4.4 Negative results — worth a subsection of their own

**Decode temperature does nothing.** One seed each on the deployment candidate:

| temp | Fail % | MCD (ok) | F0 corr | dinithi F0 | harini F0 |
|---|---|---|---|---|---|
| **0.75** | **0.0** | **62.79** | **0.443** | 0.556 | **0.330** |
| 0.70 | 1.2 | 62.96 | 0.412 | 0.536 | 0.291 |
| 0.65 | 1.2 | 62.97 | 0.430 | 0.572 | 0.291 |

Lowering the temperature **added a failure rather than removing one**, and harini's F0
correlation fell to 0.291 at both lower settings — consistent in direction, larger than the
0.014 floor. Keep 0.75 / 5.0 / 50 / 0.85.

**A unified text path costs nothing at inference.** `--text-from script` re-derives every
clip's text through the transliterator; only **4 of 80 clips changed**.

| | dinithi fail / MCD(ok) / F0 | harini fail / MCD(ok) / F0 |
|---|---|---|
| as prepared | 0.0 / 60.74 / 0.556 | 0.0 / 64.84 / **0.330** |
| unified `script` | 2.5 / 60.83 / 0.579 | 0.0 / 64.84 / **0.330** |

harini is identical to three decimals — **the control working**, since her text already came
through that path. dinithi, fed transliterated text she was not trained on, does not degrade.
So one front-end can serve both speakers at inference. It says nothing about the
training-side question.

---

## 5. MCD calibration — why 63 dB is not the disaster it looks like

This implementation's MCD is **not comparable to published figures**. To establish what its
numbers mean, `calibrate_mcd.py` scores pairs whose relationship is known, on real speech:

| Pair | MCD |
|---|---|
| file vs itself | **0.00** |
| pitch-shifted +2 semitones | 45.9 |
| band-limited to 8 kHz | 56.6 |
| **this model vs held-out reference** | **63.1** |
| clean speech + white noise at 30 dB SNR | 65.2 |
| **unrelated real speech, different sentences** | **108 – 154** |
| speech vs white noise / silence / a 440 Hz tone | 192 – 215 |

The load-bearing reading: **63 sits nowhere near the 108–154 "unrelated speech" band.** The
model lands beside *the same utterance with a mild degradation applied*, so it is tracking
target content and voice, not generating plausible noise.

Caveat to carry into the paper: the calibration was measured on 2013 radio-drama clips, not
on studio recordings. The **ordering and order of magnitude** are what to rely on.

---

## 6. Deployment — a subsection, not a contribution

| Artifact | Size | Quality | How established |
|---|---|---|---|
| training checkpoint | **5.608 GB** | — | |
| stripped fp32 | **1.868 GB** | **exactly equal** | 963 tensors verified bit-identical, *and* `compare_quality.py` returned `+0.0000` on every metric |
| fp16 storage | **0.934 GB** | not worse on any metric | 3 seeds vs fp32 (below) |

fp16 vs fp32, three seeds, means: MCD **62.776 vs 63.066**, F0 corr **0.438 vs 0.436**,
duration ratio **1.0074 vs 1.0150**, failure rate **0.8 % vs 2.1 %**, UTMOS **2.681 vs 2.677**.

> **Record but do not overclaim:** fp16 has the lower MCD at *every* seed by 0.26–0.33 dB and
> the two sets of three barely fail to overlap. There is no mechanism by which rounding
> weights to fp16 improves spectral accuracy, and a clean 3-vs-3 separation happens by chance
> about one time in twenty. State it as unexplained.

Cost to run, Tesla T4:

| | Size GB | Load s | Peak VRAM GB | RTF | First audio s |
|---|---|---|---|---|---|
| exported | 5.608 | 16.7 | **2.47** | 0.501 | 0.51 |
| slim fp32 | 1.868 | 14.0 | **2.47** | 0.494 | 0.48 |
| fp16 | 0.934 | **13.3** | **2.47** | 0.507 | 0.48 |

**Peak VRAM is identical across a 6× spread in file size**, because `load_state_dict` casts
each tensor to the dtype of the parameter receiving it — an fp16 *file* becomes an fp32
*model* in memory. Worth one sentence in the paper as a practitioner-facing correction.

**Not measured: fp16 compute.** `model.half()` collides with fp32 tensors XTTS builds inside
`generate()`. Needs `torch.autocast`. It is the only untested lever on RTF.

---

## 7. The per-speaker result, and a caution about it

fp16 at the shipped decode config, averaged over three seeds:

| Metric | dinithi | harini | Winner |
|---|---|---|---|
| MCD (ok) | **60.79** | 64.76 | dinithi |
| F0 corr | **0.568** | 0.309 | dinithi |
| SECS | 0.689 | **0.739** | **harini** |
| duration ratio | 1.027 | **0.988** | **harini** |
| failure rate | 0.8 % | 0.8 % | tie |
| UTMOS | **2.716** | 2.647 | dinithi |

The obvious reading is "harini is worse, because she has 2.11 h to dinithi's 4.69 h". Three
things argue for stating it more carefully:

1. **Two causes are confounded.** harini has less than half the data *and* the other text
   path. Separating them needs the training-side experiment (§9), not more analysis.
2. **MCD is not calibrated across speakers.** §5 establishes what this scale means for one
   corpus; nobody has measured the per-speaker floor. Two human takes of one sentence have
   nonzero MCD, and that floor depends on the speaker. **60.79 vs 64.76 across speakers is an
   unvalidated comparison.** The eval sentences also differ between speakers.
3. **Informal listening disagreed with the metrics.** A native Sinhala speaker, listening to
   the deployment candidate, judged harini the better voice — consistent with SECS and
   duration ratio, opposite to MCD and F0 corr. n=1, unblinded, so it is a hypothesis and
   not a finding. It is a strong argument for §8 gap 1, and a reason to be careful about
   reference-matching metrics standing in for quality.

---

## 8. What is missing before this is submittable

| # | Gap | Severity | Cost |
|---|---|---|---|
| 1 | **No human evaluation.** MOS/SUS with native speakers. `listening_test.py` builds the kit; it has never been run. | **fatal for a TTS paper** | recruiting, not GPU |
| 2 | **No baseline.** Everything is compared to other runs of itself. | **fatal** | see §9 |
| 3 | **MCD not comparable to literature.** Re-run with a standard implementation (`pymcd`) or report both scales. | high | CPU only |
| 4 | **Transliteration mapping unablated.** The rules in §2.1 are hand-designed. Showing they beat naive ISO romanisation turns a design decision into a finding. | medium | one training run |
| 5 | **Novelty unchecked.** Nobody has searched for prior Sinhala XTTS work. | **decides the framing** | an afternoon |

Gap 1 is also the one your own data most demands: §4.3 shows the objective metrics are either
converged or measured to be noise, and §7 shows they may be mis-ranking the two voices.

---

## 9. The VITS comparison — the strongest available baseline, with one trap

Building a VITS system and comparing is the right move. It is **not** a detour: the product
requires it anyway (`SINHALA_READER_CLAUDE.md` — XTTS cannot run offline; the desktop tier is
VITS exported to Piper's ONNX layout, ~60 MB, realtime on a laptop CPU).

It also reframes the paper into something stronger than a beauty contest:

> **What can you actually deploy for a low-resource language, and at what cost?** Two tiers,
> measured: quality against size against whether it runs at all without a GPU.

That frames a 0.934 GB GPU-only model against a ~60 MB CPU-realtime one as a *qualitative*
capability difference — offline or not — for a product whose users are blind readers.

### The trap

The VITS described in `SINHALA_READER_CLAUDE.md` is planned on **pathnirvana, ~11.8 h, single
male**. The XTTS work is **VoiceMakers, 6.81 h, two female**. Comparing those attributes any
difference to architecture when the corpus, speaker count, speaker identity, gender and data
volume all differ too. **That comparison would not survive review.**

### Two controlled designs

| | Design | Cost | Notes |
|---|---|---|---|
| **A** | Train VITS on **VoiceMakers female**, compare to the existing XTTS female model | VITS training from scratch | same data, same speakers, same eval split, same listeners |
| **B** | Use the planned **pathnirvana VITS**, and train XTTS on pathnirvana with the existing `xtts_sinhala/train_xtts_si.py` | **one XTTS run** if the VITS is already underway | same data (11.8 h single male); the XTTS pipeline for it already exists in this repo |

**B is dramatically cheaper if someone is already training that VITS.** Check before
committing to A. A has the advantage that the female model is the one already measured to
death here.

### Controls the design must carry

- **Text front-end is a second variable.** Piper phonemises with espeak-ng; this XTTS work
  uses the ASCII transliteration. If VITS uses phonemes and XTTS uses ASCII, architecture and
  front-end are confounded. Either run VITS on the same ASCII with a character front-end, or
  run both front-ends and report it as an ablation.
- **One blind listening panel rating both systems**, same listeners, same sentences, shuffled.
  This is what makes the comparison credible; two separately-run panels do not compare.
- **Same metric code for both.** Currently `evaluate_xtts.py` interleaves synthesis with
  scoring. Split it into a model-specific synthesis step and a model-agnostic scoring step
  that takes a directory of wavs. Then both systems are scored by identical code — including
  SECS via the *same* XTTS speaker encoder, which keeps that number comparable across
  architectures.
- **Report CPU RTF for both.** It is the measurement that decides the product architecture,
  and it is where the two systems differ qualitatively rather than by a few percent.

### Cost warning

VITS from scratch on ~7 h is a real undertaking — far more steps than fine-tuning a
pretrained model, plausibly several Kaggle sessions against a 30 h/week quota, with a genuine
risk that an undertrained VITS makes the comparison look unfair. **Fine-tune an existing
Piper/VITS checkpoint rather than training from scratch**, and say so in the paper.

---

## 10. Paper outline, mapped to evidence

| Section | Evidence | Status |
|---|---|---|
| Introduction — no Sinhala voice in any major TTS product | `SINHALA_READER_CLAUDE.md` | written |
| Related work | — | **not started; decides novelty** |
| Method: transliteration | §2.1, `sinhala_text.py`, selftest | complete |
| Method: data pipeline + `[UNK]` gate | §2.2, §3 | complete |
| Method: training config and deviations | §2.3 | complete |
| Method: evaluation | §2.4, §5 calibration | complete |
| Results: convergence, replicated | §4.1 | complete |
| Results: objective quality | §4.2 | complete |
| Results: noise floor | §4.3 | complete, and a contribution |
| Results: negative results | §4.4 | complete |
| Results: per-speaker | §7 | complete, with stated caveats |
| **Results: MOS/SUS** | — | **empty — gap 1** |
| **Results: baseline comparison** | — | **empty — gap 2, §9** |
| Deployment | §6 | complete |
| Ablation: transliteration rules | — | **empty — gap 4** |
| Threats to validity | §11 | drafted below |

Three of fourteen sections are empty, and two of those are the ones a reviewer checks first.

---

## 11. Threats to validity — draft

- **MCD is implementation-dependent** and this implementation's scale is unusual (§5). Not
  comparable to published Sinhala figures without re-running on a standard implementation.
- **MCD is uncalibrated across speakers**, so the dinithi/harini gap is partly unquantified
  (§7).
- **UTMOS is English-trained** and under-rates Sinhala; only the gap to real recordings is
  meaningful.
- **Two speakers, one corpus, one language, 6.81 h.** Narrow.
- **The two speakers use different text paths**, confounding text path with data volume
  throughout the per-speaker analysis.
- **Evaluation sentences differ between speakers**, so sentence difficulty is uncontrolled.
- **No human evaluation**, so no claim about perceptual quality is supported.
- **The transliteration mapping is hand-designed** by rules, not learned or ablated.
- Objective results are from **40 clips per speaker at one to three seeds**; the noise floor
  (§4.3) is the correct yardstick and several candidate findings fall inside it.

---

## 12. Reproducing every number here

```bash
# data + the [UNK] gate
python prepare_voicemakers.py --src <corpus> --out ./female_dataset \
    --speakers dinithi harini --vocab vocab.json --eval-per-speaker 40 --seed 1234

# training (Kaggle T4, ~8.5 h budget)
python train_xtts_female.py --dataset ./female_dataset --out ./run --epochs 40

# objective evaluation
python evaluate_xtts.py --run <run> --base <base> --dataset ./female_dataset --n 40 --utmos

# the noise floor (§4.3) and the sweeps (§4.4)
python sweep_eval.py --checkpoints model_slim.pth model_fp16.pth --seeds 1234,1235,1236 --utmos
python sweep_eval.py --checkpoints model_fp16.pth --temperature 0.65,0.7 --utmos
python sweep_eval.py --checkpoints model_fp16.pth --text-from script --utmos

# deployment (§6)
python optimize_checkpoint.py --in model.pth --out model_slim.pth --strip --hashes
python optimize_checkpoint.py --in model.pth --out model_slim.pth --verify
python optimize_checkpoint.py --in model.pth --out model_fp16.pth --strip --fp16
python compare_quality.py --baseline model_slim.pth --candidate model_fp16.pth --utmos
python benchmark.py --checkpoint model_fp16.pth --base <base> --ref <wav> --tag fp16

# MCD calibration (§5)
python calibrate_mcd.py --clips dataset/<corpus>/clips
```

Kaggle notebooks: `kaggle_xtts_female.ipynb` (train + evaluate + export),
`session_b/kaggle_xtts_session_b.ipynb` (experiments on a fixed checkpoint),
`listen/kaggle_xtts_listen.ipynb` (interactive Sinhala synthesis).

Two operational notes that cost four failed sessions to learn: `kaggle kernels push` **resets
the accelerator** (the API has no accelerator field), and a Kaggle P100 is **sm_60, which the
installed PyTorch cannot run at all** — it reports its capability happily and fails on the
first real operation.
