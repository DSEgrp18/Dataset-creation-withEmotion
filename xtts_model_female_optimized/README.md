# Running the Sinhala female model locally

Makes the trained model small and fast enough to run on a local GPU, and **proves
the quality did not drop** rather than assuming it.

```bash
# 1. shrink -- 5.6 GB to 1.9 GB, provably lossless
python optimize_checkpoint.py --in  ../xtts_si_female/model.pth \
                              --out ./model_slim.pth --strip --hashes

# 2. prove the weights are untouched -- no GPU, no synthesis
python optimize_checkpoint.py --in  ../xtts_si_female/model.pth \
                              --out ./model_slim.pth --verify

# 3. measure what it costs to run
python benchmark.py --checkpoint ./model_slim.pth --base <base> --ref <wav> --tag slim

# 4. prove it did not get worse
python compare_quality.py --run <run> --base <base> --dataset <ds> \
    --baseline ../xtts_si_female/model.pth --candidate ./model_slim.pth --out cmp
```

Step 4 exits non-zero on any regression beyond tolerance. That is the whole point
of this folder — "smaller and faster, but never worse" is a claim, and a claim
needs something that can fail.

## The order, and why it is an order

Size and quality are not independent, so the sequence matters: every step below is
measured against the step before it, never against the original, or two changes end
up sharing one verdict.

| | Step | Where it happens | Can it fail? |
|---|---|---|---|
| 1 | keep the current best model as the baseline | `RESULTS.md` run 4 | — |
| 2 | pick the checkpoint by **synthesis quality**, not loss | [`sweep_eval.py --all-checkpoints`](../xtts_model_female/sweep_eval.py) | ordering only |
| 3 | tune `temperature` / `repetition_penalty` / `top_k` / `top_p` | `sweep_eval.py` decode grids | ordering only |
| 4 | one Sinhala→ASCII path for both speakers | `--text-from script`, and `prepare_voicemakers.py --text-path script` for the training half | vs baseline |
| 5 | speaker-balanced sampling | `prepare_voicemakers.py --balance-speakers oversample` (needs a retrain) | vs baseline |
| 6 | **strip** to an inference-only checkpoint | `optimize_checkpoint.py --strip` | `--verify`: bit-identity |
| 7 | verify the stripped model is identical | `--verify`, then `compare_quality.py` | **yes** — must be *exactly* equal |
| 8 | fp16 storage | `--strip --fp16` | — |
| 9 | fp16 vs fp32, same clips, same seeds | `compare_quality.py` | **yes** |
| 10 | size / load / VRAM / RTF / first-audio | `benchmark.py` | never gated |
| 11 | selective int8 on GPT linears | **not implemented — gated behind step 9** | — |

Steps 2–5 change *what the model is*; 6–10 change *what the file is*. Do them in
that order, because a decode setting tuned on the wrong checkpoint is a setting
that has to be tuned again.

---

## Why the export is 5.6 GB when XTTS-v2 is 1.9 GB

The file the training run exported is a **trainer checkpoint, not a model**. Three
things in it are never touched at inference:

| In the 5.6 GB file | Needed to synthesise? |
|---|---|
| model weights | **yes** — ~1.9 GB |
| AdamW state (`exp_avg`, `exp_avg_sq`, both fp32) | no — and it is 2× the model |
| `dvae`, `torch_mel_spectrogram_dvae`, `torch_mel_spectrogram_style_encoder` | no — GPTTrainer builds targets with these |
| `scaler`, `step`, `epoch`, `model_loss` | no |

Dropping them is **numerically identical inference**, not approximately. The tensors
that remain are bit-for-bit the ones the model already used, so there is nothing to
verify beyond that the file loads. coqui's own loader discards the same three modules
on the way in; `--strip` just stops shipping them.

| | Size | Quality | |
|---|---|---|---|
| as exported | **5.608 GB** | — | Run 5 |
| `--strip` | **1.868 GB** | **exactly equal** — 963 tensors bit-identical | Run 5 |
| `--strip --fp16` | **0.934 GB** | not worse on any metric at three seeds | Session B |

Those are measurements now, not targets. The strip was verified twice by different means:
`--verify` checked all 963 kept tensors for identical dtype, shape and bytes, and
`compare_quality.py` independently returned `+0.0000` on MCD, F0 RMSE, F0 corr, SECS and
UTMOS. fp16 was compared against the fp32 slim model at seeds 1234/1235/1236 — mean MCD
62.776 vs 63.066, F0 corr 0.438 vs 0.436, UTMOS 2.681 vs 2.677, against a measured
sampling-noise floor of ±0.25 dB MCD and ±0.014 F0 corr.

**Peak VRAM is 2.47 GB for all three files.** `load_state_dict` casts each tensor to the
dtype of the parameter receiving it, so an fp16 *file* becomes an fp32 *model* in memory.
Shrinking the file buys download size and ~3.4 s of load time (16.7 s → 13.3 s); it does
not buy memory and it does not lower the GPU tier the model needs.

**`--fp16` halves the file, not the arithmetic.** `load_state_dict` casts each tensor
to the dtype of the parameter receiving it, so an fp16 file loaded into XTTS runs in
fp32 with weights rounded to fp16 precision. That is why it is cheap, and why the cost
is small rather than zero. Run `compare_quality.py` before shipping one.

The strongest signal that test gives you: a `--strip` model must come back **exactly**
equal on every metric, not merely within tolerance. `compare_quality.py` prints
`Metrics are EXACTLY equal` when that happens.

---

## Making it faster, as opposed to smaller

Weights are the wrong lever for speed — a smaller file loads quicker and frees VRAM,
but the arithmetic is unchanged. These are the levers that move RTF:

| Lever | How | Cost |
|---|---|---|
| **fp16 compute** | `benchmark.py --half` | real arithmetic change; gate it |
| **DeepSpeed kernels** | `benchmark.py --deepspeed` | extra dependency; gate it |
| **kv_cache** | already on by default in `GPTArgs` | none |
| **streaming** | `model.inference_stream()` | none — changes *when* audio arrives, not total time |

Streaming is the one worth understanding, because it improves the number a listener
actually notices. RTF is total compute per second of audio; **latency is time to the
first chunk**. A reader application can start playing at 300 ms and stay ahead of the
listener even at an RTF that looks unimpressive. `benchmark.py` reports both, separately.

The baseline run measured **RTF 0.537 on a Kaggle T4** — already faster than realtime
before any of this. If your local GPU is comparable, the problem to solve was never
throughput; it was the 5.6 GB file and the VRAM it drags in. Step 1 alone fixes that.

---

## What optimisation cannot fix

**XTTS will not run well on CPU, and no amount of this changes that.** It is a ~500 M
parameter autoregressive transformer plus a HiFiGAN decoder; on CPU it is far slower
than realtime even at 0.95 GB. The repo settled this already — see
`SINHALA_READER_CLAUDE.md`:

> **XTTS cannot run offline** — ~2 GB, far slower than realtime on CPU. Unusable as a
> book reader. Target **VITS/Piper** for the desktop; XTTS is website-only, expressive
> mode, GPU.

So the split stands: **XTTS is the GPU / expressive / server tier**, and this folder is
what makes that tier deployable on one local card. If the goal is an offline desktop
reader, the answer is a VITS model exported to Piper's ONNX layout — a different model,
roughly 60 MB, realtime on a laptop CPU — not a compressed XTTS.

## What is deliberately not implemented, and what would unlock it

| | Status | The condition |
|---|---|---|
| **selective int8** on GPT linear layers | not implemented | only after fp16 has passed the gate on real numbers. Then quantise the GPT's linear layers *selectively*, with a calibration set, and gate it like everything else — the vocoder and the attention projections are not candidates, and quantising everything because it is easy is how a model gets 20 % smaller and audibly worse |
| **int4, structural pruning** | not on the list | both need a retraining or distillation budget to recover what they remove. Neither is a storage change, and neither is reversible by a flag |
| **LoRA** | not a size claim | a LoRA adapter is small, and it is small *in addition to* the 1.9 GB base XTTS that has to be present to apply it. Deployment size is base + adapter, which is larger than the merged model, not smaller. It is a fine-tuning technique here, never a compression one |
| **distillation to a small standalone model** | not started | the condition is "0.95 GB is still too big", and that is a deployment decision nobody has made yet. If it is made, the plan is a *separate* one with the current best XTTS as teacher — and note that the repo already answers the offline case differently: a VITS model in Piper's ONNX layout, ~60 MB and realtime on a laptop CPU, is a better answer than a distilled XTTS for that tier |

Quantisation below fp16 is not a free win the way `--strip` is. XTTS's autoregressive GPT
plus vocoder does not quantise cleanly to int8 without a calibration pipeline and a
measurable quality cost, and the gate exists to reject exactly that trade when it is
made carelessly.

---

## Files

| File | Purpose |
|---|---|
| [`optimize_checkpoint.py`](optimize_checkpoint.py) | 5.6 GB trainer checkpoint → inference model; `--fp16` optional. Has a selftest |
| [`benchmark.py`](benchmark.py) | size, load time, peak VRAM, RTF, time-to-first-audio → appends a CSV row |
| [`compare_quality.py`](compare_quality.py) | runs `evaluate_xtts.py` on both checkpoints and **fails on regression** |
| [`../xtts_model_female/sweep_eval.py`](../xtts_model_female/sweep_eval.py) | the checkpoint and decode sweeps of steps 2–5, into one experiment register |

`compare_quality.py` does not re-implement a single metric — it invokes
`../xtts_model_female/evaluate_xtts.py` twice and diffs its `metrics.json`, so what is
compared is exactly what was measured for the baseline, with no second implementation
to drift.

### Tolerances the gate applies

| Metric | Direction | Allowed to worsen by |
|---|---|---|
| `mcd_db` | lower better | 2 % |
| `f0_rmse_cents` | lower better | 3 % |
| `f0_corr` | higher better | 0.02 absolute |
| `secs` | higher better | 0.01 absolute |
| `utmos` | higher better | 0.05 absolute |
| `failure_rate` | lower better | 1 percentage point |
| `duration_ratio` | closeness to 1.0 | 0.02 absolute |
| `rtf` | — | reported, never gated |

Sized for sampling noise, not for real degradation: XTTS samples at temperature 0.75, so
the moment weights differ by even fp16 rounding the token sequence diverges and every
metric moves a little. `evaluate_xtts.py` seeds `torch` and `random`, which makes a run
repeatable for identical weights — that is what makes the exact-equality check on
`--strip` meaningful. Use `--seeds 1234,1235,1236` when a result sits near a boundary.

**Decoding is passed through to both sides and must match**, or the comparison
measures the decode change instead of the checkpoint change. The flags are the same
ones `evaluate_xtts.py` takes.

### Per-speaker

The same metrics are reported for dinithi and harini separately, because an overall
mean can hold still while a change helps one speaker and hurts the other — and that
gap has persisted across all four runs. They are **warnings, not failures**, by
default: each speaker has half the clips, so the noise floor is wider than these
tolerances were sized against (harini's F0 correlation moved 0.068 across three runs
that trained identically). `--gate-per-speaker` promotes them to failures.

---

## Recording results

Append the outcome to [`../xtts_model_female/RESULTS.md`](../xtts_model_female/RESULTS.md)
— size, the benchmark row, and whether the gate passed. An optimisation that was not
measured against the baseline is not known to be free, however obviously free it looks.
