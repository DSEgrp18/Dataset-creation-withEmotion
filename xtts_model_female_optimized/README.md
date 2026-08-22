# Running the Sinhala female model locally

Makes the trained model small and fast enough to run on a local GPU, and **proves
the quality did not drop** rather than assuming it.

```bash
# 1. shrink -- 5.6 GB to 1.9 GB, provably lossless
python optimize_checkpoint.py --in  ../xtts_si_female/model.pth \
                              --out ./model_slim.pth --strip

# 2. measure what it costs to run
python benchmark.py --checkpoint ./model_slim.pth --base <base> --ref <wav> --tag slim

# 3. prove it did not get worse
python compare_quality.py --run <run> --base <base> --dataset <ds> \
    --baseline ../xtts_si_female/model.pth --candidate ./model_slim.pth --out cmp
```

Step 3 exits non-zero on any regression beyond tolerance. That is the whole point
of this folder — "smaller and faster, but never worse" is a claim, and a claim
needs something that can fail.

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

| | Size | Quality |
|---|---|---|
| as exported | 5.60 GB | — |
| `--strip` | ~1.90 GB | **identical**, provably |
| `--strip --fp16` | ~0.95 GB | small change — must be measured |

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

Quantisation below fp16 (int8, ONNX) is deliberately not here. XTTS's autoregressive GPT
plus vocoder does not quantise cleanly to int8 without a calibration pipeline and a
measurable quality cost, and the gate would reject it. It is worth revisiting only after
a run has used its full training budget.

---

## Files

| File | Purpose |
|---|---|
| [`optimize_checkpoint.py`](optimize_checkpoint.py) | 5.6 GB trainer checkpoint → inference model; `--fp16` optional. Has a selftest |
| [`benchmark.py`](benchmark.py) | size, load time, peak VRAM, RTF, time-to-first-audio → appends a CSV row |
| [`compare_quality.py`](compare_quality.py) | runs `evaluate_xtts.py` on both checkpoints and **fails on regression** |

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

---

## Recording results

Append the outcome to [`../xtts_model_female/RESULTS.md`](../xtts_model_female/RESULTS.md)
— size, the benchmark row, and whether the gate passed. An optimisation that was not
measured against the baseline is not known to be free, however obviously free it looks.
