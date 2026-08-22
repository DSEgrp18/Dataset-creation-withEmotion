#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_checkpoint.py -- turn a 5.6 GB training checkpoint into an inference model.

WHY THE EXPORT IS 5.6 GB WHEN XTTS-v2 IS 1.9 GB
-----------------------------------------------
The file the training run exported is a *trainer* checkpoint, not a model. It
holds three things inference never touches:

    optimizer state    AdamW keeps exp_avg and exp_avg_sq per parameter, both
                       fp32 -- that alone is 2x the model
    scaler / step / epoch
    training-only submodules  GPTTrainer wraps the model in a dvae and two mel
                       spectrogram encoders used to build targets. coqui's own
                       loader throws them away at load time; keeping them in the
                       file just makes the download bigger.

Dropping all three is **numerically identical inference**. Not "almost" -- the
remaining tensors are bit-for-bit the ones the model already used, so there is
nothing to verify beyond that the file loads. That is the whole of the 5.6 -> 1.9 GB
win, and it is free.

    5.6 GB   as exported by the training run
    1.9 GB   --strip           weights only, identical output
    0.95 GB  --strip --fp16    weights rounded to fp16 (see below)

WHAT --fp16 ACTUALLY DOES, AND DOES NOT DO
------------------------------------------
It halves the file on disk. It does **not** make the model compute in fp16:
`load_state_dict` casts every tensor to the dtype of the parameter receiving it,
so an fp16 file loaded into XTTS runs in fp32 with weights rounded to fp16
precision. Storage change, not an arithmetic change -- which is why it is cheap
and why the quality cost is small rather than zero.

Small is not none. Run compare_quality.py before shipping an --fp16 model; that
is what the gate is there for.

For actual speed, weights are the wrong lever -- see README.md. This script only
makes the model smaller.

USAGE
    python optimize_checkpoint.py --in  model.pth --out model_slim.pth --strip
    python optimize_checkpoint.py --in  model.pth --out model_fp16.pth --strip --fp16
    python optimize_checkpoint.py --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Exactly the modules coqui's Xtts.get_compatible_checkpoint_state_dict discards
# when it loads a GPTTrainer checkpoint (TTS/tts/models/xtts.py). They belong to
# the trainer wrapper, not the model, so a strict load would reject them anyway.
# Kept identical to upstream deliberately: if this list drifts, the slim file
# stops loading and the error is a wall of missing/unexpected keys.
TRAINER_ONLY = (
    "torch_mel_spectrogram_style_encoder",
    "torch_mel_spectrogram_dvae",
    "dvae",
)


def slim(state: dict, fp16: bool = False) -> tuple[dict, dict]:
    """Trainer checkpoint -> {"model": weights}. Returns (new_state, report)."""
    if "model" not in state:
        raise SystemExit(
            "no 'model' key -- this is not a trainer checkpoint. If it is already "
            "an inference model there is nothing here to strip.")

    weights = state["model"]
    kept, dropped, halved = {}, 0, 0
    dropped_bytes = 0

    for key, value in weights.items():
        # The trainer prefixes the wrapped model; strip it so the result loads
        # as a plain Xtts checkpoint as well as through the compat path.
        name = key[len("xtts."):] if key.startswith("xtts.") else key
        if name.split(".")[0] in TRAINER_ONLY:
            dropped += 1
            if torch.is_tensor(value):
                dropped_bytes += value.numel() * value.element_size()
            continue
        if fp16 and torch.is_tensor(value) and value.is_floating_point():
            value = value.half()
            halved += 1
        kept[name] = value

    report = {
        "tensors_kept": len(kept),
        "tensors_dropped": dropped,
        "dropped_bytes": dropped_bytes,
        "tensors_halved": halved,
        "had_optimizer": "optimizer" in state,
        "step": state.get("step"),
        "epoch": state.get("epoch"),
        "model_loss": state.get("model_loss"),
    }
    # Only "model" survives. Anything else in the file is training bookkeeping.
    return {"model": kept}, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", help="the 5.6 GB exported model.pth")
    ap.add_argument("--out", dest="dst", help="where to write the slim model")
    ap.add_argument("--strip", action="store_true",
                    help="drop optimizer state and trainer-only modules")
    ap.add_argument("--fp16", action="store_true",
                    help="also store weights as fp16 (halves the file; verify quality)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not (args.src and args.dst):
        ap.error("--in and --out are required (or use --selftest)")
    if not args.strip:
        ap.error("--strip is the point of this script; pass it explicitly")

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        print(f"no such checkpoint: {src}", file=sys.stderr)
        return 1

    # weights_only=False is required: coqui checkpoints carry a config object,
    # and torch >= 2.6 refuses to unpickle it under the new default.
    print(f"reading {src}  ({src.stat().st_size / 1e9:.2f} GB)")
    state = torch.load(str(src), map_location="cpu", weights_only=False)

    new_state, rep = slim(state, fp16=args.fp16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, str(dst))

    before, after = src.stat().st_size, dst.stat().st_size
    loss = rep["model_loss"]
    print(f"\n  trained to step {rep['step']}, epoch {rep['epoch']}"
          + ("" if loss is None else f", eval loss {loss}"))
    print(f"  optimizer state    : {'dropped' if rep['had_optimizer'] else 'was absent'}")
    print(f"  trainer-only tensors dropped : {rep['tensors_dropped']} "
          f"({rep['dropped_bytes'] / 1e9:.2f} GB)")
    print(f"  tensors kept       : {rep['tensors_kept']}"
          + (f"  ({rep['tensors_halved']} cast to fp16)" if args.fp16 else ""))
    print(f"\n{before / 1e9:.2f} GB  ->  {after / 1e9:.2f} GB   "
          f"({100 * (1 - after / before):.0f}% smaller)")
    if args.fp16:
        print("\nfp16 storage: weights are rounded, arithmetic is unchanged.\n"
              "Run compare_quality.py against the fp32 slim model before shipping this.")
    else:
        print("\nInference from this file is numerically identical to the original.")
    return 0


def _selftest() -> int:
    """Build a checkpoint shaped like the trainer's and check what survives."""
    import tempfile

    torch.manual_seed(0)
    gpt_w = torch.randn(64, 64)
    dec_w = torch.randn(32, 32)
    state = {
        "model": {
            "xtts.gpt.weight": gpt_w,                       # keep, unprefixed
            "xtts.hifigan_decoder.weight": dec_w,           # keep, unprefixed
            "dvae.codebook": torch.randn(128, 128),         # drop
            "torch_mel_spectrogram_dvae.mel": torch.randn(8, 8),        # drop
            "torch_mel_spectrogram_style_encoder.mel": torch.randn(8, 8),  # drop
            "gpt.already_unprefixed": torch.randn(4, 4),    # keep as-is
            "xtts.gpt.n_layers": torch.tensor(12),          # non-float, keep exact
        },
        "optimizer": {"exp_avg": torch.randn(2000, 2000)},  # the bulk of the file
        "scaler": {"scale": 1.0},
        "step": 1760, "epoch": 1, "model_loss": 3.0204,
    }

    slim_state, rep = slim(state, fp16=False)
    keys = set(slim_state["model"])
    assert keys == {"gpt.weight", "hifigan_decoder.weight",
                    "gpt.already_unprefixed", "gpt.n_layers"}, keys
    assert list(slim_state) == ["model"], "optimizer/scaler must not survive"
    assert rep["tensors_dropped"] == 3 and rep["had_optimizer"]
    # The whole claim of --strip: kept weights are untouched.
    assert torch.equal(slim_state["model"]["gpt.weight"], gpt_w)
    assert torch.equal(slim_state["model"]["hifigan_decoder.weight"], dec_w)

    half_state, rep16 = slim(state, fp16=True)
    assert half_state["model"]["gpt.weight"].dtype == torch.float16
    # Integer tensors must NOT be cast -- n_layers=12 is a value, not a weight.
    assert half_state["model"]["gpt.n_layers"].dtype == gpt_w.new_tensor(
        12, dtype=torch.int64).dtype
    assert rep16["tensors_halved"] == 3, rep16
    # fp16 rounding is bounded, not arbitrary.
    err = (half_state["model"]["gpt.weight"].float() - gpt_w).abs().max().item()
    assert err < 1e-2, err

    # And the file actually shrinks on disk.
    tmp = Path(tempfile.mkdtemp(prefix="slim_"))
    torch.save(state, tmp / "full.pth")
    torch.save(slim_state, tmp / "slim.pth")
    torch.save(half_state, tmp / "half.pth")
    full, sl, hf = ((tmp / n).stat().st_size for n in ("full.pth", "slim.pth", "half.pth"))
    assert sl < full / 4, (full, sl)
    assert hf < sl, (sl, hf)

    print(f"optimize_checkpoint selftest OK  "
          f"({full/1e6:.1f} MB -> {sl/1e6:.1f} MB -> {hf/1e6:.1f} MB fp16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
