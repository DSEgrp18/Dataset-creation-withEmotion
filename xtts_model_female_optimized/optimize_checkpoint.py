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

Nothing is dropped by assumption. Every top-level key that does not survive is
named in the report under the reason it was dropped, and any key this script has
not seen before is reported as `unclassified` rather than quietly discarded --
a future trainer version that parks something new in the checkpoint shows up as
a line of output instead of a silent behaviour change.

WHAT --fp16 ACTUALLY DOES, AND DOES NOT DO
------------------------------------------
It halves the file on disk. It does **not** make the model compute in fp16:
`load_state_dict` casts every tensor to the dtype of the parameter receiving it,
so an fp16 file loaded into XTTS runs in fp32 with weights rounded to fp16
precision. Storage change, not an arithmetic change -- which is why it is cheap
and why the quality cost is small rather than zero.

Small is not none, and the report says where it is largest: the cast measures the
relative round-trip error of every tensor and lists the worst, because fp16's
smallest normal value is ~6e-5 and a tensor of very small weights loses more than
a tensor of ordinary ones. That tells you which module to suspect if the quality
gate fails. It does not replace the gate -- run compare_quality.py before
shipping an --fp16 model.

For actual speed, weights are the wrong lever -- see README.md. This script only
makes the model smaller.

VERIFYING (--verify)
--------------------
`--strip` claims the kept weights are untouched. --verify proves it without a GPU
and without synthesising anything: it loads both files and checks every kept
tensor for identical dtype, shape and bytes. That is a stronger statement than
"the metrics came back equal", and it is the reason compare_quality.py's exact
equality check on a slim model means something -- bit-identical weights and a
seeded evaluation have no route to a different number.

Against an --fp16 file the same command reports the rounding error per tensor
instead, since bit-identity is not the claim there.

--verify holds both checkpoints in RAM at once (~7.5 GB for a 5.6 GB original
plus a 1.9 GB slim). That fits Kaggle's ~30 GB; it does not fit a small laptop.

USAGE
    python optimize_checkpoint.py --in  model.pth --out model_slim.pth --strip
    python optimize_checkpoint.py --in  model.pth --out model_fp16.pth --strip --fp16
    python optimize_checkpoint.py --in  model.pth --out model_slim.pth --verify
    python optimize_checkpoint.py --selftest
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# Top-level keys of a trainer checkpoint, grouped by why they do not ship. Named
# rather than inferred so the report can say what was dropped and so an
# unrecognised key becomes visible output instead of a silent discard.
DROP_GROUPS = {
    "optimizer state": ("optimizer",),
    "scheduler state": ("scheduler", "lr_scheduler", "scheduler_state_dict"),
    "gradient scaler": ("scaler", "amp_scaler", "grad_scaler"),
    "training metadata": ("step", "epoch", "model_loss", "date", "restore_path",
                          "config", "args", "best_loss", "total_steps_done",
                          "epochs_done", "criterion"),
}
KNOWN_DROPS = {k for keys in DROP_GROUPS.values() for k in keys}


def tensor_sha256(t: torch.Tensor) -> str:
    """Hash of a tensor's exact bytes, for a manifest that outlives the original.

    Contiguous first: a view's storage can hold bytes that are not part of the
    tensor, which would make the hash depend on how it was sliced. Flattened
    because a 0-dim tensor cannot be reinterpreted as bytes, and some of the
    kept entries are scalars.
    """
    return hashlib.sha256(
        t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def slim(state: dict, fp16: bool = False, hashes: bool = False) -> tuple[dict, dict]:
    """Trainer checkpoint -> {"model": weights}. Returns (new_state, report)."""
    if "model" not in state:
        raise SystemExit(
            "no 'model' key -- this is not a trainer checkpoint. If it is already "
            "an inference model there is nothing here to strip.")

    weights = state["model"]
    kept, dropped, halved = {}, 0, 0
    dropped_bytes = kept_bytes = 0
    manifest: dict[str, str] = {}
    # fp16 is lossy by a bounded amount, but not by the SAME amount everywhere:
    # a tensor whose values sit near fp16's smallest normal (~6e-5) loses far
    # more of them than one with ordinary weights. Measured here, while both
    # dtypes are in hand, because after the cast the information is gone.
    rounding: list[tuple[float, str]] = []

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
            half = value.half()
            denom = value.abs().max().item()
            if denom > 0:
                err = (half.float() - value).abs().max().item() / denom
                rounding.append((err, name))
            value = half
            halved += 1
        if torch.is_tensor(value):
            kept_bytes += value.numel() * value.element_size()
            if hashes:
                manifest[name] = tensor_sha256(value)
        kept[name] = value

    unclassified = [k for k in state
                    if k not in KNOWN_DROPS and k != "model"]
    report = {
        "tensors_kept": len(kept),
        "tensors_dropped": dropped,
        "dropped_bytes": dropped_bytes,
        "kept_bytes": kept_bytes,
        "tensors_halved": halved,
        "fp16": bool(fp16),
        "dropped_keys": {label: [k for k in keys if k in state]
                         for label, keys in DROP_GROUPS.items()},
        "unclassified_dropped_keys": unclassified,
        "step": state.get("step"),
        "epoch": state.get("epoch"),
        "model_loss": state.get("model_loss"),
    }
    if rounding:
        rounding.sort(reverse=True)
        report["fp16_max_rel_error"] = round(rounding[0][0], 6)
        report["fp16_worst_tensors"] = [
            {"name": n, "max_rel_error": round(e, 6)} for e, n in rounding[:8]]
    if hashes:
        report["sha256"] = manifest
    # Only "model" survives. Anything else in the file is training bookkeeping.
    return {"model": kept}, report


def verify(src: Path, dst: Path) -> int:
    """Check a slim file against the checkpoint it came from. 0 if it holds."""
    print(f"loading {src}  ({src.stat().st_size / 1e9:.2f} GB)")
    orig = torch.load(str(src), map_location="cpu", weights_only=False)
    print(f"loading {dst}  ({dst.stat().st_size / 1e9:.2f} GB)")
    slim_state = torch.load(str(dst), map_location="cpu", weights_only=False)

    if "model" not in orig or "model" not in slim_state:
        print("both files must carry a 'model' key", file=sys.stderr)
        return 1

    # Expected weights, named the way the slim file names them.
    expected = {}
    for key, value in orig["model"].items():
        name = key[len("xtts."):] if key.startswith("xtts.") else key
        if name.split(".")[0] in TRAINER_ONLY:
            continue
        expected[name] = value
    got = slim_state["model"]

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    is_fp16 = any(torch.is_tensor(v) and v.dtype == torch.float16 for v in got.values())

    identical = mismatched = 0
    worst_err, worst_name = 0.0, None
    problems: list[str] = []
    for name, want in expected.items():
        if name not in got:
            continue
        have = got[name]
        if not torch.is_tensor(want):
            if have != want:
                problems.append(f"{name}: non-tensor value changed")
            continue
        if have.shape != want.shape:
            problems.append(f"{name}: shape {tuple(want.shape)} -> {tuple(have.shape)}")
            continue
        if have.dtype == want.dtype:
            # Bit-for-bit, not allclose. The claim being checked is "untouched".
            if torch.equal(have, want):
                identical += 1
            else:
                mismatched += 1
                problems.append(f"{name}: same dtype but different bytes")
        elif is_fp16 and want.is_floating_point() and have.dtype == torch.float16:
            denom = want.abs().max().item()
            err = ((have.float() - want).abs().max().item() / denom) if denom else 0.0
            if err > worst_err:
                worst_err, worst_name = err, name
        else:
            problems.append(f"{name}: dtype {want.dtype} -> {have.dtype}")

    print(f"\n{'=' * 78}")
    print(f"tensors expected   : {len(expected)}")
    print(f"tensors present    : {len(got)}")
    print(f"missing            : {len(missing)}" + (f"  {missing[:5]}" if missing else ""))
    print(f"unexpected         : {len(extra)}" + (f"  {extra[:5]}" if extra else ""))
    for label, keys in DROP_GROUPS.items():
        present = [k for k in keys if k in orig]
        still = [k for k in keys if k in slim_state]
        print(f"{label:19}: {'dropped' if present and not still else ('absent' if not present else 'STILL PRESENT')}"
              + (f"  ({', '.join(present)})" if present else ""))
    if is_fp16:
        print(f"\nfp16 storage: {len(expected) - identical - mismatched} tensors rounded")
        print(f"worst relative rounding error: {worst_err:.2e}"
              + (f"  ({worst_name})" if worst_name else ""))
        print("\nBit-identity is NOT the claim for an fp16 file. Quality has to be\n"
              "measured: compare_quality.py against the fp32 slim model.")
    else:
        print(f"\nbit-identical       : {identical}")
        print(f"differing           : {mismatched}")

    if problems or missing or mismatched:
        print(f"\nFAILED -- {len(problems) + len(missing)} problem(s):", file=sys.stderr)
        for pb in problems[:20]:
            print("  -", pb, file=sys.stderr)
        for name in missing[:20]:
            print(f"  - {name}: missing from the slim file", file=sys.stderr)
        return 1
    if extra:
        print("\nNote: the slim file carries tensors the original did not. That is "
              "not a\nfailure, but it is unexpected -- check where they came from.")
    if not is_fp16:
        print("\nPASSED -- every inference weight is bit-for-bit the original's, and "
              "no\ntraining state survived. Inference from this file cannot differ.")
    else:
        print("\nPASSED -- shapes and keys match and nothing but dtype changed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", help="the 5.6 GB exported model.pth")
    ap.add_argument("--out", dest="dst", help="where to write the slim model")
    ap.add_argument("--strip", action="store_true",
                    help="drop optimizer state and trainer-only modules")
    ap.add_argument("--fp16", action="store_true",
                    help="also store weights as fp16 (halves the file; verify quality)")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing --out against --in instead of writing it")
    ap.add_argument("--hashes", action="store_true",
                    help="sha256 every kept tensor into the manifest (slow, but the "
                         "manifest then outlives the original checkpoint)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not (args.src and args.dst):
        ap.error("--in and --out are required (or use --selftest)")

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        print(f"no such checkpoint: {src}", file=sys.stderr)
        return 1

    if args.verify:
        if not dst.is_file():
            print(f"nothing to verify: {dst} does not exist", file=sys.stderr)
            return 1
        return verify(src, dst)

    if not args.strip:
        ap.error("--strip is the point of this script; pass it explicitly")
    # Overwriting a checkpoint silently is how a baseline stops existing.
    if dst.exists():
        print(f"refusing to overwrite {dst} -- delete it or choose another --out",
              file=sys.stderr)
        return 1

    # weights_only=False is required: coqui checkpoints carry a config object,
    # and torch >= 2.6 refuses to unpickle it under the new default.
    print(f"reading {src}  ({src.stat().st_size / 1e9:.2f} GB)")
    state = torch.load(str(src), map_location="cpu", weights_only=False)

    new_state, rep = slim(state, fp16=args.fp16, hashes=args.hashes)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state, str(dst))

    before, after = src.stat().st_size, dst.stat().st_size
    loss = rep["model_loss"]
    print(f"\n  trained to step {rep['step']}, epoch {rep['epoch']}"
          + ("" if loss is None else f", eval loss {loss}"))
    for label, keys in rep["dropped_keys"].items():
        print(f"  {label:19}: {'dropped -- ' + ', '.join(keys) if keys else 'absent'}")
    if rep["unclassified_dropped_keys"]:
        print(f"  unclassified       : dropped -- "
              f"{', '.join(rep['unclassified_dropped_keys'])}")
        print("    ^ this script did not recognise those keys. They are training "
              "state as\n      far as it knows; check that before shipping.")
    print(f"  trainer-only tensors dropped : {rep['tensors_dropped']} "
          f"({rep['dropped_bytes'] / 1e9:.2f} GB)")
    print(f"  tensors kept       : {rep['tensors_kept']}"
          + (f"  ({rep['tensors_halved']} cast to fp16)" if args.fp16 else ""))
    if "fp16_max_rel_error" in rep:
        print(f"  worst fp16 rounding: {rep['fp16_max_rel_error']:.2e} relative, in")
        for item in rep["fp16_worst_tensors"][:3]:
            print(f"      {item['max_rel_error']:.2e}  {item['name']}")
    print(f"\n{before / 1e9:.2f} GB  ->  {after / 1e9:.2f} GB   "
          f"({100 * (1 - after / before):.0f}% smaller)")

    # The record of this conversion, next to the file it describes. Sizes and the
    # source path are what a later comparison needs and what nobody writes down.
    manifest = {"source": str(src), "source_bytes": before,
                "output": str(dst), "output_bytes": after, **rep}
    man_path = dst.with_suffix(dst.suffix + ".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {man_path}")

    if args.fp16:
        print("\nfp16 storage: weights are rounded, arithmetic is unchanged.\n"
              "Run compare_quality.py against the fp32 slim model before shipping this.")
    else:
        print("\nInference from this file is numerically identical to the original.\n"
              f"Prove it: --in {src} --out {dst} --verify")
    return 0


def _selftest() -> int:
    """Build a checkpoint shaped like the trainer's and check what survives."""
    import tempfile

    torch.manual_seed(0)
    gpt_w = torch.randn(64, 64)
    dec_w = torch.randn(32, 32)
    tiny_w = torch.randn(16, 16) * 1e-6        # lives near fp16's denormals
    state = {
        "model": {
            "xtts.gpt.weight": gpt_w,                       # keep, unprefixed
            "xtts.hifigan_decoder.weight": dec_w,           # keep, unprefixed
            "xtts.gpt.tiny": tiny_w,                        # keep; fp16 hurts here
            "dvae.codebook": torch.randn(128, 128),         # drop
            "torch_mel_spectrogram_dvae.mel": torch.randn(8, 8),        # drop
            "torch_mel_spectrogram_style_encoder.mel": torch.randn(8, 8),  # drop
            "gpt.already_unprefixed": torch.randn(4, 4),    # keep as-is
            "xtts.gpt.n_layers": torch.tensor(12),          # non-float, keep exact
        },
        "optimizer": {"exp_avg": torch.randn(2000, 2000)},  # the bulk of the file
        "scaler": {"scale": 1.0},
        "scheduler": {"last_epoch": 6},
        "step": 1760, "epoch": 1, "model_loss": 3.0204,
    }

    slim_state, rep = slim(state, fp16=False, hashes=True)
    keys = set(slim_state["model"])
    assert keys == {"gpt.weight", "hifigan_decoder.weight", "gpt.tiny",
                    "gpt.already_unprefixed", "gpt.n_layers"}, keys
    assert list(slim_state) == ["model"], "optimizer/scaler must not survive"
    assert rep["tensors_dropped"] == 3
    # Every training-state key is reported under its own heading, not implied.
    assert rep["dropped_keys"]["optimizer state"] == ["optimizer"]
    assert rep["dropped_keys"]["gradient scaler"] == ["scaler"]
    assert rep["dropped_keys"]["scheduler state"] == ["scheduler"]
    assert set(rep["dropped_keys"]["training metadata"]) == {"step", "epoch",
                                                             "model_loss"}
    assert rep["unclassified_dropped_keys"] == []
    # The whole claim of --strip: kept weights are untouched.
    assert torch.equal(slim_state["model"]["gpt.weight"], gpt_w)
    assert torch.equal(slim_state["model"]["hifigan_decoder.weight"], dec_w)
    # The manifest hashes what was kept, and the hash tracks the bytes.
    assert rep["sha256"]["gpt.weight"] == tensor_sha256(gpt_w)
    assert rep["sha256"]["gpt.weight"] != rep["sha256"]["hifigan_decoder.weight"]

    # An unknown top-level key must be visible, not silently discarded.
    odd = {**state, "something_new": {"a": 1}}
    _, rep_odd = slim(odd)
    assert rep_odd["unclassified_dropped_keys"] == ["something_new"], rep_odd

    half_state, rep16 = slim(state, fp16=True)
    assert half_state["model"]["gpt.weight"].dtype == torch.float16
    # Integer tensors must NOT be cast -- n_layers=12 is a value, not a weight.
    assert half_state["model"]["gpt.n_layers"].dtype == gpt_w.new_tensor(
        12, dtype=torch.int64).dtype
    assert rep16["tensors_halved"] == 4, rep16
    # fp16 rounding is bounded, not arbitrary.
    err = (half_state["model"]["gpt.weight"].float() - gpt_w).abs().max().item()
    assert err < 1e-2, err
    # And the report points at where rounding is worst, which is the tensor of
    # very small values -- that is the whole reason the column exists.
    assert rep16["fp16_worst_tensors"][0]["name"] == "gpt.tiny", rep16
    assert rep16["fp16_max_rel_error"] > 0

    # And the file actually shrinks on disk.
    tmp = Path(tempfile.mkdtemp(prefix="slim_"))
    torch.save(state, tmp / "full.pth")
    torch.save(slim_state, tmp / "slim.pth")
    torch.save(half_state, tmp / "half.pth")
    full, sl, hf = ((tmp / n).stat().st_size for n in ("full.pth", "slim.pth", "half.pth"))
    assert sl < full / 4, (full, sl)
    assert hf < sl, (sl, hf)

    # --verify, on four files. The last two MUST fail, so their reports say
    # FAILED -- banners here because an unlabelled FAILED in the output of a
    # passing selftest is indistinguishable from a broken one.
    def expect(code, label, *args):
        print(f"\n--- selftest: {label} (expecting "
              f"{'PASS' if code == 0 else 'FAILURE'}) ---")
        got = verify(*args)
        assert got == code, f"{label}: verify returned {got}, expected {code}"

    expect(0, "the real slim file", tmp / "full.pth", tmp / "slim.pth")
    expect(0, "the fp16 file", tmp / "full.pth", tmp / "half.pth")
    # A weight altered by 1e-7 -- far below any tolerance, which is the point:
    # bit-identity has no tolerance, so this has to be caught.
    tampered = {"model": dict(slim_state["model"])}
    tampered["model"]["gpt.weight"] = gpt_w + 1e-7
    torch.save(tampered, tmp / "tampered.pth")
    expect(1, "a weight altered by 1e-7", tmp / "full.pth", tmp / "tampered.pth")
    short = {"model": {k: v for k, v in slim_state["model"].items()
                       if k != "hifigan_decoder.weight"}}
    torch.save(short, tmp / "short.pth")
    expect(1, "a missing tensor", tmp / "full.pth", tmp / "short.pth")

    def size(n):
        return f"{n/1e6:.1f} MB" if n >= 1e6 else f"{n/1e3:.0f} kB"

    print(f"\noptimize_checkpoint selftest OK  "
          f"({size(full)} -> {size(sl)} strip -> {size(hf)} fp16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
