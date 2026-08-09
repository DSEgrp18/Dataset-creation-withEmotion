#!/usr/bin/env python3
"""
test_checkpoint.py -- synthesise speech from a trained run, after /kaggle/temp is gone.

WHY THIS EXISTS
    export_checkpoint.py needs --xtts-dir, the vocab-extended base, to copy
    config.json / vocab.json / speakers_xtts.pth next to the exported weights.
    That directory lives in /kaggle/temp, which Kaggle does NOT persist -- so
    after the session ends you have checkpoints and no way to load them.

    Nothing is actually lost. The trained checkpoint already contains the
    resized embeddings; only the *vocabulary file* is missing, and that is
    deterministic: base XTTS vocab (6681 tokens) plus the Sinhala tokens
    extend_vocab.py appended, in order. This script redownloads the base and
    replays that append, then hands off to the repo's own export + infer.

    The rebuild is verified against the checkpoint: if the reconstructed vocab
    size does not match the checkpoint's text_embedding rows, it refuses to
    continue rather than emit a model that decodes to the wrong characters.

USAGE (on Kaggle, with the run folder attached or in /kaggle/working)
    python test_checkpoint.py --run /kaggle/working/run/GPT_XTTS_si-*
    python test_checkpoint.py --run <run> --checkpoint best_model_1082.pth
    python test_checkpoint.py --run <run> --ref my_voice.wav --text "..."
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# The tokens extend_vocab.py appended, in the EXACT order its log reported:
#   vocab: 6681 -> 6755  (+74 tokens)
#   added: [si] ං ඃ අ ආ ...
#
# ORDER IS THE WHOLE POINT. Token IDs are assigned by append order, so a list
# with the right 74 tokens in the wrong sequence passes the size check and then
# decodes every Sinhala character to the wrong glyph. Kept as one literal
# sequence, copied verbatim from the run log, so it cannot drift.
_ADDED = (
    "[si] ං ඃ අ ආ ඇ ඈ ඉ ඊ උ ඌ ඍ එ ඒ ඓ ඔ ඕ ඖ ක ඛ ග ඝ ඞ ඟ ච ඡ ජ ඣ ඤ ඥ ට ඨ ඩ ඪ "
    "ණ ඬ ත ථ ද ධ න ඳ ප ඵ බ භ ම ඹ ය ර ල ව ශ ෂ ස හ ළ ෆ ් ා ැ ෑ ි ී ු ූ ෘ ෙ ේ "
    "ෛ ො ෝ ෞ ෲ"
)
ADDED_TOKENS = _ADDED.split()

BASE_FILES = ["config.json", "vocab.json", "speakers_xtts.pth"]
HF_REPO = "coqui/XTTS-v2"


def rebuild_vocab(base_dir: Path, out_dir: Path, expect_size: int | None) -> Path:
    """Append the Sinhala tokens to the base vocab, exactly as extend_vocab did."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = json.loads((base_dir / "vocab.json").read_text(encoding="utf-8"))
    model = vocab.get("model", vocab)
    tokens = model["vocab"]
    before = len(tokens)

    for tok in ADDED_TOKENS:
        if tok not in tokens:
            tokens[tok] = len(tokens)
    after = len(tokens)
    print(f"vocab: {before} -> {after}  (+{after - before} tokens)")

    if expect_size is not None and after != expect_size:
        raise SystemExit(
            f"\nERROR: rebuilt vocab is {after} tokens but the checkpoint expects "
            f"{expect_size}.\nThe token list in this script does not match the run. "
            f"Copy the 'added:' line from that run's extend_vocab output into "
            f"ADDED_TOKENS, or re-run extend_vocab.py with the original charset.")

    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    for name in ("config.json", "speakers_xtts.pth"):
        shutil.copy2(base_dir / name, out_dir / name)
    return out_dir


def checkpoint_vocab_size(ckpt: Path) -> int | None:
    """Read the trained text-embedding size straight out of the checkpoint."""
    import torch
    try:
        blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  (could not inspect checkpoint: {type(exc).__name__})")
        return None
    sd = blob.get("model", blob)
    for key in ("xtts.gpt.text_embedding.weight", "gpt.text_embedding.weight"):
        if key in sd:
            return int(sd[key].shape[0])
    for key, val in sd.items():
        if key.endswith("text_embedding.weight"):
            return int(val.shape[0])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="trainer run folder (has *.pth + config.json)")
    ap.add_argument("--checkpoint", default=None,
                    help="which .pth (default: best_model.pth, else newest)")
    ap.add_argument("--src", default="/kaggle/working/XTTS_V2_Baseline/src",
                    help="the cloned repo's src/")
    ap.add_argument("--base", default="/kaggle/working/xtts_base_dl")
    ap.add_argument("--work", default="/kaggle/working/xtts_test")
    ap.add_argument("--ref", default=None,
                    help="reference wav for the voice; defaults to a training clip")
    ap.add_argument("--text", action="append", default=None,
                    help="repeatable Sinhala line to synthesise")
    ap.add_argument("--out", default="/kaggle/working/samples")
    args = ap.parse_args()

    run = Path(args.run)
    if not run.is_dir():
        # Allow a glob like .../GPT_XTTS_si-*
        matches = sorted(Path(run.parent).glob(run.name))
        if not matches:
            print(f"ERROR: no run folder at {run}", file=sys.stderr)
            return 2
        run = matches[-1]
    print(f"run: {run}")

    if args.checkpoint:
        ckpt = run / args.checkpoint if not Path(args.checkpoint).is_absolute() \
            else Path(args.checkpoint)
    else:
        best = run / "best_model.pth"
        ckpt = best if best.is_file() else max(
            run.glob("*.pth"), key=lambda p: p.stat().st_mtime, default=None)
    if ckpt is None or not ckpt.is_file():
        print(f"ERROR: no checkpoint in {run}", file=sys.stderr)
        return 2
    print(f"checkpoint: {ckpt.name}  ({ckpt.stat().st_size/1e6:.0f} MB)")

    want = checkpoint_vocab_size(ckpt)
    print(f"checkpoint text vocab: {want}")

    # 1. base XTTS files (small: config + vocab + speakers, not model.pth)
    base = Path(args.base)
    base.mkdir(parents=True, exist_ok=True)
    if not all((base / f).is_file() for f in BASE_FILES):
        from huggingface_hub import hf_hub_download
        for f in BASE_FILES:
            print(f"downloading {f} ...")
            p = hf_hub_download(repo_id=HF_REPO, filename=f, local_dir=str(base))
            print(f"  {p}")

    # 2. replay the vocab extension
    xtts_si = rebuild_vocab(base, Path(args.work) / "xtts_si", want)

    # 3. the repo's own exporter turns a trainer checkpoint into a loadable model
    export = Path(args.work) / "export"
    cmd = [sys.executable, f"{args.src}/export_checkpoint.py",
           "--run", str(run), "--xtts-dir", str(xtts_si),
           "--out", str(export), "--checkpoint", str(ckpt)]
    print("\n" + " ".join(cmd))
    if subprocess.run(cmd).returncode != 0:
        return 1

    # 4. a reference clip is required -- XTTS clones the voice from it
    ref = args.ref
    if not ref:
        for pat in ("/kaggle/working/xtts_data/wavs/*.wav",
                    "/kaggle/temp/dataset/wavs/*.wav",
                    "/kaggle/input/**/clips/*.wav"):
            hits = sorted(Path("/").glob(pat.lstrip("/")))
            if hits:
                ref = str(hits[0])
                break
    if not ref:
        print("\nERROR: no reference wav found. Pass --ref <a 6-30 s clip>.",
              file=sys.stderr)
        print("XTTS clones the voice from this clip; it cannot synthesise without one.",
              file=sys.stderr)
        return 2
    print(f"reference: {ref}")

    cmd = [sys.executable, f"{args.src}/infer.py",
           "--model-dir", str(export), "--ref", ref, "--out", args.out]
    for t in (args.text or []):
        cmd += ["--text", t]
    print("\n" + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"\nwavs in {args.out} -- listen before reading anything into the loss curves.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
