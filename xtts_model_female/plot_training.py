#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_training.py -- draw the loss curves and say whether the run overfit.

WHY A SEPARATE SCRIPT
---------------------
The trainer writes TensorBoard events, but reading them needs TensorBoard running
against a directory that Kaggle deletes with the session. train.log is a plain
file that survives in the notebook output, so the curves are recoverable from a
finished run months later. Parsing lives in train_log.py, which has a selftest;
this script only draws.

WHAT TO READ OFF THE PLOT
-------------------------
loss_mel_ce is the acoustic reconstruction term and the only loss that tracks
audio quality (loss_text_ce carries weight 0.01 in the total). Two curves matter
and they answer different questions:

    train   is the model fitting at all? A flat train curve means the run is not
            learning -- wrong LR, or the text is [UNK] and there is nothing to
            learn. It should fall steeply and then slowly.

    eval    is it generalising? This is the one that decides which checkpoint to
            export. While eval falls, more training helps. Once eval turns up
            while train keeps falling, the model is memorising the training
            clips and every later checkpoint is worse than one you already have.

The gap between them is the overfitting signal. On ~7 h of audio with a 500 M
parameter model, expect it to open eventually -- the question is only whether it
happened before or after your run stopped.

USAGE
    python plot_training.py --log train.log
    python plot_training.py --log train.log --out curves.png --smooth 25
    python plot_training.py --selftest        # render from a synthetic log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # no display on Kaggle or in CI
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_log  # noqa: E402


def rolling(values, window: int):
    """Centred moving average, shrinking the window at the edges.

    Not scipy or pandas: this file should import on a bare Kaggle kernel without
    pulling anything in, and the whole operation is four lines.
    """
    if window <= 1 or len(values) < 2:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot(text: str, out_path: Path, smooth: int = 15, title: str = "") -> dict:
    s = train_log.series(text)
    train, evals, best = s["train"], s["eval"], s["best"]
    if not train:
        raise SystemExit("no loss_mel_ce values in the log -- training never started")

    label, why = train_log.verdict(evals)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [2, 1]})

    # ---- panel 1: both curves ------------------------------------------
    tx, ty = [p[0] for p in train], [p[1] for p in train]
    ax1.plot(tx, ty, lw=0.8, alpha=0.30, color="tab:blue", label="train (raw)")
    if len(ty) > 3:
        ax1.plot(tx, rolling(ty, smooth), lw=2.0, color="tab:blue",
                 label=f"train (mean of {smooth})")
    if evals:
        ex, ey = [p[0] for p in evals], [p[1] for p in evals]
        ax1.plot(ex, ey, "o-", lw=2.0, ms=5, color="tab:red", label="eval (held out)")
        i_best = min(range(len(ey)), key=ey.__getitem__)
        ax1.plot(ex[i_best], ey[i_best], "*", ms=18, color="gold",
                 markeredgecolor="black", zorder=5,
                 label=f"best eval {ey[i_best]:.4f} @ step {ex[i_best]}")
    ax1.set_ylabel("loss_mel_ce")
    ax1.set_title(title or "XTTS fine-tune — acoustic reconstruction loss")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    colour = {"overfitting": "tab:red", "improving": "tab:green",
              "plateau": "tab:orange", "too-short": "grey"}[label]
    ax1.text(0.01, 0.02, f"{label.upper()} — {why}", transform=ax1.transAxes,
             fontsize=9, color="white", va="bottom",
             bbox=dict(boxstyle="round", facecolor=colour, alpha=0.85))

    # ---- panel 2: the eval curve on its own -----------------------------
    # On the shared axis the eval curve is a handful of points lost inside
    # thousands of training points, and its turn upward -- the thing that
    # decides which checkpoint to ship -- is invisible.
    if len(evals) >= 2:
        ex, ey = [p[0] for p in evals], [p[1] for p in evals]
        ax2.plot(ex, ey, "o-", lw=2.0, ms=6, color="tab:red")
        i_best = min(range(len(ey)), key=ey.__getitem__)
        ax2.axhline(ey[i_best], ls="--", lw=1, color="grey")
        ax2.plot(ex[i_best], ey[i_best], "*", ms=18, color="gold",
                 markeredgecolor="black", zorder=5)
        ax2.set_ylabel("eval loss_mel_ce")
        for x in best:
            ax2.axvline(x, color="tab:green", alpha=0.25, lw=1)
        ax2.set_title("held-out loss only — green lines are checkpoints written as best",
                      fontsize=10)
    else:
        ax2.text(0.5, 0.5, "fewer than 2 eval points — nothing to read yet",
                 ha="center", va="center", transform=ax2.transAxes, color="grey")
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.set_xlabel("global step")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    return {"verdict": label, "why": why, "n_train": len(train),
            "n_eval": len(evals), "last_step": train[-1][0],
            "best_steps": best, "png": str(out_path)}


def _selftest() -> int:
    """Render from a synthetic log that overfits, and check the verdict."""
    import tempfile
    lines, step, best_so_far = [], 0, float("inf")
    for epoch in range(6):
        for _ in range(8):
            step += 50
            # train falls the whole way; eval bottoms at epoch 2 then climbs,
            # which is the textbook overfitting shape this must detect.
            lines.append(f"--> TIME: x -- STEP: {step}/400 -- GLOBAL_STEP: {step}")
            lines.append(f"     | > loss_mel_ce: {3.5 - 0.05 * step / 50:.4f}  (x)")
        ev = 3.2 - 0.25 * epoch + (0.40 * max(0, epoch - 2))
        lines.append("\x1b[1m > EVALUATION \x1b[0m")
        lines.append(f"     | > avg_loss_mel_ce:\x1b[92m {ev:.4f} \x1b[0m(x)")
        if ev < best_so_far:                 # the trainer only writes on improvement
            best_so_far = ev
            lines.append(f" > BEST MODEL : /x/best_model_{step}.pth")
        lines.append(f" > EPOCH: {epoch + 1}/6")

    out = Path(tempfile.mkdtemp(prefix="curves_")) / "curves.png"
    info = plot("\n".join(lines), out, smooth=5, title="selftest — synthetic run")
    assert info["verdict"] == "overfitting", info
    assert out.is_file() and out.stat().st_size > 10_000, "png looks empty"
    assert info["n_eval"] == 6 and info["n_train"] == 48, info
    # Only the three improving epochs wrote a checkpoint.
    assert info["best_steps"] == [400, 800, 1200], info
    print(f"plot_training selftest OK -- {out} ({out.stat().st_size/1000:.0f} kB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="/kaggle/working/train.log")
    ap.add_argument("--out", default=None, help="PNG path (default: beside the log)")
    ap.add_argument("--smooth", type=int, default=15,
                    help="window for the training-curve moving average")
    ap.add_argument("--title", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    log = Path(args.log)
    if not log.is_file():
        print(f"no such log: {log}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else log.with_name("curves.png")

    info = plot(log.read_text(encoding="utf-8", errors="replace"),
                out, smooth=args.smooth, title=args.title)
    print(f"train points : {info['n_train']}   eval points : {info['n_eval']}")
    print(f"last step    : {info['last_step']}")
    print(f"best written : {info['best_steps'][-5:] or 'none'}")
    print(f"\nVERDICT: {info['verdict']} -- {info['why']}")
    print(f"\nwrote {info['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
