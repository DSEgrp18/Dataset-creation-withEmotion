#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_log.py -- read the trainer's stdout well enough to decide "is this run alive?"

Kept out of the notebook so it can be tested against real logs. The notebook has
burned a full session on a NaN run once and aborted a healthy run once, and both
failures were in ten lines of regex that nothing exercised.

WHAT MAKES THIS FIDDLY
    The coqui Trainer colours its epoch summaries, so the raw bytes are

        | > avg_loss_mel_ce:\x1b[92m 3.6631 \x1b[0m(-0.1749)

    Match on the raw text and the "value" you capture is \x1b[92m, which is not
    a number, which reads as divergence. Strip the escapes first, always.

USAGE
    from train_log import losses, nonfinite, diverged, interesting
"""

from __future__ import annotations

import re

# CSI sequences: colour is \x1b[92m, but tqdm also emits \x1b[A and \x1b[2K.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

# Deliberately also matches avg_loss_mel_ce -- an epoch average is a real signal
# and it is the line that carries the colour codes.
_LOSS = re.compile(r"loss_mel_ce:\s*(\S+)")

# A finite number and nothing else: rejects nan, inf, -inf and any stray escape.
_FINITE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

_KEYS = ("loss_mel_ce", "EPOCH", "EVAL", "BEST", "STEP:")
_NOISE = ("it/s", "iB/s")

# For the curves. The trainer prints the step on its own line, ahead of the
# losses it belongs to:
#     --> TIME: ... -- STEP: 350/879 -- GLOBAL_STEP: 350
#          | > loss_mel_ce: 3.4930  (3.7744)
_STEP = re.compile(r"GLOBAL_STEP:\s*(\d+)")
# "> loss_mel_ce" only. "> avg_loss_mel_ce" cannot match this, because the
# characters right after "> " are "avg_" -- which is what separates the running
# training loss from an epoch or eval average.
_TRAIN_LOSS = re.compile(r">\s*loss_mel_ce:\s*(\S+)")
_AVG_LOSS = re.compile(r">\s*avg_loss_mel_ce:\s*(\S+)")
_BEST = re.compile(r"BEST MODEL\s*:.*?best_model_(\d+)\.pth")


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def is_finite(value: str) -> bool:
    return bool(_FINITE.match(value))


def losses(text: str) -> list[str]:
    """Every loss_mel_ce value in the log, oldest first, as written."""
    return _LOSS.findall(strip_ansi(text))


def nonfinite(values) -> list[str]:
    return [v for v in values if not is_finite(v)]


def diverged(values, need: int = 3) -> bool:
    """True when the run is nan and is never coming back.

    One nan among finite neighbours is a blip -- gradient clipping recovers from
    those. It is `need` non-finite values in a row AT THE TAIL that is terminal,
    so look only at the tail: a run that was nan an hour ago and is finite now
    is a run that recovered, and aborting it would be the same mistake in the
    other direction.
    """
    recent = list(values)[-need:]
    return len(recent) == need and all(not is_finite(v) for v in recent)


def interesting(text: str, limit: int | None = None) -> list[str]:
    """Log lines worth printing in a heartbeat; tqdm bars are not."""
    lines = [l.strip() for l in strip_ansi(text).splitlines()
             if any(k in l for k in _KEYS) and not any(n in l for n in _NOISE)]
    return lines[-limit:] if limit else lines


def series(text: str) -> dict:
    """Loss curves, as {"train": [(step, loss)], "eval": [...], "best": [step]}.

    The trainer prints an average under two different headings and they mean
    opposite things: under `> EVALUATION` it is held-out loss, and anywhere else
    it is the training epoch average. Plotting the training average as if it were
    eval would hide overfitting completely -- the two curves would be the same
    curve -- so the block heading is tracked rather than the line matched alone.
    """
    train: list[tuple[int, float]] = []
    evals: list[tuple[int, float]] = []
    best: list[int] = []
    step, in_eval = 0, False

    for line in strip_ansi(text).splitlines():
        m = _STEP.search(line)
        if m:
            step, in_eval = int(m.group(1)), False
        if "EVALUATION" in line:
            in_eval = True
        elif "EPOCH:" in line:
            in_eval = False

        m = _BEST.search(line)
        if m:
            best.append(int(m.group(1)))
            continue
        m = _AVG_LOSS.search(line)
        if m:
            if is_finite(m.group(1)):
                (evals if in_eval else train).append((step, float(m.group(1))))
            continue
        m = _TRAIN_LOSS.search(line)
        if m and is_finite(m.group(1)):
            train.append((step, float(m.group(1))))

    return {"train": train, "eval": evals, "best": best}


def verdict(evals, patience: int = 2, rise: float = 0.01) -> tuple[str, str]:
    """Read the eval curve: (label, one-line explanation).

    Labels are `improving`, `plateau`, `overfitting`, `too-short`. The point of
    the distinction is which checkpoint to export -- on `overfitting` the last
    checkpoint is worse than one you already have, so export `best_model.pth`
    and not the final step.

    `rise` is relative, because loss scale differs between runs; 0.01 is 1 %
    above the minimum, comfortably outside eval noise on this model.
    """
    pts = [(s, v) for s, v in evals if v == v]        # drop nan
    if len(pts) < 3:
        return "too-short", f"only {len(pts)} eval points -- train longer before reading this"

    losses_ = [v for _, v in pts]
    i_best = min(range(len(losses_)), key=losses_.__getitem__)
    best_v, last_v = losses_[i_best], losses_[-1]
    after = len(losses_) - 1 - i_best

    if after == 0:
        return "improving", (f"eval loss is still at its minimum ({best_v:.4f}) at the last "
                             f"point -- the run stopped early, not because it converged")
    if after >= patience and last_v > best_v * (1 + rise):
        return "overfitting", (
            f"eval bottomed at {best_v:.4f} (step {pts[i_best][0]}) and has risen to "
            f"{last_v:.4f} over {after} evals -- export best_model.pth, not the last step")
    return "plateau", (f"eval best {best_v:.4f} at step {pts[i_best][0]}, now {last_v:.4f} "
                       f"-- flat within noise for {after} evals")


def _selftest() -> None:
    # Verbatim from the run that this module exists because of: healthy losses,
    # then a coloured epoch average that the old regex read as divergence.
    healthy = (
        "\x1b[4m\x1b[1m > EPOCH: 0/1\x1b[0m\n"
        "| > loss_mel_ce: 4.310885906219482  (4.310885906219482)\n"
        "| > loss_mel_ce: 3.7885243892669678  (3.9431158542633056)\n"
        "| > avg_loss_mel_ce: 3.8380595048268638 \x1b[0m(+0.0)\n"
        "| > avg_loss_mel_ce:\x1b[92m 3.6631078720092773 \x1b[0m(-0.1749516328175864)\n"
    )
    vals = losses(healthy)
    assert vals == ["4.310885906219482", "3.7885243892669678",
                    "3.8380595048268638", "3.6631078720092773"], vals
    assert nonfinite(vals) == []
    assert not diverged(vals)

    # Verbatim from the fp16 run that produced NaN checkpoints for 8 hours.
    nan_log = (
        " > loss_mel_ce: nan  (nan)\n"
        " > loss_mel_ce: nan  (nan)\n"
        " > loss_mel_ce: nan  (nan)\n"
    )
    assert nonfinite(losses(nan_log)) == ["nan", "nan", "nan"]
    assert diverged(losses(nan_log))

    # A blip that recovers must not abort a good run.
    assert not diverged(["4.1", "nan", "3.9", "3.8", "3.7", "3.6"])
    # Two at the tail is not yet enough to be sure.
    assert not diverged(["4.1", "3.9", "3.8", "3.7", "nan", "nan"])
    # Three is.
    assert diverged(["4.1", "3.9", "3.8", "nan", "nan", "nan"])
    # inf counts as divergence too.
    assert diverged(["inf", "-inf", "nan"])
    # Scientific notation is a number.
    assert is_finite("1.5e-07") and is_finite("-3e5") and is_finite(".5")
    assert not is_finite("nan") and not is_finite("\x1b[92m") and not is_finite("")

    assert interesting(healthy, 2) == [
        "| > avg_loss_mel_ce: 3.8380595048268638 (+0.0)",
        "| > avg_loss_mel_ce: 3.6631078720092773 (-0.1749516328175864)"]

    # --- series() -------------------------------------------------------
    # Verbatim shape of the real run, including the coloured eval average and
    # the BEST MODEL line that carries the exact step.
    run = (
        "--> TIME: 2026-08-19 10:59:45 -- STEP: 350/879 -- GLOBAL_STEP: 350\n"
        "     | > loss_mel_ce: 3.493067979812622  (3.774454493522644)\n"
        "--> TIME: 2026-08-19 11:10:23 -- STEP: 800/879 -- GLOBAL_STEP: 800\n"
        "     | > loss_mel_ce: 3.054823160171509  (3.491334294974804)\n"
        "\x1b[1m > EVALUATION \x1b[0m\n"
        "  \x1b[1m--> EVAL PERFORMANCE\x1b[0m\n"
        "     | > avg_loss_mel_ce:\x1b[92m 3.189875941527517 \x1b[0m(+0.0)\n"
        " > BEST MODEL : /kaggle/temp/run/training/x/best_model_880.pth\n"
        "\x1b[4m\x1b[1m > EPOCH: 1/39\x1b[0m\n"
        "--> TIME: 2026-08-19 11:20:46 -- STEP: 320/879 -- GLOBAL_STEP: 1200\n"
        "     | > loss_mel_ce: 3.3386614322662354  (3.0876808032393463)\n"
        "\x1b[1m > EVALUATION \x1b[0m\n"
        "     | > avg_loss_mel_ce:\x1b[92m 3.020407877470318 \x1b[0m(-0.169)\n"
        " > BEST MODEL : /kaggle/temp/run/training/x/best_model_1760.pth\n"
    )
    s = series(run)
    assert s["train"] == [(350, 3.493067979812622), (800, 3.054823160171509),
                          (1200, 3.3386614322662354)], s["train"]
    # The eval averages must NOT have leaked into the training curve, and must
    # carry the step of the training point they follow.
    assert s["eval"] == [(800, 3.189875941527517), (1200, 3.020407877470318)], s["eval"]
    assert s["best"] == [880, 1760], s["best"]

    # A training epoch average is not an eval point.
    epoch_avg = (" > EPOCH: 2/39\n"
                 "     | > avg_loss_mel_ce: 2.9 (+0.0)\n")
    assert series(epoch_avg)["eval"] == []
    assert series(epoch_avg)["train"] == [(0, 2.9)]

    # nan values are dropped from the curves rather than plotted as gaps.
    assert series(" | > loss_mel_ce: nan  (nan)\n")["train"] == []

    # --- verdict() ------------------------------------------------------
    assert verdict([(1, 3.0), (2, 2.9)])[0] == "too-short"
    assert verdict([(1, 3.0), (2, 2.9), (3, 2.8)])[0] == "improving"
    # Risen well clear of the minimum, and stayed there.
    assert verdict([(1, 3.0), (2, 2.5), (3, 2.7), (4, 2.9)])[0] == "overfitting"
    # Within 1 % of the minimum is noise, not overfitting.
    assert verdict([(1, 3.0), (2, 2.5), (3, 2.505), (4, 2.51)])[0] == "plateau"
    # One bad eval after the minimum is not yet a verdict of overfitting.
    assert verdict([(1, 3.0), (2, 2.9), (3, 2.5), (4, 2.9)], patience=2)[0] == "plateau"

    print("train_log selftest OK")


if __name__ == "__main__":
    _selftest()
