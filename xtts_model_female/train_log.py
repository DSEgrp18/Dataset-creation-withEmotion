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

    print("train_log selftest OK")


if __name__ == "__main__":
    _selftest()
