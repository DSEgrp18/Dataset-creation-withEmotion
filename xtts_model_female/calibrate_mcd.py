#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_mcd.py -- find out what this repo's MCD numbers actually mean.

WHY THIS EXISTS
---------------
evaluate_xtts.py reports MCD, and a run scored 63.13 dB. Published Sinhala TTS
papers report 13-20 dB. Those two numbers cannot be compared: MCD is defined over
mel-cepstra, and every implementation differs in how it obtains them -- mel band
count, log base, DCT normalisation, the silence floor. evaluate_xtts.py says so
in its own output, and the README repeats it, but a warning does not tell you
whether 63 is good.

So measure the scale instead of guessing at it. Score pairs of REAL speech whose
relationship is already known -- identical, mildly degraded, unrelated, not
speech at all -- and read the run's number against those landmarks.

The clips come from dataset/<episode>/clips/, built by the pipeline one level up.
They are lossy multi-speaker broadcast, not studio recordings, so the landmarks
shift somewhat on cleaner audio. The ORDERING and the ORDER OF MAGNITUDE are what
to rely on; treat the boundaries as approximate.

USAGE
    python calibrate_mcd.py                        # uses dataset/, writes to a tmpdir
    python calibrate_mcd.py --clips path/to/clips  # any directory of wavs

Results are recorded in RESULTS.md. Re-run this after touching mcd_and_f0().
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_xtts import SR, mcd_and_f0  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=None,
                    help="directory of wavs (default: the first dataset/*/clips found)")
    ap.add_argument("--min-sec", type=float, default=3.0,
                    help="ignore clips shorter than this")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    clip_dir = Path(args.clips) if args.clips else next(
        iter(sorted(root.glob("dataset/*/clips"))), None)
    if clip_dir is None or not clip_dir.is_dir():
        print("no clips directory found -- pass --clips", file=sys.stderr)
        return 1

    loaded = []
    for c in sorted(clip_dir.glob("*.wav")):
        y, _ = librosa.load(str(c), sr=SR, mono=True)
        if len(y) > SR * args.min_sec:
            loaded.append((c, y))
        if len(loaded) >= 3:
            break
    if len(loaded) < 3:
        print(f"need 3 clips over {args.min_sec}s in {clip_dir}", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="mcd_cal_"))
    rng = np.random.default_rng(0)
    (c0, y0), (c1, _), (c2, _) = loaded

    def wav(name, y):
        p = tmp / name
        sf.write(str(p), y, SR)
        return p

    def run(label, a, b):
        m = mcd_and_f0(a, b)
        val = f"{m['mcd_db']:8.2f} dB" if m else "  (too short)"
        print(f"  {label:<44s} {val}")

    print(f"clips: {clip_dir}   SR={SR}\n")

    print("IDENTICAL -- the floor of the scale")
    run("same file vs itself", c0, c0)

    print("\nSAME UTTERANCE, DEGRADED -- small real differences")
    for snr in (30, 20, 10):
        amp = np.sqrt(np.mean(y0 ** 2)) * 10 ** (-snr / 20)
        run(f"+ white noise at {snr} dB SNR",
            c0, wav(f"n{snr}.wav", y0 + rng.normal(0, amp, len(y0))))
    run("resampled 22k->8k->22k (band-limited)", c0, wav("bl.wav", librosa.resample(
        librosa.resample(y0, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR)))
    run("pitch-shifted +2 semitones",
        c0, wav("ps.wav", librosa.effects.pitch_shift(y0, sr=SR, n_steps=2)))

    # This band is the one that matters: a synthesiser scoring here is producing
    # speech unrelated to the target, whatever its loss curve says.
    print("\nDIFFERENT CONTENT -- what 'unrelated speech' scores")
    run("clip A vs clip B", c0, c1)
    run("clip A vs clip C", c0, c2)
    run("clip B vs clip C", c1, c2)

    print("\nNOT SPEECH -- the ceiling of the scale")
    run("clip A vs white noise", c0, wav("wn.wav", rng.normal(0, 0.05, len(y0))))
    run("clip A vs silence", c0, wav("sil.wav", np.full(len(y0), 1e-6)))
    run("clip A vs a 440 Hz tone", c0, wav(
        "tone.wav", 0.2 * np.sin(2 * np.pi * 440 * np.arange(len(y0)) / SR)))

    print(f"\nscratch wavs in {tmp}")
    print("Record the run's MCD against these landmarks in RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
