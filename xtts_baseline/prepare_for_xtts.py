#!/usr/bin/env python3
"""
prepare_for_xtts.py -- turn this repo's manifest into input for XTTS_V2_Baseline.

    dataset/all_manifests.csv  ->  xtts_metadata.csv  (audio_file|text|speaker_name)

That file feeds DSEgrp18/XTTS_V2_Baseline's prepare_dataset.py, which does the
resampling, trimming and train/eval split. This script only decides WHICH clips
go in, because that repo's preparer knows nothing about our `quality` column or
about XTTS's hard length ceiling.

THREE FILTERS, and the second one is not optional
-------------------------------------------------
1. QUALITY   The corpus has two tiers. The 201-episode serial is 11 kHz / 16 kbps,
             so its clips are band-limited to 5.5 kHz -- fricatives are largely
             gone. Training voice quality on that teaches the model to generate
             muffled speech. Default is hifi-only; --quality all to include both.

2. LENGTH    XTTS GPTArgs.max_wav_length is 255995 samples @22050 Hz = 11.61 s.
             Longer clips are dropped SILENTLY by the dataloader. Our clips run
             to 30 s, so ~16% of the audio would vanish without warning. They are
             dropped here instead, where the count is printed.

3. SNAPPED   Clips whose timestamps were never confirmed against the VAD
             (`snapped=False`) can start or end mid-word. Excluded by default;
             --allow-unsnapped keeps them.

SPEAKER LABELS -- READ THIS
---------------------------
Radio drama is multi-speaker and nothing in this pipeline diarizes it, so every
clip is written under ONE speaker name. XTTS is a voice-cloning model: during
training it samples another clip from the same speaker as the conditioning
reference. If that "speaker" is actually thirty different actors, the model
learns that the reference clip does NOT predict the output voice -- which is
precisely the capability being fine-tuned.

Expect this run to teach Sinhala phonetics and prosody, and NOT to give
controllable voice identity. Speaker diarization is the fix, and it belongs
upstream in the dataset build.

USAGE
    python prepare_for_xtts.py --src dataset --out xtts_metadata.csv
    python prepare_for_xtts.py --src dataset --quality all --max-seconds 11.4
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

# GPTArgs.max_wav_length = 255995 @ 22050 Hz = 11.61 s. Stay under it.
XTTS_MAX_SECONDS = 11.4
XTTS_MIN_SECONDS = 0.9
XTTS_MAX_CHARS = 190       # GPTArgs.max_text_length = 200 tokens; chars ~ tokens


def load_rows(src: Path) -> list[dict]:
    """Read all_manifests.csv, or every per-episode manifest if it is absent."""
    combined = src / "all_manifests.csv"
    if combined.exists():
        with combined.open(encoding="utf-8-sig") as fh:
            return list(csv.DictReader(fh))
    rows: list[dict] = []
    for man in sorted(src.glob("*/manifest.csv")):
        with man.open(encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="dataset root holding <episode>/ dirs + all_manifests.csv")
    ap.add_argument("--out", default="xtts_metadata.csv")
    ap.add_argument("--quality", default="hifi", choices=["hifi", "lofi", "all"])
    ap.add_argument("--max-seconds", type=float, default=XTTS_MAX_SECONDS)
    ap.add_argument("--min-seconds", type=float, default=XTTS_MIN_SECONDS)
    ap.add_argument("--max-chars", type=int, default=XTTS_MAX_CHARS)
    ap.add_argument("--allow-unsnapped", action="store_true",
                    help="keep clips whose cut points were never VAD-verified")
    ap.add_argument("--speaker", default="muwan_palassa",
                    help="single speaker label (see the caveat in the docstring)")
    args = ap.parse_args()

    src = Path(args.src)
    rows = load_rows(src)
    if not rows:
        print(f"ERROR: no manifest under {src}", file=sys.stderr)
        return 2
    print(f"{len(rows)} clips in manifest")

    drop = Counter()
    kept: list[tuple[str, str]] = []
    for r in rows:
        q = r.get("quality", "hifi")
        if args.quality != "all" and q != args.quality:
            drop[f"quality!={args.quality}"] += 1
            continue
        if not args.allow_unsnapped and str(r.get("snapped", "True")) != "True":
            drop["unsnapped"] += 1
            continue
        try:
            dur = float(r["duration"])
        except (KeyError, ValueError):
            drop["bad_duration"] += 1
            continue
        if dur > args.max_seconds:
            drop[f"longer_than_{args.max_seconds:g}s"] += 1
            continue
        if dur < args.min_seconds:
            drop[f"shorter_than_{args.min_seconds:g}s"] += 1
            continue
        text = (r.get("text") or "").strip()
        if not text:
            drop["empty_text"] += 1
            continue
        if len(text) > args.max_chars:
            drop[f"text_over_{args.max_chars}_chars"] += 1
            continue
        kept.append((r["clip_id"], text))

    if not kept:
        print("ERROR: every clip was filtered out.", file=sys.stderr)
        print(json.dumps(drop, indent=2), file=sys.stderr)
        return 3

    # Pipe-separated with this exact header: XTTS_V2_Baseline/prepare_dataset.py
    # keys off "audio_file"/"text", and picks '|' as the delimiter when the
    # first line has more pipes than commas.
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("audio_file|text|speaker_name\n")
        for cid, text in kept:
            fh.write(f"{cid}|{text}|{args.speaker}\n")

    # Report against the manifest's own durations, so the yield is visible
    # before an hour of GPU time is spent discovering it.
    by_id = {r["clip_id"]: r for r in rows}
    secs = [float(by_id[c]["duration"]) for c, _ in kept]
    total_h = sum(secs) / 3600
    print(f"\nwrote {out}  ({len(kept)} clips, {total_h:.2f} h)")
    print(f"  duration : median {st.median(secs):.2f}s  max {max(secs):.2f}s")
    print(f"  speaker  : {args.speaker!r} (single label -- see docstring)")
    if drop:
        print("\ndropped:")
        for reason, n in drop.most_common():
            print(f"  {n:6d}  {reason}")

    if total_h < 1.0:
        print("\nNOTE: under 1 hour. Enough to validate the pipeline end to end,"
              "\n      not enough to judge output quality. Keep building episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
