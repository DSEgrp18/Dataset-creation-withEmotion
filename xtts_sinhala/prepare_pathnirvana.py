#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_pathnirvana.py -- pathnirvana/sinhala-tts-dataset -> XTTS training input.

    sinhala_dataset.tar.bz2 (wavs/) + metadata.csv
        -> <out>/wavs/*.wav
           <out>/metadata_train.csv     audio_file|text|speaker_name
           <out>/metadata_eval.csv

The output is the `coqui` formatter's format, which is what the official
TTS/demos/xtts_ft_demo trainer consumes.

WHAT THIS SCRIPT DECIDES
------------------------
1. TEXT      Column 1 of metadata.csv (the romanisation) folded to ascii by
             sinhala_text.fold. Column 2 (Sinhala script) is NOT used: several
             hundred rows abbreviate it to the "-පෙ-" ellipsis marker while the
             audio speaks the full sentence, so column 1 is the only column
             that reliably matches the waveform.

2. SPEAKER   Default `mettananda` only (~5400 clips, ~11.8 h, male).
             XTTS conditions on a second clip *from the same speaker* during
             training. Mixing speakers under one label teaches the model that
             the reference does not predict the voice, which is the capability
             being fine-tuned. Use --speaker both only if you also keep the
             real per-speaker labels (this script does).

3. LENGTH    GPTArgs.max_wav_length = 255995 samples @ 22050 Hz = 11.61 s.
             Longer clips are dropped SILENTLY by the dataloader. pathnirvana
             clips run to 15 s, so they are dropped here instead, where you can
             see the count.

4. TOKENS    GPTArgs.max_text_length = 200 tokens. Measured max on this corpus
             is 117, so nothing should trip this -- it is checked anyway.

USAGE
    python prepare_pathnirvana.py --out ./si_dataset              # downloads
    python prepare_pathnirvana.py --out ./si_dataset --wavs /path/to/wavs
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import statistics
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinhala_text import fold  # noqa: E402

RELEASE = "https://github.com/pnfo/sinhala-tts-dataset/releases/download/v2.1/sinhala_dataset.tar.bz2"
METADATA = "https://raw.githubusercontent.com/pathnirvana/sinhala-tts-dataset/master/metadata.csv"

XTTS_MAX_SECONDS = 11.6      # GPTArgs.max_wav_length / 22050
XTTS_MIN_SECONDS = 1.0
XTTS_MAX_CHARS = 250         # VoiceBpeTokenizer.char_limits["en"]


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def hook(blocks, size, total):
        if total > 0 and blocks % 400 == 0:
            pct = 100 * blocks * size / total
            print(f"\r    {min(pct, 100):5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    print()
    tmp.rename(dest)
    return dest


def _find_wavs(root: Path) -> Path | None:
    """The tarball layout has moved between releases; search rather than guess."""
    if (root / "wavs").is_dir():
        return root / "wavs"
    for cand in root.rglob("wavs"):
        if cand.is_dir() and any(cand.glob("*.wav")):
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset directory to build")
    ap.add_argument("--cache", default=None,
                    help="where to keep the downloaded tarball (default <out>/../_cache)")
    ap.add_argument("--wavs", default=None,
                    help="existing wavs/ directory; skips the 1.7 GB download")
    ap.add_argument("--metadata", default=None,
                    help="existing metadata.csv; downloaded if omitted")
    ap.add_argument("--speaker", default="mettananda",
                    choices=["mettananda", "oshadi", "both"])
    ap.add_argument("--max-seconds", type=float, default=XTTS_MAX_SECONDS)
    ap.add_argument("--min-seconds", type=float, default=XTTS_MIN_SECONDS)
    ap.add_argument("--eval-size", type=int, default=128)
    ap.add_argument("--copy-wavs", action="store_true",
                    help="copy the kept wavs into <out>/wavs instead of symlinking")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    cache = Path(args.cache) if args.cache else out.parent / "_cache"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- audio
    print("[1/4] audio")
    if args.wavs:
        wavs = Path(args.wavs).resolve()
    else:
        tar = _download(RELEASE, cache / "sinhala_dataset.tar.bz2")
        extract = cache / "extracted"
        if not _find_wavs(extract):
            print("  extracting (a few minutes)")
            extract.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar, "r:bz2") as tf:
                tf.extractall(extract)
        wavs = _find_wavs(extract)
    if wavs is None or not wavs.is_dir():
        print(f"ERROR: no wavs/ directory found (looked under {wavs})", file=sys.stderr)
        return 2
    print(f"  wavs: {wavs}  ({sum(1 for _ in wavs.glob('*.wav'))} files)")

    # ------------------------------------------------------------- metadata
    print("[2/4] metadata")
    meta = Path(args.metadata) if args.metadata else _download(METADATA, cache / "metadata.csv")
    rows = [l.rstrip("\n").split("|")
            for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if len(r) >= 4]
    print(f"  {len(rows)} lines   speakers: {dict(Counter(r[3] for r in rows))}")

    # --------------------------------------------------------------- filter
    print("[3/4] filtering")
    import soundfile as sf

    wanted = {"mettananda", "oshadi"} if args.speaker == "both" else {args.speaker}
    drop: Counter = Counter()
    kept: list[tuple[str, str, str, float]] = []

    for r in rows:
        clip_id, roman, _script, speaker = r[0], r[1], r[2], r[3]
        if speaker not in wanted:
            drop[f"speaker!={'/'.join(sorted(wanted))}"] += 1
            continue
        wav = wavs / f"{clip_id}.wav"
        if not wav.is_file():
            drop["missing_wav"] += 1
            continue
        try:
            info = sf.info(str(wav))            # header only, no decode
        except Exception:
            drop["unreadable_wav"] += 1
            continue
        dur = info.frames / info.samplerate
        if dur > args.max_seconds:
            drop[f"longer_than_{args.max_seconds:g}s"] += 1
            continue
        if dur < args.min_seconds:
            drop[f"shorter_than_{args.min_seconds:g}s"] += 1
            continue
        text = fold(roman)
        if not text:
            drop["empty_text"] += 1
            continue
        if len(text) > XTTS_MAX_CHARS:
            drop[f"text_over_{XTTS_MAX_CHARS}_chars"] += 1
            continue
        kept.append((clip_id, text, speaker, dur))

    if not kept:
        print("ERROR: everything was filtered out", file=sys.stderr)
        print(dict(drop), file=sys.stderr)
        return 3

    # ---------------------------------------------------------------- write
    print("[4/4] writing")
    dest_wavs = out / "wavs"
    dest_wavs.mkdir(exist_ok=True)
    for clip_id, *_ in kept:
        src, dst = wavs / f"{clip_id}.wav", dest_wavs / f"{clip_id}.wav"
        if dst.exists() or dst.is_symlink():
            continue
        if args.copy_wavs:
            shutil.copy2(src, dst)
        else:
            try:
                dst.symlink_to(src)
            except OSError:          # Windows without developer mode
                shutil.copy2(src, dst)

    rng = random.Random(args.seed)
    order = kept[:]
    rng.shuffle(order)
    n_eval = min(args.eval_size, len(order) // 10)
    eval_rows, train_rows = order[:n_eval], order[n_eval:]

    for name, part in (("metadata_train.csv", train_rows),
                       ("metadata_eval.csv", eval_rows)):
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_MINIMAL,
                           lineterminator="\n")
            w.writerow(["audio_file", "text", "speaker_name"])
            for clip_id, text, speaker, _dur in part:
                w.writerow([f"wavs/{clip_id}.wav", text, speaker])

    # ---------------------------------------------------------------- report
    secs = [d for *_, d in kept]
    srs = Counter(sf.info(str(dest_wavs / f'{c}.wav')).samplerate for c, *_ in kept[:200])
    print(f"\n{out}")
    print(f"  train {len(train_rows)}   eval {len(eval_rows)}   "
          f"total {sum(secs) / 3600:.2f} h")
    print(f"  duration : median {statistics.median(secs):.2f}s  max {max(secs):.2f}s")
    print(f"  speakers : {dict(Counter(s for _, _, s, _ in kept))}")
    print(f"  samplerate (first 200): {dict(srs)}   (XttsAudioConfig wants 22050)")
    if drop:
        print("  dropped:")
        for reason, n in drop.most_common():
            print(f"    {n:6d}  {reason}")
    print("\n  first three lines of metadata_train.csv:")
    for line in (out / "metadata_train.csv").read_text(encoding="utf-8").splitlines()[:4]:
        print("   ", line)

    hours = sum(secs) / 3600
    if hours < 4:
        print("\nNOTE: under 4 h. XTTS can adapt to a new sound inventory from this"
              "\n      much audio, but expect accent artefacts on unseen words.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
