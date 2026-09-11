#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_vits.py -- the XTTS dataset, made ready for a VITS fine-tune.

This deliberately does NOT re-do the work prepare_voicemakers.py already did.
Discovery, filtering, the digit rule, the character limit, the [UNK] gate and
the train/eval split all happened there, and the whole point of comparing two
architectures is that they see **the same data, split the same way, with the
same text**. Re-preparing from the corpus would quietly reintroduce differences.

So this takes prepare_voicemakers.py's output directory and changes exactly two
things:

  RESAMPLE   VoiceMakers ships 44.1 kHz. XTTS resamples to 22.05 kHz internally
             on load; VITS does not, and a sample-rate mismatch against the
             pretrained checkpoint is silent and ruinous. So the audio is
             written out once at the target rate rather than left to chance.

  SPLIT BY SPEAKER   Piper ships one model per voice, and fine-tuning a
             single-speaker pretrained VITS into a single-speaker VITS needs no
             surgery on a speaker-embedding table. See train_vits_female.py.
             The multi-speaker file is written too, for anyone who wants it.

Everything else -- which clips, which text, which 80 held out -- is inherited
unchanged, which is what makes the eventual comparison mean something.

USAGE
    python prepare_vits.py --dataset ../female_dataset --out ./vits_dataset
    python prepare_vits.py --dataset ../female_dataset --out ./vits_dataset --sample-rate 22050
    python prepare_vits.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The pretrained LJSpeech VITS this is meant to be fine-tuned from is 22.05 kHz.
# Changing this means changing the checkpoint you restore from as well.
DEFAULT_SR = 22050


def read_metadata(path: Path) -> tuple[list[list[str]], list[str] | None]:
    """coqui-format rows, header separated if present."""
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines(),
                           delimiter="|"))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    header = None
    if rows and rows[0][0].strip().lower() in ("audio_file", "audio"):
        header, rows = rows[0], rows[1:]
    return rows, header


def charset_of(rows: list[list[str]]) -> Counter:
    """Every character the text actually uses, counted."""
    c: Counter = Counter()
    for r in rows:
        if len(r) > 1:
            c.update(r[1])
    return c


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="prepare_voicemakers.py output directory")
    ap.add_argument("--out", help="VITS dataset directory to build")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not (args.dataset and args.out):
        ap.error("--dataset and --out are required (or --selftest)")

    src, out = Path(args.dataset).resolve(), Path(args.out).resolve()
    if not (src / "metadata_train.csv").is_file():
        print(f"no metadata_train.csv under {src} -- run prepare_voicemakers.py first",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    import librosa

    # ------------------------------------------------------------ audio
    print(f"[1/3] resampling audio to {args.sample_rate} Hz")
    wavs = sorted(src.glob("wavs/*/*.wav"))
    if not wavs:
        print(f"no wavs under {src}/wavs", file=sys.stderr)
        return 2
    rates_in: Counter = Counter()
    done = skipped = 0
    for i, w in enumerate(wavs, 1):
        dst = out / "wavs" / w.parent.name / w.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_file():
            skipped += 1
            continue
        info = sf.info(str(w))
        rates_in[info.samplerate] += 1
        if info.samplerate == args.sample_rate:
            shutil.copy2(w, dst)
        else:
            y, _ = librosa.load(str(w), sr=args.sample_rate, mono=True)
            sf.write(str(dst), y, args.sample_rate, subtype="PCM_16")
        done += 1
        if i % 500 == 0 or i == len(wavs):
            print(f"  {i}/{len(wavs)}")
    print(f"  wrote {done}, already present {skipped}")
    print(f"  source rates: {dict(rates_in) or 'all already at target'}")

    # --------------------------------------------------------- metadata
    print("\n[2/3] metadata")
    written: dict[str, dict[str, int]] = defaultdict(dict)
    all_rows: dict[str, list[list[str]]] = {}
    for split in ("train", "eval"):
        rows, header = read_metadata(src / f"metadata_{split}.csv")
        all_rows[split] = rows
        by_spk: dict[str, list[list[str]]] = defaultdict(list)
        for r in rows:
            if len(r) >= 3:
                by_spk[r[2].strip()].append(r)

        # multi-speaker file, verbatim
        with (out / f"metadata_{split}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="|", lineterminator="\n")
            if header:
                w.writerow(header)
            w.writerows(rows)

        # one file per speaker -- what a per-voice fine-tune consumes
        for spk, spk_rows in by_spk.items():
            with (out / f"metadata_{split}_{spk}.csv").open(
                    "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh, delimiter="|", lineterminator="\n")
                if header:
                    w.writerow(header)
                w.writerows(spk_rows)
            written[spk][split] = len(spk_rows)
        print(f"  {split}: {len(rows)} rows, "
              + ", ".join(f"{s}={len(v)}" for s, v in sorted(by_spk.items())))

    # eval_reference.json is what evaluate_xtts.py scores against -- the same
    # 80 held-out clips, so both architectures are judged on identical items.
    ref = src / "eval_reference.json"
    if ref.is_file():
        shutil.copy2(ref, out / "eval_reference.json")
        print("  copied eval_reference.json (the shared held-out set)")

    # ---------------------------------------------------------- charset
    print("\n[3/3] character inventory")
    chars = charset_of(all_rows["train"]) + charset_of(all_rows["eval"])
    inventory = "".join(sorted(chars))
    non_ascii = [c for c in inventory if ord(c) > 127]
    print(f"  {len(chars)} distinct characters: {inventory!r}")
    if non_ascii:
        print(f"  FATAL: non-ASCII characters survived: {non_ascii}", file=sys.stderr)
        print("  The transliteration is supposed to guarantee ASCII. Fix "
              "sinhala_text.py rather than working around it here.", file=sys.stderr)
        return 3
    print("  OK -- pure ASCII, as the transliteration promises")

    report = {
        "source_dataset": str(src),
        "sample_rate": args.sample_rate,
        "clips": len(wavs),
        "per_speaker": {s: dict(v) for s, v in sorted(written.items())},
        "charset": inventory,
        "char_counts": dict(chars.most_common()),
    }
    (out / "vits_prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{out}")
    for spk, v in sorted(written.items()):
        print(f"  {spk:10s} train {v.get('train', 0):5d}   eval {v.get('eval', 0):3d}")
    print("\nTrain one voice with:")
    print(f"  python train_vits_female.py --dataset {out} --speaker "
          f"{sorted(written)[0]} --out ./run")
    return 0


def _selftest() -> int:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="vitsprep_"))
    rows = [["audio_file", "text", "speaker_name"],
            ["wavs/a/1.wav", "ayubovan obata kohomadha?", "a"],
            ["wavs/b/2.wav", "mata puluvan.", "b"],
            ["wavs/a/3.wav", "hondhayi.", "a"]]
    meta = tmp / "metadata_train.csv"
    with meta.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="|", lineterminator="\n").writerows(rows)

    parsed, header = read_metadata(meta)
    assert header == rows[0], header
    assert len(parsed) == 3, parsed
    # A file without a header must not lose its first data row.
    meta2 = tmp / "noheader.csv"
    with meta2.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, delimiter="|", lineterminator="\n").writerows(rows[1:])
    parsed2, header2 = read_metadata(meta2)
    assert header2 is None and len(parsed2) == 3, (header2, parsed2)

    chars = charset_of(parsed)
    assert all(ord(c) < 128 for c in chars), chars
    assert chars["a"] > 0 and "?" in chars, chars
    # The inventory is what the model's embedding must cover, so an unexpected
    # character has to be visible rather than silently dropped later.
    assert set("abcdefghijklmnopqrstuvwxyz .,?!'-:;") >= set(chars), set(chars)

    print("prepare_vits selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
