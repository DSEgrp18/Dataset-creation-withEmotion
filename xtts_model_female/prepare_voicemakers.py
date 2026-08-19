#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_voicemakers.py -- VoiceMakers Sinhala TTS corpus -> XTTS training input.

    kaggle.com/datasets/safnask/sinhalatts-dataset-publication-by-voicemakers
        -> <out>/wavs/<speaker>/*.wav
           <out>/metadata_train.csv   audio_file|text|speaker_name
           <out>/metadata_eval.csv
           <out>/prepare_report.json

Default speakers are the two female voices, Dinithi (4.82 h) and Harini (2.14 h),
for ~7 h across two genuinely distinct speakers. That matters more than the hours:
XTTS samples a conditioning clip from the SAME speaker during training, so a
correctly-labelled two-speaker set teaches the reference-to-voice mapping that a
single pooled label actively destroys.

NOTHING ABOUT THE LAYOUT IS ASSUMED
-----------------------------------
The published folders are inconsistent -- `Isuru-44100Hz` vs `Yasindu-44100`,
with at least one speaker directory nested inside a duplicate of itself. So this
script discovers rather than assumes:

  * speaker directories  : matched by name substring, recursively
  * metadata.csv         : found by rglob under each speaker directory
  * wav files            : found by rglob, indexed by filename stem
  * delimiter            : scored across , | tab ; and chosen by consistency
  * column order         : the Sinhala-script column is found by codepoint range,
                           the romanised column by its diacritics -- not by index
  * header row           : detected and skipped, or absent, either is fine

Every discovery decision is printed. If the dataset layout changes, you see it.

WHY THE ROMANISED COLUMN
------------------------
XTTS-v2's vocab.json is a whitespace-pretokenised BPE with an [UNK] fallback, and
contains no Sinhala codepoint -- nor the diacritics of this corpus's romanisation
(ā ī ū ē ṭ ḍ ṇ ḷ ṁ are all absent). Either column fed raw makes every word a single
[UNK]. sinhala_text.fold() maps the romanisation to plain ASCII that the pretrained
vocabulary already covers; verified here at 0 [UNK] before a GPU is touched.

Both columns fold to identical ASCII on this corpus, so the romanised column is
used and the script column is kept only for the evaluation reports.

FILTERS
    duration   1.0 - 11.6 s   GPTArgs.max_wav_length = 255995 @ 22050 Hz; the
                              dataloader drops longer clips SILENTLY
    digits     rows dropped   a digit in the text means the audio speaks a number
                              the text does not contain; XTTS's "en" cleaner would
                              expand it to ENGLISH words inside a Sinhala sentence
    chars      <= 250         VoiceBpeTokenizer.char_limits["en"]

USAGE
    python prepare_voicemakers.py --src /kaggle/input/sinhalatts-dataset-publication-by-voicemakers --out ./female_dataset
    python prepare_voicemakers.py --src ... --out ... --speakers dinithi harini isuru
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# sinhala_text.py lives in the sibling xtts_sinhala/ package; fall back to a local
# copy so this directory also works when lifted out of the repo on its own.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "xtts_sinhala"):
    if (_cand / "sinhala_text.py").is_file():
        sys.path.insert(0, str(_cand))
        break
try:
    from sinhala_text import ROMAN_TO_ASCII, fold  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit("ERROR: sinhala_text.py not found in this directory or ../xtts_sinhala/")

# fold() finishes by DROPPING every character outside its keep-set, so an
# unmapped diacritic raises nothing -- it silently turns "vaṟdanak" into
# "vadanak", a different word, and poisons training without a warning. The check
# therefore has to run on the RAW romanisation, against what the map covers.
KNOWN_NON_ASCII = {c for k in ROMAN_TO_ASCII for c in k if ord(c) > 127}
KNOWN_NON_ASCII |= set("‘’“”‍ ")


def unmapped_chars(roman: str) -> list[str]:
    """Non-ASCII characters in the source text that ROMAN_TO_ASCII does not cover."""
    return [c for c in roman.lower() if ord(c) > 127 and c not in KNOWN_NON_ASCII]

XTTS_MAX_SECONDS = 11.6      # GPTArgs.max_wav_length / 22050
XTTS_MIN_SECONDS = 1.0
XTTS_MAX_CHARS = 250         # VoiceBpeTokenizer.char_limits["en"]

FEMALE_SPEAKERS = ["dinithi", "harini"]
VOCAB_URL = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"

SINHALA = re.compile(r"[඀-෿]")
DIACRITICS = re.compile(r"[āīūēōæǣñśşḍḥḷṁṅṇṉṛṝṭ]", re.IGNORECASE)
DIGITS = re.compile(r"[0-9෦-෯]")   # ASCII and Sinhala digits


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def find_speaker_dirs(src: Path, wanted: list[str]) -> dict[str, Path]:
    """Match speaker directories by name substring, shallowest match wins."""
    found: dict[str, Path] = {}
    candidates = [p for p in src.rglob("*") if p.is_dir()]
    candidates.sort(key=lambda p: (len(p.relative_to(src).parts), str(p)))
    for name in wanted:
        for d in candidates:
            if name.lower() in d.name.lower() and any(d.rglob("*.wav")):
                found[name] = d
                break
    return found


def sniff_rows(meta: Path) -> list[list[str]]:
    """Parse the metadata file without trusting its delimiter or column order."""
    raw = meta.read_text(encoding="utf-8-sig", errors="replace")
    best, best_score = None, -1.0
    for delim in [",", "|", "\t", ";"]:
        try:
            rows = [r for r in csv.reader(raw.splitlines(), delimiter=delim) if r]
        except csv.Error:
            continue
        if not rows:
            continue
        widths = Counter(len(r) for r in rows)
        width, n = widths.most_common(1)[0]
        if width < 2:
            continue
        # consistency of column count, plus a bonus for finding a Sinhala column
        score = n / len(rows)
        good = [r for r in rows if len(r) == width]
        if any(SINHALA.search(c) for c in good[0]):
            score += 1.0
        if score > best_score:
            best, best_score = good, score
    if best is None:
        raise SystemExit(f"ERROR: could not parse {meta} with any known delimiter")
    return best


def locate_columns(rows: list[list[str]]) -> tuple[int, int, int]:
    """Return (id_col, script_col, roman_col) by inspecting content, not position."""
    n = len(rows[0])
    sinhala_hits = [0] * n
    roman_hits = [0] * n
    for r in rows[: min(200, len(rows))]:
        for i, cell in enumerate(r):
            if SINHALA.search(cell):
                sinhala_hits[i] += 1
            elif DIACRITICS.search(cell):
                roman_hits[i] += 1
    script_col = max(range(n), key=lambda i: sinhala_hits[i])
    roman_col = max(range(n), key=lambda i: roman_hits[i])
    if sinhala_hits[script_col] == 0:
        raise SystemExit("ERROR: no Sinhala-script column found in the metadata")
    if roman_hits[roman_col] == 0 or roman_col == script_col:
        raise SystemExit("ERROR: no romanised column found in the metadata")
    # the id column is whichever remaining column is pure ASCII everywhere
    others = [i for i in range(n) if i not in (script_col, roman_col)]
    id_col = others[0] if others else 0
    return id_col, script_col, roman_col


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dataset root (the Kaggle input dir)")
    ap.add_argument("--out", required=True, help="dataset directory to build")
    ap.add_argument("--speakers", nargs="+", default=FEMALE_SPEAKERS,
                    help=f"speaker name substrings (default: {' '.join(FEMALE_SPEAKERS)})")
    ap.add_argument("--vocab", default=None,
                    help="path to XTTS vocab.json for the [UNK] check (strongly recommended)")
    ap.add_argument("--max-seconds", type=float, default=XTTS_MAX_SECONDS)
    ap.add_argument("--min-seconds", type=float, default=XTTS_MIN_SECONDS)
    ap.add_argument("--eval-per-speaker", type=int, default=40,
                    help="held-out clips per speaker, used by evaluate_xtts.py")
    ap.add_argument("--keep-digits", action="store_true",
                    help="keep rows containing digits (see the docstring -- do not)")
    ap.add_argument("--copy-wavs", action="store_true",
                    help="copy instead of symlink (needed on filesystems without links)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    # ------------------------------------------------------------ discovery
    print("[1/5] discovering speaker directories")
    spk_dirs = find_speaker_dirs(src, args.speakers)
    missing = [s for s in args.speakers if s not in spk_dirs]
    if missing:
        print(f"  ERROR: no directory with wavs matched {missing}", file=sys.stderr)
        print("  directories present under --src:", file=sys.stderr)
        for d in sorted({p.name for p in src.rglob("*") if p.is_dir()})[:40]:
            print(f"    {d}", file=sys.stderr)
        return 2
    for name, d in spk_dirs.items():
        print(f"  {name:10s} -> {d.relative_to(src)}")

    # ------------------------------------------------------------- parsing
    print("\n[2/5] reading metadata")
    drop: Counter = Counter()
    kept: list[tuple[str, str, str, str, float, Path]] = []   # id, ascii, script, spk, dur, wav
    unmapped: Counter = Counter()

    for speaker, sdir in spk_dirs.items():
        metas = sorted(sdir.rglob("metadata.csv")) or sorted(sdir.rglob("*.csv"))
        if not metas:
            print(f"  {speaker}: NO metadata csv found -- skipped", file=sys.stderr)
            continue
        meta = metas[0]
        rows = sniff_rows(meta)
        id_col, script_col, roman_col = locate_columns(rows)
        wavs = {p.stem: p for p in sdir.rglob("*.wav")}
        print(f"  {speaker:10s} {meta.name}: {len(rows)} rows, {len(wavs)} wavs, "
              f"cols id={id_col} script={script_col} roman={roman_col}")

        for r in rows:
            clip_id = r[id_col].strip()
            script = r[script_col].strip()
            roman = r[roman_col].strip()
            if not clip_id or not SINHALA.search(script):
                drop["header_or_malformed"] += 1
                continue
            wav = wavs.get(clip_id)
            if wav is None:
                drop["missing_wav"] += 1
                continue
            if not args.keep_digits and (DIGITS.search(roman) or DIGITS.search(script)):
                drop["contains_digits"] += 1
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
            bad = unmapped_chars(roman)
            if bad:
                unmapped.update(bad)
                drop["unmapped_characters"] += 1
                continue
            text = fold(roman)
            if not text:
                drop["empty_after_fold"] += 1
                continue
            if len(text) > XTTS_MAX_CHARS:
                drop[f"text_over_{XTTS_MAX_CHARS}_chars"] += 1
                continue
            kept.append((clip_id, text, script, speaker, dur, wav))

    if not kept:
        print("\nERROR: nothing survived filtering", file=sys.stderr)
        print(json.dumps(drop, indent=2), file=sys.stderr)
        return 3

    if unmapped:
        print("\n  FATAL: the romanisation uses characters sinhala_text.py does not "
              "map. fold() would DELETE them silently, changing the words:",
              file=sys.stderr)
        for ch, n in unmapped.most_common():
            print(f"    {ch!r} U+{ord(ch):04X} {unicodedata.name(ch, '?')}  x{n}",
                  file=sys.stderr)
        print(f"  {drop['unmapped_characters']} affected rows were dropped.\n"
              "  Add these to ROMAN_TO_ASCII in sinhala_text.py, then re-run.",
              file=sys.stderr)
        return 4

    # ------------------------------------------------- tokenizer sanity check
    print("\n[3/5] tokenizer check")
    if args.vocab and Path(args.vocab).is_file():
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(args.vocab)
        unk = total = 0
        lens = []
        worst: list[str] = []
        for _, text, *_ in kept:
            ids = tk.encode("[en]" + text.replace(" ", "[SPACE]"))
            c = ids.tokens.count("[UNK]")
            if c and len(worst) < 5:
                worst.append(text)
            unk += c
            total += len(ids.ids)
            lens.append(len(ids.ids))
        lens.sort()
        print(f"  {total} tokens, [UNK] = {unk}")
        print(f"  length: median {lens[len(lens)//2]}  p95 {lens[int(.95*len(lens))]}  "
              f"max {lens[-1]}   (GPTArgs.max_text_length = 200)")
        over = sum(1 for x in lens if x > 200)
        if over:
            print(f"  WARNING: {over} clips exceed 200 tokens and will be truncated")
        if unk:
            print(f"  FATAL: {unk} [UNK] tokens. Examples:", file=sys.stderr)
            for w in worst:
                print(f"    {w}", file=sys.stderr)
            return 5
        print("  OK -- every token is a pretrained token")
    else:
        print(f"  SKIPPED. Download the vocab and pass --vocab:\n    {VOCAB_URL}")

    # ---------------------------------------------------------------- write
    print("\n[4/5] linking audio")
    for clip_id, _, _, speaker, _, wav in kept:
        dst_dir = out / "wavs" / speaker
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{clip_id}.wav"
        if dst.exists() or dst.is_symlink():
            continue
        if args.copy_wavs:
            shutil.copy2(wav, dst)
        else:
            try:
                dst.symlink_to(wav)
            except OSError:
                shutil.copy2(wav, dst)

    print("[5/5] writing metadata")
    rng = random.Random(args.seed)
    by_speaker: dict[str, list] = defaultdict(list)
    for row in kept:
        by_speaker[row[3]].append(row)

    eval_rows: list = []
    train_rows: list = []
    for speaker, rows_s in by_speaker.items():
        rng.shuffle(rows_s)
        n_eval = min(args.eval_per_speaker, len(rows_s) // 10)
        eval_rows += rows_s[:n_eval]
        train_rows += rows_s[n_eval:]
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    for name, part in (("metadata_train.csv", train_rows),
                       ("metadata_eval.csv", eval_rows)):
        with (out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="|", lineterminator="\n")
            w.writerow(["audio_file", "text", "speaker_name"])
            for clip_id, text, _script, speaker, _dur, _wav in part:
                w.writerow([f"wavs/{speaker}/{clip_id}.wav", text, speaker])

    # The evaluator needs the Sinhala script and true durations, which the coqui
    # format has nowhere to put.
    with (out / "eval_reference.json").open("w", encoding="utf-8") as fh:
        json.dump([{"clip_id": c, "ascii": t, "sinhala": s, "speaker": sp,
                    "duration": d, "wav": f"wavs/{sp}/{c}.wav"}
                   for c, t, s, sp, d, _ in eval_rows], fh, ensure_ascii=False, indent=1)

    # -------------------------------------------------------------- report
    secs = [d for *_, d, _ in kept]
    per_spk = {s: {"clips": len(v), "hours": round(sum(r[4] for r in v) / 3600, 2)}
               for s, v in by_speaker.items()}
    srates = Counter(sf.info(str(w)).samplerate for *_, w in kept[:200])
    report = {
        "speakers": per_spk,
        "train_clips": len(train_rows),
        "eval_clips": len(eval_rows),
        "total_hours": round(sum(secs) / 3600, 2),
        "duration_median_s": round(statistics.median(secs), 2),
        "duration_max_s": round(max(secs), 2),
        "source_sample_rates": dict(srates),
        "dropped": dict(drop),
    }
    (out / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{out}")
    print(f"  train {len(train_rows)}   eval {len(eval_rows)}   "
          f"total {report['total_hours']} h")
    for s, v in per_spk.items():
        print(f"    {s:10s} {v['clips']:5d} clips   {v['hours']:.2f} h")
    print(f"  duration : median {report['duration_median_s']}s  "
          f"max {report['duration_max_s']}s")
    print(f"  source sample rates: {dict(srates)}  "
          f"(the trainer resamples to 22050 on load)")
    if drop:
        print("  dropped:")
        for reason, n in drop.most_common():
            print(f"    {n:6d}  {reason}")
    print("\n  first lines of metadata_train.csv:")
    for line in (out / "metadata_train.csv").read_text(encoding="utf-8").splitlines()[:4]:
        print("   ", line)

    if len(by_speaker) < 2:
        print("\nNOTE: only one speaker. XTTS conditions on a same-speaker reference,"
              "\n      so multi-speaker data is what makes that mapping learnable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
