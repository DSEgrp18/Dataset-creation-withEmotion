#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_listening.py -- turn returned rater CSVs into MOS and SUS numbers.

    python score_listening.py --key ./listening_test/answer_key.json \
                              --responses ./returned/*.csv

MOS   Mean of the 5-point ratings, reported per scale (intelligibility,
      naturalness), with the standard deviation and the per-rater spread. Also
      reported as a percentage (mean / 5 x 100) because SPECOM 2025 publishes it
      that way -- both forms are printed so you can compare either direction.

SUS   Word accuracy from a word-level alignment against the answer key:
      (N - substitutions - deletions - insertions) / N, floored at zero, which is
      the usual "percentage of words correctly identified".

GROUPS
      Blind and sighted raters are scored separately as well as together.
      Nanayakkara et al. (2018) found visually impaired listeners scored the same
      system 66% where sighted listeners scored ~70%, and argued the sighted
      group was simply less practised at listening to synthetic speech. If your
      panel is mixed, reporting only the pooled number hides that.

INTER-RATER AGREEMENT
      Krippendorff-style agreement is overkill for a 12-person panel, so this
      reports the per-item standard deviation across raters instead. An item
      where raters disagree wildly is usually a broken clip, not a hard call --
      check the high-sigma items before trusting the mean.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def word_accuracy(ref: str, hyp: str) -> float:
    """1 - WER, floored at 0, on whitespace-split tokens."""
    a, b = ref.split(), hyp.split()
    if not a:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(a))


def block(vals: list[float], scale5: bool = False) -> str:
    if not vals:
        return "no data"
    m, sd = st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)
    if scale5:
        return f"{m:.2f} / 5   ({100*m/5:.2f}%)   sd {sd:.2f}   n={len(vals)}"
    return f"{100*m:.2f}%   sd {100*sd:.2f}   n={len(vals)}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", required=True, help="answer_key.json from listening_test.py")
    ap.add_argument("--responses", nargs="+", required=True,
                    help="rater CSVs; globs are expanded")
    ap.add_argument("--out", default="./listening_results.json")
    args = ap.parse_args()

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    sus_key = {str(k["item"]): k for k in key if k["part"] == "SUS"}
    mos_key = {str(k["item"]): k for k in key if k["part"] == "MOS"}

    paths: list[str] = []
    for pat in args.responses:
        paths += glob.glob(pat) or ([pat] if Path(pat).is_file() else [])
    if not paths:
        print("no response files matched")
        return 2

    mos_int: dict[str, list] = defaultdict(list)   # by group
    mos_nat: dict[str, list] = defaultdict(list)
    sus_acc: dict[str, list] = defaultdict(list)
    per_item_int: dict[str, list] = defaultdict(list)
    per_item_sus: dict[str, list] = defaultdict(list)
    raters: set[str] = set()
    groups: dict[str, str] = {}

    for p in sorted(paths):
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rater = (row.get("rater") or "").strip()
                if not rater:
                    continue
                grp = (row.get("group") or "sighted").strip()
                raters.add(rater)
                groups[rater] = grp
                item = (row.get("item") or "").strip()
                if row.get("part") == "MOS":
                    for field, bucket, per_item in (
                            ("intelligibility", mos_int, per_item_int),
                            ("naturalness", mos_nat, None)):
                        v = (row.get(field) or "").strip()
                        if v.isdigit():
                            bucket[grp].append(int(v))
                            bucket["all"].append(int(v))
                            if per_item is not None:
                                per_item[item].append(int(v))
                elif row.get("part") == "SUS":
                    ref = sus_key.get(item)
                    hyp = (row.get("transcript") or "").strip()
                    if ref and hyp:
                        a = word_accuracy(ref["sinhala"], hyp)
                        sus_acc[grp].append(a)
                        sus_acc["all"].append(a)
                        per_item_sus[item].append(a)

    print(f"{len(raters)} raters from {len(paths)} files")
    by_group = defaultdict(int)
    for g in groups.values():
        by_group[g] += 1
    print("  " + ", ".join(f"{g}: {n}" for g, n in by_group.items()))

    print("\n--- MOS -------------------------------------------------")
    for g in ["all"] + sorted(x for x in mos_int if x != "all"):
        print(f"  intelligibility [{g:18s}] {block(mos_int[g], scale5=True)}")
    print()
    for g in ["all"] + sorted(x for x in mos_nat if x != "all"):
        print(f"  naturalness     [{g:18s}] {block(mos_nat[g], scale5=True)}")

    print("\n--- SUS word accuracy -----------------------------------")
    for g in ["all"] + sorted(x for x in sus_acc if x != "all"):
        print(f"  [{g:18s}] {block(sus_acc[g])}")

    flagged = [(i, st.pstdev(v)) for i, v in per_item_int.items() if len(v) > 2]
    flagged.sort(key=lambda x: -x[1])
    if flagged:
        print("\n--- items raters disagreed on most (check these clips) ---")
        for item, sd in flagged[:5]:
            k = mos_key.get(item, {})
            print(f"  item {item:>3s}  sd {sd:.2f}   {k.get('sinhala','')[:52]}")

    worst = sorted(((i, st.mean(v)) for i, v in per_item_sus.items() if v),
                   key=lambda x: x[1])
    if worst:
        print("\n--- least intelligible SUS sentences ---")
        for item, acc in worst[:5]:
            k = sus_key.get(item, {})
            print(f"  item {item:>3s}  {100*acc:5.1f}%   {k.get('sinhala','')[:52]}")

    result = {
        "raters": len(raters),
        "groups": dict(by_group),
        "mos_intelligibility": {g: {"mean_5": round(st.mean(v), 3),
                                    "percent": round(100 * st.mean(v) / 5, 2),
                                    "n": len(v)}
                                for g, v in mos_int.items() if v},
        "mos_naturalness": {g: {"mean_5": round(st.mean(v), 3),
                                "percent": round(100 * st.mean(v) / 5, 2),
                                "n": len(v)}
                            for g, v in mos_nat.items() if v},
        "sus_word_accuracy": {g: {"percent": round(100 * st.mean(v), 2), "n": len(v)}
                              for g, v in sus_acc.items() if v},
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    if len(raters) < 12:
        print(f"NOTE: {len(raters)} raters. SPECOM 2025 used 12; below that the "
              "confidence interval is wide enough to swallow most differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
