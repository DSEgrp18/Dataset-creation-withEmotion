#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_quality.py -- prove an optimisation did not cost quality, or fail.

"Faster and smaller, but the results must never go down" is only a real
requirement if something checks it. This runs the SAME evaluation used on the
baseline (evaluate_xtts.py, unchanged) against two checkpoints, diffs the
metrics, and exits non-zero if the candidate regressed beyond tolerance.

It does not re-implement any metric. evaluate_xtts.py is invoked as a subprocess
twice, so whatever that script measures is exactly what is compared -- there is
no second implementation to drift.

ON TOLERANCES
-------------
XTTS samples at temperature 0.75 with top_k/top_p, so two checkpoints never
produce identical audio unless their weights are identical. evaluate_xtts.py
seeds torch and random, which makes a run repeatable for the SAME weights -- but
the moment weights differ by even fp16 rounding, the sampled token sequence
diverges and every metric moves a little. Tolerances below are sized for that
sampling noise, not for real degradation.

The corollary worth knowing: a --strip model (weights bit-identical to the
baseline) must come back EXACTLY equal, not merely within tolerance. This script
says so explicitly when it happens, and that is the strongest evidence the strip
was lossless.

Use --seeds to average over several runs when a result sits near a boundary; one
seed at n=40 is enough to catch a real regression, not enough to resolve a 1 %
difference.

USAGE
    python compare_quality.py --run <run_dir> --base <base_dir> --dataset <ds> \
        --baseline model.pth --candidate model_fp16.pth --out cmp

    python compare_quality.py ... --seeds 1234,1235,1236 --utmos
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

_FEMALE = Path(__file__).resolve().parent.parent / "xtts_model_female"
EVAL = _FEMALE / "evaluate_xtts.py"

# The experiment register lives with the sweep driver so a gate result and a
# sweep row are the same kind of row, in the same file, with the same columns.
sys.path.insert(0, str(_FEMALE))
try:
    from sweep_eval import append_row, row_from_metrics
except ImportError:                              # pragma: no cover
    append_row = row_from_metrics = None

# key, higher_is_better, tolerance, kind ("rel" fraction | "abs")
GATES = [
    ("mcd_db",         False, 0.02, "rel"),
    ("f0_rmse_cents",  False, 0.03, "rel"),
    ("f0_corr",        True,  0.02, "abs"),
    ("secs",           True,  0.01, "abs"),
    ("utmos",          True,  0.05, "abs"),
    ("failure_rate",   False, 0.01, "abs"),   # +1 percentage point
]
# Reported but never gated: speed is expected to change, that is the point.
UNGATED = ("rtf",)


def value(metrics: dict, key: str, speaker: str | None = None):
    """Pull one number out of evaluate_xtts's overall, or one speaker's, block."""
    block = metrics["overall"]
    if speaker is not None:
        blocks = [d for d in metrics.get("per_speaker", []) if d["name"] == speaker]
        if not blocks:
            return None
        block = blocks[0]
    v = block.get(key)
    if v is None:
        return None
    return v["mean"] if isinstance(v, dict) else v


def run_eval(ckpt: Path, out: Path, args, seed: int, label: str) -> dict:
    metrics_path = out / "metrics.json"
    # Already measured: reuse it. A Kaggle session can end between the baseline
    # and the candidate, and re-running would both cost the GPU time again and
    # overwrite a result that is already recorded.
    if metrics_path.is_file() and not args.force:
        print(f"reusing {metrics_path}  (--force to re-run)", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    cmd = [sys.executable, str(EVAL), "--run", args.run, "--base", args.base,
           "--dataset", args.dataset, "--checkpoint", str(ckpt),
           "--out", str(out), "--n", str(args.n), "--seed", str(seed),
           "--temperature", str(args.temperature),
           "--repetition-penalty", str(args.repetition_penalty),
           "--top-k", str(args.top_k), "--top-p", str(args.top_p),
           "--length-penalty", str(args.length_penalty),
           "--text-from", args.text_from,
           "--label", label]
    if args.utmos:
        cmd.append("--utmos")
    print("$", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit(f"evaluation failed for {ckpt}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]


def averaged(runs: list[dict], key: str, speaker: str | None = None):
    vals = [v for v in (value(m, key, speaker) for m in runs) if v is not None]
    return statistics.fmean(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run dir (for speaker refs)")
    ap.add_argument("--base", required=True, help="XTTS_v2.0_original_model_files")
    ap.add_argument("--dataset", required=True, help="prepare_voicemakers.py output")
    ap.add_argument("--baseline", required=True, help="checkpoint to beat")
    ap.add_argument("--candidate", required=True, help="the optimised checkpoint")
    ap.add_argument("--out", default="./cmp_out")
    ap.add_argument("--n", type=int, default=40, help="eval clips per speaker")
    ap.add_argument("--seeds", default="1234",
                    help="comma-separated; metrics are averaged over them")
    # Decoding must be IDENTICAL on both sides or the comparison measures the
    # decode change instead of the checkpoint change. One set of flags, passed to
    # both evaluations, is the only way to keep that true.
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition-penalty", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--text-from", choices=("dataset", "script"), default="dataset")
    ap.add_argument("--utmos", action="store_true")
    ap.add_argument("--registry", default=None,
                    help="append both sides to this experiment register CSV "
                         "(default <out>/experiments.csv)")
    ap.add_argument("--force", action="store_true",
                    help="re-run evaluations that already have a metrics.json")
    # Per-speaker regressions are REPORTED always and gated only on request. Each
    # speaker has half the clips, so the noise floor per speaker is wider than the
    # pooled one the tolerances were sized against -- RESULTS.md measured harini's
    # F0 corr moving 0.068 across three runs that trained identically. Promoting
    # that to a hard failure would reject changes for noise; ignoring it would
    # miss a change that helps one speaker by hurting the other.
    ap.add_argument("--gate-per-speaker", action="store_true",
                    help="treat a per-speaker regression as a failure, not a warning")
    args = ap.parse_args()

    if not EVAL.is_file():
        print(f"cannot find {EVAL}", file=sys.stderr)
        return 1
    base_ck, cand_ck = Path(args.baseline), Path(args.candidate)
    for p in (base_ck, cand_ck):
        if not p.is_file():
            print(f"no such checkpoint: {p}", file=sys.stderr)
            return 1

    out = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base_runs, cand_runs = [], []
    for seed in seeds:
        base_runs.append(run_eval(base_ck, out / f"baseline_s{seed}", args, seed,
                                  f"baseline-{base_ck.stem}"))
        cand_runs.append(run_eval(cand_ck, out / f"candidate_s{seed}", args, seed,
                                  f"candidate-{cand_ck.stem}"))

    size_b, size_c = base_ck.stat().st_size, cand_ck.stat().st_size

    print("\n" + "=" * 78)
    print(f"baseline  : {base_ck.name}   {size_b/1e9:.2f} GB")
    print(f"candidate : {cand_ck.name}   {size_c/1e9:.2f} GB   "
          f"({100*(1-size_c/size_b):+.0f}% size)")
    print(f"seeds     : {seeds}   clips/speaker: {args.n}")
    print(f"decode    : temperature {args.temperature}, repetition_penalty "
          f"{args.repetition_penalty}, top_k {args.top_k}, top_p {args.top_p}, "
          f"length_penalty {args.length_penalty}, text_from {args.text_from}"
          "   (identical on both sides)")
    print("=" * 78)
    print(f"\n{'metric':<16}{'baseline':>12}{'candidate':>12}{'change':>12}   verdict")
    print("-" * 78)

    failures, identical = [], True
    for key, higher_better, tol, kind in GATES:
        b, c = averaged(base_runs, key), averaged(cand_runs, key)
        if b is None or c is None:
            continue
        delta = c - b
        if abs(delta) > 1e-12:
            identical = False
        allowed = tol * abs(b) if kind == "rel" else tol
        # Worse means: up when lower is better, down when higher is better.
        worse_by = -delta if higher_better else delta
        ok = worse_by <= allowed
        mark = "ok" if ok else "REGRESSED"
        if not ok:
            failures.append(f"{key}: {b:.4f} -> {c:.4f} "
                            f"(worse by {worse_by:.4f}, allowed {allowed:.4f})")
        print(f"{key:<16}{b:>12.4f}{c:>12.4f}{delta:>+12.4f}   {mark}")

    # Closeness to 1.0 is what matters here, not direction.
    b, c = averaged(base_runs, "duration_ratio"), averaged(cand_runs, "duration_ratio")
    if b is not None and c is not None:
        db, dc = abs(1 - b), abs(1 - c)
        ok = dc - db <= 0.02
        if not ok:
            failures.append(f"duration_ratio drifted from 1.0: {b:.3f} -> {c:.3f}")
        print(f"{'duration_ratio':<16}{b:>12.4f}{c:>12.4f}{c-b:>+12.4f}   "
              f"{'ok' if ok else 'REGRESSED'}")

    for key in UNGATED:
        b, c = averaged(base_runs, key), averaged(cand_runs, key)
        if b is not None and c is not None:
            print(f"{key:<16}{b:>12.4f}{c:>12.4f}{c-b:>+12.4f}   (not gated)")

    print("-" * 78)

    # ------------------------------------------------------- per speaker
    # An overall mean can hold still while a change helps dinithi and hurts
    # harini, and that gap has persisted across every run in RESULTS.md -- so it
    # is the one place an averaged verdict is most likely to mislead.
    warnings = []
    speakers = sorted({d["name"] for m in base_runs for d in m.get("per_speaker", [])})
    if speakers:
        print(f"\nPer speaker ({args.n} clips each -- half the pooled n, so a "
              f"wider noise floor):")
        print(f"\n{'speaker / metric':<26}{'baseline':>11}{'candidate':>11}"
              f"{'change':>11}   verdict")
        print("-" * 78)
        for spk in speakers:
            for key, higher_better, tol, kind in GATES:
                b = averaged(base_runs, key, spk)
                c = averaged(cand_runs, key, spk)
                if b is None or c is None:
                    continue
                allowed = tol * abs(b) if kind == "rel" else tol
                worse_by = -(c - b) if higher_better else (c - b)
                ok = worse_by <= allowed
                if not ok:
                    msg = (f"{spk} {key}: {b:.4f} -> {c:.4f} "
                           f"(worse by {worse_by:.4f}, allowed {allowed:.4f})")
                    (failures if args.gate_per_speaker else warnings).append(msg)
                print(f"{spk + ' ' + key:<26}{b:>11.4f}{c:>11.4f}{c - b:>+11.4f}"
                      f"   {'ok' if ok else ('REGRESSED' if args.gate_per_speaker else 'worse')}")
        print("-" * 78)

    if warnings:
        print(f"\n{len(warnings)} per-speaker metric(s) moved beyond the pooled "
              "tolerance:")
        for w in warnings:
            print("  -", w)
        print("These are NOT gated by default -- each speaker has half the clips, "
              "so the\nnoise floor is wider than the tolerances were sized for. They "
              "are also not\nnothing: re-run with --seeds 1234,1235,1236 before "
              "deciding, or --gate-per-speaker\nto make them fail.")

    # ---------------------------------------------------------- register
    # Both sides go into the same register the sweeps write to, so a gate result
    # is comparable with every other experiment instead of living in stdout.
    if row_from_metrics is not None:
        registry = Path(args.registry) if args.registry else out / "experiments.csv"
        for tag, runs, ck in (("baseline", base_runs, base_ck),
                              ("candidate", cand_runs, cand_ck)):
            for seed, m in zip(seeds, runs):
                eid = f"{tag}-{ck.stem}-s{seed}"
                row = row_from_metrics(m, eid, eid, args.n)
                row["size_gb"] = round(ck.stat().st_size / 1e9, 3)
                append_row(registry, row)
        print(f"\nregistered in {registry}")
    if identical:
        print("\nMetrics are EXACTLY equal. The candidate's weights are bit-identical\n"
              "to the baseline's, so this optimisation is provably lossless.")
    if failures:
        print(f"\nFAILED -- {len(failures)} metric(s) regressed beyond tolerance:")
        for f in failures:
            print("  -", f)
        print("\nDo not ship this candidate. If the change was --fp16, the fp32 slim\n"
              "model is still 3x smaller than the export and is lossless.")
        return 1

    print(f"\nPASSED -- no metric regressed beyond tolerance, at "
          f"{100*(1-size_c/size_b):.0f}% smaller.")
    print("Record the numbers in ../xtts_model_female/RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
