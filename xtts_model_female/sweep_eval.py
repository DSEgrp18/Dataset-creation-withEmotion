#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_eval.py -- run evaluate_xtts.py over several checkpoints and decode
configurations, and keep one register of what every experiment measured.

WHY THIS EXISTS
---------------
RESULTS.md run 4 is the argument for it. That run trained 3.8x longer than run 3,
reached the best held-out loss of any run, and came back WORSE on the two metrics
that track autoregressive stability: failure rate 0 -> 3.8 % and duration ratio
0.982 -> 1.026. Two separate things follow, and neither is about weights:

  * "best eval loss" is not known to be the right export criterion for this
    model, so the checkpoints have to be scored by synthesis quality instead of
    ranked by the loss the trainer happened to record;
  * truncation, looping and over-generation are DECODING behaviour, which means
    temperature / repetition_penalty / top_k / top_p can move them on the
    existing weights for no GPU hours.

Both are sweeps, both are measured with evaluate_xtts.py, and a sweep whose
results are not written down in a comparable form is a sweep that has to be run
again. This script is the driver and the register; it does not compute a single
metric of its own.

WHAT IT DOES NOT DO
-------------------
It does not declare a winner. `--rank-by` orders the table, and the default
ordering is argued for below, but picking a checkpoint to ship is a judgement
over the whole row including per-speaker columns -- see `compare_quality.py` for
the part that can fail a candidate automatically.

THE DEFAULT ORDERING, AND WHY
-----------------------------
    1. failure_rate          ascending
    2. |duration_ratio - 1|  ascending
    3. mcd_db_ok             ascending

Failures first because a truncated or looping clip is not a slightly worse clip,
it is an unusable one, and because run-to-run noise on failure rate (0 -> 1.2 ->
0 -> 3.8 %) is smaller than the movement worth acting on. Duration ratio second
because over-generation is the same defect measured before it becomes a failure.
MCD last, and the `_ok` variant of it: the pooled mean includes failed clips,
which score near the "unrelated speech" band of the calibration table in
RESULTS.md, so three blown-up clips in 80 move it further than a real change in
the model does.

RESUMING
--------
An experiment whose metrics.json already exists is SKIPPED, not re-run. Kaggle
sessions end at 12 h and a 40-clip evaluation is not free, so a sweep is expected
to span sessions; and re-running would overwrite a result that is already
recorded. Pass --force to re-run deliberately.

USAGE
    # step 2 -- which checkpoint actually synthesises best
    python sweep_eval.py --run <run> --base <base> --dataset <ds> \
        --all-checkpoints --out ./sweeps/checkpoints

    # step 3 -- decode parameters on the checkpoint chosen above
    python sweep_eval.py --run <run> --base <base> --dataset <ds> \
        --checkpoints <run>/best_model.pth \
        --temperature 0.75,0.65,0.6 --repetition-penalty 5.0,7.5,10.0 \
        --out ./sweeps/decode

    # step 4 -- one text path for both speakers
    python sweep_eval.py ... --text-from dataset,script --out ./sweeps/textpath

    python sweep_eval.py ... --dry-run          # print the plan, run nothing
    python sweep_eval.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluate_xtts.py"

# Written for every experiment, in this order. Per-speaker columns are appended
# after these and vary with the dataset, so the writer unions fieldnames rather
# than assuming a fixed header -- see append_row.
CORE_FIELDS = [
    "exp_id", "label", "checkpoint", "n", "seed",
    "temperature", "repetition_penalty", "top_k", "top_p", "length_penalty",
    "text_from",
    "clips", "clips_ok", "failure_rate", "duration_ratio",
    "mcd_db", "mcd_db_ok", "f0_rmse_cents", "f0_rmse_cents_ok",
    "f0_corr", "f0_corr_ok", "secs", "secs_ok",
    "utmos", "utmos_real", "rtf",
]

# The table a human reads. (column, header, decimals)
SHOW = [
    ("failure_rate", "fail%", 1),
    ("duration_ratio", "dur", 3),
    ("mcd_db_ok", "MCD(ok)", 2),
    ("f0_corr_ok", "F0corr", 3),
    ("secs_ok", "SECS", 3),
    ("utmos", "UTMOS", 2),
    ("rtf", "RTF", 3),
]


def mean_of(block: dict, key: str):
    """evaluate_xtts writes either a scalar or an agg dict; take the number."""
    v = block.get(key)
    if v is None:
        return None
    return v.get("mean") if isinstance(v, dict) else v


def row_from_metrics(metrics: dict, exp_id: str, label: str, n: int) -> dict:
    """One register row out of an evaluate_xtts.py metrics.json 'metrics' block.

    Kept separate from the running of experiments so compare_quality.py records
    its runs into the same register with the same columns, and so the selftest
    can exercise it without a GPU.
    """
    overall = metrics["overall"]
    decode = metrics.get("decode", {})
    row = {
        "exp_id": exp_id,
        "label": label,
        "checkpoint": Path(metrics.get("checkpoint", "")).name,
        "n": n,
        "seed": metrics.get("seed"),
        "temperature": decode.get("temperature", metrics.get("temperature")),
        "repetition_penalty": decode.get("repetition_penalty"),
        "top_k": decode.get("top_k"),
        "top_p": decode.get("top_p"),
        "length_penalty": decode.get("length_penalty"),
        "text_from": decode.get("text_from"),
        "clips": overall.get("clips"),
        "clips_ok": overall.get("clips_ok"),
        "failure_rate": overall.get("failure_rate"),
    }
    for key in ("duration_ratio", "mcd_db", "mcd_db_ok", "f0_rmse_cents",
                "f0_rmse_cents_ok", "f0_corr", "f0_corr_ok", "secs", "secs_ok",
                "utmos", "utmos_real", "rtf"):
        row[key] = mean_of(overall, key)
    # Per-speaker, because the dinithi/harini gap has persisted across every run
    # and an overall mean hides a change that only helps one of them.
    for spk in metrics.get("per_speaker", []):
        p = spk["name"]
        row[f"{p}_failure_rate"] = spk.get("failure_rate")
        for key in ("duration_ratio", "mcd_db_ok", "f0_corr_ok", "secs_ok", "utmos"):
            row[f"{p}_{key}"] = mean_of(spk, key)
    return row


def append_row(path: Path, row: dict, key: str | None = "exp_id") -> None:
    """Append one row, widening the header if this row carries new columns.

    Per-speaker column names depend on the dataset, so a fixed header would
    either drop them or have to be guessed. The register is a few dozen rows of
    text, so rewriting it is cheaper than being clever.

    `key` names the column that identifies an experiment; a row replaces any
    existing row with the same value, so re-running one does not leave two
    contradictory rows behind. key=None appends unconditionally, which is what a
    table of repeated timings wants -- there the spread between repeats IS the
    result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, fields = [], list(CORE_FIELDS)
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fields = list(reader.fieldnames or fields)
    for col in row:
        if col not in fields:
            fields.append(col)
    if key is not None and key in row:
        rows = [r for r in rows if r.get(key) != str(row[key])
                and r.get(key) != row[key]]        # re-runs replace
    rows.append({k: row.get(k) for k in fields})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def rank_key(row: dict, rank_by: str | None):
    """Ordering for the summary table. See THE DEFAULT ORDERING in the docstring."""
    def num(key, default=float("inf")):
        v = row.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    if rank_by:
        return (num(rank_by),)
    return (num("failure_rate"), abs(num("duration_ratio", 99.0) - 1.0), num("mcd_db_ok"))


def exp_id_for(ckpt: Path, cfg: dict) -> str:
    """A short, stable, filesystem-safe name for one experiment."""
    parts = [ckpt.stem,
             f"t{cfg['temperature']:g}", f"rp{cfg['repetition_penalty']:g}",
             f"k{cfg['top_k']}", f"p{cfg['top_p']:g}", f"lp{cfg['length_penalty']:g}"]
    if cfg["text_from"] != "dataset":
        parts.append(cfg["text_from"])
    # Only non-default seeds are named, so ids already in a register stay valid
    # and a re-run at seed 1234 still resolves to the same experiment.
    if cfg.get("seed", 1234) != 1234:
        parts.append(f"s{cfg['seed']}")
    return "__".join(parts)


def find_checkpoints(run: Path) -> list[Path]:
    """Every checkpoint in a run directory, oldest step first.

    best_model.pth is a byte copy of the best best_model_<step>.pth, so it is
    only included when no stepped best exists -- scoring the same weights twice
    costs a full evaluation and tells you the noise floor, not a checkpoint.
    """
    stepped = sorted(run.glob("best_model_*.pth"),
                     key=lambda p: int(p.stem.split("_")[-1]))
    periodic = sorted(run.glob("checkpoint_*.pth"),
                      key=lambda p: int(p.stem.split("_")[-1]))
    found = stepped + periodic
    if not stepped and (run / "best_model.pth").is_file():
        found.insert(0, run / "best_model.pth")
    return found


def parse_list(text: str, cast):
    return [cast(x) for x in str(text).split(",") if str(x).strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="GPT_XTTS_si_female-<stamp> directory")
    ap.add_argument("--base", help="XTTS_v2.0_original_model_files")
    ap.add_argument("--dataset", help="output of prepare_voicemakers.py")
    ap.add_argument("--out", default="./sweep_out", help="one subdir per experiment")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit .pth paths (default: the run's best_model.pth)")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="every checkpoint in --run -- this is step 2")
    ap.add_argument("--n", type=int, default=40, help="eval clips per speaker")
    ap.add_argument("--seed", type=int, default=1234)
    # Two results inside the noise floor cannot be separated by measuring them
    # more precisely -- they differ because XTTS samples. Resolving one costs
    # SEEDS, not a bigger --n: the same config at three seeds says whether a
    # difference survives resampling. Every config is run at every seed.
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds; multiplies the grid")
    ap.add_argument("--temperature", default="0.75")
    ap.add_argument("--repetition-penalty", default="5.0")
    ap.add_argument("--top-k", default="50")
    ap.add_argument("--top-p", default="0.85")
    ap.add_argument("--length-penalty", default="1.0")
    ap.add_argument("--text-from", default="dataset")
    ap.add_argument("--utmos", action="store_true")
    ap.add_argument("--registry", default=None,
                    help="register CSV (default <out>/experiments.csv)")
    # A 40-clip evaluation is minutes of GPU, and the product of five lists grows
    # fast enough to swallow a whole session by accident. Refuse rather than
    # discover it at hour six.
    ap.add_argument("--max-experiments", type=int, default=12,
                    help="refuse a grid larger than this (0 disables)")
    ap.add_argument("--force", action="store_true",
                    help="re-run experiments that already have metrics.json")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--rank-by", default=None,
                    help="order the table by this column, ascending, instead of "
                         "the default failure_rate / duration / MCD ordering")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    for name in ("run", "base", "dataset"):
        if not getattr(args, name):
            ap.error(f"--{name} is required (or use --selftest)")

    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    registry = Path(args.registry) if args.registry else out / "experiments.csv"

    if args.all_checkpoints:
        ckpts = find_checkpoints(run)
        if not ckpts:
            print(f"no checkpoints under {run}", file=sys.stderr)
            return 2
    elif args.checkpoints:
        ckpts = [Path(c).resolve() for c in args.checkpoints]
    else:
        ckpts = [run / "best_model.pth"]
    missing = [c for c in ckpts if not c.is_file()]
    if missing:
        for c in missing:
            print(f"no such checkpoint: {c}", file=sys.stderr)
        return 2

    seeds = parse_list(args.seeds, int) if args.seeds else [args.seed]
    grid = [
        dict(zip(("temperature", "repetition_penalty", "top_k", "top_p",
                  "length_penalty", "text_from", "seed"), combo))
        for combo in itertools.product(
            parse_list(args.temperature, float),
            parse_list(args.repetition_penalty, float),
            parse_list(args.top_k, int),
            parse_list(args.top_p, float),
            parse_list(args.length_penalty, float),
            parse_list(args.text_from, str),
            seeds)
    ]
    plan = [(c, cfg) for c in ckpts for cfg in grid]

    print(f"{len(ckpts)} checkpoint(s) x {len(grid)} config(s) "
          f"(seeds {seeds}) = {len(plan)} experiment(s), "
          f"{args.n} clips per speaker each")
    for ckpt, cfg in plan:
        eid = exp_id_for(ckpt, cfg)
        done = (out / eid / "metrics.json").is_file()
        print(f"  {eid}" + ("   [done, will skip]" if done and not args.force else ""))
    if args.max_experiments and len(plan) > args.max_experiments:
        print(f"\nrefusing {len(plan)} experiments: over --max-experiments "
              f"{args.max_experiments}. Each one is a full {args.n}-clip evaluation, "
              "so narrow the grid (sweep one parameter at a time against a fixed\n"
              "baseline) or raise the limit deliberately.", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n--dry-run: nothing executed")
        return 0

    rows, failed = [], []
    for i, (ckpt, cfg) in enumerate(plan, 1):
        eid = exp_id_for(ckpt, cfg)
        exp_out = out / eid
        metrics_path = exp_out / "metrics.json"
        print(f"\n{'=' * 78}\n[{i}/{len(plan)}] {eid}\n{'=' * 78}", flush=True)

        if metrics_path.is_file() and not args.force:
            print("already measured -- skipping. Pass --force to re-run.")
        else:
            cmd = [sys.executable, str(EVAL),
                   "--run", str(run), "--base", args.base, "--dataset", args.dataset,
                   "--checkpoint", str(ckpt), "--out", str(exp_out),
                   "--n", str(args.n), "--seed", str(cfg["seed"]), "--label", eid,
                   "--temperature", str(cfg["temperature"]),
                   "--repetition-penalty", str(cfg["repetition_penalty"]),
                   "--top-k", str(cfg["top_k"]), "--top-p", str(cfg["top_p"]),
                   "--length-penalty", str(cfg["length_penalty"]),
                   "--text-from", cfg["text_from"]]
            if args.utmos:
                cmd.append("--utmos")
            print("$ " + " ".join(cmd), flush=True)
            # One failed experiment must not lose the ones already measured: the
            # register is written after every experiment, and a sweep reports what
            # it managed rather than raising out of the loop.
            if subprocess.run(cmd).returncode != 0:
                print(f"FAILED: {eid}", file=sys.stderr)
                failed.append(eid)
                continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
        row = row_from_metrics(metrics, eid, eid, args.n)
        append_row(registry, row)
        rows.append(row)

    if not rows:
        print("\nnothing measured", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: rank_key(r, args.rank_by))
    table = render(rows, args.rank_by)
    (out / "summary.md").write_text(table, encoding="utf-8")
    print("\n" + table)
    print(f"\nregister : {registry}")
    print(f"summary  : {out / 'summary.md'}")
    if failed:
        print(f"\n{len(failed)} experiment(s) FAILED and are not in the table: "
              + ", ".join(failed), file=sys.stderr)
        return 1
    print("\nThis table orders experiments; it does not choose one. Read the "
          "per-speaker\ncolumns in the register too, then gate the choice with "
          "compare_quality.py.")
    return 0


def render(rows: list[dict], rank_by: str | None) -> str:
    """The summary table, as markdown."""
    def fmt(row, key, nd):
        v = row.get(key)
        if v in (None, ""):
            return "n/a"
        v = float(v)
        # Every failure_rate column is a fraction, including the per-speaker ones,
        # and all of them are read as a percentage.
        return f"{100 * v:.1f}" if key.endswith("failure_rate") else f"{v:.{nd}f}"

    head = ["experiment"] + [h for _, h, _ in SHOW]
    lines = [f"# sweep_eval -- {len(rows)} experiment(s)",
             "",
             f"ordered by {rank_by or 'failure_rate, |duration_ratio - 1|, mcd_db_ok'} "
             "(best first). Ordering is not a verdict -- see the docstring.",
             "",
             "| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            [str(r["exp_id"])] + [fmt(r, k, nd) for k, _, nd in SHOW]) + " |")
    spk_cols = sorted({k for r in rows for k in r
                       if k.endswith("_failure_rate") and k != "failure_rate"})
    if spk_cols:
        speakers = [c[: -len("_failure_rate")] for c in spk_cols]
        lines += ["", "Per speaker -- fail % / MCD(ok) / F0 corr:", "",
                  "| experiment | " + " | ".join(speakers) + " |",
                  "|" + "|".join(["---"] * (len(speakers) + 1)) + "|"]
        for r in rows:
            cells = []
            for s in speakers:
                cells.append(f"{fmt(r, s + '_failure_rate', 1)} / "
                             f"{fmt(r, s + '_mcd_db_ok', 2)} / "
                             f"{fmt(r, s + '_f0_corr_ok', 3)}")
            lines.append(f"| {r['exp_id']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def _selftest() -> int:
    """Exercise everything that does not need a GPU: rows, register, ranking."""
    import tempfile

    metrics = {
        "checkpoint": "/x/best_model_15800.pth", "seed": 1234,
        "decode": {"temperature": 0.65, "repetition_penalty": 7.5, "top_k": 50,
                   "top_p": 0.85, "length_penalty": 1.0, "text_from": "script"},
        "overall": {"name": "o", "clips": 80, "clips_ok": 77, "failure_rate": 0.0375,
                    "mcd_db": {"mean": 63.67}, "mcd_db_ok": {"mean": 62.9},
                    "duration_ratio": {"mean": 1.026}, "f0_corr_ok": {"mean": 0.41},
                    "secs_ok": {"mean": 0.704}, "utmos": {"mean": 2.68},
                    "rtf": {"mean": 0.495}},
        "per_speaker": [
            {"name": "dinithi", "failure_rate": 0.05, "mcd_db_ok": {"mean": 61.9},
             "f0_corr_ok": {"mean": 0.528}, "duration_ratio": {"mean": 1.057}},
            {"name": "harini", "failure_rate": 0.025, "mcd_db_ok": {"mean": 65.5},
             "f0_corr_ok": {"mean": 0.287}, "duration_ratio": {"mean": 0.995}},
        ],
    }
    row = row_from_metrics(metrics, "exp-a", "exp-a", 40)
    assert row["checkpoint"] == "best_model_15800.pth"
    assert row["temperature"] == 0.65 and row["repetition_penalty"] == 7.5
    assert row["text_from"] == "script"
    assert row["mcd_db"] == 63.67 and row["mcd_db_ok"] == 62.9
    assert row["failure_rate"] == 0.0375 and row["clips_ok"] == 77
    # Per-speaker must survive into the register; the dinithi/harini gap is the
    # finding most likely to be hidden by an overall mean.
    assert row["dinithi_failure_rate"] == 0.05 and row["harini_mcd_db_ok"] == 65.5

    # A metrics.json from before --decode existed must still produce a row.
    legacy = {"checkpoint": "old.pth", "temperature": 0.75, "seed": 1234,
              "overall": {"clips": 80, "clips_ok": 80, "failure_rate": 0.0,
                          "mcd_db": {"mean": 63.0}}}
    old_row = row_from_metrics(legacy, "exp-legacy", "legacy", 40)
    assert old_row["temperature"] == 0.75 and old_row["top_k"] is None

    tmp = Path(tempfile.mkdtemp(prefix="sweep_"))
    reg = tmp / "experiments.csv"
    append_row(reg, old_row)                 # narrow row first
    append_row(reg, row)                     # then one with per-speaker columns
    text = reg.read_text(encoding="utf-8")
    assert "dinithi_failure_rate" in text.splitlines()[0], "header must widen"
    assert len(text.strip().splitlines()) == 3, text
    # Re-running an experiment replaces its row instead of appending a duplicate.
    append_row(reg, {**row, "mcd_db_ok": 61.0})
    body = reg.read_text(encoding="utf-8").strip().splitlines()
    assert len(body) == 3, body
    assert "61.0" in body[-1]

    # Ranking: failures dominate, then over-generation, then MCD.
    a = {"exp_id": "a", "failure_rate": 0.0, "duration_ratio": 1.05, "mcd_db_ok": 64.0}
    b = {"exp_id": "b", "failure_rate": 0.04, "duration_ratio": 1.00, "mcd_db_ok": 62.0}
    c = {"exp_id": "c", "failure_rate": 0.0, "duration_ratio": 1.01, "mcd_db_ok": 65.0}
    order = [r["exp_id"] for r in sorted([a, b, c], key=lambda r: rank_key(r, None))]
    assert order == ["c", "a", "b"], order
    # An explicit key overrides it.
    order = [r["exp_id"] for r in sorted([a, b, c],
                                         key=lambda r: rank_key(r, "mcd_db_ok"))]
    assert order == ["b", "a", "c"], order
    # A missing column must sort last, not crash.
    rank_key({"exp_id": "d"}, None)

    assert "best_model__t0.75__rp5__k50__p0.85__lp1" == exp_id_for(
        Path("best_model.pth"),
        {"temperature": 0.75, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.85,
         "length_penalty": 1.0, "text_from": "dataset"})
    assert exp_id_for(Path("best_model.pth"),
                      {"temperature": 0.75, "repetition_penalty": 5.0, "top_k": 50,
                       "top_p": 0.85, "length_penalty": 1.0,
                       "text_from": "script"}).endswith("__script")

    # Seeds: the default stays unnamed so ids already in a register keep
    # resolving, and a non-default seed gets its own id -- without which three
    # seeds of one config collide into one directory and two are silently
    # skipped as "already measured".
    base_cfg = {"temperature": 0.75, "repetition_penalty": 5.0, "top_k": 50,
                "top_p": 0.85, "length_penalty": 1.0, "text_from": "dataset"}
    assert exp_id_for(Path("m.pth"), base_cfg) ==         exp_id_for(Path("m.pth"), {**base_cfg, "seed": 1234})
    assert exp_id_for(Path("m.pth"), {**base_cfg, "seed": 1235}).endswith("__s1235")
    ids = {exp_id_for(Path("m.pth"), {**base_cfg, "seed": sd})
           for sd in (1234, 1235, 1236)}
    assert len(ids) == 3, ids

    # find_checkpoints: stepped bests preferred, best_model.pth not double-scored.
    (tmp / "run").mkdir()
    for name in ("best_model.pth", "best_model_15800.pth", "best_model_5250.pth",
                 "checkpoint_22000.pth"):
        (tmp / "run" / name).write_bytes(b"x")
    got = [p.name for p in find_checkpoints(tmp / "run")]
    assert got == ["best_model_5250.pth", "best_model_15800.pth",
                   "checkpoint_22000.pth"], got
    (tmp / "bare").mkdir()
    (tmp / "bare" / "best_model.pth").write_bytes(b"x")
    assert [p.name for p in find_checkpoints(tmp / "bare")] == ["best_model.pth"]

    table = render([row], None)
    assert "fail%" in table and "dinithi" in table
    # Failure rates are fractions in the register and percentages in the table --
    # per-speaker columns included. dinithi's 0.05 must read as 5.0, not as 0.1.
    assert "| 5.0 / 61.90 / 0.528 |" in table, table
    assert "| 3.8 |" in table, table          # overall 0.0375 -> 3.8 %

    print("sweep_eval selftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
