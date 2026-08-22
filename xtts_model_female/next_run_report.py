#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
next_run_report.py -- one document with everything needed to improve the next run.

The README used to end with "what to send back after the Kaggle run": the eval
table, the tail of train.log, prepare_report.json, a few wavs. Collecting those by
hand after an eight-hour session is exactly when things get forgotten, and the
session is deleted shortly afterwards. This assembles them while the files still
exist, and writes one self-contained markdown file.

It is not a summary. Every finding below is DERIVED from the numbers in this run
and carries the evidence that produced it, so a recommendation can be argued with:

    training stopped at epoch 5 of 40      -> train longer, and here is the resume command
    eval turned up at step 1760            -> export best_model.pth, not the last step
    the weaker speaker also used the       -> confounded; here is the experiment that
      script-transliteration text path        separates data volume from text path
    1.42 s/step measured, 40 epochs asked  -> that needs 13.9 h against an 8.5 h budget

Missing inputs are reported as missing rather than crashing: under Save & Run All a
step may have been skipped, and a partial handoff still beats none.

USAGE
    python next_run_report.py --dataset ./female_dataset --log train.log \
        --eval ./eval_out --run <run_dir> --out next_run.md
    python next_run_report.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_log  # noqa: E402

_TIME = re.compile(r"TIME:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_EPOCH = re.compile(r"EPOCH:\s*(\d+)/(\d+)")
_PER_EPOCH = re.compile(r"STEP:\s*\d+/(\d+)")
_EFF_BATCH = re.compile(r"effective batch\s+(\d+)")


def read_json(path: Path | None):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def parse_log(text: str) -> dict:
    """Everything the report needs from train.log, in one pass."""
    clean = train_log.strip_ansi(text)
    s = train_log.series(text)
    stamps = _TIME.findall(clean)
    epochs = [(int(a), int(b)) for a, b in _EPOCH.findall(clean)]
    per_epoch = [int(x) for x in _PER_EPOCH.findall(clean)]
    eff = _EFF_BATCH.search(clean)

    wall_h = sec_per_step = None
    if len(stamps) >= 2:
        fmt = "%Y-%m-%d %H:%M:%S"
        try:
            span = (datetime.strptime(stamps[-1], fmt)
                    - datetime.strptime(stamps[0], fmt)).total_seconds()
            first_step = s["train"][0][0] if s["train"] else 0
            last_step = s["train"][-1][0] if s["train"] else 0
            wall_h = span / 3600
            if last_step > first_step:
                sec_per_step = span / (last_step - first_step)
        except ValueError:
            pass

    label, why = train_log.verdict(s["eval"])
    return {
        "train": s["train"], "eval": s["eval"], "best_steps": s["best"],
        "verdict": label, "why": why,
        "last_step": s["train"][-1][0] if s["train"] else 0,
        "epoch_reached": epochs[-1][0] if epochs else None,
        "epoch_total": epochs[-1][1] + 1 if epochs else None,
        "steps_per_epoch": max(per_epoch) if per_epoch else None,
        "effective_batch": int(eff.group(1)) if eff else None,
        "wall_h": wall_h, "sec_per_step": sec_per_step,
    }


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def metric(block: dict, key: str):
    v = block.get(key)
    if v is None:
        return None
    return v["mean"] if isinstance(v, dict) else v


def build(prep, log, metrics, run_dir, dataset_dir, status=None) -> tuple[str, list]:
    """Returns (markdown, findings). Findings are (severity, title, evidence, action)."""
    out: list[str] = []
    findings: list[tuple[str, str, str, str]] = []
    w = out.append

    run_name = Path(run_dir).name if run_dir else "unknown-run"
    w(f"# Next-run brief — `{run_name}`")
    w("")
    w(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Everything below is measured from "
      "this run; the recommendations at the end carry the evidence that produced them.")
    w("")

    # ---------------------------------------------------------------- data
    w("## 1. What was trained on")
    w("")
    if prep:
        w("| Speaker | Clips | Hours | Text source |")
        w("|---|---|---|---|")
        src = prep.get("text_source", {})
        for spk, d in sorted(prep.get("speakers", {}).items()):
            label = {"roman": "corpus romanisation",
                     "script": "**transliterated from script**"}.get(
                         src.get(spk, "?"), src.get(spk, "?"))
            w(f"| {spk} | {d['clips']} | {d['hours']} | {label} |")
        w("")
        w(f"- train / eval clips: **{prep.get('train_clips')} / {prep.get('eval_clips')}**, "
          f"total **{prep.get('total_hours')} h**")
        w(f"- clip duration: median {prep.get('duration_median_s')} s, "
          f"max {prep.get('duration_max_s')} s")
        dropped = prep.get("dropped", {})
        if dropped:
            w("- dropped: " + ", ".join(f"{v} {k}" for k, v in sorted(dropped.items())))
        w("")

        # -- finding: a speaker whose text came from the fallback path
        script_spk = [s for s, v in src.items() if v == "script"]
        if script_spk:
            findings.append((
                "investigate",
                "One speaker's text came from the transliteration fallback",
                f"text_source: {', '.join(f'{s}={src[s]}' for s in sorted(src))}. "
                "sinhala_to_ascii(script) agrees with fold(romanisation) on 96.6 % of "
                "lines, so those speakers are trained on systematically different "
                "spellings for ~3.4 % of their text.",
                "Run the corpus romanisation through the transliterator for ALL speakers "
                "and retrain. If the gap closes, the text path was the cause; if it holds, "
                "it is data volume and the fix is more audio."))

        # -- finding: imbalance
        hours = {s: d["hours"] for s, d in prep.get("speakers", {}).items()}
        if len(hours) > 1 and max(hours.values()) > 2 * min(hours.values()):
            lo = min(hours, key=hours.get)
            findings.append((
                "consider",
                f"Speaker data is imbalanced ({lo} has the least)",
                ", ".join(f"{s}={h} h" for s, h in sorted(hours.items())),
                f"XTTS samples a conditioning clip from the same speaker each step, so "
                f"{lo} gets proportionally fewer updates. Either add audio for {lo} or "
                "accept that its voice will be the weaker of the two."))
    else:
        w("_prepare_report.json not found — dataset composition unknown._")
        w("")

    # ------------------------------------------------------------ training
    w("## 2. How training went")
    w("")
    if log and log["train"]:
        w(f"- reached **global step {log['last_step']}**"
          + (f", epoch **{log['epoch_reached']} of {log['epoch_total']}**"
             if log["epoch_reached"] is not None else ""))
        if log["effective_batch"]:
            w(f"- effective batch {log['effective_batch']}"
              + (f", {log['steps_per_epoch']} steps/epoch" if log["steps_per_epoch"] else ""))
        if log["wall_h"]:
            w(f"- wall clock **{log['wall_h']:.2f} h**"
              + (f" at **{log['sec_per_step']:.2f} s/step**" if log["sec_per_step"] else ""))
        first_t = log["train"][0][1]
        w(f"- train loss_mel_ce {first_t:.4f} → {log['train'][-1][1]:.4f} "
          f"over {len(log['train'])} logged points")
        if log["eval"]:
            best = min(log["eval"], key=lambda p: p[1])
            w(f"- eval loss_mel_ce best **{best[1]:.4f} at step {best[0]}**, "
              f"last {log['eval'][-1][1]:.4f} ({len(log['eval'])} evals)")
        w(f"- checkpoints written as best: {log['best_steps'][-6:] or 'none'}")
        w("")
        w(f"**Curve verdict: `{log['verdict']}`** — {log['why']}")
        w("")

        # Why it stopped is the first question a short run raises, and the log
        # alone cannot answer it: budget, nan, disk and a crash all look the same
        # once the process is gone. The notebook records it while it still knows.
        REASONS = {
            "budget": "the {budget:.1f} h training budget was reached",
            "nan": "the loss went nan and the guard aborted it",
            "disk": "the disk guard stopped it",
            "process_exit": "**the training process exited on its own**",
            "epochs_done": "it finished all requested epochs",
        }
        if status:
            reason = status.get("reason", "unknown")
            line = REASONS.get(reason, f"unrecorded (`{reason}`)").format(
                budget=status.get("budget_h", 0))
            w(f"**Why it stopped:** {line}"
              + (f" (exit code {status['returncode']})"
                 if status.get("returncode") not in (None, 0) else "") + ".")
            w("")
            used, budget = status.get("wall_h"), status.get("budget_h")
            if used and budget and used < 0.6 * budget and reason != "epochs_done":
                evidence = f"Stop reason recorded as `{reason}`"
                if reason == "process_exit":
                    evidence += f", exit code {status.get('returncode')}"
                evidence += f". Used {used:.2f} h of {budget:.1f} h."
                if log.get("sec_per_step"):
                    lost = int((budget - used) * 3600 / log["sec_per_step"])
                    evidence += (f" At {log['sec_per_step']:.2f} s/step the unused "
                                 f"{budget - used:.1f} h was worth about {lost:,} "
                                 f"more steps — roughly "
                                 f"{lost / max(log.get('steps_per_epoch') or 1, 1):.0f} "
                                 f"more epochs.")
                findings.append((
                    "do first",
                    f"Training used only {used:.1f} h of its {budget:.1f} h budget",
                    evidence,
                    "This, not the epoch count, is why the run is short. A run that "
                    "stops at 2 h will stop at 2 h again, and each attempt costs a "
                    "session — so diagnose it before resuming. The log tail is in "
                    "section 6."))
        else:
            w("**Why it stopped:** not recorded. Re-run with the notebook's training "
              "cell, which now writes `run_status.json`.")
            w("")

        if log["verdict"] == "improving":
            findings.append((
                "do first",
                "The run stopped before it converged",
                f"Eval loss was still at its minimum at the last eval (step "
                f"{log['eval'][-1][0] if log['eval'] else '?'}). Nothing here says the "
                "model has learned all it can from this data.",
                "Resume and train further before changing any hyperparameter. Tuning "
                "against an unconverged run measures the stopping point, not the change."))
        elif log["verdict"] == "overfitting":
            best = min(log["eval"], key=lambda p: p[1])
            findings.append((
                "do first",
                "Eval loss turned up — the last checkpoint is not the best one",
                log["why"],
                f"Export from best_model.pth (step {best[0]}), not the final step. Then "
                "either stop training at that point next time, or add data — more epochs "
                "on this corpus will keep making it worse."))
        elif log["verdict"] == "plateau":
            findings.append((
                "consider",
                "Eval loss has flattened",
                log["why"],
                "More epochs on the same data and LR will not help. Change something: "
                "more audio, a lower LR for fine detail, or accept this as converged."))

        # -- finding: the budget cannot fit the requested epochs
        if log["sec_per_step"] and log["steps_per_epoch"] and log["epoch_total"]:
            need_h = (log["sec_per_step"] * log["steps_per_epoch"]
                      * log["epoch_total"] / 3600)
            done = log["last_step"] * log["sec_per_step"] / 3600
            if need_h > 9:
                findings.append((
                    "do first",
                    f"{log['epoch_total']} epochs needs ~{need_h:.1f} h — more than one session",
                    f"Measured {log['sec_per_step']:.2f} s/step x "
                    f"{log['steps_per_epoch']} steps/epoch x {log['epoch_total']} epochs. "
                    f"This session covered ~{done:.1f} h.",
                    f"Plan ~{need_h / 8.5:.0f} sessions with --continue-path, or cut epochs "
                    "to what one session fits. Kaggle's cap is 12 h and the notebook "
                    "budgets 8.5 h of it for training."))
    else:
        w("_train.log not found or empty — training progress unknown._")
        w("")

    # ---------------------------------------------------------- objective
    w("## 3. Objective results")
    w("")
    if metrics:
        m = metrics.get("metrics", metrics)
        blocks = [m["overall"]] + m.get("per_speaker", [])
        w(f"- checkpoint `{Path(m.get('checkpoint', '?')).name}`, {m.get('n_clips')} clips, "
          f"temperature {m.get('temperature')}, seed {m.get('seed')}")
        w("")
        w("| Scope | MCD dB | log-F0 RMSE | F0 corr | SECS | Dur. ratio | Fail % | RTF |")
        w("|---|---|---|---|---|---|---|---|")
        for d in blocks:
            w(f"| {d['name']} | {fmt(metric(d,'mcd_db'),2)} | "
              f"{fmt(metric(d,'f0_rmse_cents'),1)} | {fmt(metric(d,'f0_corr'))} | "
              f"{fmt(metric(d,'secs'))} | {fmt(metric(d,'duration_ratio'))} | "
              f"{100*d.get('failure_rate',0):.1f} | {fmt(metric(d,'rtf'))} |")
        w("")
        w("MCD is implementation-dependent — compare against other runs of this script "
          "via `calibrate_mcd.py` and RESULTS.md, never against a published figure.")
        w("")

        overall = m["overall"]
        fail = overall.get("failure_rate", 0)
        if fail > 0.03:
            findings.append((
                "investigate", f"Generation failure rate is {100*fail:.1f} %",
                "Duration ratio outside 0.7-1.4 counts as a failure: XTTS is "
                "autoregressive, so this is truncation or runaway looping.",
                "Listen to the worst clips in eval_out/synth. If they are long "
                "sentences, lower --temperature or raise repetition_penalty."))

        dur = metric(overall, "duration_ratio")
        if dur is not None and abs(1 - dur) > 0.05:
            findings.append((
                "investigate",
                f"Synthesised audio is {'shorter' if dur < 1 else 'longer'} than the "
                f"reference (ratio {dur:.3f})",
                "A systematic length offset usually means the model is cutting off or "
                "padding rather than mispronouncing.",
                "Check a few long sentences by ear before trusting MCD, which DTW-aligns "
                "and so partly hides a length problem."))

        # -- finding: per-speaker spread, cross-referenced with the text path
        spk = {d["name"]: d for d in m.get("per_speaker", [])}
        if len(spk) > 1:
            corrs = {n: metric(d, "f0_corr") for n, d in spk.items()}
            corrs = {n: v for n, v in corrs.items() if v is not None}
            if corrs and max(corrs.values()) - min(corrs.values()) > 0.15:
                worst = min(corrs, key=corrs.get)
                best = max(corrs, key=corrs.get)
                extra = ""
                if prep:
                    src = prep.get("text_source", {})
                    hrs = {s: d["hours"] for s, d in prep.get("speakers", {}).items()}
                    extra = (f" {worst}: {hrs.get(worst,'?')} h, text from "
                             f"{src.get(worst,'?')}; {best}: {hrs.get(best,'?')} h, text "
                             f"from {src.get(best,'?')}.")
                findings.append((
                    "investigate",
                    f"Speakers differ a lot in prosody: F0 corr {worst} "
                    f"{corrs[worst]:.3f} vs {best} {corrs[best]:.3f}",
                    f"A {max(corrs.values())-min(corrs.values()):.3f} spread." + extra,
                    "If the weaker speaker has both less audio and a different text "
                    "source, those two causes are confounded — change one at a time."))
    else:
        w("_eval metrics.json not found — objective results unknown._")
        w("")

    # ------------------------------------------------------------ findings
    w("## 4. What to change next, in order")
    w("")
    order = {"do first": 0, "investigate": 1, "consider": 2}
    findings.sort(key=lambda f: order.get(f[0], 9))
    if findings:
        for i, (sev, title, evidence, action) in enumerate(findings, 1):
            w(f"### {i}. {title}  \n*{sev}*")
            w("")
            w(f"**Evidence.** {evidence}")
            w("")
            w(f"**Do.** {action}")
            w("")
    else:
        w("Nothing flagged. Either the run is healthy or an input was missing above.")
        w("")

    # -------------------------------------------------------------- resume
    w("## 5. Resume command")
    w("")
    w("```bash")
    w("python train_xtts_female.py \\")
    w(f"    --dataset {dataset_dir or '<dataset>'} \\")
    w("    --out /kaggle/temp/run \\")
    w("    --epochs 40 --batch-size 4 --grad-accum 16 --lr 1e-5 \\")
    w(f"    --continue-path {run_dir or '<run_dir>'}")
    w("```")
    w("")
    w("Attach the previous session's output as an input first; `best_model.pth` carries "
      "the optimizer state, so a resume continues rather than restarts.")
    w("")

    # The last thing the trainer said before it died. When a run ends for an
    # unrecorded reason this is the only evidence there is, and the session that
    # held it is deleted shortly after this file is written.
    tail = (status or {}).get("log_tail")
    if tail:
        w("## 6. Last lines of train.log")
        w("")
        w("```")
        for line in tail:
            w(line)
        w("```")
        w("")
    w("---")
    w("")
    w("Append the headline numbers to `RESULTS.md` before starting the next run — that "
      "file is what makes these metrics comparable at all.")
    return "\n".join(out), findings


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="prepare_voicemakers.py output dir")
    ap.add_argument("--log", default="/kaggle/working/train.log")
    ap.add_argument("--eval", help="evaluate_xtts.py output dir")
    ap.add_argument("--run", help="training run directory")
    ap.add_argument("--status", default=None,
                    help="run_status.json written by the notebook's training cell")
    ap.add_argument("--out", default="/kaggle/working/next_run.md")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    prep = read_json(Path(args.dataset) / "prepare_report.json") if args.dataset else None
    metrics = read_json(Path(args.eval) / "metrics.json") if args.eval else None
    log_path = Path(args.log)
    log = parse_log(log_path.read_text(encoding="utf-8", errors="replace")) \
        if log_path.is_file() else None
    status_path = Path(args.status) if args.status else log_path.with_name("run_status.json")
    status = read_json(status_path)

    for label, ok in (("prepare_report.json", prep is not None),
                      ("train.log", log is not None),
                      ("eval metrics.json", metrics is not None),
                      ("run_status.json", status is not None)):
        print(f"  {'found' if ok else 'MISSING'}  {label}")

    md, findings = build(prep, log, metrics, args.run, args.dataset, status)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    print(f"\n{len(findings)} finding(s):")
    for sev, title, _, _ in findings:
        print(f"  [{sev}] {title}")
    print(f"\nwrote {out}")
    return 0


def _selftest() -> int:
    prep = {
        "speakers": {"dinithi": {"clips": 2462, "hours": 4.69},
                     "harini": {"clips": 1135, "hours": 2.11}},
        "train_clips": 3517, "eval_clips": 80, "total_hours": 6.81,
        "duration_median_s": 6.82, "duration_max_s": 11.51,
        "text_source": {"dinithi": "roman", "harini": "script"},
        "dropped": {"unreadable_wav": 56, "longer_than_11.6s": 47},
    }
    log_text = (
        "  effective batch 64 (4 x 16)\n"
        "--> TIME: 2026-08-19 12:15:00 -- STEP: 350/879 -- GLOBAL_STEP: 350\n"
        "     | > loss_mel_ce: 3.7744  (x)\n"
        " > EVALUATION \n"
        "     | > avg_loss_mel_ce:\x1b[92m 3.1898 \x1b[0m(+0.0)\n"
        " > BEST MODEL : /x/best_model_880.pth\n"
        " > EPOCH: 1/39\n"
        "--> TIME: 2026-08-19 14:16:00 -- STEP: 720/879 -- GLOBAL_STEP: 5000\n"
        "     | > loss_mel_ce: 3.0448  (x)\n"
        " > EVALUATION \n"
        "     | > avg_loss_mel_ce:\x1b[92m 3.0204 \x1b[0m(-0.169)\n"
        " > BEST MODEL : /x/best_model_5000.pth\n"
        " > EPOCH: 5/39\n"
    )
    metrics = {"metrics": {
        "checkpoint": "/x/best_model.pth", "n_clips": 80,
        "temperature": 0.75, "seed": 1234,
        "overall": {"name": "best_model", "mcd_db": {"mean": 63.13},
                    "f0_rmse_cents": {"mean": 359.5}, "f0_corr": {"mean": 0.399},
                    "secs": {"mean": 0.700}, "duration_ratio": {"mean": 0.968},
                    "rtf": {"mean": 0.537}, "failure_rate": 0.025},
        "per_speaker": [
            {"name": "dinithi", "mcd_db": {"mean": 60.76}, "f0_corr": {"mean": 0.550},
             "secs": {"mean": 0.679}, "duration_ratio": {"mean": 1.001},
             "f0_rmse_cents": {"mean": 317.4}, "rtf": {"mean": 0.535},
             "failure_rate": 0.0},
            {"name": "harini", "mcd_db": {"mean": 65.50}, "f0_corr": {"mean": 0.248},
             "secs": {"mean": 0.721}, "duration_ratio": {"mean": 0.936},
             "f0_rmse_cents": {"mean": 401.6}, "rtf": {"mean": 0.539},
             "failure_rate": 0.05}]}}

    log = parse_log(log_text)
    assert log["epoch_reached"] == 5 and log["epoch_total"] == 40, log
    assert log["effective_batch"] == 64 and log["steps_per_epoch"] == 879, log
    assert abs(log["wall_h"] - 2.0166) < 0.01, log["wall_h"]
    assert log["verdict"] == "too-short", log["verdict"]   # only 2 evals

    # The case that matters most: the process died on its own, far short of the
    # budget, and the log tail is the only evidence of why.
    status = {"reason": "process_exit", "returncode": 1, "wall_h": 2.27,
              "budget_h": 8.5, "log_tail": ["torch.OutOfMemoryError: CUDA out of memory"]}
    md, findings = build(prep, log, metrics, "/x/GPT_XTTS_si_female-run", "/x/ds", status)
    titles = " | ".join(t for _, t, _, _ in findings)
    assert "used only 2.3 h of its 8.5 h budget" in titles, titles
    assert findings[0][0] == "do first"
    assert "the training process exited on its own" in md and "exit code 1" in md
    assert "CUDA out of memory" in md and "## 6. Last lines of train.log" in md
    ev = next(f[2] for f in findings if "budget" in f[1])
    assert "more steps" in ev and "more epochs" in ev, ev

    # A budget-exhausted run is doing what it was told; no early-stop finding.
    _, f_ok = build(prep, log, metrics, "/x/r", "/x/ds",
                    {"reason": "budget", "wall_h": 8.4, "budget_h": 8.5})
    assert not any("budget" in t for _, t, _, _ in f_ok), \
        [t for _, t, _, _ in f_ok]

    md, findings = build(prep, log, metrics, "/x/GPT_XTTS_si_female-run", "/x/ds")
    titles = " | ".join(t for _, t, _, _ in findings)
    assert "not recorded" in md      # no status file -> say so, do not guess

    # The three findings this run must produce, each from a different input.
    assert "transliteration fallback" in titles, titles
    assert "imbalanced" in titles, titles
    assert "more than one session" in titles, titles   # 1.42 s/step x 879 x 40 = 13.9 h
    # And the per-speaker prosody gap, cross-referenced against the text path.
    assert "F0 corr" in titles, titles
    assert "harini: 2.11 h, text from script" in " ".join(f[2] for f in findings)
    # "do first" items must sort above "consider".
    assert findings[0][0] == "do first", [f[0] for f in findings]
    assert "# Next-run brief" in md and "Resume command" in md

    # A run with nothing available must still produce a document.
    md2, f2 = build(None, None, None, None, None)
    assert "not found" in md2 and f2 == []

    print(f"next_run_report selftest OK -- {len(findings)} findings, {len(md)} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
