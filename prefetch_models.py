#!/usr/bin/env python3
"""
prefetch_models.py -- download bench models into the HF cache, resumably.

The bench loads models in-process, so a dropped connection mid-download kills
that engine's whole column. This connection drops often (observed: read
timeouts on cas-bridge.xethub.hf.co, then DNS failure on huggingface.co), so
fetch the weights separately with retries FIRST, then run the bench offline
against the warm cache.

huggingface_hub resumes partial downloads, so a retry continues rather than
restarting -- which matters when a single file is 1.2 GB.

    python prefetch_models.py                 # smallest first
    python prefetch_models.py --only large-v3
    python prefetch_models.py --tries 20
"""

from __future__ import annotations

import argparse
import sys
import time

# (label, repo, approx MB) -- ordered smallest-first so a flaky link still
# makes progress on something.
TARGETS = [
    ("whisper-small-si",   "Lingalingeswaran/whisper-small-sinhala_v3",                 970),
    ("xlsr-300m-si",       "SpideyDLK/wav2vec2-large-xls-r-300m-sinhala-low-LR-part1", 1150),
    ("large-v3",           "Systran/faster-whisper-large-v3",                          3090),
]


def fetch(repo: str, tries: int, label: str) -> bool:
    from huggingface_hub import snapshot_download
    for attempt in range(1, tries + 1):
        try:
            print(f"[{label}] attempt {attempt}/{tries}: {repo}", flush=True)
            t0 = time.time()
            path = snapshot_download(repo_id=repo, max_workers=2)
            print(f"[{label}] OK in {time.time()-t0:.0f}s -> {path}", flush=True)
            return True
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            # Backoff is capped: these failures are transient link drops, not
            # rate limits, so waiting minutes buys nothing.
            wait = min(15 * attempt, 60)
            print(f"[{label}] FAILED ({type(ex).__name__}: {str(ex)[:140]}); "
                  f"retry in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    print(f"[{label}] GIVING UP after {tries} attempts", file=sys.stderr, flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=12)
    ap.add_argument("--only", help="substring: fetch only matching labels/repos")
    args = ap.parse_args()

    targets = TARGETS
    if args.only:
        needle = args.only.lower()
        targets = [t for t in TARGETS
                   if needle in t[0].lower() or needle in t[1].lower()]
        if not targets:
            print(f"no target matches {args.only!r}", file=sys.stderr)
            return 2

    total_mb = sum(mb for _, _, mb in targets)
    print(f"fetching {len(targets)} model(s), ~{total_mb} MB total\n", flush=True)

    results = {}
    for label, repo, mb in targets:
        print(f"=== {label}  (~{mb} MB) ===", flush=True)
        results[label] = fetch(repo, args.tries, label)
        print(flush=True)

    print("=== summary ===")
    for label, ok in results.items():
        print(f"  {'OK    ' if ok else 'FAILED'}  {label}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
