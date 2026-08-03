#!/usr/bin/env python3
"""
check_quota.py -- when does the Gemini free-tier quota reset, and is it back yet?

Google's docs say: "Requests per day (RPD) quotas reset at midnight Pacific
time." Pacific observes daylight saving, so the wall-clock moment in your
timezone shifts by an hour twice a year -- this computes it rather than
hardcoding it.

Two things it reports:

  1. WHEN  -- the next midnight Pacific, converted to your local clock.
  2. WHAT  -- which models can actually be called RIGHT NOW. This is the
              reliable answer: a 429 body naming a *PerDay* quota means that
              model is done for the day, and no amount of waiting inside a
              session changes it.

Cost: a successful probe spends ONE request of that model's daily allowance
(1 of ~20). Run it to decide whether launching a build is worth it, not on a
loop. --no-probe skips the API entirely and only prints the reset time.

USAGE
    python check_quota.py
    python check_quota.py --no-probe
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import sys
from pathlib import Path

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/{model}:generateContent")


def resolve_api_key() -> str:
    import os
    f = Path(__file__).with_name("api_key.txt")
    if f.exists() and f.read_text(encoding="utf-8").strip():
        return f.read_text(encoding="utf-8").strip()
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY") or "").strip()


def next_reset() -> tuple[dt.datetime, dt.datetime, str]:
    """Return (reset_utc, reset_local, tz_label) for the next midnight Pacific."""
    try:
        from zoneinfo import ZoneInfo
        pac = ZoneInfo("America/Los_Angeles")
    except Exception:
        # No tz database (bare Windows Pythons sometimes lack one). Fall back to
        # a fixed -8, which is wrong by an hour during daylight saving -- say so.
        pac = dt.timezone(dt.timedelta(hours=-8))
    now_pac = dt.datetime.now(pac)
    tomorrow = (now_pac + dt.timedelta(days=1)).date()
    reset_pac = dt.datetime.combine(tomorrow, dt.time(0, 0), tzinfo=pac)
    label = reset_pac.tzname() or "PT"
    return reset_pac.astimezone(dt.timezone.utc), reset_pac.astimezone(), label


def probe(model: str, api_key: str) -> tuple[str, str]:
    """Return (status, detail) for one model. Costs 1 request if it succeeds."""
    import numpy as np
    import requests
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype="float32"), 16000,
             format="FLAC", subtype="PCM_16")
    body = {
        "contents": [{"parts": [
            {"text": "Reply with OK."},
            {"inline_data": {"mime_type": "audio/flac",
                             "data": base64.b64encode(buf.getvalue()).decode()}},
        ]}],
        "generationConfig": {"maxOutputTokens": 4, "temperature": 0.0},
    }
    try:
        r = requests.post(GEMINI_URL.format(model=model), json=body,
                          timeout=90, headers={"x-goog-api-key": api_key})
    except Exception as ex:
        return "ERROR", type(ex).__name__
    if r.status_code == 200:
        return "AVAILABLE", "quota remaining"
    if r.status_code == 429:
        try:
            for d in r.json().get("error", {}).get("details", []):
                for v in d.get("violations", []):
                    qid = str(v.get("quotaId", ""))
                    if "PerDay" in qid:
                        return "DAILY CAP", f"limit {v.get('quotaValue')}"
                    if "PerMinute" in qid:
                        return "per-minute", "retry shortly"
        except Exception:
            pass
        return "RATE LIMITED", "429, reason unclear"
    return f"HTTP {r.status_code}", r.text[:60].replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-probe", action="store_true",
                    help="only show the reset time; make no API calls")
    args = ap.parse_args()

    reset_utc, reset_local, label = next_reset()
    left = reset_utc - dt.datetime.now(dt.timezone.utc)
    hrs, rem = divmod(int(left.total_seconds()), 3600)

    print("=" * 60)
    print("  QUOTA RESET  (midnight Pacific, per Google's docs)")
    print("=" * 60)
    print(f"  next reset : {reset_local:%Y-%m-%d %H:%M} "
          f"{reset_local.tzname() or 'local'}   [{label} midnight]")
    print(f"  in         : {hrs} h {rem//60} min")
    if label in ("PT", "UTC-08:00"):
        print("  NOTE: no timezone database found; assumed UTC-8. During US"
              " daylight saving the real reset is an hour earlier.")

    if args.no_probe:
        return 0

    api_key = resolve_api_key()
    if not api_key:
        print("\n  no API key (api_key.txt or GEMINI_API_KEY); skipping probe")
        return 2

    print("\n" + "=" * 60)
    print("  MODEL AVAILABILITY  (each success costs 1 of ~20 daily)")
    print("=" * 60)
    ready = 0
    for m in MODELS:
        status, detail = probe(m, api_key)
        if status == "AVAILABLE":
            ready += 1
        print(f"  {status:12s} {m:24s} {detail}")

    print("=" * 60)
    if ready:
        print(f"  {ready}/{len(MODELS)} model(s) usable -> a build run is worth"
              f" starting (~{ready*20} requests, ~{ready*4} episodes).")
    else:
        print(f"  Nothing left today. Start the next run after"
              f" {reset_local:%H:%M}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
