#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_vits.py -- the held-out set, spoken by VITS, into a directory.

The output is deliberately dumb: one `<clip_id>.wav` per held-out item, named so
that `evaluate_xtts.py --synth-dir` can score it. That is the whole mechanism by
which the two architectures get compared honestly -- **every metric is computed
by one implementation**, over the same 80 clips, with SECS measured by the same
XTTS speaker encoder for both systems. A second scoring implementation would be
comparing metric code as much as models.

What this script does NOT do is score anything. It synthesises and stops.

    python synthesize_vits.py --run <vits run dir> --dataset ./vits_dataset \\
        --out ./vits_synth --speaker dinithi

    python ../xtts_model_female/evaluate_xtts.py \\
        --run <xtts run dir> --base <xtts base> --dataset ../female_dataset \\
        --synth-dir ./vits_synth --out ./vits_scored --n 40 --utmos \\
        --label vits-dinithi

Note the second command still needs an XTTS run and base: the model is loaded
for its speaker encoder, not to synthesise anything. That is the point.

ONE SPEAKER AT A TIME. A per-voice VITS only knows its own speaker, so a run
covers the 40 held-out clips belonging to it. Synthesise each voice into the
SAME output directory to score both at once, exactly as the XTTS evaluation
covers both speakers in one pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "xtts_sinhala"):
    if (_cand / "sinhala_text.py").is_file():
        sys.path.insert(0, str(_cand))
        break
from sinhala_text import to_ascii  # noqa: E402


def find_checkpoint(run: Path) -> Path | None:
    best = run / "best_model.pth"
    if best.is_file():
        return best
    cks = sorted(run.glob("checkpoint_*.pth"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    return cks[-1] if cks else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="VITS run directory")
    ap.add_argument("--dataset", required=True,
                    help="prepare_vits.py output (for eval_reference.json)")
    ap.add_argument("--out", required=True, help="directory to write <clip_id>.wav into")
    ap.add_argument("--speaker", default=None,
                    help="only synthesise this speaker's held-out clips")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None, help="default: <run>/config.json")
    ap.add_argument("--n", type=int, default=40, help="clips per speaker")
    args = ap.parse_args()

    import torch
    import torchaudio
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.models.vits import Vits
    from TTS.utils.audio import AudioProcessor

    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    dataset = Path(args.dataset).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else find_checkpoint(run)
    if ckpt is None or not ckpt.is_file():
        print(f"no checkpoint under {run}", file=sys.stderr)
        return 2
    cfg_path = Path(args.config) if args.config else run / "config.json"
    if not cfg_path.is_file():
        print(f"no config at {cfg_path}", file=sys.stderr)
        return 2
    print(f"checkpoint : {ckpt}")

    ref_path = dataset / "eval_reference.json"
    if not ref_path.is_file():
        print(f"no eval_reference.json in {dataset} -- it carries the held-out "
              "items and is what makes this comparable to the XTTS run",
              file=sys.stderr)
        return 2
    items = json.loads(ref_path.read_text(encoding="utf-8"))

    # Same selection rule as evaluate_xtts.py: sort by clip_id, take the first n
    # per speaker. If that rule ever diverges the two systems stop being scored
    # on the same sentences, which is the one thing this must not get wrong.
    by_spk: dict[str, list] = {}
    for it in items:
        by_spk.setdefault(it["speaker"], []).append(it)
    chosen = []
    for spk, lst in by_spk.items():
        if args.speaker and spk != args.speaker:
            continue
        chosen += sorted(lst, key=lambda x: x["clip_id"])[: args.n]
    if not chosen:
        print(f"no held-out clips for speaker {args.speaker!r}; "
              f"present: {sorted(by_spk)}", file=sys.stderr)
        return 2
    print(f"items      : {len(chosen)}"
          + (f"  (speaker {args.speaker})" if args.speaker else f"  across {sorted(by_spk)}"))

    config = VitsConfig()
    config.load_json(str(cfg_path))
    ap_audio = AudioProcessor.init_from_config(config)
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, str(ckpt), eval=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sr = config.audio.sample_rate
    print(f"sample rate: {sr}")

    total_audio = total_time = 0.0
    for i, it in enumerate(chosen, 1):
        # The text is taken from eval_reference.json, exactly as evaluate_xtts.py
        # takes it, so both systems speak the same string. to_ascii is a no-op on
        # text that is already ASCII and a safety net if it is not.
        text = to_ascii(it["ascii"])
        t0 = time.time()
        wav = model.tts(text)
        elapsed = time.time() - t0

        tensor = torch.as_tensor(wav, dtype=torch.float32).reshape(1, -1)
        torchaudio.save(str(out / f"{it['clip_id']}.wav"), tensor, sr)
        secs = tensor.shape[-1] / sr
        total_audio += secs
        total_time += elapsed
        if i % 10 == 0 or i == len(chosen):
            print(f"  {i}/{len(chosen)}")

    print(f"\nwrote {len(chosen)} wavs to {out}")
    print(f"RTF (this GPU): {total_time / max(total_audio, 1e-6):.3f}  "
          f"over {total_audio:.1f} s of audio")
    print("\nscore it with the SAME code that scored XTTS:")
    print(f"  python ../xtts_model_female/evaluate_xtts.py --run <xtts_run> "
          f"--base <xtts_base> --dataset <xtts_dataset> \\\n"
          f"      --synth-dir {out} --out ./vits_scored --n {args.n} --utmos "
          f"--label vits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
