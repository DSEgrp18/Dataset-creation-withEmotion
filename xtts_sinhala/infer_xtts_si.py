#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_xtts_si.py -- synthesise Sinhala with the fine-tuned checkpoint.

You type Sinhala script; sinhala_text.sinhala_to_ascii folds it to the same
ascii the model was trained on. Skipping that step feeds the tokenizer
codepoints that are not in vocab.json, every word becomes [UNK], and you get
noise -- so the fold is not optional at inference either.

The Trainer writes a full XTTS state dict under the "xtts." prefix, and
Xtts.get_compatible_checkpoint_state_dict strips it, so best_model.pth loads
directly against the original config.json and vocab.json. Nothing to convert.

USAGE
    python infer_xtts_si.py \
        --run  ./run/training/GPT_XTTS_si-August-08-2026_10+00AM \
        --base ./run/training/XTTS_v2.0_original_model_files \
        --ref  ./si_dataset/wavs/sinh_0042.wav \
        --text "මට සිංහල භාෂාවෙන් කතා කරන්න පුළුවන්."

    # bundle model.pth + config.json + vocab.json into one portable folder
    python infer_xtts_si.py --run ... --base ... --ref ... --export ./xtts_si
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinhala_text import sinhala_to_ascii, to_ascii  # noqa: E402

DEFAULT_TEXTS = [
    "මට සිංහල භාෂාවෙන් කතා කරන්න පුළුවන්.",
    "අද දවස ලස්සන දවසක්. හෙට වැස්ස එයි කියලා හිතනවා.",
    "ශ්‍රී ලංකාවේ අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ ය.",
]


def pick_checkpoint(run: Path) -> Path:
    best = run / "best_model.pth"
    if best.is_file():
        return best
    cks = sorted(run.glob("checkpoint_*.pth"),
                 key=lambda p: int(p.stem.split("_")[-1]))
    if not cks:
        raise SystemExit(f"no best_model.pth or checkpoint_*.pth under {run}")
    return cks[-1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="GPT_XTTS_si-<stamp> directory")
    ap.add_argument("--base", required=True,
                    help="XTTS_v2.0_original_model_files (config.json + vocab.json)")
    ap.add_argument("--ref", required=True, help="speaker reference wav, 6-20 s")
    ap.add_argument("--text", action="append", default=None,
                    help="Sinhala script; repeatable. Defaults to three samples.")
    ap.add_argument("--out", default="./samples")
    ap.add_argument("--checkpoint", default=None, help="override the .pth choice")
    ap.add_argument("--export", default=None,
                    help="also write a self-contained model directory here")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=5.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.85)
    args = ap.parse_args()

    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    run, base = Path(args.run).resolve(), Path(args.base).resolve()
    ckpt = Path(args.checkpoint) if args.checkpoint else pick_checkpoint(run)
    print(f"checkpoint : {ckpt}")

    config = XttsConfig()
    config.load_json(str(base / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(ckpt),
        vocab_path=str(base / "vocab.json"),
        use_deepspeed=False,
    )
    if torch.cuda.is_available():
        model.cuda()

    print(f"reference  : {args.ref}")
    gpt_latent, speaker_emb = model.get_conditioning_latents(
        audio_path=[args.ref],
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, sinhala in enumerate(args.text or DEFAULT_TEXTS, 1):
        ascii_text = to_ascii(sinhala)
        print(f"\n[{i}] {sinhala}")
        print(f"    -> {ascii_text}")
        result = model.inference(
            text=ascii_text,
            language="en",              # tokenizer branch, see train_xtts_si.py
            gpt_cond_latent=gpt_latent,
            speaker_embedding=speaker_emb,
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
            enable_text_splitting=True,
        )
        wav = torch.tensor(result["wav"]).unsqueeze(0)
        path = out / f"sample_{i:02d}.wav"
        torchaudio.save(str(path), wav, 24000)   # XttsAudioConfig.output_sample_rate
        print(f"    -> {path}  ({wav.shape[-1] / 24000:.2f} s)")

    if args.export:
        exp = Path(args.export)
        exp.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, exp / "model.pth")
        shutil.copy2(base / "config.json", exp / "config.json")
        shutil.copy2(base / "vocab.json", exp / "vocab.json")
        shutil.copy2(Path(__file__).parent / "sinhala_text.py", exp / "sinhala_text.py")
        print(f"\nexported {exp}  (load with Xtts.load_checkpoint(checkpoint_dir=...))")
        print("Remember: always run text through sinhala_text.to_ascii() first.")
    return 0


if __name__ == "__main__":
    _ = sinhala_to_ascii  # re-exported for callers importing from this module
    raise SystemExit(main())
