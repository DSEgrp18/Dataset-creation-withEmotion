#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_xtts_si.py -- the official Coqui XTTS-v2 fine-tune recipe, with only the
changes a single-GPU Sinhala run actually needs.

Base recipe:
  recipes/ljspeech/xtts_v2/train_gpt_xtts.py
  TTS/demos/xtts_ft_demo/utils/gpt_train.py
Every GPTArgs / XttsAudioConfig / optimizer value below is copied verbatim from
those two files. The deviations are listed here so you can see all of them:

  language="en"        Not a lie about the audio -- it selects the tokenizer
                       branch. VoiceBpeTokenizer.preprocess_text raises
                       NotImplementedError for anything outside its 17-language
                       set, and "[si]" is not a token in vocab.json. The text
                       has already been folded to ascii by sinhala_text.py, so
                       the "en" cleaner path (lowercase, number expansion,
                       whitespace) is exactly right for it.

  grad_accum           Upstream uses BATCH_SIZE * GRAD_ACUMM_STEPS = 252, which
                       is the right number when you have a datacentre. On one
                       T4 that is ~100 s per optimiser step and you would get
                       ~400 steps out of a Kaggle session -- not enough to move
                       a new sound inventory. Default here is 4 x 16 = 64,
                       which trades gradient noise for ~4x more steps. Raise it
                       if you have the hours.

  lr=1e-5              Upstream is 5e-6 at effective batch 252. At batch 64 the
                       per-step signal is smaller and the budget is tighter, so
                       this doubles it. Drop back to 5e-6 if loss_mel_ce is
                       noisy or rising.

  mixed_precision      Off upstream, on here: ~2x throughput on a T4. If the
                       smoke run prints nan, pass --no-mixed-precision.

  scheduler            Left exactly as upstream. Note that Trainer defaults to
                       scheduler_after_epoch=True and the milestones are
                       [900000, 2700000, 5400000] *epochs*, so the LR is
                       constant in practice. That is intentional, not a bug.

WHAT TO WATCH
  loss_mel_ce is the acoustic reconstruction term and the only one that tracks
  audio quality. loss_text_ce carries weight 0.01 in the total.

USAGE
    python train_xtts_si.py --dataset ./si_dataset --out ./run --smoke
    python train_xtts_si.py --dataset ./si_dataset --out ./run --epochs 40
    python train_xtts_si.py --dataset ./si_dataset --out ./run --continue-path ./run/training/GPT_XTTS_si-<stamp>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sinhala_text import sinhala_to_ascii  # noqa: E402

# XTTS v2.0.2 base files, as in the upstream recipe.
LINKS = {
    "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
    "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
}

# Two sentences in Sinhala script, synthesised into tensorboard every eval.
TEST_SENTENCES_SI = [
    "මට සිංහල භාෂාවෙන් කතා කරන්න පුළුවන්.",
    "අද දවස ලස්සන දවසක්. හෙට වැස්ස එයි කියලා හිතනවා.",
]


def fetch_base(dest: Path) -> dict[str, str]:
    from TTS.utils.manage import ModelManager

    dest.mkdir(parents=True, exist_ok=True)
    missing = [url for name, url in LINKS.items() if not (dest / name).is_file()]
    if missing:
        print(" > downloading XTTS v2 base files")
        ModelManager._download_model_files(missing, str(dest), progress_bar=True)
    return {name: str(dest / name) for name in LINKS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="output of prepare_pathnirvana.py")
    ap.add_argument("--out", default="./run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-step", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--no-mixed-precision", dest="amp", action="store_false")
    ap.add_argument("--continue-path", default=None,
                    help="resume a previous run directory (Kaggle 12 h limit)")
    ap.add_argument("--smoke", action="store_true",
                    help="a handful of steps on a tiny slice, to prove the wiring")
    args = ap.parse_args()

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
    from TTS.tts.models.xtts import XttsAudioConfig

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve() / "training"
    out.mkdir(parents=True, exist_ok=True)
    base = fetch_base(out / "XTTS_v2.0_original_model_files")

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="pathnirvana_si",
        path=str(dataset),
        meta_file_train=str(dataset / "metadata_train.csv"),
        meta_file_val=str(dataset / "metadata_eval.csv"),
        language="en",          # tokenizer branch, not a claim about the audio
    )

    # --- verbatim from the upstream recipe -------------------------------
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 s
        min_conditioning_length=66150,    # 3 s
        debug_loading_failures=False,
        max_wav_length=255995,            # ~11.6 s
        max_text_length=200,
        mel_norm_file=base["mel_stats.pth"],
        dvae_checkpoint=base["dvae.pth"],
        xtts_checkpoint=base["model.pth"],
        tokenizer_file=base["vocab.json"],
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # ---------------------------------------------------------------------

    train_samples, eval_samples = load_tts_samples(
        [config_dataset], eval_split=True, eval_split_max_size=256, eval_split_size=0.01)
    if args.smoke:
        train_samples, eval_samples = train_samples[:64], eval_samples[:8]

    # The longest transcript is the safest conditioning reference: it is the
    # clip most likely to carry the full phonetic range of the speaker.
    speaker_ref = max(train_samples, key=lambda s: len(s["text"]))["audio_file"]

    steps_per_epoch = max(1, len(train_samples) // (args.batch_size * args.grad_accum))
    print(f"\n  train {len(train_samples)}  eval {len(eval_samples)}")
    print(f"  effective batch {args.batch_size * args.grad_accum} "
          f"({args.batch_size} x {args.grad_accum})")
    print(f"  ~{steps_per_epoch} optimiser steps/epoch, "
          f"~{steps_per_epoch * args.epochs} total")
    print(f"  reference wav {speaker_ref}\n")

    config = GPTTrainerConfig(
        epochs=2 if args.smoke else args.epochs,
        output_path=str(out),
        model_args=model_args,
        run_name="GPT_XTTS_si",
        project_name="XTTS_si",
        run_description="XTTS v2 fine-tune, Sinhala romanised to ascii",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=args.batch_size,
        batch_group_size=48,
        eval_batch_size=args.batch_size,
        num_loader_workers=0 if args.smoke else args.workers,
        eval_split_max_size=256,
        print_step=10 if args.smoke else 50,
        plot_step=100,
        log_model_step=1000,
        save_step=args.save_step,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        mixed_precision=args.amp,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,   # set False for multi-GPU
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18],
                             "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {"text": sinhala_to_ascii(s), "speaker_wav": [speaker_ref], "language": "en"}
            for s in TEST_SENTENCES_SI
        ],
    )

    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,      # the base weights come in via xtts_checkpoint
            continue_path=args.continue_path,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=args.grad_accum,
        ),
        config,
        output_path=str(out),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    print(f"\nrun directory: {trainer.output_path}")
    print("infer with:")
    print(f"  python infer_xtts_si.py --run {trainer.output_path} "
          f"--base {out / 'XTTS_v2.0_original_model_files'} --ref {speaker_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
