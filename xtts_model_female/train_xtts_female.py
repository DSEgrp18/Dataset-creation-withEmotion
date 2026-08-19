#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_xtts_female.py -- fine-tune XTTS-v2 on the two VoiceMakers female voices.

The official Coqui recipe (recipes/ljspeech/xtts_v2/train_gpt_xtts.py and
TTS/demos/xtts_ft_demo/utils/gpt_train.py) with three deviations from upstream, all
deliberate and all listed here, plus one setting that matches upstream and is
listed anyway because it is the thing most likely to be changed back.

  language="en"      Selects a tokenizer branch, not a claim about the audio.
                     VoiceBpeTokenizer.preprocess_text raises NotImplementedError
                     outside its 17-language set and "[si]" is not a token in
                     vocab.json. prepare_voicemakers.py has already folded the
                     text to ASCII, so the "en" cleaner path -- lowercase, number
                     expansion, whitespace collapse -- is exactly right for it.

  effective batch    Upstream wants BATCH_SIZE * GRAD_ACUMM_STEPS >= 252, correct
                     advice when you have a datacentre. On one T4 that is ~100 s
                     per optimiser step, so a 12 h session buys ~400 steps -- not
                     enough to move the model onto a new sound inventory. 4 x 16
                     = 64 trades gradient noise for ~4x the steps.

  lr=1e-5            Upstream is 5e-6 at effective batch 252. Halving the batch
                     four times over shrinks the per-step signal, so this doubles
                     the rate. Drop to 5e-6 if loss_mel_ce gets noisy or rises.

  mixed_precision    Off, as upstream. fp16 roughly doubles throughput on a T4
                     but drives loss_mel_ce to nan on the first step of this
                     model and never recovers, and Turing has no bf16 -- so
                     there is no stable mixed-precision option on that card.
                     --mixed-precision opts in on a GPU where you have checked.

MULTI-SPEAKER, AND WHY IT MATTERS HERE
--------------------------------------
XTTS samples a second clip from the SAME speaker as the conditioning reference on
every training step. Two correctly-labelled speakers therefore teach "the
reference predicts the output voice", which is the capability being fine-tuned.
Pooling both under one label teaches the opposite, and no amount of data fixes it.
The `coqui` formatter reads the speaker_name column, so the labels written by
prepare_voicemakers.py carry through untouched.

WHAT TO WATCH
    loss_mel_ce is the acoustic reconstruction term and the only loss that tracks
    audio quality. loss_text_ce carries weight 0.01 in the total.

USAGE
    python train_xtts_female.py --dataset ./female_dataset --out ./run --smoke
    python train_xtts_female.py --dataset ./female_dataset --out ./run --epochs 40
    python train_xtts_female.py --dataset ./female_dataset --out ./run \
        --continue-path ./run/training/GPT_XTTS_si_female-<stamp>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The coqui Trainer refuses to start when more than one GPU is visible:
#   RuntimeError: [!] 2 active GPUs. Define the target GPU by CUDA_VISIBLE_DEVICES.
# So Kaggle's "GPU T4 x2" fails outright unless we pin. Real multi-GPU needs
# `python -m trainer.distribute` plus optimizer_wd_only_on_weights=False; that
# roughly doubles throughput but adds a whole failure surface, so get a baseline
# on one GPU first. setdefault means an explicit CUDA_VISIBLE_DEVICES still wins.
# This must run before torch is imported, hence module level rather than main().
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "xtts_sinhala"):
    if (_cand / "sinhala_text.py").is_file():
        sys.path.insert(0, str(_cand))
        break
from sinhala_text import to_ascii  # noqa: E402

LINKS = {
    "mel_stats.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth",
    "dvae.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth",
    "vocab.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "model.pth": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config.json": "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json",
}

# Synthesised into tensorboard at every eval, one per speaker reference.
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
    ap.add_argument("--dataset", required=True, help="output of prepare_voicemakers.py")
    ap.add_argument("--out", default="./run")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-step", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    # fp16 is OFF by default. It roughly doubles throughput, but on a Kaggle T4
    # it drives loss_mel_ce straight to nan on this model and never recovers --
    # an 8 hour run that produces only NaN checkpoints. Turing has no bf16, so
    # there is no stable mixed-precision option on that card. Opt in only if you
    # have verified finite losses on your GPU.
    ap.add_argument("--mixed-precision", dest="amp", action="store_true",
                    help="enable fp16 (~2x faster; produces nan on T4 -- verify first)")
    ap.add_argument("--no-mixed-precision", dest="amp", action="store_false")
    ap.set_defaults(amp=False)
    ap.add_argument("--continue-path", default=None,
                    help="resume a run directory (Kaggle's 12 h session limit)")
    ap.add_argument("--smoke", action="store_true",
                    help="a few steps on a tiny slice, to prove the wiring")
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
        dataset_name="voicemakers_female",
        path=str(dataset),
        meta_file_train=str(dataset / "metadata_train.csv"),
        meta_file_val=str(dataset / "metadata_eval.csv"),
        language="en",          # tokenizer branch, not a claim about the audio
    )

    # ---- verbatim from the upstream recipe ------------------------------
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

    # One reference per speaker: the longest transcript that speaker has, since
    # it is the clip most likely to span their full phonetic range.
    refs: dict[str, str] = {}
    for s in train_samples:
        spk = s.get("speaker_name") or "spk"
        if spk not in refs or len(s["text"]) > refs[spk][1]:
            refs[spk] = (s["audio_file"], len(s["text"]))
    refs = {k: v[0] for k, v in refs.items()}

    steps_per_epoch = max(1, len(train_samples) // (args.batch_size * args.grad_accum))
    print(f"\n  train {len(train_samples)}   eval {len(eval_samples)}")
    print(f"  speakers: {list(refs)}")
    print(f"  effective batch {args.batch_size * args.grad_accum} "
          f"({args.batch_size} x {args.grad_accum})")
    print(f"  ~{steps_per_epoch} optimiser steps/epoch, "
          f"~{steps_per_epoch * args.epochs} total")
    for spk, wav in refs.items():
        print(f"  reference[{spk}] = {wav}")
    print()

    test_sentences = []
    for spk, wav in refs.items():
        for s in TEST_SENTENCES_SI:
            test_sentences.append(
                {"text": to_ascii(s), "speaker_wav": [wav], "language": "en"})

    config = GPTTrainerConfig(
        epochs=2 if args.smoke else args.epochs,
        output_path=str(out),
        model_args=model_args,
        run_name="GPT_XTTS_si_female",
        project_name="XTTS_si_female",
        run_description="XTTS v2 fine-tune, VoiceMakers Dinithi + Harini, Sinhala as ASCII",
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
        # Trainer defaults to scheduler_after_epoch=True, so these milestones are
        # epochs and the LR is constant in practice. Upstream behaviour, kept.
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18],
                             "gamma": 0.5, "last_epoch": -1},
        test_sentences=test_sentences,
    )

    model = GPTTrainer.init_from_config(config)
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,      # base weights arrive via xtts_checkpoint
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

    run_dir = Path(trainer.output_path)
    (run_dir / "speaker_refs.json").write_text(
        json.dumps(refs, indent=2), encoding="utf-8")

    print(f"\nrun directory: {run_dir}")
    print("evaluate with:")
    print(f"  python evaluate_xtts.py --run {run_dir} "
          f"--base {out / 'XTTS_v2.0_original_model_files'} --dataset {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
