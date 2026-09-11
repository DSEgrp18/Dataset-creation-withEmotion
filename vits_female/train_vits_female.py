#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vits_female.py -- a VITS voice for one Sinhala speaker, by fine-tuning.

WHY FINE-TUNE RATHER THAN TRAIN FROM SCRATCH
--------------------------------------------
VITS from scratch is a ~1M-step proposition. On 4.69 h of audio and a Kaggle
T4 you would get perhaps 150k steps across several sessions, and it would
sound like it. That is not a baseline, it is a strawman: a comparison against
an undertrained VITS tells you nothing about the architecture and would not
survive review.

So this restores a pretrained English VITS and adapts it. **The reason that is
possible at all is the same trick the XTTS work rests on:** the text has already
been transliterated to ASCII, so an English front-end can read it. The phoneme
inventory the pretrained model learned is the one our text produces, nothing is
randomly initialised, and the task is again "learn an accent", not "learn a
language". The transliteration buys transfer for both architectures.

SINGLE SPEAKER, DELIBERATELY
----------------------------
One model per voice. Two reasons, and neither is laziness:

  * it is what the product ships. Piper serves one ONNX voice per speaker, and
    SINHALA_READER_CLAUDE.md already names that as the offline tier.
  * fine-tuning single-speaker -> single-speaker needs no surgery. Restoring a
    109-speaker VCTK checkpoint for 2 new speakers means resizing `emb_g`, and a
    randomly-initialised speaker table is exactly the kind of thing that makes a
    "VITS is worse" result meaningless.

Pass --multi-speaker to train both voices in one model instead; it is supported
and it trains from scratch, with the quality consequences above.

WHAT IS HELD CONSTANT AGAINST THE XTTS RUN
------------------------------------------
Everything that would otherwise confound the comparison: the same clips, the
same 80 held-out items, the same ASCII text, and a training budget expressed in
the same wall-clock terms. See prepare_vits.py. The remaining differences are
the architecture and the front-end, and the front-end difference is stated
rather than hidden -- XTTS reads the ASCII as BPE tokens, VITS reads it as
phonemes through espeak.

MIXED PRECISION IS ON HERE, UNLIKE THE XTTS RECIPE
--------------------------------------------------
train_xtts_female.py disables fp16 because it drives loss to nan on a T4 within
one step. That is a property of that model, not of the card: VITS fp16 is
standard and roughly doubles throughput, which matters a great deal when the
budget is a Kaggle session. The smoke test checks for nan before the real run
commits hours to it. --no-mixed-precision if you see nan anyway.

USAGE
    python train_vits_female.py --dataset ./vits_dataset --speaker dinithi --out ./run --smoke
    python train_vits_female.py --dataset ./vits_dataset --speaker dinithi --out ./run
    python train_vits_female.py --dataset ./vits_dataset --multi-speaker --out ./run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Same reasoning as the XTTS trainer: the coqui Trainer refuses to start when
# more than one GPU is visible, so Kaggle's "T4 x2" fails outright unless pinned.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for _var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
    os.environ.setdefault(_var, "expandable_segments:True")

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "xtts_sinhala"):
    if (_cand / "sinhala_text.py").is_file():
        sys.path.insert(0, str(_cand))
        break
from sinhala_text import to_ascii  # noqa: E402

# Single-speaker, 22.05 kHz, phoneme front-end. Its config carries the symbol
# set the checkpoint was trained with, which is why the config is loaded FROM
# the download rather than written here -- a hand-written symbol set that
# differs by one character silently invalidates the restored embedding.
PRETRAINED = "tts_models/en/ljspeech/vits"

TEST_SENTENCES_SI = [
    "මට සිංහල භාෂාවෙන් කතා කරන්න පුළුවන්.",
    "අද දවස ලස්සන දවසක්. හෙට වැස්ස එයි කියලා හිතනවා.",
]


def free_gb(path: Path) -> float:
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="prepare_vits.py output")
    ap.add_argument("--out", default="./run")
    ap.add_argument("--speaker", default=None,
                    help="train one voice (default: the largest in the dataset)")
    ap.add_argument("--multi-speaker", action="store_true",
                    help="one model for every speaker -- trains from scratch")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="VITS default; lower to 1e-4 if fine-tuning destabilises")
    ap.add_argument("--save-step", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--mixed-precision", dest="amp", action="store_true")
    ap.add_argument("--no-mixed-precision", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--restore-path", default=None,
                    help="a .pth to fine-tune from (default: download " + PRETRAINED + ")")
    ap.add_argument("--from-scratch", action="store_true",
                    help="do not restore anything -- see the docstring on why not")
    ap.add_argument("--continue-path", default=None, help="resume a run directory")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--smoke", action="store_true",
                    help="a few steps on a tiny slice, to prove the wiring")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve() / "training"

    have = free_gb(out)
    print(f"disk: {have:.1f} GB free at {out}")
    if args.min_free_gb and have < args.min_free_gb:
        print(f"\nERROR: {have:.1f} GB free, need {args.min_free_gb:.0f} GB.\n"
              "  A VITS checkpoint is far smaller than XTTS's ~5.5 GB, but the\n"
              "  trainer still holds several. On Kaggle train into /kaggle/temp.",
              file=sys.stderr)
        return 1

    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits
    from TTS.tts.utils.speakers import SpeakerManager
    from TTS.utils.audio import AudioProcessor
    from TTS.utils.manage import ModelManager

    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------- which speaker
    if args.multi_speaker:
        train_meta, eval_meta, tag = "metadata_train.csv", "metadata_eval.csv", "multi"
    else:
        speaker = args.speaker
        if speaker is None:
            cands = sorted(dataset.glob("metadata_train_*.csv"),
                           key=lambda p: p.stat().st_size, reverse=True)
            if not cands:
                print(f"no per-speaker metadata in {dataset}; run prepare_vits.py",
                      file=sys.stderr)
                return 2
            speaker = cands[0].stem.replace("metadata_train_", "")
            print(f"no --speaker given; using the largest: {speaker}")
        train_meta = f"metadata_train_{speaker}.csv"
        eval_meta = f"metadata_eval_{speaker}.csv"
        tag = speaker
        if not (dataset / train_meta).is_file():
            print(f"no such speaker file: {dataset / train_meta}", file=sys.stderr)
            return 2

    # ------------------------------------------- pretrained config first
    restore_path = args.restore_path
    config = VitsConfig()
    if args.from_scratch:
        print("training from scratch -- see the docstring on why this is a bad "
              "baseline at this data scale")
        config.audio.sample_rate = 22050
        config.use_phonemes = True
        config.phoneme_language = "en-us"
        config.text_cleaner = "english_cleaners"
    else:
        print(f"fetching {PRETRAINED} for its config and weights")
        mm = ModelManager()
        model_file, config_file, _ = mm.download_model(PRETRAINED)
        # The config comes FROM the checkpoint. Its symbol set, audio settings
        # and phonemizer are what the restored weights were trained against, and
        # a single mismatched character silently invalidates the text embedding.
        config.load_json(str(config_file))
        if restore_path is None:
            restore_path = str(model_file)
        print(f"  weights : {restore_path}")
        print(f"  sample rate {config.audio.sample_rate}, "
              f"phonemes={config.use_phonemes} ({config.phoneme_language})")

    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="voicemakers_female_vits",
        path=str(dataset),
        meta_file_train=str(dataset / train_meta),
        meta_file_val=str(dataset / eval_meta),
        language="en",
    )

    # ------------------------------------------------ training settings
    config.output_path = str(out)
    config.datasets = [dataset_config]
    config.run_name = f"vits_si_female_{tag}"
    config.project_name = "VITS_si_female"
    config.run_description = (
        "VITS fine-tune, VoiceMakers Sinhala female, ASCII transliteration, "
        f"speaker={tag}")
    config.batch_size = args.batch_size
    config.eval_batch_size = max(2, args.batch_size // 2)
    config.batch_group_size = 5
    config.num_loader_workers = 0 if args.smoke else args.workers
    config.num_eval_loader_workers = 0 if args.smoke else max(1, args.workers // 2)
    config.run_eval = True
    config.test_delay_epochs = -1
    config.epochs = 2 if args.smoke else args.epochs
    config.lr_gen = args.lr
    config.lr_disc = args.lr
    config.print_step = 10 if args.smoke else 50
    config.plot_step = 200
    config.save_step = args.save_step
    config.save_n_checkpoints = 2
    config.save_best_after = 0
    config.print_eval = False
    config.mixed_precision = args.amp
    config.cudnn_benchmark = False
    config.compute_input_seq_cache = True
    config.phoneme_cache_path = str(out / "phoneme_cache")
    config.test_sentences = [to_ascii(s) for s in TEST_SENTENCES_SI]

    train_samples, eval_samples = load_tts_samples(
        [dataset_config], eval_split=True, eval_split_max_size=256, eval_split_size=0.01)
    if args.smoke:
        train_samples, eval_samples = train_samples[:32], eval_samples[:4]

    speaker_manager = None
    if args.multi_speaker:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples, parse_key="speaker_name")
        config.model_args.use_speaker_embedding = True
        config.model_args.num_speakers = speaker_manager.num_speakers
        config.num_speakers = speaker_manager.num_speakers
        print(f"  multi-speaker: {speaker_manager.num_speakers} speakers")
        if not args.from_scratch:
            print("  NOTE: restoring a single-speaker checkpoint into a "
                  "multi-speaker model will not load the speaker table; "
                  "consider --from-scratch or a per-speaker run instead")

    print(f"\n  speaker  : {tag}")
    print(f"  train {len(train_samples)}   eval {len(eval_samples)}")
    print(f"  batch {config.batch_size}, lr {args.lr}, fp16 {config.mixed_precision}")
    print(f"  restore  : {restore_path or 'nothing (from scratch)'}\n")

    ap_audio = AudioProcessor.init_from_config(config)
    model = Vits(config, ap_audio, speaker_manager=speaker_manager)

    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            continue_path=args.continue_path,
            skip_train_epoch=False,
            start_with_eval=False,
        ),
        config,
        output_path=str(out),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    run_dir = Path(trainer.output_path)
    print(f"\nrun directory: {run_dir}")
    print("synthesise the held-out set with:")
    print(f"  python synthesize_vits.py --run {run_dir} --dataset {dataset} "
          f"--out ./vits_synth --speaker {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
