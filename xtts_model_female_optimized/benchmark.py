#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py -- what a checkpoint actually costs to run: size, VRAM, speed.

One configuration per invocation, appending a row to a CSV, so several runs build
a comparison table without this script having to own a matrix of options. Run it
once per configuration you care about and read results.csv at the end.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
    file size        what a user downloads
    load seconds     cold start of the app
    peak VRAM        whether it fits alongside anything else on the card
    RTF              seconds of compute per second of audio. Below 1.0 is
                     faster than realtime; a reader app needs well below 1.0
                     because the user also waits for the first chunk.
    latency          time to the FIRST audio chunk, via inference_stream. This
                     is the number a listener experiences as "responsive", and
                     it is not RTF -- a model can stream the first 200 ms
                     quickly and still have a poor RTF overall.

WARMUP MATTERS. The first inference on a CUDA device pays for kernel autotuning
and allocator growth, and on this model that is seconds. Timing it would make
every configuration look equally bad, so --warmup runs are discarded.

Speed is NOT gated anywhere. compare_quality.py gates quality; this only
measures. A configuration that is faster and passes the quality gate is a win,
and one that is faster and fails it is not a tradeoff worth making silently.

USAGE
    python benchmark.py --checkpoint model.pth       --base <base> --ref <wav> --tag exported
    python benchmark.py --checkpoint model_slim.pth  --base <base> --ref <wav> --tag slim
    python benchmark.py --checkpoint model_fp16.pth  --base <base> --ref <wav> --tag fp16-file
    python benchmark.py --checkpoint model_slim.pth  --base <base> --ref <wav> --tag fp16-compute --half
    python benchmark.py --checkpoint model_slim.pth  --base <base> --ref <wav> --tag deepspeed --deepspeed
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "xtts_sinhala"))

SENTENCES_SI = [
    "ආයුබෝවන්, ඔබට කොහොමද?",
    "මම අද උදේ පාසල් ගියා.",
    "මේ ගැන නම් මට සහතික වෙන්න බැහැ, මම හරියටම දැක්කෙ නැති නිසා.",
    "එනම් මේ දැවැන්ත ගල් කනු වල කොටා ඇති සත්ත්ව රූප මගින් පිළිබිඹු කර ඇත්තේ විවිධ තාරකා",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base", required=True, help="dir with config.json and vocab.json")
    ap.add_argument("--ref", required=True, help="speaker reference wav")
    ap.add_argument("--tag", required=True, help="name for this row")
    ap.add_argument("--out", default="./results.csv")
    ap.add_argument("--half", action="store_true",
                    help="run the model in fp16 (changes arithmetic, not just storage)")
    ap.add_argument("--deepspeed", action="store_true", help="DeepSpeed inference kernels")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.75)
    args = ap.parse_args()

    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from sinhala_text import to_ascii

    ck, base = Path(args.checkpoint), Path(args.base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA. XTTS on CPU is far slower than realtime -- these\n"
              "numbers will not resemble GPU performance. See README.md on why the\n"
              "offline target is VITS/Piper and not XTTS.\n", file=sys.stderr)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    config = XttsConfig()
    config.load_json(str(base / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ck),
                          vocab_path=str(base / "vocab.json"),
                          use_deepspeed=args.deepspeed)
    model.to(device)
    if args.half:
        model.half()
    load_s = time.time() - t0

    gpt_latent, spk_emb = model.get_conditioning_latents(
        audio_path=[args.ref], gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len, sound_norm_refs=config.sound_norm_refs)
    if args.half:
        gpt_latent, spk_emb = gpt_latent.half(), spk_emb.half()

    texts = [to_ascii(s) for s in SENTENCES_SI]

    def synth(text):
        return model.inference(
            text=text, language="en", gpt_cond_latent=gpt_latent,
            speaker_embedding=spk_emb, temperature=args.temperature,
            length_penalty=1.0, repetition_penalty=5.0, top_k=50, top_p=0.85,
            enable_text_splitting=False)

    # Discarded: the first call pays for CUDA kernel autotuning and allocator
    # growth, which is seconds on this model and belongs to no configuration.
    for _ in range(args.warmup):
        synth(texts[0])
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    rtfs, audio_s = [], 0.0
    for _ in range(args.repeat):
        for text in texts:
            t = time.time()
            res = synth(text)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t
            secs = len(res["wav"]) / 24000
            audio_s += secs
            rtfs.append(elapsed / secs)

    # Time to first audio -- what a listener perceives, distinct from RTF.
    latency = None
    try:
        t = time.time()
        for _ in model.inference_stream(
                text=texts[2], language="en", gpt_cond_latent=gpt_latent,
                speaker_embedding=spk_emb, temperature=args.temperature):
            if device == "cuda":
                torch.cuda.synchronize()
            latency = time.time() - t
            break
    except Exception as exc:                      # streaming is optional
        print(f"  (streaming unavailable: {exc})")

    vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    row = {
        "tag": args.tag,
        "checkpoint": ck.name,
        "size_gb": round(ck.stat().st_size / 1e9, 3),
        "half": int(args.half),
        "deepspeed": int(args.deepspeed),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "load_s": round(load_s, 1),
        "peak_vram_gb": round(vram, 2),
        "rtf_mean": round(statistics.fmean(rtfs), 3),
        "rtf_p95": round(sorted(rtfs)[int(0.95 * (len(rtfs) - 1))], 3),
        "first_audio_s": None if latency is None else round(latency, 2),
        "audio_s": round(audio_s, 1),
    }

    out = Path(args.out)
    new = not out.is_file()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    print("\n" + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"\nappended to {out}")
    if args.half or args.deepspeed:
        print("This configuration changes arithmetic. Run compare_quality.py before\n"
              "treating it as a free win.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
