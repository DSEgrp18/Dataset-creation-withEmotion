#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_xtts.py -- every metric the Sinhala TTS literature uses that a machine
can compute, run over the held-out eval split.

WHAT IS AND IS NOT AUTOMATABLE
------------------------------
MOS and SUS require human listeners; there is no substitute and claiming one is
how papers end up reporting "98% accuracy" for a synthesiser. Those two come from
listening_test.py, which builds the panel kit. Everything below runs unattended:

  MCD (dB)            Mel Cepstral Distortion, DTW-aligned. The only objective
                      metric any Sinhala TTS paper has reported -- SPECOM 2025
                      gives 13.27 dB for its best voice. Lower is better.
                      NOTE: MCD is implementation-dependent. Compare runs of THIS
                      script against each other; do not read across papers.

  log-F0 RMSE         Pitch accuracy in cents over DTW-aligned voiced frames.
  F0 correlation      Whether the intonation CONTOUR tracks the reference, which
                      RMSE alone misses -- a flat monotone can score decent RMSE.

  SECS                Speaker Encoder Cosine Similarity, using XTTS's own speaker
                      encoder. This is the voice-cloning metric: it answers "does
                      the output sound like the conditioning speaker". Measured
                      against a HELD-OUT clip of the same speaker, never the
                      conditioning clip itself, which would inflate it.

  Duration ratio      synth length / reference length. XTTS is autoregressive and
                      its classic failures are truncation and runaway looping;
                      both show here before you hear them.
  Failure rate        Share of clips outside a sane duration ratio band.

  RTF                 Real-time factor. Matters if this ever ships in a reader.

  UTMOS      (--utmos)  Learned MOS predictor. Trained on English, so treat it as
                        a relative signal between checkpoints, not an absolute MOS.

  ASR CER    (--asr)    Intelligibility proxy. Sinhala ASR is weak, so absolute CER
                        is meaningless -- the script therefore transcribes the REAL
                        held-out audio through the same model and reports the GAP.
                        The gap controls for the ASR's own error rate.

USAGE
    python evaluate_xtts.py --run ./run/training/GPT_XTTS_si_female-<stamp> \
        --base ./run/training/XTTS_v2.0_original_model_files \
        --dataset ./female_dataset --n 40
    python evaluate_xtts.py ... --utmos --asr openai/whisper-large-v3
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

HOP = 256
N_MFCC = 13
FMIN, FMAX = 80.0, 500.0        # female range, generous on both sides
SR = 22050
FAIL_LOW, FAIL_HIGH = 0.7, 1.4  # duration-ratio band outside which a clip failed


# --------------------------------------------------------------------------
def mcd_and_f0(ref_path: Path, syn_path: Path):
    """DTW-align two waveforms on MFCCs, then score spectrum and pitch on that path."""
    import librosa
    import numpy as np

    r, _ = librosa.load(str(ref_path), sr=SR, mono=True)
    s, _ = librosa.load(str(syn_path), sr=SR, mono=True)
    if len(r) < HOP * 4 or len(s) < HOP * 4:
        return None

    # Mel-cepstra on the CONVENTIONAL scale. librosa.feature.mfcc defaults to a
    # dB (10*log10) mel spectrogram, but MCD is defined over natural-log
    # cepstra -- feeding it dB inflates every distance by ~10/ln10 and puts the
    # numbers nowhere near the range anyone reports. So take the log ourselves.
    def melcep(y, floor_ref):
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=80)
        # Floor at -80 dB relative to a SHARED reference. Without a floor the
        # log of a near-silent mel bin runs off to -20 and dominates the
        # distance; with a per-file floor the two signals get different floors
        # and quiet frames alone produce a large spurious MCD.
        mel = np.maximum(mel, floor_ref * 1e-8)
        return librosa.feature.mfcc(S=np.log(mel), n_mfcc=N_MFCC)[1:]  # drop c0

    ref_power = max(
        float(librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=1024, hop_length=HOP, n_mels=80).max())
        for x in (r, s))
    R, S = melcep(r, ref_power), melcep(s, ref_power)
    _, wp = librosa.sequence.dtw(X=R, Y=S, metric="euclidean")
    wp = wp[::-1]                                   # librosa returns it reversed

    diff = R[:, wp[:, 0]] - S[:, wp[:, 1]]
    mcd = (10.0 / np.log(10)) * np.sqrt(2.0) * float(
        np.mean(np.sqrt((diff ** 2).sum(axis=0))))

    # pitch on the same frame grid, compared only where BOTH are voiced
    f0r, vr, _ = librosa.pyin(r, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP)
    f0s, vs, _ = librosa.pyin(s, fmin=FMIN, fmax=FMAX, sr=SR, hop_length=HOP)
    nr, ns = len(f0r), len(f0s)
    ok = (wp[:, 0] < nr) & (wp[:, 1] < ns)
    pr, ps = wp[ok, 0], wp[ok, 1]
    voiced = (vr[pr] & vs[ps] & ~np.isnan(f0r[pr]) & ~np.isnan(f0s[ps]))

    f0_rmse_cents = f0_corr = None
    if voiced.sum() >= 10:
        a, b = f0r[pr][voiced], f0s[ps][voiced]
        cents = 1200.0 * np.log2(b / a)
        f0_rmse_cents = float(np.sqrt(np.mean(cents ** 2)))
        if a.std() > 1e-6 and b.std() > 1e-6:
            f0_corr = float(np.corrcoef(np.log(a), np.log(b))[0, 1])

    return {
        "mcd_db": mcd,
        "f0_rmse_cents": f0_rmse_cents,
        "f0_corr": f0_corr,
        "voiced_frames": int(voiced.sum()),
        "ref_seconds": len(r) / SR,
        "syn_seconds": len(s) / SR,
    }


def cer(ref: str, hyp: str) -> float:
    """Character error rate via Levenshtein, on whitespace-normalised strings."""
    a = " ".join(ref.split())
    b = " ".join(hyp.split())
    if not a:
        return 1.0 if b else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def agg(values: list) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 4),
        "median": round(st.median(vals), 4),
        "std": round(st.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="GPT_XTTS_si_female-<stamp> directory")
    ap.add_argument("--base", required=True, help="XTTS_v2.0_original_model_files")
    ap.add_argument("--dataset", required=True, help="output of prepare_voicemakers.py")
    ap.add_argument("--out", default="./eval_out")
    ap.add_argument("--checkpoint", default=None, help="override the .pth choice")
    ap.add_argument("--n", type=int, default=40, help="eval clips per speaker")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--utmos", action="store_true", help="add the UTMOS predictor")
    ap.add_argument("--asr", default=None, metavar="MODEL",
                    help="e.g. openai/whisper-large-v3 -- adds the CER gap")
    ap.add_argument("--label", default=None, help="name for this run in the report")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    run, base = Path(args.run).resolve(), Path(args.base).resolve()
    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    (out / "synth").mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    if ckpt is None:
        best = run / "best_model.pth"
        if best.is_file():
            ckpt = best
        else:
            cks = sorted(run.glob("checkpoint_*.pth"),
                         key=lambda p: int(p.stem.split("_")[-1]))
            if not cks:
                return print(f"no checkpoint under {run}", file=sys.stderr) or 2
            ckpt = cks[-1]
    print(f"checkpoint : {ckpt}")

    ref_items = json.loads((dataset / "eval_reference.json").read_text(encoding="utf-8"))
    by_spk = defaultdict(list)
    for it in ref_items:
        by_spk[it["speaker"]].append(it)
    items = []
    for spk, lst in by_spk.items():
        lst = sorted(lst, key=lambda x: x["clip_id"])
        items += lst[: args.n]
    print(f"eval items : {len(items)}  across {list(by_spk)}")

    # ------------------------------------------------------------- model
    config = XttsConfig()
    config.load_json(str(base / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt),
                          vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ------------------------------------------------------- synthesise
    print("\nsynthesising")
    rows = []
    for i, it in enumerate(items, 1):
        target_wav = dataset / it["wav"]
        # Condition on a DIFFERENT clip from the same speaker. Using the target
        # itself would leak the answer into SECS and MCD alike.
        pool = [x for x in by_spk[it["speaker"]] if x["clip_id"] != it["clip_id"]]
        cond = dataset / random.Random(args.seed + i).choice(pool)["wav"]

        gpt_latent, spk_emb = model.get_conditioning_latents(
            audio_path=[str(cond)],
            gpt_cond_len=config.gpt_cond_len,
            max_ref_length=config.max_ref_len,
            sound_norm_refs=config.sound_norm_refs,
        )
        t0 = time.time()
        res = model.inference(
            text=it["ascii"], language="en",
            gpt_cond_latent=gpt_latent, speaker_embedding=spk_emb,
            temperature=args.temperature, length_penalty=1.0,
            repetition_penalty=5.0, top_k=50, top_p=0.85,
            enable_text_splitting=False,
        )
        elapsed = time.time() - t0
        wav = torch.tensor(res["wav"]).unsqueeze(0)
        syn_path = out / "synth" / f"{it['clip_id']}.wav"
        torchaudio.save(str(syn_path), wav, 24000)

        # SECS against the held-out target, not the conditioning clip
        _, emb_syn = model.get_conditioning_latents(
            audio_path=[str(syn_path)], gpt_cond_len=config.gpt_cond_len,
            max_ref_length=config.max_ref_len, sound_norm_refs=config.sound_norm_refs)
        _, emb_tgt = model.get_conditioning_latents(
            audio_path=[str(target_wav)], gpt_cond_len=config.gpt_cond_len,
            max_ref_length=config.max_ref_len, sound_norm_refs=config.sound_norm_refs)
        secs = float(torch.nn.functional.cosine_similarity(
            emb_syn.flatten().unsqueeze(0), emb_tgt.flatten().unsqueeze(0)).item())

        syn_sec = wav.shape[-1] / 24000
        rows.append({**it, "synth_wav": str(syn_path), "cond_wav": str(cond),
                     "secs": secs, "synth_seconds": syn_sec,
                     "rtf": elapsed / max(syn_sec, 1e-6)})
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------- signal metrics
    print("\nscoring MCD and F0 (pyin is slow, ~2 s per clip)")
    for i, r in enumerate(rows, 1):
        m = mcd_and_f0(dataset / r["wav"], Path(r["synth_wav"]))
        if m:
            r.update(m)
            r["duration_ratio"] = m["syn_seconds"] / max(m["ref_seconds"], 1e-6)
            r["failed"] = not (FAIL_LOW <= r["duration_ratio"] <= FAIL_HIGH)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    # ----------------------------------------------------------- UTMOS
    if args.utmos:
        print("\nUTMOS")
        try:
            import librosa
            import torch as _t
            predictor = _t.hub.load("tarepan/SpeechMOS", "utmos22_strong",
                                    trust_repo=True)
            for r in rows:
                w, _ = librosa.load(r["synth_wav"], sr=16000, mono=True)
                r["utmos"] = float(predictor(_t.from_numpy(w).unsqueeze(0), 16000).item())
                wr, _ = librosa.load(str(dataset / r["wav"]), sr=16000, mono=True)
                r["utmos_real"] = float(predictor(_t.from_numpy(wr).unsqueeze(0), 16000).item())
        except Exception as e:
            print(f"  skipped: {e}")

    # ------------------------------------------------------------- ASR
    if args.asr:
        print(f"\nASR intelligibility via {args.asr}")
        try:
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model=args.asr,
                           device=0 if device == "cuda" else -1,
                           generate_kwargs={"language": "sinhala", "task": "transcribe"})
            for i, r in enumerate(rows, 1):
                r["asr_synth"] = asr(r["synth_wav"])["text"].strip()
                r["asr_real"] = asr(str(dataset / r["wav"]))["text"].strip()
                r["cer_synth"] = cer(r["sinhala"], r["asr_synth"])
                r["cer_real"] = cer(r["sinhala"], r["asr_real"])
                if i % 10 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)}")
        except Exception as e:
            print(f"  skipped: {e}")

    # ---------------------------------------------------------- report
    def summarise(subset: list, name: str) -> dict:
        d = {
            "name": name,
            "clips": len(subset),
            "mcd_db": agg([r.get("mcd_db") for r in subset]),
            "f0_rmse_cents": agg([r.get("f0_rmse_cents") for r in subset]),
            "f0_corr": agg([r.get("f0_corr") for r in subset]),
            "secs": agg([r.get("secs") for r in subset]),
            "duration_ratio": agg([r.get("duration_ratio") for r in subset]),
            "rtf": agg([r.get("rtf") for r in subset]),
            "failure_rate": round(
                sum(1 for r in subset if r.get("failed")) / max(len(subset), 1), 4),
        }
        if any("utmos" in r for r in subset):
            d["utmos"] = agg([r.get("utmos") for r in subset])
            d["utmos_real"] = agg([r.get("utmos_real") for r in subset])
        if any("cer_synth" in r for r in subset):
            d["cer_synth"] = agg([r.get("cer_synth") for r in subset])
            d["cer_real"] = agg([r.get("cer_real") for r in subset])
            gap = [r["cer_synth"] - r["cer_real"] for r in subset if "cer_synth" in r]
            d["cer_gap"] = agg(gap)
        return d

    overall = summarise(rows, args.label or ckpt.stem)
    per_speaker = [summarise([r for r in rows if r["speaker"] == s], s)
                   for s in sorted(by_spk)]
    metrics = {"checkpoint": str(ckpt), "n_clips": len(rows),
               "temperature": args.temperature, "seed": args.seed,
               "overall": overall, "per_speaker": per_speaker}
    (out / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "clips": rows}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    def cell(a, k="mean", nd=2):
        return "n/a" if not a else f"{a[k]:.{nd}f}"

    lines = [
        f"# XTTS Sinhala female -- objective evaluation",
        "",
        f"- checkpoint: `{ckpt.name}`",
        f"- clips: {len(rows)}  (temperature {args.temperature}, seed {args.seed})",
        "",
        "| Scope | MCD dB | log-F0 RMSE (cents) | F0 corr | SECS | Dur. ratio | Fail % | RTF |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in [overall] + per_speaker:
        lines.append(
            f"| {d['name']} | {cell(d['mcd_db'])} | {cell(d['f0_rmse_cents'], nd=1)} | "
            f"{cell(d['f0_corr'], nd=3)} | {cell(d['secs'], nd=3)} | "
            f"{cell(d['duration_ratio'], nd=3)} | {100*d['failure_rate']:.1f} | "
            f"{cell(d['rtf'], nd=3)} |")
    if "utmos" in overall:
        lines += ["", "| Scope | UTMOS synth | UTMOS real recordings |", "|---|---|---|"]
        for d in [overall] + per_speaker:
            lines.append(f"| {d['name']} | {cell(d.get('utmos'))} | {cell(d.get('utmos_real'))} |")
    if "cer_synth" in overall:
        lines += ["", "| Scope | CER synth | CER real | Gap |", "|---|---|---|---|"]
        for d in [overall] + per_speaker:
            lines.append(f"| {d['name']} | {cell(d.get('cer_synth'), nd=3)} | "
                         f"{cell(d.get('cer_real'), nd=3)} | {cell(d.get('cer_gap'), nd=3)} |")
    lines += [
        "",
        "MCD is implementation-dependent -- compare against other runs of this "
        "script, not against published figures. MOS and SUS need human listeners; "
        "build the panel with `listening_test.py`.",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\nwrote {out/'metrics.json'} and {out/'report.md'}")
    print(f"synth audio in {out/'synth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
