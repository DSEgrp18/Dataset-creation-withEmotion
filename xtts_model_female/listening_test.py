#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
listening_test.py -- build the MOS + SUS panel kit that evaluate_xtts.py cannot.

MOS and SUS are the two metrics the Sinhala TTS literature actually compares on,
and both require human listeners. There is no automatic substitute. This script
synthesises the stimuli and writes ONE self-contained HTML file you send to
native-speaker raters; they open it in a browser, listen, rate, and download a
CSV. No server, no internet, audio embedded as data URIs.

PROTOCOL -- matched to SPECOM 2025 so the numbers are comparable
---------------------------------------------------------------
MOS   15 sentences drawn from the held-out eval split, five each of short,
      medium and long. Two 5-point Likert scales per clip: intelligibility and
      naturalness. Raters never see the text -- showing it makes intelligibility
      unmeasurable.

SUS   10 Semantically Unpredictable Sentences. The rater transcribes what they
      hear; intelligibility is word accuracy. SUS exists because meaningful
      sentences let a listener reconstruct words they did not actually hear, so
      MOS-style intelligibility is always optimistic.

      These are built by interleaving the words of two real corpus sentences of
      equal length. That preserves inflection and syntactic shape while
      destroying meaning, which is the standard construction. It is mechanical,
      so HAVE A NATIVE SPEAKER READ THE LIST before you run the panel and drop
      any that came out ungrammatical -- an ungrammatical SUS item measures
      nothing.

SAMPLE SIZE
      SPECOM 2025 used 12 listeners (6M, 6F); TacoSi used 10; Nanayakkara et al.
      used 30, split across visually impaired and sighted groups and reported
      them separately -- which found a 4-point intelligibility gap between the
      groups and is the most interesting result in that paper. Aim for 12+, and
      if any of your raters are blind, report them as their own group.

USAGE
    python listening_test.py --run ./run/training/GPT_XTTS_si_female-<stamp> \
        --base ./run/training/XTTS_v2.0_original_model_files \
        --dataset ./female_dataset --out ./listening_test
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Pin to one GPU, matching train_xtts_female.py.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / "xtts_sinhala"):
    if (_cand / "sinhala_text.py").is_file():
        sys.path.insert(0, str(_cand))
        break
from sinhala_text import sinhala_to_ascii  # noqa: E402


def build_sus(items: list[dict], n: int, seed: int) -> list[dict]:
    """Interleave word-for-word between two real sentences of the same length."""
    rng = random.Random(seed)
    by_len = defaultdict(list)
    for it in items:
        w = it["sinhala"].split()
        if 5 <= len(w) <= 10:
            by_len[len(w)].append(w)
    out: list[dict] = []
    lengths = [L for L, v in by_len.items() if len(v) >= 2]
    rng.shuffle(lengths)
    for L in lengths:
        pool = by_len[L][:]
        rng.shuffle(pool)
        for i in range(0, len(pool) - 1, 2):
            a, b = pool[i], pool[i + 1]
            mixed = [(a[j] if j % 2 == 0 else b[j]) for j in range(L)]
            sent = " ".join(mixed)
            if not sent.rstrip().endswith("."):
                sent = sent.rstrip(" .,") + "."
            out.append({"sinhala": sent, "ascii": sinhala_to_ascii(sent),
                        "words": len(mixed)})
            if len(out) >= n:
                return out
    return out


def b64_wav(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


PAGE = """<!doctype html>
<html lang="si"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sinhala TTS listening test</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+Sinhala:wght@400;600&display=swap">
<style>
 :root{--ink:#14202a;--muted:#5b7080;--rule:#d3dde4;--bg:#eef2f5;--card:#fff;--acc:#0c6d74}
 @media(prefers-color-scheme:dark){:root{--ink:#e4ebf0;--muted:#8ea3b0;--rule:#25333c;--bg:#0d151a;--card:#151f26;--acc:#45b2b4}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
 .si{font-family:"Noto Serif Sinhala","Iskoola Pota",serif}
 .wrap{max-width:52rem;margin:0 auto;padding:2rem 1.2rem 5rem}
 h1{font-size:1.7rem;margin:0 0 .4rem}
 h2{font-size:1.25rem;margin:2.5rem 0 .3rem;border-bottom:2px solid var(--ink);padding-bottom:.4rem}
 .lede{color:var(--muted);margin:0 0 1.4rem}
 .card{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:1rem 1.1rem;margin-bottom:1rem}
 .n{font:700 .78rem/1 ui-monospace,monospace;color:var(--acc);letter-spacing:.08em}
 audio{width:100%;margin:.7rem 0}
 label{display:block;font-size:.85rem;color:var(--muted);margin:.7rem 0 .25rem}
 .scale{display:flex;flex-wrap:wrap;gap:.4rem}
 .scale label{display:inline-flex;align-items:center;gap:.3rem;margin:0;font-size:.9rem;color:var(--ink);
   border:1px solid var(--rule);border-radius:4px;padding:.35rem .6rem;cursor:pointer}
 .scale input{margin:0}
 .scale label:focus-within{outline:2px solid var(--acc)}
 input[type=text]{width:100%;padding:.6rem;font-size:1rem;border:1px solid var(--rule);
   border-radius:4px;background:var(--bg);color:var(--ink);font-family:"Noto Serif Sinhala",serif}
 .meta{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:1rem 1.1rem;margin-bottom:1.5rem}
 .meta input,.meta select{padding:.45rem;border:1px solid var(--rule);border-radius:4px;
   background:var(--bg);color:var(--ink);font-size:.95rem;margin-right:.6rem}
 button{background:var(--acc);color:#fff;border:0;border-radius:5px;padding:.8rem 1.4rem;
   font-size:1rem;font-weight:600;cursor:pointer}
 button:hover{filter:brightness(1.1)}
 .bar{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--rule);
   padding:.8rem 1.2rem;display:flex;gap:1rem;align-items:center;justify-content:center}
 .warn{font-size:.85rem;color:var(--muted)}
 textarea{width:100%;height:8rem;font:12px/1.5 ui-monospace,monospace;margin-top:.8rem;
   border:1px solid var(--rule);border-radius:4px;background:var(--bg);color:var(--ink);padding:.6rem}
</style></head><body><div class="wrap">
<h1>Sinhala TTS listening test</h1>
<p class="lede">About 15 minutes. Use headphones in a quiet room. Play each clip as many
times as you like, but <b>do not skip any</b>. There are no right answers in Part&nbsp;1.</p>

<div class="meta">
  <label for="rater">Your name or initials</label>
  <input id="rater" type="text" style="width:14rem" placeholder="required">
  <label for="group">Are you blind or visually impaired?</label>
  <select id="group">
    <option value="sighted">No</option>
    <option value="visually_impaired">Yes</option>
  </select>
  <label for="native">Is Sinhala your first language?</label>
  <select id="native"><option value="yes">Yes</option><option value="no">No</option></select>
</div>

<h2>Part 1 &mdash; quality rating (MOS)</h2>
<p class="lede">For each clip give two ratings. <b>Intelligibility</b>: how easily could you
understand the words? <b>Naturalness</b>: how close to a real human voice did it sound?
A clip can be perfectly clear and still sound robotic &mdash; rate them independently.</p>
__MOS__

<h2>Part 2 &mdash; transcription (SUS)</h2>
<p class="lede">These sentences are grammatical but meaningless, so you cannot guess the
words from the sense &mdash; that is the point. Type exactly what you hear, in Sinhala.
If a word is unclear, write your best guess or leave a dash.</p>
__SUS__

<div class="bar">
  <button id="save">Download my answers</button>
  <span class="warn" id="status">Nothing saved yet</span>
</div>
<textarea id="fallback" hidden readonly></textarea>
</div>
<script>
const MOS_N=__MOSN__, SUS_N=__SUSN__;
document.getElementById('save').addEventListener('click',()=>{
  const rater=document.getElementById('rater').value.trim();
  if(!rater){alert('Please enter your name or initials first.');return;}
  const rows=[['rater','group','native','part','item','clip_id','intelligibility','naturalness','transcript']];
  const group=document.getElementById('group').value, native=document.getElementById('native').value;
  let missing=0;
  for(let i=0;i<MOS_N;i++){
    const int_=document.querySelector(`input[name=mos_int_${i}]:checked`);
    const nat=document.querySelector(`input[name=mos_nat_${i}]:checked`);
    if(!int_||!nat) missing++;
    rows.push([rater,group,native,'MOS',i,document.getElementById('mos_id_'+i).textContent,
               int_?int_.value:'',nat?nat.value:'','']);
  }
  for(let i=0;i<SUS_N;i++){
    const t=document.getElementById('sus_'+i).value.trim();
    if(!t) missing++;
    rows.push([rater,group,native,'SUS',i,document.getElementById('sus_id_'+i).textContent,'','',t]);
  }
  if(missing && !confirm(missing+' answers are blank. Download anyway?')) return;
  const csv=rows.map(r=>r.map(c=>'"'+String(c).replace(/"/g,'""')+'"').join(',')).join('\\n');
  const fb=document.getElementById('fallback');
  fb.value=csv; fb.hidden=false;
  try{
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    a.download='listening_'+rater.replace(/[^A-Za-z0-9]/g,'_')+'.csv';
    document.body.appendChild(a); a.click(); a.remove();
    document.getElementById('status').textContent='Saved. Send the CSV back.';
  }catch(e){
    document.getElementById('status').textContent='Download blocked \\u2014 copy the text box below instead.';
  }
});
</script></body></html>
"""


def scale(name: str, low: str, high: str) -> str:
    opts = "".join(
        f'<label><input type="radio" name="{name}" value="{v}"> {v}</label>'
        for v in range(1, 6))
    return (f'<div class="scale">{opts}</div>'
            f'<div class="warn">1 = {low} &nbsp;&middot;&nbsp; 5 = {high}</div>')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="./listening_test")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mos-n", type=int, default=15)
    ap.add_argument("--sus-n", type=int, default=10)
    ap.add_argument("--speaker", default=None,
                    help="restrict to one speaker (default: balanced across all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    import torchaudio
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    run, base = Path(args.run).resolve(), Path(args.base).resolve()
    dataset = Path(args.dataset).resolve()
    out = Path(args.out).resolve()
    (out / "clips").mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint) if args.checkpoint else (run / "best_model.pth")
    if not ckpt.is_file():
        cks = sorted(run.glob("checkpoint_*.pth"),
                     key=lambda p: int(p.stem.split("_")[-1]))
        if not cks:
            print(f"no checkpoint under {run}", file=sys.stderr)
            return 2
        ckpt = cks[-1]
    print(f"checkpoint : {ckpt}")

    items = json.loads((dataset / "eval_reference.json").read_text(encoding="utf-8"))
    if args.speaker:
        items = [i for i in items if i["speaker"] == args.speaker]
    by_spk = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it)

    # MOS stimuli: five short, five medium, five long, balanced across speakers
    ranked = sorted(items, key=lambda x: len(x["sinhala"].split()))
    third = max(1, len(ranked) // 3)
    buckets = [ranked[:third], ranked[third:2 * third], ranked[2 * third:]]
    per_bucket = max(1, args.mos_n // 3)
    mos_items: list[dict] = []
    for b in buckets:
        mos_items += rng.sample(b, min(per_bucket, len(b)))
    mos_items = mos_items[: args.mos_n]
    rng.shuffle(mos_items)

    sus_items = build_sus(items, args.sus_n, args.seed)
    print(f"MOS stimuli: {len(mos_items)}   SUS stimuli: {len(sus_items)}")

    config = XttsConfig()
    config.load_json(str(base / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=str(ckpt),
                          vocab_path=str(base / "vocab.json"), use_deepspeed=False)
    if torch.cuda.is_available():
        model.cuda()

    def synth(text_ascii: str, speaker: str, name: str) -> Path:
        cond = dataset / rng.choice(by_spk[speaker])["wav"]
        lat, emb = model.get_conditioning_latents(
            audio_path=[str(cond)], gpt_cond_len=config.gpt_cond_len,
            max_ref_length=config.max_ref_len, sound_norm_refs=config.sound_norm_refs)
        res = model.inference(text=text_ascii, language="en", gpt_cond_latent=lat,
                              speaker_embedding=emb, temperature=0.75,
                              length_penalty=1.0, repetition_penalty=5.0,
                              top_k=50, top_p=0.85, enable_text_splitting=False)
        p = out / "clips" / f"{name}.wav"
        torchaudio.save(str(p), torch.tensor(res["wav"]).unsqueeze(0), 24000)
        return p

    speakers = sorted(by_spk)
    mos_html, key = [], []
    print("synthesising MOS stimuli")
    for i, it in enumerate(mos_items):
        p = synth(it["ascii"], it["speaker"], f"mos_{i:02d}")
        key.append({"part": "MOS", "item": i, "clip_id": it["clip_id"],
                    "speaker": it["speaker"], "sinhala": it["sinhala"]})
        mos_html.append(
            f'<div class="card"><span class="n">CLIP {i+1} OF {len(mos_items)}</span>'
            f'<span id="mos_id_{i}" hidden>{html.escape(it["clip_id"])}</span>'
            f'<audio controls preload="none" src="{b64_wav(p)}"></audio>'
            f'<label>Intelligibility &mdash; how easily could you understand it?</label>'
            f'{scale(f"mos_int_{i}", "could not understand", "completely clear")}'
            f'<label>Naturalness &mdash; how human did it sound?</label>'
            f'{scale(f"mos_nat_{i}", "clearly a machine", "indistinguishable from a person")}'
            f'</div>')

    sus_html = []
    print("synthesising SUS stimuli")
    for i, it in enumerate(sus_items):
        spk = speakers[i % len(speakers)]
        p = synth(it["ascii"], spk, f"sus_{i:02d}")
        key.append({"part": "SUS", "item": i, "clip_id": f"sus_{i:02d}",
                    "speaker": spk, "sinhala": it["sinhala"], "words": it["words"]})
        sus_html.append(
            f'<div class="card"><span class="n">SENTENCE {i+1} OF {len(sus_items)}</span>'
            f'<span id="sus_id_{i}" hidden>sus_{i:02d}</span>'
            f'<audio controls preload="none" src="{b64_wav(p)}"></audio>'
            f'<label>Type exactly what you heard</label>'
            f'<input id="sus_{i}" type="text" class="si" autocomplete="off"></div>')

    page = (PAGE.replace("__MOS__", "\n".join(mos_html))
                .replace("__SUS__", "\n".join(sus_html))
                .replace("__MOSN__", str(len(mos_items)))
                .replace("__SUSN__", str(len(sus_items))))
    html_path = out / "listening_test.html"
    html_path.write_text(page, encoding="utf-8")
    (out / "answer_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    mb = html_path.stat().st_size / 1e6
    print(f"\nwrote {html_path}  ({mb:.1f} MB, audio embedded)")
    print(f"      {out/'answer_key.json'}  -- SUS ground truth, do NOT send to raters")
    print("\nNext:")
    print("  1. Have a native speaker check answer_key.json for ungrammatical SUS items.")
    print("  2. Send the HTML to 12+ native listeners. Collect their CSVs.")
    print("  3. python score_listening.py --key answer_key.json --responses *.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
