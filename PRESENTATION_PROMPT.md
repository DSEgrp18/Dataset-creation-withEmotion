You are helping me build a technical presentation for tomorrow. I will give you the
full context of my project below. Produce a **slide-by-slide deck outline** (title +
bullet points + speaker notes for each slide), aimed at a university/technical audience
with mixed ML background. Target ~15-18 slides for a 20-minute talk. Lead with the
problem, make the engineering decisions the star of the talk, and be honest about
limitations. Use the real numbers I give you — do not invent any.

===============================================================================
PROJECT: Emotional Sinhala Speech Dataset + Sinhala Text-to-Speech
===============================================================================

## 1. The problem

Sinhala is a low-resource language for speech technology:
- **No public Sinhala emotional-speech dataset exists.** Every available corpus is
  neutral read speech.
- No Sinhala screen reader exists. NVDA, JAWS, VoiceOver, Windows SAPI and the Web
  Speech API ship **zero** Sinhala voices. Sinhala speakers with visual impairment
  have no assistive reading option in their own language.
- Existing corpus: PathNirvana, ~11.8–13.6 h, a single male speaker, studio read
  speech. Good for intelligibility, useless for expressiveness.

Our end goal: an expressive Sinhala TTS voice, and downstream, an offline Sinhala
PDF reader for blind users.

## 2. Approach in one line

Build the missing emotional corpus from **SLBC radio dramas** on the Internet Archive
(naturally expressive — shouting, crying, whispering, arguing), then fine-tune
**XTTS-v2** for Sinhala across three progressively-corrected training tracks.

## 3. Dataset pipeline architecture

Source: archive.org SLBC radio drama collection — **201 episodes, 82.0 hours**.

```
archive.org  ->  download episode (mp3)
                      |
             5-minute chunks, FLAC-encoded payloads
                      |
        +-------------+--------------+
        |                            |
   Gemini API                   Silero VAD
   WHAT was said,               EXACTLY where speech
   roughly WHERE                starts and stops
   (sentence text +             (acoustic evidence)
    rough timestamps)
        |                            |
        +------------+---------------+
                     |
            OWNERSHIP SNAPPING
   each speech region awarded to exactly ONE sentence
   (the one whose rough span overlaps it most)
                     |
      monotonic assignment + neighbour-aware padding
                     |
  manifest.csv + clips/*.wav (24kHz mono PCM16) + review.html
```

### The central technical insight
An LLM's timestamps **drift** — trust them and every clip starts mid-syllable. A VAD
cannot find **sentences** — it hears pauses, not grammar, so one speech region may
hold three sentences or half of one. So each source is used only for what it is
reliable at, and the two are fused by region ownership.

Because speech regions are **disjoint and singly-owned**, clips *cannot* duplicate
audio. This is asserted at the end of every run (clip-minutes vs source-minutes).

### Two refinements that turned out to be necessary
- **Monotonic assignment.** Over a 5-minute chunk the model's absolute timestamps
  drift by seconds, but the *order* of sentences stays exactly right. Regions are
  walked in time order and the sentence pointer never moves backwards — so a drifted
  sentence cannot claim a distant region and splice two speakers together.
- **Neighbour-aware padding.** Clips get 0.1 s of breathing room, bounded by the
  previous clip's *padded* end — otherwise two pads meet in the middle and reintroduce
  the overlap that ownership just eliminated.

Sentences that win no region keep the model's raw times, are flagged `snapped=False`
(6.8%), and are surfaced in `review.html` for human QC rather than silently discarded.

## 4. Results on the built episode (Muwan Palassa 140113, 25.3 min source)

| Metric | Value |
|---|---|
| Clips | 192 |
| Usable audio | 16.2 min = **64.2% of source** |
| Clip length | median 4.3 s, mean 5.1 s, range 1.1–15.0 s |
| Transcript | 1,691 words, mean 8.8 words/clip |
| Timestamps verified against audio | **93.2%** (179/192) |
| **Overlapping clips** | **0** |
| Projected yield over 201 episodes | **~52 h of clean speech** |

**64% coverage is the headline.** The original Whisper-based design predicted under
10–20% usable yield. The remainder is music, sound effects and silence.
**0 overlaps matters more than it looks** — overlapping clips ship the same audio
repeatedly under different transcripts and quietly poison a TTS training set.

## 5. Why a hosted LLM instead of an open-weight ASR model

Every Sinhala ASR checkpoint on HuggingFace was benchmarked on this audio. They all
failed — **not because of model choice, because of domain mismatch.** They are
fine-tuned on clean single-speaker *read* speech; this is lossy 2013 broadcast with
music, speaker overlap and theatrical delivery.

Same clip, same audio:
- `hlasith/whisper-sinhala-small` -> "එක්යන්මෙතාමතේ රස්නේ ඉවරවිලන් නෑ"  (word salad)
- **Gemini** -> "ඒ කියන්නේ තාමත් ඒ ප්‍රශ්නය ඉවර වෙලා නෑ."  (grammatical, correct
  conjuncts and punctuation)

Ruled out against the **model files**, not the model cards:
- SeamlessM4T-v2 — no Sinhala (`sin` absent from its 98 output languages)
- MMS-1B-all — no Sinhala adapter (`sin` absent from all vocab files)
- Whisper large-v3 — supports `si`, but it is one of its thinnest languages

Google Chirp 2 and Azure do list `si-LK`, but they are paid; the free tier removed
the need.

## 6. The real engineering constraint: API quota, not compute

The bottleneck was **requests per day**, not GPU or bandwidth:
`GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20`

| | |
|---|---|
| Requests needed for all 201 episodes | 991 |
| Free-tier allowance | ~100/day (5 models x 20) |
| Sessions needed | ~10 days |

Three mitigations made it survivable:
- **5-minute chunks + FLAC payloads** — an episode costs 5 requests, not 25
- **Per-model failover** — quota is per model, so exhausting one moves to the next
- **Automatic waiting** — when all models are spent, sleep 20 min and retry. It does
  not try to predict Google's reset time; a successful call is the proof quota
  returned. An 11-hour session sits through a daily reset unattended.

Kaggle is used **only for its network** (a 25-min episode downloads in ~1 min there
vs ~50 min on a home connection; accelerator: None, because the job is network-bound).
Kaggle wipes `/kaggle/working`, so each run is handed to the next via Save & Run All
and re-attaching the previous output; finished episodes are adopted and skipped free.

## 7. Output format (the deliverable is the manifest, not the wavs)

```
dataset/<episode>/
  manifest.csv         <- the deliverable
  clips/<clip_id>.wav  <- 24 kHz, mono, PCM 16-bit
  review.html          <- audio players beside text, for QC
  .chunks/             <- resume cache (raw API output)
```

`clip_id, source_file, start_sec, end_sec, duration, text, snapped`

Timestamps are **absolute seconds in the source**, so the corpus regenerates from
archive.org without redistributing audio. CSVs are UTF-8-BOM so Excel renders Sinhala.

Audio is deliberately **not loudness-normalised** — flattening a shouted line and a
whispered one to the same level would destroy the exact emotional dynamics the
dataset exists to capture.

## 8. TTS training: three tracks, and what each one taught us

### Track A - `xtts_baseline`: fine-tune XTTS-v2 on our radio-drama corpus
Outcome: **it failed, and the failure was diagnostic.** Two reasons:
1. XTTS-v2's tokenizer contains no Sinhala codepoint (see next section).
2. Radio drama is multi-speaker and nothing diarised it, so every clip carried one
   speaker label. XTTS samples a conditioning clip **from the same speaker** on every
   training step — so a single pooled label teaches the model that the reference clip
   does *not* predict the output voice, i.e. the exact opposite of the capability
   being fine-tuned.

Also discovered: XTTS **silently discards clips over 11.6 s**
(`GPTArgs.max_wav_length = 255995` @ 22050 Hz). No warning is raised.
On a measured episode: 141 clips / 16.0 min as built -> 129 clips / 13.4 min usable.
**~16% of the audio was being wasted invisibly.**

### The tokenizer discovery — the most important finding in the project
Checked against the real `vocab.json`:
```
pre_tokenizer : Whitespace
model         : BPE
unk_token     : "[UNK]"
vocab size    : 6681
```
It is a **whitespace-pretokenised BPE with an `[UNK]` fallback**, not a byte-level
BPE. When a word contains any out-of-vocabulary character, **the entire word** becomes
one `[UNK]` token — not the character, the whole word. Sinhala (U+0D80–U+0DFF) is
absent, so every word became `[UNK]` and the model was trained to map
"unknown unknown unknown" onto audio.

**The loss falls convincingly the whole way down and the samples are babble.** This is
the trap: your training curves look healthy while the model learns nothing.

The obvious fix — `extend_vocab.py`, adding ~80 Sinhala rows to the token embedding
matrix — was the right diagnosis but the wrong remedy at our scale: those rows start
randomly initialised, and published new-language XTTS ports that take that route
(viXTTS et al.) use hundreds of hours and tens of thousands of optimiser steps. Our
run had ~1.5 h of audio and roughly 100 optimiser steps.

(VITS did better on Sinhala precisely because VITS trains a character embedding table
from scratch, so Sinhala script costs it nothing.)

### Track B - `xtts_sinhala`: spell Sinhala with characters XTTS already knows
Instead of teaching the model a new script, **romanise into ASCII** using Sri Lankan
"Singlish" conventions chosen so pretrained English tokens already carry roughly the
right sound:

| Sinhala | | ASCII | why |
|---|---|---|---|
| ට ඩ | retroflex | t d | English t/d are the nearest match |
| ත ද | dental | th dh | the h-digraph, as Sri Lankans write it |
| ඛ ඝ ඡ ඨ ථ ධ ඵ භ | aspirated | dropped | modern spoken Sinhala has no phonemic aspiration |
| ා ී ූ ේ ෝ | long vowels | aa ii uu ee oo | length written as doubling |
| ණ ළ | | n l | homophones of න and ල |
| ං ඞ ඟ | | ng | |
| ශ ෂ | | sh | English "sh" *is* /ʃ/ |

Measured over all 6,386 corpus lines:
```
output charset : " !',-.:;?abcdefghijklmnoprstuvy"   pure ASCII
[UNK] tokens   : 0  out of  380,730
tokens/line    : median 59, p95 96, max 117   (GPTArgs limit is 200)
```
Nothing is randomly initialised, no embedding matrix is resized, and the task drops
from **"learn a new script"** to **"learn a new accent"** — which ~8 h of audio can
actually do. The Sinhala-script-to-ASCII path agrees with the corpus romanisation
on **96.6%** of lines.

Data: PathNirvana `mettananda` split — one voice, studio, already 22050 Hz mono,
~5,400 clips / ~11.8 h, so speaker conditioning finally means something.

### Track C - `xtts_model_female`: the current run
Data: **VoiceMakers SinhalaTTS** — Dinithi (4.82 h) + Harini (2.14 h), ~7 h across two
genuinely distinct female speakers, **correctly labelled per speaker**.

Motivation: SPECOM 2025 trained VITS on PathNirvana and its female voice was the
weakest configuration by a wide margin — **MCD 20.56 dB vs 13.27 for male** — which the
authors attribute to insufficient female audio (~2 h). We have ~7 h of it.

## 9. Robustness engineering (this is where most of the real work went)

**Nothing about the dataset layout is assumed.** The published VoiceMakers folders are
inconsistent (`Isuru-44100Hz` beside `Yasindu-44100`, one speaker directory nested
inside a duplicate of itself). So the prepare step **discovers and prints every
decision**: speaker dirs by name substring recursively; `metadata.csv` by rglob;
delimiter scored across `|` tab `;` `,` by whether the split actually *separates*
script from romanisation; column order found by **content** (Sinhala by codepoint
range, romanised by its diacritics) never by index; header row detected or absent.

Two of those are not hypothetical:
- Delimiter scoring originally used column-count consistency alone and **failed on the
  real corpus** — the files are pipe-delimited, but splitting a pipe-delimited line on
  commas *also* yields a consistent column count when each sentence contains one comma,
  putting Sinhala on both sides of the split.
- **Harini's file has no romanised column at all**, so the Sinhala-to-ASCII
  transliterator handles it instead. Both paths verify at 0 `[UNK]`.

**The silent-corruption guard.** The ASCII fold drops anything outside its keep-set, so
an unmapped diacritic raises nothing — it silently turns `vaṟdanak` into `vadanak`, a
different word, poisoning training with no warning. So the check runs on the **raw**
romanisation against what the map actually covers, and aborts with a list of offending
codepoints. *Caught by a deliberate negative test, not by reasoning.*

**Rows dropped, and why:** duration > 11.6 s (XTTS's silent ceiling), duration < 1.0 s
(too short to condition on), contains a digit (XTTS's `en` cleaner would expand it to
**English** words inside a Sinhala sentence), unmapped character, > 250 characters
(`VoiceBpeTokenizer.char_limits["en"]`).

**The NaN guard.** `mixed_precision=True` (fp16) roughly doubles throughput but drives
`loss_mel_ce` to `nan` on a Kaggle T4 on the first step, and it never recovers: an
8-hour run that writes only NaN checkpoints **and exits with status 0**. Turing has no
bf16, so there is no stable mixed-precision option on that card. A real run burned an
entire session at `nan` before the guard was added. The smoke test (a handful of steps
on 64 clips, ~2 min) now parses the log and raises on any non-finite loss; the long
run repeats the check on its heartbeat and aborts after three consecutive `nan`s.

**And the guard itself has failed in both directions** — once by missing a `nan` run,
once by aborting a perfectly healthy one. The trainer prints epoch averages in colour,
so the raw bytes read `avg_loss_mel_ce:<ESC>[92m 3.66` and a naive regex captured
`[92m` as the value — not a number, looks exactly like divergence. Log parsing now
lives in its own module (`train_log.py`), strips ANSI first, and carries a selftest
built from both real logs.

**GPU pinning.** All scripts set `CUDA_VISIBLE_DEVICES=0` before torch loads —
otherwise the Coqui Trainer refuses to start on Kaggle's "GPU T4 x2":
`RuntimeError: [!] 2 active GPUs. Define the target GPU by CUDA_VISIBLE_DEVICES.`

## 10. Deviations from the official Coqui recipe (and their justifications)

Everything in `GPTArgs`, `XttsAudioConfig` and the optimizer block is copied verbatim
from `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`. Three things differ:

- **`language="en"`** — this selects a *tokenizer branch*, not a claim about the audio.
  `VoiceBpeTokenizer.preprocess_text` raises `NotImplementedError` outside its
  17-language set and `[si]` is not a token in `vocab.json`. The text is already ASCII
  by then, so the `en` cleaner path is exactly right.
- **Effective batch 64, not 252.** Upstream's advice is correct for a datacentre. On
  one T4, an effective batch of 252 is ~100 s per optimiser step, so a full session
  buys ~400 steps — nowhere near enough to move the model onto a new sound inventory.
  `4 x 16` trades gradient noise for roughly 4x the steps.
- **`lr=1e-5`, not `5e-6`**, compensating for the smaller batch.

One thing that *looks* like a deviation but is not: the scheduler milestones are
`[900000, 2700000, 5400000]` and `Trainer` defaults to `scheduler_after_epoch=True`,
so those are **epochs** and the learning rate never decays. That is upstream behaviour.

## 11. Evaluation framework

### Objective, unattended (`evaluate_xtts.py`)
| Metric | What it catches |
|---|---|
| **MCD (dB)**, DTW-aligned | spectral distance to the real recording; the only objective metric Sinhala TTS papers report |
| **log-F0 RMSE (cents)** | pitch accuracy over aligned voiced frames |
| **F0 correlation** | whether the intonation *contour* tracks the reference — a flat monotone can score decent RMSE and still sound dead |
| **SECS** | speaker-encoder cosine similarity, measured against a **held-out** clip, never the conditioning clip (which would inflate it) |
| **Duration ratio + failure rate** | XTTS is autoregressive; truncation and runaway looping show here before you hear them |
| **RTF** | synthesis speed |
| `--utmos` | learned MOS predictor, English-trained — a relative signal between checkpoints, not an absolute MOS |
| `--asr` | intelligibility proxy: Sinhala ASR is weak, so we transcribe the **real** held-out audio through the same model and report the **gap**, controlling for the ASR's own error rate |

The MCD implementation was unit-tested on synthetic signals: identical inputs give
exactly 0.00 dB; a small perturbation ~13.5 dB; monotone in spectral distance. F0 RMSE
recovers a known 200->260 Hz shift as 456.5 cents against a true 454.2.
**MCD is implementation-dependent — compare runs of this script to each other, never
to a published figure.**

### Subjective, human (`listening_test.py`)
MOS and SUS are what the literature actually compares on, and there is no automatic
substitute. Claiming one is how papers end up reporting "98% accuracy" for a
synthesiser, which is not a meaningful synthesis metric at all.

The script synthesises the stimuli and writes **one self-contained HTML file** — audio
embedded, no server, no internet. Raters open it in a browser, listen, rate, download
a CSV. Protocol matches SPECOM 2025 for comparability: 15 MOS sentences (5 short,
5 medium, 5 long) rated on separate 5-point scales for intelligibility and
naturalness, plus 10 SUS sentences transcribed by ear.

- **SUS sentences are generated mechanically**, by interleaving the words of two real
  corpus sentences of equal length — preserving inflection while destroying meaning,
  the standard construction. A native speaker must vet them first; an ungrammatical
  SUS item measures nothing.
- **Blind and sighted raters are reported separately.** Nanayakkara et al. found
  visually impaired listeners scored the same system 66% where sighted listeners scored
  ~70%, and argued the sighted group was simply less practised at synthetic speech.
  Pooling hides that — and for this project the blind listeners are the ones who matter.

## 12. Downstream target: offline Sinhala PDF reader for blind users

One Spring Boot application, two profiles; the desktop app **is** the backend on
localhost.

```
        shared frontend (TypeScript + pdf.js, no framework)
     reader view . voice command bar . TTS playground . a11y layer
                |                              |
     DESKTOP profile                    CLOUD profile
     jpackage installer + tray          Spring Boot on a VPS
     LocalTtsEngine (espeak/ONNX)       RemoteTtsEngine --HTTP--> voice-service
     EnrolledCommandRecognizer          installer download        (FastAPI, GPU)
                |
     reader-core (Java, no Spring):
     PDFBox . legacy-font fix . reading order . segmentation
```

Key decisions:
- **The desktop app opens the user's default browser; it does not embed a webview.**
  Blind users already have NVDA/JAWS configured for their browser — a real browser
  gives a real accessibility tree, working `MediaRecorder` and ARIA live regions that
  actually announce. Consequence: the frontend is byte-identical across desktop and
  web; only the active `TtsEngine` bean differs.
- **espeak-ng is the permanent fallback tier**, not scaffolding — robotic but genuinely
  intelligible Sinhala, ~5 MB, offline, all platforms. With no neural model downloaded
  the app still reads books. That is what makes first run work before a 60 MB download.
- **Piper = VITS exported to ONNX, phonemised by espeak-ng**, so the placeholder and
  the real model share the same text front-end; only the synthesis back half swaps.
- **XTTS cannot run offline** (~2 GB, far slower than realtime on CPU) — so VITS/Piper
  targets the desktop, and XTTS is website-only, expressive mode, GPU.

Two traps handled explicitly:
1. **Legacy FM-Abhaya fonts.** A large share of Sinhala PDFs map Sinhala glyphs onto
   ASCII `0x20-0x7E`. The font *draws* Sinhala; the bytes *say* `Bugjkq`. PDFBox
   extracts that ASCII faithfully, no error is raised anywhere, and the reader speaks
   gibberish. For a blind user who cannot see it is garbage, this is the worst possible
   failure. Detect by font name + ">90% printable ASCII in a document with no Latin
   content", convert with per-font mapping tables, and **announce** the conversion.
   Never read undecoded text aloud.
2. **No Sinhala speech recognition exists** (Vosk has none; Whisper and Google Web
   Speech are unreliable) — yet voice commands are the primary interface by user
   requirement. Solution: **speaker-enrolled keyword spotting**. The user says each of
   ~12 commands three times on first run; store MFCC templates; recognise by DTW
   distance with a rejection threshold. This is *better* than a generic model here, not
   a compromise: one user, one device, a closed vocabulary — speaker-adapted by
   construction, no training corpus, no GPU, fully offline, ~300 lines, and it
   sidesteps dialect variation. Every command also has a keyboard equivalent, always:
   recognition fails, and a blind user must never be trapped with no way out.

## 13. Limitations, stated honestly

- 7 h across two speakers is enough for intelligible, correctly-accented Sinhala in the
  trained voices, and enough to judge whether the approach works. It is **not** enough
  for reliable zero-shot cloning of unseen speakers — that capability comes from the
  pretrained model and this fine-tune erodes it somewhat.
- Expect artefacts on words whose letter sequences never appeared in training.
- The radio-drama emotional corpus is **not yet diarised**, which is the blocker for
  using it in XTTS. Diarisation belongs upstream in the dataset build; retrofitting it
  across a finished corpus is far more expensive than adding it now.
- The romanisation fold is lossy by design (aspiration dropped, ණ/න and ළ/ල collapsed).
  It matches modern spoken Sinhala phonology, but it cannot round-trip orthography.
- "Perfect" is not on the table from 7 hours. Clearly better than the previous attempt
  is.

## 14. Roadmap

1. Finish the 201-episode dataset build (~10 quota sessions, ~52 h projected).
2. Diarise the radio-drama corpus so each actor gets a real speaker label.
3. Append it to PathNirvana with the same ASCII fold — PathNirvana holds the phonetics
   steady while the smaller emotional set teaches prosody.
4. Train VITS for the offline Piper/ONNX desktop voice; keep XTTS for the expressive
   web tier.
5. Run the MOS/SUS panel with blind and sighted raters reported separately.
6. Accessibility hardening with actual blind users — a requirement, not a nice-to-have.

===============================================================================
WHAT I WANT FROM YOU
===============================================================================

Produce the deck outline with:
1. A title slide and a one-sentence thesis for the whole talk.
2. Slides in a narrative arc: problem -> why existing tools fail -> our dataset
   approach -> the LLM+VAD fusion insight -> results -> the tokenizer discovery ->
   the three training tracks -> evaluation -> downstream application -> limitations
   -> roadmap.
3. For each slide: a headline, 3-5 tight bullets, and 2-3 sentences of speaker notes.
4. Flag which 3 slides are the "wow" moments and should get the most stage time.
5. Suggest one diagram per major section and describe what it should show.
6. Anticipate 6 likely audience questions with short answers — especially:
   "why not just use Whisper?", "isn't using a proprietary API a weakness?",
   "how do you know the transcripts are correct?", and "why romanise instead of
   extending the vocabulary?"
