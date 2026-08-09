# CLAUDE.md

> Copy this file to the new repository's root as `CLAUDE.md`.

## What this project is

A **voice-driven, fully offline Sinhala PDF reader for blind users**, shipped as a
desktop installer, plus a **website** that does Sinhala text-to-speech in the browser
and serves the installer.

No Sinhala screen reader exists. NVDA, JAWS, VoiceOver, Windows SAPI and the Web
Speech API all ship **zero** Sinhala voices. That gap is the entire reason for this
project — do not propose "just use the system TTS", it does not exist for this language.

The Sinhala neural voice is being trained separately (VITS, on the
[pathnirvana corpus](https://github.com/pathnirvana/sinhala-tts-dataset), ~11.8 h single
male speaker). **It is not finished.** Everything here is built against a speech
interface with a working placeholder behind it, so the app is demoable and
accessibility-testable today and the trained model drops in later as a file swap.

---

## Hard-won facts — verified, do not re-derive

These were each checked against primary sources. Re-litigating them wastes time and
usually lands on the wrong answer.

| Fact | Consequence |
|---|---|
| **espeak-ng supports Sinhala** (`si`, Indic family) | This is the placeholder TTS *and* the permanent fallback tier. Robotic but genuinely intelligible Sinhala, ~5 MB, offline, all platforms. |
| **Piper = VITS exported to ONNX, phonemized by espeak-ng** | The placeholder and the real model share the same text front-end. Only the synthesis back half swaps. `piper-jni` is a real Java JNI binding, used in production by openHAB. |
| **XTTS cannot run offline** | ~2 GB, far slower than realtime on CPU. Unusable as a book reader. Target **VITS/Piper** for the desktop; XTTS is website-only, expressive mode, GPU. |
| **Vosk has no Sinhala model** | Checked its full 40+ language list. Neither Whisper nor Google Web Speech handles Sinhala reliably. Voice commands cannot use any off-the-shelf recogniser — see the keyword-spotting section. |
| **Most Sinhala PDFs are not Unicode** | Legacy fonts (FM Abhaya, DL-Manel, Malithi) map Sinhala glyphs onto ASCII `0x20–0x7E`. See the trap section — this is the #1 failure mode. |
| **XTTS's tokenizer is a whitespace-pretokenised BPE with `[UNK]`** | Irrelevant here, but it is why XTTS fine-tuning on Sinhala script fails: one unknown codepoint turns the *whole word* into `[UNK]`. Mentioned so nobody re-runs that experiment. |

---

## Architecture

One Spring Boot application, two profiles. **The desktop app *is* the backend**,
running on localhost.

```
              ┌─────────── shared frontend (TypeScript + pdf.js, no framework) ───────────┐
              │   reader view · voice command bar · TTS playground · accessibility layer  │
              └──────────────┬──────────────────────────────────────┬────────────────────┘
                             │ localhost:8731                       │ https://
        ┌────────────────────┴─────────┐            ┌───────────────┴──────────────┐
        │ DESKTOP   profile: desktop   │            │ CLOUD   profile: cloud       │
        │ jpackage installer + tray    │            │ Spring Boot on a VPS         │
        │ LocalTtsEngine (espeak/ONNX) │            │ RemoteTtsEngine ─HTTP─┐      │
        │ EnrolledCommandRecognizer    │            │ installer download    │      │
        └──────────────┬───────────────┘            └───────────────────────┼──────┘
                       │                                                    │
         ┌─────────────┴──────────────────────┐            ┌────────────────┴───────┐
         │ reader-core (Java, no Spring)      │            │ voice-service          │
         │ PDFBox · legacy-font fix ·         │            │ (Python/FastAPI)       │
         │ reading order · segmentation       │            │ VITS + XTTS on GPU     │
         └────────────────────────────────────┘            └────────────────────────┘
```

**The desktop app opens the user's default browser, it does not embed a webview.**
Blind users already have NVDA/JAWS configured for their browser. A real browser gives
a real accessibility tree, working `MediaRecorder`, and ARIA live regions that
actually announce. JavaFX WebView gives none of that reliably. Consequence: the
frontend is byte-identical between desktop and website — only the active `TtsEngine`
bean differs.

### Modules

| Module | Language | Responsibility |
|---|---|---|
| `reader-core` | Java | PDF → ordered, clean Sinhala sentences. **No Spring, no framework deps.** |
| `speech-spi` | Java | `TtsEngine`, `CommandRecognizer`, `Phonemizer` + DTOs. **Zero implementations.** |
| `speech-local` | Java | espeak-ng engine, `piper-jni` ONNX engine, enrolled keyword spotter. |
| `speech-remote` | Java | HTTP client implementing the same SPI against `voice-service`. |
| `app-web` | Spring Boot | REST, session, job queue, static frontend, installer download. |
| `frontend` | TypeScript + pdf.js | Reader UI, voice bar, TTS playground, a11y layer. |
| `voice-service` | Python/FastAPI | GPU inference. Cloud profile only. |
| `packaging` | jpackage | `.msi` / `.deb` / `.dmg` with bundled JRE, espeak-ng, models. |

### The interface everything hangs off

```java
public interface TtsEngine {
    String id();                    // "espeak-si", "piper-si-mettananda"
    boolean isReady();              // model present and loadable
    Quality quality();              // PLACEHOLDER | NEURAL
    PcmAudio synthesize(String sinhalaText, VoiceOptions opts);
    Stream<PcmChunk> stream(String sinhalaText, VoiceOptions opts);  // long books
}
```

`EngineRegistry` selects the highest-quality **ready** engine and falls back down the
chain. With no neural model downloaded the app still reads books, robotically. **That
fallback is permanent, not scaffolding** — it is what makes first run work before a
60 MB download.

Rule: nothing outside `speech-local` / `speech-remote` may reference a concrete
engine. If you find yourself importing `EspeakTtsEngine` in `app-web`, stop.

---

## The two traps

### 1. Legacy FM-Abhaya fonts

A large share of Sinhala PDFs — government documents, older books, most pre-2015
material — use legacy fonts that map Sinhala glyphs onto **ASCII codepoints
0x20–0x7E**. The font *draws* Sinhala; the bytes *say* `Bugjkq`. PDFBox extracts that
ASCII faithfully, no error is raised anywhere, and the reader speaks gibberish.

For a blind user who cannot see that it is garbage, this is the worst possible
failure. Handle it in `reader-core` from day one:

- **Detect**: font resource name matches `FM*` / `DL-*` / `Malithi*`, combined with
  the heuristic *"extracted text is >90 % printable ASCII in a document with no Latin
  content"*.
- **Convert**: per-font-family legacy→Unicode mapping table. Reference sources are the
  UCSC Language Technology Research Laboratory tables and the pitaka.lk rule set.
- **Announce**: the UI must say *"legacy font converted"* or *"this PDF's text could
  not be decoded"*. Never read undecoded text aloud.

### 2. No Sinhala speech recognition exists

Voice commands are the **primary interface** by user requirement, and there is no
off-the-shelf recogniser for Sinhala.

**Solution: speaker-enrolled keyword spotting.** On first run the user says each of
~12 commands three times. Store MFCC templates; recognise by DTW distance with a
rejection threshold.

This is *better* than a generic model here, not a compromise: one user, one device, a
closed vocabulary. Speaker-adapted by construction, no training corpus, no GPU, fully
offline, ~300 lines. It also sidesteps dialect variation. A trained Sinhala KWS model
can implement the same `CommandRecognizer` later if wider coverage is wanted.

Command set:

| Sinhala | Meaning | Sinhala | Meaning |
|---|---|---|---|
| කියවන්න | read | වේගය වැඩි | faster |
| නවත්වන්න | stop | වේගය අඩු | slower |
| ඊළඟ | next | පිටුව | page |
| පෙර | previous | පොත විවෘත කරන්න | open book |
| නැවත | again | සලකුණ | bookmark |
| උදව් | help | නවතන්න | quit |

**Every command must also have a keyboard equivalent, always.** Recognition fails; a
blind user must never be trapped with no way out. Push-to-talk on a held key is the
default capture mode — more reliable than always-on and far less battery-hungry.
Tune the threshold so **false accepts are rarer than misses**: a wrongly-triggered
"quit" is much worse than a missed "next".

---

## Sinhala language reference

Ported from a validated Python implementation (96.6 % agreement against 6,386 paired
romanisation/script lines from the pathnirvana corpus). Use these tables as-is.

**Script range**: `U+0D80`–`U+0DFF`. Detect Sinhala with `'඀' <= c <= '෿'`.

**Structure**: a consonant carries an *inherent* vowel `a` unless followed by the
virama `්` (U+0DCA, kills it) or a vowel sign (replaces it). `ZWJ` (U+200D) is the
conjunct joiner and carries no sound — strip it before processing.

```
VIRAMA   ්  U+0DCA      ANUSVARA  ං  U+0D82
VISARGA  ඃ  U+0D83      ZWJ         U+200D
```

**Consonants** (with a Latin approximation useful for fallback and for tests):

```
ක k   ඛ k   ග g   ඝ g   ඞ ng  ඟ ng
ච ch  ඡ ch  ජ j   ඣ j   ඤ ny  ඥ gn
ට t   ඨ t   ඩ d   ඪ d   ණ n   ඬ nd
ත th  ථ th  ද dh  ධ dh  න n   ඳ ndh
ප p   ඵ p   බ b   භ b   ම m   ඹ mb
ය y   ර r   ල l   ව v   ශ sh  ෂ sh
ස s   හ h   ළ l   ෆ f
```

**Independent vowels**: `අ a  ආ aa  ඇ ae  ඈ aae  ඉ i  ඊ ii  උ u  ඌ uu  ඍ ru  එ e  ඒ ee  ඓ ai  ඔ o  ඕ oo  ඖ au`

**Vowel signs**: `ා aa  ැ ae  ෑ aae  ි i  ී ii  ු u  ූ uu  ෘ ru  ෲ ruu  ෙ e  ේ ee  ෛ ai  ො o  ෝ oo  ෞ au`

Three phonological facts that matter for normalisation and testing:

- **Modern spoken Sinhala has no phonemic aspiration.** ථ and ත are the same sound, ඛ
  and ක are the same sound. Aspirated letters exist in spelling only.
- **ණ/න and ළ/ල are homophones** (`n` and `l`). Retroflex/dental *stops* (ට/ත, ඩ/ද)
  **are** contrastive and must not be collapsed.
- **ඍ / ෘ is pronounced `/ru/`**, not as a vocalic r.

---

## Conventions

- **Java 21**, Gradle multi-module. `reader-core` and `speech-spi` must stay
  framework-free so they can be unit-tested without a Spring context.
- **Frontend: plain TypeScript + Vite, no React/Vue.** Fewer layers means fewer ARIA
  pitfalls, and accessibility is the product here.
- **All user-facing strings in Sinhala**, with English only in code, logs and tests.
- **Every UI action reachable by keyboard**, announced via ARIA live regions.
- Never speak text the pipeline could not verify as decoded Sinhala. Say so instead.
- Comments explain *why*, matching the density of the surrounding file.

---

## Phases and current status

- [ ] **Phase 0 — spikes.** (a) espeak-ng `si` from Java, judged intelligible by a
      native speaker; (b) PDFBox on a Unicode vs an FM-Abhaya PDF; (c) MFCC+DTW
      separates three enrolled Sinhala words at >90 %. **Also: collect ~20 real
      Sinhala PDFs as the test corpus** — every correctness claim in `reader-core`
      rests on it. Exit gate: if a native listener judges espeak-ng Sinhala
      unintelligible, the placeholder strategy must be rethought before Phase 1.
- [ ] **Phase 1 — `reader-core`.** PDFBox extraction with reading-order sorting, font
      inspection, legacy conversion, sentence segmentation, text normalisation
      (digits, dates, abbreviations, embedded English). Emits `List<Utterance>` with
      text, page and bounding boxes for highlighting.
- [ ] **Phase 2 — `speech-spi` + `speech-local`.** Write `PiperTtsEngine` **now** and
      prove it against any published Piper voice, so Phase 5 is a file swap and not a
      debugging session.
- [ ] **Phase 3 — desktop app.** The deliverable that matters most; it can ship
      without the website.
- [ ] **Phase 4 — website.** TTS playground (type Sinhala → play + download), PDF →
      audiobook, installer download.
- [ ] **Phase 5 — model swap-in.** Export the trained VITS to Piper's ONNX layout
      (`.onnx` + `.onnx.json`), drop into the model dir, `EngineRegistry` promotes it
      automatically.
- [ ] **Phase 6 — accessibility hardening.** NVDA/JAWS/Orca, keyboard-only traversal,
      **and testing with actual blind users** — a requirement, not a nice-to-have.

---

## Verification

| What | How |
|---|---|
| `reader-core` | JUnit over the Phase 0 corpus: Unicode, FM-Abhaya, multi-column, scanned. Hand-transcribe at least five and assert exact text. |
| Legacy font conversion | Round-trip Unicode → FM-Abhaya bytes → converter → assert identity. |
| Segmentation | Run the Java segmenter and the reference Python one over the same 6,386 pathnirvana lines; assert agreement. |
| `TtsEngine` | One contract test run against **every** implementation: non-empty PCM, correct sample rate, streamed chunks concatenate to the batch result. |
| Command recognition | Held-out enrollment recordings; report per-command accuracy **and false-accept rate**. |
| Offline guarantee | Disable networking at the OS level and complete a full book read. This is the product's core claim — make it a CI job. |
| Accessibility | axe-core in CI; manual NVDA and JAWS scripts per release. |
| End-to-end | Install the `.msi` on a clean Windows VM, open a Sinhala PDF, drive it by voice only. |

---

## Open questions

1. **Does the website host have a GPU?** No GPU means the site also runs Piper on CPU
   and XTTS expressive mode is dropped. Affects Phase 4 only.
2. **Is the Phase 0 PDF corpus collected?** Nothing in `reader-core` can be trusted
   without it.
3. **Team size and deadline.** Phases 1, 2 and the frontend parallelise across three
   people; solo, cut Phase 4 to the TTS playground alone.
