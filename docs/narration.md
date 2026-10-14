# Narration: turning the closed vocabulary into speech

The device has no runtime neural voice. Narration is pre-recorded clips in flash, and
the vocabulary is closed (3,509 ids = 59 control tokens + 3,450 words), so "every word
the device can say" is a finite list that can be rendered once and shipped.

This is the note that goes with `web/narration.html` — the page that plays those clips
in the browser, with the same playback rule the firmware will use.

> **These are synthetic takes.** Kokoro-82M renders them offline (Apache-2.0, code and
> weights). The plan of record is still to re-record with a human voice; the generated
> clips are a distributable stand-in that unblocks narration today.

---

## 1. What is rendered

| tier | clips | audio | what it is for |
|---|---:|---:|---|
| **word clips** | 3,450 | 37.8 min | every word of the closed table — closes the "line has no narration mapping" gap (`G1` in `docs/review_gaps_and_improvements.md`) |
| **hot-line clips** | 100 | 6.4 min | the 100 most-used lines in the corpus (60.8 % of all spoken lines) — natural sentence prosody where it matters most |
| engine templates (openings/closings/spine/problems) | 291 | 20.6 min | costed but not shipped: the templates carry `{name}`/`{prop}`/`{place}` slots, so they need a splice rule before they can be played |
| every engine line (5 worlds × 8 modes × 7 places × 9 themes) | 11,649 | 13.8 h | **infeasible** (94 MB as Opus 16 kbps) — line-level clips only scale for a hot subset |

Measured with Kokoro-82M, voice `af_heart`, speed 0.9 (a calm bedtime pace), 24 kHz,
trimmed to speech (−45 dB threshold, 25 ms pad), peak-normalised to −1 dBFS:

* word clips: mean **0.658 s**, median 0.640, p90 0.780, max 1.120
* hot-line clips: mean **3.84 s**
* whole vocabulary renders in **~9 minutes of CPU** and is deterministic for a fixed
  voice, so re-rendering for another voice or a fix is cheap

## 2. What it costs on the device

The 8 MB part leaves roughly **4.4 MB** for voice (app ≈2.25 MB + INT4 model 1.05 MB +
metadata ≈0.2 MB); the plan currently allocates 2 MB.

| configuration | 16 kHz IMA ADPCM | 8 kHz IMA ADPCM | Opus 16 kbps | Opus 12 kbps | 8 MB part |
|---|---:|---:|---:|---:|---|
| 100 hot lines only | 3.00 MB | 1.50 MB | 0.73 MB | 0.55 MB | fits |
| all 3,450 words | 17.73 MB | 8.86 MB | 4.33 MB | 3.25 MB | only Opus |
| all words + 100 hot lines | 20.73 MB | 10.36 MB | 5.06 MB | 3.79 MB | Opus 12 kbps, tight |
| top-1,164 words (99 % of spoken words) + hot lines | 8.99 MB | 4.49 MB | 2.19 MB | 1.65 MB | 8 kHz ADPCM, marginal |
| top-700 words (95 %) + hot lines | 6.54 MB | 3.27 MB | 1.60 MB | 1.20 MB | comfortable |

Consequences:

* **With IMA ADPCM the full vocabulary does not fit** (8.86 MB at 8 kHz). Either accept
  Opus-class coding (2 kB/s) or clip only the frequent words.
* **Rendering a word inside a carrier phrase and cutting it out halves the library**
  (measured 0.32 s vs 0.658 s per word): a word spoken inside a sentence has no
  phrase-final lengthening. With boundary-safe cuts (~0.40 s) the full vocabulary is
  ≈5.0–5.7 MB at 8 kHz ADPCM and ≈1.9–2.1 MB at Opus 12 kbps — which is what makes
  ADPCM viable again. The cutter must expand the aligned span to the nearest energy
  minimum first, or it clips onsets (measured head-window energy up to 1.9× the clip RMS).
* The website ships the clips as **two CBR MP3 blobs** the browser can slice and decode
  with Web Audio: `web/audio/words.mp3` (7.4 MB, 24 kbps) + `web/audio/lines.mp3`
  (1.5 MB, 32 kbps), each with a `.index.json` (byte offset, length, duration per clip).

## 3. How it will sound — measured proxies

I cannot listen to the output; these are acoustic measurements, and the naturalness
verdict belongs to a human listener. Play the A/B on `narration.html`
(`story1_words.wav` vs `story1_both.wav` in `artifacts/tts-preview/`).

| property | words only | words + hot lines |
|---|---|---|
| a 240-word story | **176.6 s** | **118.0 s** (reference human pace: 111 s) |
| line source | every line assembled from word clips | 13/22 lines (59 %) are whole-line clips |
| pitch | a bare word is a complete utterance: **75 %** of the 3,450 clips end on a fall > 20 Hz; median within-clip pitch range 104 Hz | whole-line clips keep natural sentence prosody |
| voice consistency | median F0 spread across the 3,450 clips: **σ = 13 Hz** (no drift) | same |
| joins | clip onsets start from silence (sample step 0.00 vs 0.038 inside clips) — **no clicks** | same |
| loudness | per-clip RMS spread 1.49× (p90/p10): add per-clip gain before packing | same |
| intelligibility (proxy) | Whisper-tiny.en WER **11.2 %** | **6.2 %** (the ASR's own floor inflates both) |

So: complete coverage, one steady voice, no artefacts — but word-by-word narration has
list rhythm and runs ~60 % slower than natural speech, and the fix is the hot-line tier
(natural prosody for the lines the engine actually uses most).

## 4. Pipeline

```bash
# once: a Python 3.13 venv (Kokoro's G2P has no 3.14 wheels)
python3.13 -m venv .venv-tts
.venv-tts/bin/pip install kokoro soundfile numpy scipy lameenc
.venv-tts/bin/pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# render (word clips + the hot lines listed in data/narration_lines.txt)
.venv-tts/bin/python training/render_narration.py --what words  --out /tmp/narration
.venv-tts/bin/python training/render_narration.py --what lines  --out /tmp/narration \
    --lines data/narration_lines.txt

# pack for the website (two blobs + indexes) and re-render the WAV previews
.venv-tts/bin/python training/pack_narration.py --words /tmp/narration/words \
    --lines /tmp/narration/lines --line-keys data/narration_lines.txt --out web/audio
```

`make narration` runs both steps. The page reads `web/audio/*.index.json` and slices
clips out of the blob; playback gaps follow the device rules — 45 ms between words,
140 ms after a comma, 320 ms at the end of a line.

## 5. Open items

1. **Opus decode on the ESP32-S3** — is there CPU/latency budget for a decoder? If not,
   the ADPCM path forces either the carrier-cut render or the top-1,164-word subset.
2. **Carrier-cut cutter** — needs a forced-aligner (Whisper timestamps work; the sample
   test used `mlx-whisper`) plus boundary expansion; not yet in the pipeline.
3. **Per-clip gain** — 1.49× RMS spread across word clips.
4. **Listening verdict** — voice choice (`af_heart` vs the seven auditions) and
   words-only vs hybrid, from a human, on the device speaker.
5. **Human re-record** — the tiers and sizes above are exactly what a human voice actor
   would need to cover (3,450 words + the hot lines).
