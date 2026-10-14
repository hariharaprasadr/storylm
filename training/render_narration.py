#!/usr/bin/env python3
"""Render the closed vocabulary (and the hot lines) with Kokoro-82M, offline.

The device cannot speak with a neural voice at runtime, so narration is a set of
pre-rendered clips in flash; this is the first half of producing them. The second
half is `training/pack_narration.py`, which turns the WAVs into the two blobs the
website streams.

    python3.13 -m venv .venv-tts
    .venv-tts/bin/pip install kokoro soundfile numpy scipy lameenc
    .venv-tts/bin/pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    .venv-tts/bin/python training/render_narration.py --what words
    .venv-tts/bin/python training/render_narration.py --what lines --lines data/narration_lines.txt

Why a separate venv: Kokoro's English G2P (misaki -> spaCy -> blis/thinc) has no
wheels for Python 3.14, which the model's own .venv uses. Python 3.13 works.

Measured on an M-series laptop: the whole 3,450-word library renders in ~9 minutes
of CPU (37.8 min of audio, mean 0.658 s per word) and is deterministic for a fixed
voice, so a re-render is reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 24000
CARRIER = "The word is {}."
WORD_RE = re.compile(r"[a-z]+")


def load_words(vocab_path):
    with open(vocab_path) as handle:
        tokens = json.load(handle)["tokens"]
    return [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]


def trim(x, sr=SR, thresh_db=-45.0, pad_ms=25):
    """Speech only: cut the leading/trailing silence, leave a small pad, remove DC."""
    x = x - float(np.mean(x))
    if not len(x):
        return x
    win = max(1, int(0.01 * sr))
    frames = np.abs(x[: len(x) // win * win]).reshape(-1, win).max(axis=1)
    if not len(frames):
        return x
    thresh = 10 ** (thresh_db / 20) * max(float(np.max(frames)), 1e-9)
    loud = np.where(frames > thresh)[0]
    if not len(loud):
        return x
    return x[max(0, loud[0] * win - int(pad_ms / 1000 * sr)):
             min(len(x), (loud[-1] + 1) * win + int(pad_ms / 1000 * sr))]


def peak_norm(x, target=0.89):  # ~ -1 dBFS
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (target / peak) if peak > 1e-9 else x


def f0_stats(x, sr=SR):
    """Crude autocorrelation F0 per frame: enough to see a sentence-final fall."""
    win, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    vals = []
    for i in range(0, max(0, len(x) - win), hop):
        seg = x[i:i + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        top = min(hi, len(ac) - 1)
        if top <= lo:
            continue
        lag = lo + int(np.argmax(ac[lo:top]))
        if ac[lag] > 0.35:
            vals.append(sr / lag)
    if len(vals) < 3:
        return None
    v = np.array(vals)
    return {"f0_median": float(np.median(v)), "f0_range": float(v.max() - v.min()),
            "f0_fall": float(v[0] - v[-1])}


def render(pipeline, text, voice, speed):
    chunks = list(pipeline(text, voice=voice, speed=speed))
    audio = torch.cat([c.audio for c in chunks]).numpy() if chunks else np.zeros(1, dtype="float32")
    return peak_norm(trim(audio))


def main():
    ap = argparse.ArgumentParser(description="Render narration clips with Kokoro-82M")
    ap.add_argument("--what", choices=["words", "lines"], default="words")
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data/vocab.json"))
    ap.add_argument("--lines", default=os.path.join(ROOT, "data/narration_lines.txt"),
                    help="one line per row for --what lines (the hot-line tier)")
    ap.add_argument("--out", default="/tmp/narration")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=0.9, help="0.9 = the calm bedtime pace")
    ap.add_argument("--limit", type=int, default=0, help="0 = everything")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, args.what), exist_ok=True)

    texts = load_words(args.vocab) if args.what == "words" else \
        [l.strip() for l in open(args.lines) if l.strip()]
    if args.limit:
        texts = texts[: args.limit]
    print(f"{args.what}: {len(texts)} clips · voice {args.voice} · speed {args.speed}")

    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    manifest, started = [], time.time()
    for i, text in enumerate(texts, 1):
        audio = render(pipeline, text, args.voice, args.speed)
        key = text if args.what == "words" else " ".join(WORD_RE.findall(text.lower()))[:120]
        sf.write(os.path.join(args.out, args.what, f"{key}.wav"), audio, SR)
        manifest.append({"word" if args.what == "words" else "line": key,
                         "iso_s": len(audio) / SR, "iso_rms": float(np.sqrt(np.mean(audio ** 2))),
                         "iso_f0": f0_stats(audio)})
        if i % 250 == 0:
            print(f"  {i}/{len(texts)}  {time.time() - started:.0f}s", flush=True)

    with open(os.path.join(args.out, f"{args.what}.manifest.json"), "w") as handle:
        json.dump(manifest, handle)
    durations = np.array([m["iso_s"] for m in manifest])
    print(f"rendered {len(manifest)} clips in {time.time() - started:.0f}s · "
          f"{durations.sum() / 60:.1f} min of audio · mean {durations.mean():.3f}s")
    for label, bps in (("16 kHz IMA ADPCM", 8192), ("8 kHz IMA ADPCM", 4096),
                       ("Opus 16 kbps", 2000), ("Opus 12 kbps", 1500)):
        print(f"  at {label:16} this tier is {durations.sum() * bps / 1048576:6.2f} MB")


if __name__ == "__main__":
    raise SystemExit(main())
