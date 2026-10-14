#!/usr/bin/env python3
"""Pack rendered narration clips into the two blobs the website streams.

    .venv-tts/bin/python training/pack_narration.py \
        --words /tmp/narration/words --lines /tmp/narration/lines \
        --line-keys data/narration_lines.txt --out web/audio

Each blob is one concatenated CBR MP3 with a JSON index beside it, so the browser
can slice a single clip out of it and schedule it with Web Audio:

    web/audio/words.mp3 + words.index.json   every word in the closed table
    web/audio/lines.mp3 + lines.index.json   the hot whole-line clips

CBR keeps the byte-offset -> time mapping exact (bytes / (bitrate / 8)), which is
what makes one file per tier possible instead of 3,450 little files.

Measured output (16 kHz mono, MP3): words 24 kbps -> 7.4 MB, lines 32 kbps -> 1.5 MB
for the 100-line hot set. The device would store the same clips as IMA ADPCM (no
decoder needed) or Opus (four times smaller, needs a decoder); see docs/narration.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from math import gcd

import lameenc
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR_OUT = 16000
WORD_RE = re.compile(r"[a-z]+")


def to_mp3(x, sr, bitrate):
    if sr != SR_OUT:
        g = gcd(int(sr), SR_OUT)
        x = resample_poly(x, SR_OUT // g, int(sr) // g)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(SR_OUT)
    encoder.set_channels(1)
    encoder.set_quality(6)
    return encoder.encode(pcm) + encoder.flush()


def clip_key(name):
    """words are keyed by the word; lines by the normalised line text"""
    return name[:-4]


def pack(folder, bitrate, only_keys=None):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
    if only_keys is not None:
        keep = set()
        for line in open(only_keys):
            key = " ".join(WORD_RE.findall(line.lower()))[:120]
            if key:
                keep.add(key + ".wav")
        files = [f for f in files if f in keep]
    blob, index = bytearray(), {}
    for name in files:
        x, sr = sf.read(os.path.join(folder, name), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        data = to_mp3(x, sr, bitrate)
        index[clip_key(name)] = {"o": len(blob), "n": len(data), "d": round(len(x) / sr, 3)}
        blob += data
    meta = {"sr": SR_OUT, "bitrate_kbps": bitrate, "clips": len(index), "bytes": len(blob),
            "seconds": round(sum(v["d"] for v in index.values()), 1), "index": index}
    return bytes(blob), meta


def main():
    ap = argparse.ArgumentParser(description="Pack narration clips for the website")
    ap.add_argument("--words", default="/tmp/narration/words")
    ap.add_argument("--lines", default="/tmp/narration/lines")
    ap.add_argument("--line-keys", default=None,
                    help="text file of lines to keep in the lines blob (the hot set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words-bitrate", type=int, default=24)
    ap.add_argument("--lines-bitrate", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, folder, bitrate, only in (("words", args.words, args.words_bitrate, None),
                                        ("lines", args.lines, args.lines_bitrate, args.line_keys)):
        if not os.path.isdir(folder):
            print(f"skip {name}: {folder} not found")
            continue
        blob, meta = pack(folder, bitrate, only)
        with open(os.path.join(args.out, f"{name}.mp3"), "wb") as handle:
            handle.write(blob)
        with open(os.path.join(args.out, f"{name}.index.json"), "w") as handle:
            json.dump(meta, handle, separators=(",", ":"))
        print(f"{name:6} {meta['clips']:5} clips  {meta['seconds']:8.1f}s  "
              f"{len(blob) / 1048576:5.2f} MB  ({bitrate} kbps {SR_OUT} Hz mono)")


if __name__ == "__main__":
    raise SystemExit(main())
