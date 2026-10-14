#!/usr/bin/env python3
"""Composed narration: cut the templates' phrases out of real sentences instead of
speaking the words one at a time.

Word clips alone sound like a reader who does not know the language, because every clip
is a complete little utterance with its own falling pitch, so a line comes out as a list
of one-word sentences. Whole lines sound right — but the device's text is generated, so
they cannot all be pre-rendered.

The middle ground is available because the text is *not* open-ended: every line the
device can say is `template.format(name=..., friend=..., place=..., prop=..., word=...)`
(see `storylm/corpus_builder.py`). So render each template once as a whole sentence, then
cut the fixed phrases between its slots out of that render with word alignment: the
phrases keep real sentence prosody, including the way they run on into a slot, which is
what a standalone word clip can never have. Slot values come from words cut the same way
out of a carrier sentence.

    python training/render_compose.py --stage plan       --out /tmp/narration_compose
    python training/render_compose.py --stage render     --out /tmp/narration_compose
    python training/render_compose.py --stage cut        --out /tmp/narration_compose

`cut` needs `mlx-whisper` for the alignment (the tiny.en model is enough: the text is
known, this is forced alignment, not transcription).

Output: `units/<key>.wav` — one clip per phrase and per spoken word take, plus
`manifest.json` with durations and the alignment quality, ready for
`training/pack_narration.py`.
"""
from __future__ import annotations

import argparse
import collections
import difflib
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
sys.path.insert(0, ROOT)
from storylm import corpus_builder as cb  # noqa: E402

SR = 24000
SLOT_RE = re.compile(r"\{(\w+)\}")
WORD_RE = re.compile(r"[a-z0-9]+")
VOICE = "af_heart"
SPEED = 0.9

# A slot word is cut out of a sentence that keeps going, so it keeps its flat
# continuation intonation instead of the falling pitch of a sentence-final word.
CARRIERS = {
    "name": "Then {w} smiled and walked inside.",
    "friend": "Then {w} smiled and walked inside.",
    "place": "We walked to the {w} together.",
    "prop": "We put {w} beside the bed.",
    "object": "We found the {w} inside the room.",
    "quality": "The little room felt {w} tonight.",
    "action_past_transitive": "Then she {w} the little box.",
    "manner": "She spoke {w} to the little one.",
    "abstract": "It was a {w} that nobody mentioned.",
    "action_ing": "She was {w} beside the window.",
    "action_past_intransitive": "Then it {w} behind the house.",
    "action_base_intransitive": "Maybe she should {w} before bed.",
    "action_base_transitive": "Maybe she should {w} the box.",
}


def norm(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# plan: every template, once, with the word spans of each phrase
# ---------------------------------------------------------------------------

def slot_values() -> dict[str, list[str]]:
    """Values for the four slot names, per world, plus the detail-word pools."""
    return {
        "name": sorted({c["story_name"] for c in cb.SERIES_CONFIG}),
        "friend": sorted({f for c in cb.SERIES_CONFIG for f in c["friends"]}),
        "place": sorted({p for c in cb.SERIES_CONFIG for p in c["places"]}),
        "prop": sorted({p for c in cb.SERIES_CONFIG for p in c["props"]}),
    }


def detail_ranked(limit: int) -> list[str]:
    """Detail words by how often the corpus actually uses them, longest first."""
    path = os.path.join(ROOT, "data/processed/train.jsonl")
    counts: collections.Counter = collections.Counter()
    if os.path.exists(path):
        with open(path) as handle:
            for row in map(json.loads, handle):
                for line in row["lines"]:
                    counts.update(words(line))
    pool = {w for ws in cb.detail_pools().values() for w in ws}
    return [w for w, _ in counts.most_common() if w in pool][:limit]


def templates():
    """(id, bank, template, world, mode, theme) for every line the builder can emit."""
    out = []

    def add(bank, text, world="bluey", mode="bedtime", theme="help"):
        if isinstance(text, str) and "{" in text or (isinstance(text, str) and text):
            out.append({"id": f"{bank}_{len(out):04d}", "bank": bank, "text": text,
                        "world": world, "mode": mode, "theme": theme})

    for (world, mode), pool in cb.OPENINGS.items():
        for t in pool:
            add("OPENINGS", t, world, mode)
    for world, themes in cb.CHARACTER_PROBLEMS.items():
        for theme, pool in themes.items():
            for t in pool:
                add("CHARACTER_PROBLEMS", t, world, mode="bedtime", theme=theme)
    for (world, mode), t in cb.CLOSINGS.items():
        add("CLOSINGS", t, world, mode)
    for bank in ("SETTLE_BEATS", "SKY_BEATS", "HELPER_BEATS", "QUIET_BEATS", "SLEEP_BEATS"):
        for t in getattr(cb, bank):
            add(bank, t)
    for theme, pool in cb.TRY_BEATS.items():
        for t in pool:
            add("TRY_BEATS", t, theme=theme)
    for world, pool in cb.WARM_BEATS.items():
        for t in pool:
            add("WARM_BEATS", t, world=world)
    for cls, pool in cb.DETAIL_FRAMES.items():
        for t in pool:
            add("DETAIL_FRAMES", t, theme=cls)
    return out


def fill_for(tpl: dict, values: dict[str, list[str]]) -> tuple[str, dict[str, str]]:
    """One grammatical filling per template, used only to make it speak naturally."""
    cfg = next((c for c in cb.SERIES_CONFIG if c["id"] == tpl["world"]), cb.SERIES_CONFIG[0])
    picks = {
        "name": cfg["story_name"],
        "friend": cfg["friends"][0],
        "place": cfg["places"][0],
        "prop": cfg["props"][0],
        "word": (cb.detail_pools().get(tpl["theme"]) or ["shadow"])[0],
    }
    return tpl["text"].format(**picks), picks


def split_spans(text: str, picks: dict[str, str]) -> list[dict]:
    """The phrases between slots, with their word offsets inside the filled sentence."""
    parts, last, marks = [], 0, list(SLOT_RE.finditer(text))
    for m in marks:
        parts.append(text[last:m.start()])
        last = m.end()
    parts.append(text[last:])
    units, cursor = [], 0
    for i, part in enumerate(parts):
        clean = " ".join(part.split())
        n = len(words(clean))
        if n:
            units.append({"key": norm(clean), "text": clean, "kind": "frag",
                          "tag": "end" if i == len(parts) - 1 else "cont",
                          "w0": cursor, "w1": cursor + n})
            cursor += n
        if i < len(marks):
            cursor += len(words(picks[marks[i].group(1)]))
    return units


def build_plan(args):
    values, templates_ = slot_values(), templates()
    plan = {"templates": [], "words": []}
    for tpl in templates_:
        fill, picks = fill_for(tpl, values)
        units = split_spans(tpl["text"], picks)
        if not units:
            continue
        plan["templates"].append({"id": tpl["id"], "bank": tpl["bank"], "text": fill,
                                  "words": words(fill), "units": units})
    detail = detail_ranked(args.detail_words)
    pool_class = {w: c for c, ws in cb.detail_pools().items() for w in ws}
    takes = ([("name", v) for v in values["name"]] + [("friend", v) for v in values["friend"]] +
             [("place", v) for v in values["place"]] + [("prop", v) for v in values["prop"]] +
             [(pool_class.get(w, "object"), w) for w in detail])
    seen = set()
    for cls, word in takes:
        if norm(word) in seen:
            continue
        seen.add(norm(word))
        frame = CARRIERS.get(cls, CARRIERS["object"])
        carrier = frame.format(w=word)
        cw, expect = words(carrier), len(words(frame.split("{w}")[0]))
        at = [i for i, w in enumerate(cw) if w == words(word)[0]]
        best = min(at, key=lambda i: abs(i - expect)) if at else expect
        plan["words"].append({"id": f"w_{norm(word)}", "key": norm(word), "text": word,
                              "carrier": carrier, "words": cw, "slot": cls,
                              "w0": best, "w1": best + len(words(word))})
    os.makedirs(args.out, exist_ok=True)
    json.dump(plan, open(os.path.join(args.out, "plan.json"), "w"))
    frag_keys = {(u["key"], u["tag"]) for t in plan["templates"] for u in t["units"]}
    print(f"plan: {len(plan['templates'])} template takes, {len(plan['words'])} word takes, "
          f"{len(frag_keys)} phrase units ({sum(1 for k, t in frag_keys if t == 'cont')} continuing)")
    return plan


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def render(pipeline, text, voice=VOICE, speed=SPEED):
    chunks = list(pipeline(text, voice=voice, speed=speed))
    x = torch.cat([c.audio for c in chunks]).numpy() if chunks else np.zeros(1, dtype="float32")
    return x.astype("float32")


def stage_render(args):
    plan = json.load(open(os.path.join(args.out, "plan.json")))
    takes = os.path.join(args.out, "takes")
    os.makedirs(takes, exist_ok=True)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    todo = [("tpl_" + t["id"], t["text"]) for t in plan["templates"]]
    todo += [("wrd_" + w["id"], w["carrier"]) for w in plan["words"]]
    todo = [t for t in todo if not os.path.exists(os.path.join(takes, t[0] + ".wav"))]
    if args.limit:
        todo = todo[:args.limit]
    started, audio = time.time(), 0.0
    for i, (name, text) in enumerate(todo, 1):
        x = render(pipeline, text)
        sf.write(os.path.join(takes, name + ".wav"), x, SR)
        audio += len(x) / SR
        if i % 25 == 0:
            done = audio / max(1e-9, time.time() - started)
            print(f"  {i}/{len(todo)} takes · {audio / 60:5.1f} min audio · {done:4.1f}x realtime · "
                  f"{audio / max(1e-9, time.time() - started):.0f}", flush=True)
    print(f"rendered {len(todo)} takes · {audio / 60:.1f} min audio in {time.time() - started:.0f}s")


# ---------------------------------------------------------------------------
# cut
# ---------------------------------------------------------------------------

def load16k(path):
    """mlx-whisper shells out to ffmpeg for file paths; hand it samples instead."""
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        from scipy.signal import resample_poly
        x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    return np.ascontiguousarray(x)


def asr_spans(path, model):
    if not os.path.exists(path):
        return []
    import mlx_whisper
    info = mlx_whisper.transcribe(load16k(path), path_or_hf_repo=model, word_timestamps=True)
    out = []
    for seg in info.get("segments", []):
        for w in seg.get("words", []):
            token = norm(w.get("word", ""))
            if token:
                out.append((token.split()[0], float(w["start"]), float(w["end"])))
    return out


def align(expected: list[str], heard: list[tuple[str, float, float]]) -> tuple[list[tuple[float, float]], float]:
    """Time span for every expected word. Known text + ASR timestamps = forced alignment."""
    times = np.zeros((len(expected), 2), dtype="float64")
    hits = 0
    matcher = difflib.SequenceMatcher(None, [h[0] for h in heard], expected, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                times[j1 + k] = (heard[i1 + k][1], heard[i1 + k][2])
                hits += 1
    if len(heard) > 1:
        start, stop = heard[0][1], heard[-1][2]
    else:
        start, stop = 0.0, max(0.1, len(expected) * 0.3)
    # fill the gaps proportionally, so a mis-heard word still gets a plausible span
    known = [i for i in range(len(expected)) if times[i][1] > times[i][0]]
    anchors = [-1] + known + [len(expected)]
    edges = {i: (times[i][0], times[i][1]) for i in known}
    edges[-1] = (start, start)
    edges[len(expected)] = (stop, stop)
    for a, b in zip(anchors, anchors[1:]):
        if b - a <= 1:
            continue
        t0 = edges[a][1] if a >= 0 else start
        t1 = edges[b][0] if b < len(expected) else stop
        weights = np.array([len(w) + 1 for w in expected[a + 1:b]], dtype="float64")
        weights /= weights.sum()
        cursor = t0
        for k, frac in zip(range(a + 1, b), weights):
            span = (t1 - t0) * frac
            times[k] = (cursor, cursor + span)
            cursor += span
    return [(float(t0), float(t1)) for t0, t1 in times], hits / max(1, len(expected))


def snap_to_valley(x, t, sr, window=0.08):
    """Move a cut point to the quietest nearby sample, so a neighbour cannot be clipped."""
    lo = max(0, int((t - window) * sr))
    hi = min(len(x), int((t + window) * sr))
    if hi - lo < 8:
        return t
    seg = np.abs(x[lo:hi])
    win = max(1, int(0.005 * sr))
    frames = seg[: len(seg) // win * win].reshape(-1, win).mean(axis=1) if len(seg) >= win else seg
    return (lo + int(np.argmin(frames)) * win) / sr


def trim(x, sr=SR, thresh_db=-40.0, pad_ms=15):
    x = x - float(np.mean(x))
    win = max(1, int(0.01 * sr))
    frames = np.abs(x[: len(x) // win * win]).reshape(-1, win).max(axis=1)
    if not len(frames):
        return x
    th = 10 ** (thresh_db / 20) * max(float(frames.max()), 1e-9)
    loud = np.where(frames > th)[0]
    if not len(loud):
        return x
    return x[max(0, loud[0] * win - int(pad_ms / 1000 * sr)):
             min(len(x), (loud[-1] + 1) * win + int(pad_ms / 1000 * sr))]


def peak_norm(x, target=0.89):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (target / peak) if peak > 1e-9 else x


def cut_take(path, expected, spans, units, sr=SR):
    """Cut each unit out of the take, expanding to quiet samples at both edges."""
    x, file_sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if file_sr != sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(file_sr, sr)
        x = resample_poly(x, sr // g, file_sr // g).astype("float32")
    out = []
    for unit in units:
        w0, w1 = unit["w0"], unit["w1"]
        if w0 >= len(spans) or w1 > len(spans):
            continue
        t0 = snap_to_valley(x, spans[w0][0], sr) - 0.015
        t1 = snap_to_valley(x, spans[w1 - 1][1], sr) + 0.02
        t0, t1 = max(0.0, t0), min(len(x) / sr, t1)
        if t1 - t0 < 0.12:
            continue
        clip = peak_norm(trim(x[int(t0 * sr):int(t1 * sr)]))
        if len(clip) < int(0.10 * sr):
            continue
        out.append((unit, clip))
    return out


def head_energy(clip, sr=SR):
    """A tight cut can start on a consonant: compare the first 30 ms with the clip RMS."""
    head = clip[: int(0.03 * sr)]
    rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
    head_rms = float(np.sqrt(np.mean(head ** 2))) if len(head) else 0.0
    return head_rms / max(rms, 1e-9)


def stage_cut(args):
    plan = json.load(open(os.path.join(args.out, "plan.json")))
    takes = os.path.join(args.out, "takes")
    units_dir = os.path.join(args.out, "units")
    os.makedirs(units_dir, exist_ok=True)
    manifest = {}
    stats, skipped, aligned_all = collections.Counter(), 0, []

    for i, tpl in enumerate(plan["templates"], 1):
        path = os.path.join(takes, f"tpl_{tpl['id']}.wav")
        heard = asr_spans(path, args.model)
        spans, rate = align(tpl["words"], heard)
        aligned_all.append(rate)
        for unit, clip in cut_take(path, tpl["words"], spans, tpl["units"]):
            # a phrase can appear in several templates; keep its longest take
            key = unit["key"] + ("#e" if unit["tag"] == "end" else "")
            old = manifest.get(key)
            if old and old["d"] >= len(clip) / SR:
                continue
            manifest[key] = {"t": unit["text"], "k": unit["kind"], "tag": unit["tag"],
                             "take": tpl["id"], "d": round(len(clip) / SR, 3),
                             "head": round(head_energy(clip), 2), "align": round(rate, 2)}
            sf.write(os.path.join(units_dir, key + ".wav"), clip, SR)
            stats["frag"] += 1
        if i % 50 == 0:
            print(f"  cut {i}/{len(plan['templates'])} templates · {len(manifest)} units · "
                  f"alignment {np.mean(aligned_all) * 100:.0f}%", flush=True)

    for i, word in enumerate(plan["words"], 1):
        path = os.path.join(takes, f"wrd_{word['id']}.wav")
        heard = asr_spans(path, args.model)
        spans, rate = align(word["words"], heard)
        aligned_all.append(rate)
        for unit, clip in cut_take(path, word["words"], spans, [{"key": word["key"], "text": word["text"],
                                                                 "kind": "word", "tag": "word",
                                                                 "w0": word["w0"], "w1": word["w1"]}]):
            old = manifest.get(unit["key"])
            if old and old["k"] == "frag":
                continue                      # a phrase take of the same text is better than a carrier cut
            if old and old["d"] >= len(clip) / SR and old["align"] >= rate:
                continue
            manifest[unit["key"]] = {"t": unit["text"], "k": "word", "tag": "word",
                                     "take": word["id"], "d": round(len(clip) / SR, 3),
                                     "head": round(head_energy(clip), 2), "align": round(rate, 2)}
            sf.write(os.path.join(units_dir, unit["key"] + ".wav"), clip, SR)
            stats["word"] += 1

    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=0, sort_keys=True)
    clips = list(manifest.values())
    print(f"\n{len(clips)} units · {sum(c['d'] for c in clips) / 60:.1f} min audio · "
          f"alignment {np.mean(aligned_all) * 100:.1f}% · "
          f"mean head energy {np.mean([c['head'] for c in clips]):.2f} (1.0 = uniform)")
    for label, bps in (("16 kHz IMA ADPCM", 8000), ("8 kHz IMA ADPCM", 4000),
                       ("Opus 16 kbps", 2000), ("Opus 12 kbps", 1500)):
        print(f"  {label:16} {sum(c['d'] for c in clips) * bps / 1048576:5.2f} MB")
    frags = [c for c in clips if c["k"] == "frag"]
    words_ = [c for c in clips if c["k"] == "word"]
    print(f"  phrases: {len(frags)} clips, mean {np.mean([c['d'] for c in frags]):.2f}s  ·  "
          f"words: {len(words_)} clips, mean {np.mean([c['d'] for c in words_]):.2f}s")


def main():
    ap = argparse.ArgumentParser(description="Composed narration: phrases cut from real sentences")
    ap.add_argument("--stage", choices=["plan", "render", "cut"], default="plan")
    ap.add_argument("--out", default="/tmp/narration_compose")
    ap.add_argument("--detail-words", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="mlx-community/whisper-tiny.en-mlx")
    args = ap.parse_args()
    if args.stage == "plan":
        build_plan(args)
    elif args.stage == "render":
        stage_render(args)
    else:
        stage_cut(args)


if __name__ == "__main__":
    raise SystemExit(main())
