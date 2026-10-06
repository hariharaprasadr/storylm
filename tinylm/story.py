#!/usr/bin/env python3
"""Generate one story from the trained model.

    python training/story.py                                  # a bedtime Bluey story
    python training/story.py --world gabby --theme brave
    python training/story.py --list                           # what can I ask for?
    python training/story.py --seed 42                        # reproducible
    python training/story.py --raw                            # the draft, before the engine renders it

Inputs are the picker's own words (world, mode, place, theme), not token ids.
The story is printed through the device's path: the model drafts, the engine's
canonicaliser renders the line, the firmware validator has the last word.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from .model import Config, TinyLM, decode_lines, generate_story
except (ImportError, ValueError):
    try:
        from tinylm.model import Config, TinyLM, decode_lines, generate_story
    except ImportError:
        from model import Config, TinyLM, decode_lines, generate_story
try:
    from sample import canonicalize
except ImportError:
    canonicalize = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINTS = ["checkpoints/embedded_big.pt",
                       "checkpoints/embedded.pt",
                       "artifacts/checkpoints/embedded_big.pt",
                       "artifacts/checkpoints/embedded.pt"]
MODES = ["bedtime", "adventure", "funny", "kindness"]
THEMES = ["help", "share", "brave", "try"]


def load_series():
    with open(os.path.join(ROOT, "data/series.json")) as handle:
        return {entry["id"]: entry for entry in json.load(handle)}


def resolve(series, needle, kind):
    """Accepts an id, a display label, a protagonist name or a place id."""
    needle = needle.strip().lower()
    if kind == "world":
        for sid, entry in series.items():
            names = {sid, entry["name"].lower(), entry["story_name"].lower()}
            if needle in names:
                return sid
        return None
    for sid, entry in series.items():
        for value in entry.get(kind, []):
            if needle == value.lower():
                return sid, value
    return None


def list_choices(series):
    print("worlds:")
    for sid, entry in series.items():
        places = ", ".join(entry["places"])
        cast = ", ".join(entry["friends"])
        print(f"  {sid:<13} {entry['name']:<16} with {cast}")
        print(f"  {'':<13} places: {places}")
    print(f"\nmodes:  {', '.join(MODES)}")
    print(f"themes: {', '.join(THEMES)}")
    print("\nexample: python training/story.py --world gabby --mode bedtime "
          "--place window --theme brave")


def main():
    ap = argparse.ArgumentParser(description="Generate one bedtime story.")
    ap.add_argument("--world", default="bluey", help="world id, label or protagonist name")
    ap.add_argument("--mode", default=None, help="bedtime (default), adventure, funny, kindness")
    ap.add_argument("--place", default=None, help="a place id of that world")
    ap.add_argument("--theme", default="help", help="help, share, brave or try")
    ap.add_argument("--seed", type=int, default=0, help="fix the sampler")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--raw", action="store_true", help="show the draft, not the rendered story")
    ap.add_argument("--list", action="store_true", help="show the available inputs")
    args = ap.parse_args()

    series = load_series()
    if args.list:
        list_choices(series)
        return 0

    world = resolve(series, args.world, "world")
    if world is None:
        print(f"unknown world {args.world!r}. Try --list.", file=sys.stderr)
        return 2
    entry = series[world]

    mode = (args.mode or entry.get("default_mode") or "bedtime").lower()
    if mode not in MODES:
        print(f"unknown mode {mode!r}. Choose from: {', '.join(MODES)}", file=sys.stderr)
        return 2
    theme = args.theme.lower()
    if theme not in THEMES:
        print(f"unknown theme {theme!r}. Choose from: {', '.join(THEMES)}", file=sys.stderr)
        return 2
    place = args.place or entry["places"][0]
    if place not in entry["places"]:
        print(f"unknown place {place!r} for {world}. Choose from: "
              f"{', '.join(entry['places'])}", file=sys.stderr)
        return 2

    checkpoint = args.checkpoint
    if checkpoint is None:
        for candidate in DEFAULT_CHECKPOINTS:
            if os.path.exists(os.path.join(ROOT, candidate)):
                checkpoint = os.path.join(ROOT, candidate)
                break
    if checkpoint is None or not os.path.exists(checkpoint):
        print("no checkpoint found; train one first (see training/README.md)", file=sys.stderr)
        return 2

    with open(os.path.join(ROOT, "data/vocab.json")) as handle:
        tokens = json.load(handle)["tokens"]
    to_id = {token: i for i, token in enumerate(tokens)}

    torch.manual_seed(args.seed)
    ckpt = torch.load(checkpoint, weights_only=False)
    model = TinyLM(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    tag = "<CHAR=ONE>" if world == "one" else f"<CHAR={world.upper()}>"
    prompt_tokens = [tag, f"<MODE={mode.upper()}>", f"<PLACE={place.upper()}>",
                     f"<THEME={theme.upper()}>"]
    ids = [to_id["<BOS>"]] + [to_id[t] for t in prompt_tokens]
    out = generate_story(model, ids, tokens, model.cfg.ctx, temperature=args.temperature)

    proper_names = {entry["story_name"].lower()} | {f.lower() for f in entry["friends"]}
    draft = decode_lines(out, tokens, proper_names)

    if args.raw or canonicalize is None:
        lines, note = draft, "draft (not canonicalised)"
    else:
        try:
            lines, stats = canonicalize(draft, world, place, theme, mode)
        except Exception:
            lines, note = draft, "draft (canonicaliser unavailable)"
        note = stats or "canonicalised"

    print(f"# {entry['name']} - {mode} - {place} - {theme}   seed {args.seed}")
    for line in lines:
        print(line)
    print(f"\n{len(lines)} lines, {sum(len(l.split()) for l in lines)} words | {note}")

    if not args.raw and os.path.exists(os.path.join(ROOT, "build/storygen")):
        payload = "".join(f"<LINE>{line}</LINE>\n" for line in lines) + "<END>\n"
        verdict = subprocess.run(
            ["./build/storygen", "--check", "--series", world, "--mode", mode,
             "--place", place, "--theme", theme],
            input=payload, capture_output=True, text=True, cwd=ROOT,
        )
        print("validator:", verdict.stdout.strip())
        if "unmatched 0" not in note:
            print("note: unmatched lines are drift, or a beat written for another "
                  f"theme (this story asked for {theme})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
