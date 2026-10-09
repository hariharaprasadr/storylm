#!/usr/bin/env python3
"""Generate one story from a trained checkpoint.

    python cli.py story                                       # a bedtime Bluey story
    python cli.py story --world gabby --theme brave
    python cli.py story --list                                # what can I ask for?
    python cli.py story --seed 42                             # reproducible
    python cli.py story --raw                                 # the draft, before the engine renders it

Inputs are the picker's own words (world, mode, place, theme), not token ids.
The checkpoint decides everything else: its own token table fixes the prompt
space (`storylm/vocab.py` pairs the two by provenance and refuses a mismatch),
its series file lists the worlds and places, and its generation's table names
the modes and themes.

The story is printed through the device's path: the model drafts, the engine's
canonicaliser renders the line, the firmware validator has the last word.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from .model import Config, StoryLM, decode_lines, generate_story
except (ImportError, ValueError):
    try:
        from storylm.model import Config, StoryLM, decode_lines, generate_story
    except ImportError:
        from model import Config, StoryLM, decode_lines, generate_story

from storylm.validator import run_canonicalize, run_check
from storylm.vocab import (TableMismatch, control_values, default_checkpoint,
                           require, series_of, table_for_checkpoint)


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


def list_choices(series, generation, modes, themes):
    print(f"generation: {generation.name} ({generation.vocab})")
    print("worlds:")
    for sid, entry in series.items():
        places = ", ".join(entry["places"])
        cast = ", ".join(entry["friends"])
        print(f"  {sid:<13} {entry['name']:<16} with {cast}")
        print(f"  {'':<13} places: {places}")
    print(f"\nmodes:  {', '.join(modes)}")
    print(f"themes: {', '.join(themes)}")
    example = series.get("bluey") or next(iter(series.values()))
    print(f"\nexample: python cli.py story --world {example['id']} --mode {modes[0]} "
          f"--place {example['places'][0]} --theme {themes[0]}")


def main():
    ap = argparse.ArgumentParser(description="Generate one bedtime story.")
    ap.add_argument("--world", default="bluey", help="world id, label or protagonist name")
    ap.add_argument("--mode", default=None, help="bedtime (default); --list shows the rest")
    ap.add_argument("--place", default=None, help="a place id of that world")
    ap.add_argument("--theme", default="help", help="help (default); --list shows the rest")
    ap.add_argument("--seed", type=int, default=0, help="fix the sampler")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--vocab", default=None,
                    help="token table to pair with the checkpoint (default: the one it "
                         "was trained on, matched by id count)")
    ap.add_argument("--raw", action="store_true", help="show the draft, not the rendered story")
    ap.add_argument("--list", action="store_true", help="show the available inputs")
    args = ap.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    if checkpoint is None or not os.path.exists(checkpoint):
        print("no checkpoint found; train one first (see training/README.md)", file=sys.stderr)
        return 2

    try:
        tokens, generation = table_for_checkpoint(checkpoint, args.vocab)
    except TableMismatch as error:
        print(error, file=sys.stderr)
        return 2
    modes = control_values(tokens, "MODE")
    themes = control_values(tokens, "THEME")
    series = {entry["id"]: entry for entry in series_of(generation.series_path())}
    to_id = {token: i for i, token in enumerate(tokens)}

    if args.list:
        list_choices(series, generation, modes, themes)
        return 0

    world = resolve(series, args.world, "world")
    if world is None:
        print(f"unknown world {args.world!r} in the {generation.name} table. Try --list.",
              file=sys.stderr)
        return 2
    entry = series[world]

    mode = (args.mode or entry.get("default_mode") or modes[0]).lower()
    if mode not in modes:
        print(f"unknown mode {mode!r}. Choose from: {', '.join(modes)}", file=sys.stderr)
        return 2
    theme = args.theme.lower()
    if theme not in themes:
        print(f"unknown theme {theme!r}. Choose from: {', '.join(themes)}", file=sys.stderr)
        return 2
    place = args.place or entry["places"][0]
    if place not in entry["places"]:
        print(f"unknown place {place!r} for {world}. Choose from: "
              f"{', '.join(entry['places'])}", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    ckpt = torch.load(checkpoint, weights_only=False)
    model = StoryLM(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt_tokens = [entry["token"], f"<MODE={mode.upper()}>", f"<PLACE={place.upper()}>",
                     f"<THEME={theme.upper()}>"]
    try:
        require(tokens, ["<BOS>"] + prompt_tokens, f"{generation.name} table")
    except TableMismatch as error:
        print(error, file=sys.stderr)
        return 2
    ids = [to_id["<BOS>"]] + [to_id[t] for t in prompt_tokens]
    out = generate_story(model, ids, tokens, model.cfg.ctx, temperature=args.temperature)

    proper_names = {entry["story_name"].lower()} | {f.lower() for f in entry["friends"]}
    draft = decode_lines(out, tokens, proper_names)

    lines, note = run_canonicalize(world, mode, place, theme, draft,
                                   vocab_path=generation.vocab_path())
    if args.raw:
        lines, note = draft, "draft (not canonicalised)"

    print(f"# {entry['name']} - {mode} - {place} - {theme}   seed {args.seed}"
          f"   [{generation.name}: {os.path.relpath(checkpoint, ROOT)} + {generation.vocab}]")
    for line in lines:
        print(line)
    print(f"\n{len(lines)} lines, {sum(len(l.split()) for l in lines)} words | {note}")

    if not args.raw:
        print("validator:", run_check(world, mode, place, theme, lines, lines,
                                      vocab_path=generation.vocab_path()))
        if "unmatched 0" not in note:
            print("note: unmatched lines are drift, or a beat written for another "
                  f"theme (this story asked for {theme})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
