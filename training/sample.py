#!/usr/bin/env python3
"""Generate stories from a trained checkpoint and run the runtime path on them.

    python training/sample.py                       # one story per world
    python training/sample.py --per-world 4

The pipeline is the device's: the model emits words, the engine's canonicaliser
renders each line (punctuation, capitals, and a word-level repair against the
beat inventory), and only then does the firmware validator see it. Every story
reports three numbers - pass rate, word-level fidelity, and unmatched lines -
because a pass rate alone cannot see drift.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from storylm.model import Config, StoryLM, decode_lines, generate_story
from storylm.vocab import (TableMismatch, control_values, default_checkpoint,
                           series_of, table_for_checkpoint)

WORLDS = [
    ("bluey", "BEDROOM"), ("gabby", "BEDROOM"), ("one", "PLAYROOM"),
    ("amelia", "KITCHEN"), ("junie", "BEDROOM"),
]
PLACE_IDS = {"bluey": "bedroom", "gabby": "bedroom", "one": "playroom",
             "amelia": "kitchen", "junie": "bedroom"}
SERIES_IDS = {"bluey": "bluey", "gabby": "gabby", "one": "numberblocks",
              "amelia": "amelia", "junie": "junie"}


def words_of(line):
    """Words only: the table is letters-only, so punctuation is not drift."""
    return " ".join(re.findall(r"[a-z]+", line.lower()))


def canonicalize(lines, series_id, place_id, theme, mode="bedtime",
                 vocab_path="data/vocab.json"):
    """Run the engine's canonicaliser over a draft story.

    The inventory must be built for the *same prompt* the story came from: a
    story generated for one mode does not contain another mode's opening or
    closing, so passing the wrong mode makes those lines unmatched.

    Uses the C++ engine when it is built on the current token table, and the
    Python validator (`storylm/validator.py`) otherwise.
    """
    from storylm.validator import run_canonicalize
    return run_canonicalize(series_id, mode, place_id, theme, lines, vocab_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint to sample (default: the newest in checkpoints/)")
    ap.add_argument("--vocab", default=None,
                    help="token table (default: the one the checkpoint was trained on)")
    ap.add_argument("--corpus", default="data/processed/train.jsonl",
                    help="corpus for the word-level fidelity count")
    ap.add_argument("--per-world", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1,
                    help="stories per prompt, for a tighter estimate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0,
                    help="fix the sampler so a story can be reproduced exactly")
    args = ap.parse_args()

    torch.manual_seed(args.seed)  # reproducible stories for debugging

    checkpoint = args.checkpoint or default_checkpoint()
    if checkpoint is None or not os.path.exists(checkpoint):
        print("no checkpoint found; train one first (see training/README.md)", file=sys.stderr)
        return 2

    try:
        tokens, generation = table_for_checkpoint(checkpoint, args.vocab)
    except TableMismatch as error:
        print(error, file=sys.stderr)
        return 2
    vocab_path = generation.vocab_path() if args.vocab is None else args.vocab

    ckpt = torch.load(checkpoint, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = StoryLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    to_id = {token: i for i, token in enumerate(tokens)}
    series = {entry["id"]: entry for entry in series_of(generation.series_path())}
    themes = [t.upper() for t in control_values(tokens, "THEME")]
    print(f"# {os.path.relpath(checkpoint, ROOT)}  [{generation.name} table "
          f"{generation.vocab}, {len(themes)} themes]")

    corpus_lines = set()
    with open(args.corpus) as handle:
        for row in map(json.loads, handle):
            for line in row["lines"]:
                corpus_lines.add(words_of(line))

    passed = total = matched_lines = total_lines = unmatched_lines = 0
    for world, place_token in WORLDS:
        for theme in themes[: args.per_world]:
          for _repeat in range(args.repeats):
                entry = series[SERIES_IDS[world]]
                char_token = entry["token"]
                prompt = (f"{char_token}<MODE=BEDTIME>"
                          f"<PLACE={place_token}><THEME={theme}>")
                ids = [to_id["<BOS>"]] + [to_id[t] for t in
                                          (char_token, "<MODE=BEDTIME>",
                                           f"<PLACE={place_token}>", f"<THEME={theme}>")]
                out = generate_story(model, ids, tokens, cfg.ctx,
                                     temperature=args.temperature, top_k=40)
                draft = decode_lines(out, tokens, {entry["story_name"].lower()} |
                                     {f.lower() for f in entry["friends"]})
                total += 1

                lines, stats = canonicalize(draft, SERIES_IDS[world], PLACE_IDS[world], theme.lower(),
                                            vocab_path=vocab_path)
                matched = sum(1 for line in lines if words_of(line) in corpus_lines)
                unmatched = int(stats.split("unmatched ")[-1].split(" ")[0])
                matched_lines += matched
                total_lines += len(lines)
                unmatched_lines += unmatched

                from storylm.validator import run_check
                verdict_text = run_check(SERIES_IDS[world], "bedtime", PLACE_IDS[world],
                                         theme.lower(), lines, lines, vocab_path=vocab_path)
                verdict_ok = verdict_text.startswith("ok")
                passed += verdict_ok

                print(f"=== {prompt}  ({len(lines)} lines, "
                      f"{sum(len(line.split()) for line in lines)} words)")
                for line in lines:
                    print("   ", line)
                print(f"    corpus-exact: {matched}/{len(lines)} lines"
                      f"    canonicalize: {stats}"
                      f"    validator: {verdict_text}")

    print(f"\n{passed}/{total} stories passed the firmware validator")
    print(f"{matched_lines}/{total_lines} lines match an engine line word for word "
          f"({100 * matched_lines / max(total_lines, 1):.1f} %)")
    print(f"{unmatched_lines}/{total_lines} lines are not in this prompt's inventory "
          f"({100 * unmatched_lines / max(total_lines, 1):.2f} %: drift or a beat from the "
          f"wrong theme) - only those need the engine's fallback tier")


if __name__ == "__main__":
    main()
