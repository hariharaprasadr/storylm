#!/usr/bin/env python3
"""KV-cache equivalence and speed test.

    .venv/bin/python training/test_cache.py

A cache is the classic place to introduce silent numerical wrongness, so this
checks two things and then measures the gain:

1. the next-token logits from a cached prefill/step match a full recompute;
2. greedy stories are token-for-token identical with and without the cache.

Exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    from .model import Config, KVBuf, StoryLM, generate_story
except (ImportError, ValueError):
    try:
        from storylm.model import Config, KVBuf, StoryLM, generate_story
    except ImportError:
        from model import Config, KVBuf, StoryLM, generate_story

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from storylm.vocab import (TableMismatch, default_checkpoint, require,
                           table_for_checkpoint)

PROMPT = ["<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BEDROOM>", "<THEME=HELP>"]


def load(checkpoint, vocab):
    payload = torch.load(checkpoint, weights_only=False)
    model = StoryLM(Config(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    tokens, generation = table_for_checkpoint(checkpoint, vocab)
    require(tokens, ["<BOS>"] + PROMPT, f"{generation.name} table")
    return model, tokens, generation


def logits_match(model, tokens, tol=1e-4):
    """Two comparisons, both against a full recompute over the same prefix:

    * prefill: cache filled in one call (is_causal over the whole prefix)
    * step: cache filled by prefill + one appended token (the decoding path)
    """
    to_id = {t: i for i, t in enumerate(tokens)}
    prompt = [to_id["<BOS>"]] + [to_id[t] for t in PROMPT]
    worst = 0.0
    with torch.no_grad():
        for length in range(2, 34):
            if length <= len(prompt):
                prefix = prompt[:length]
            else:
                prefix = prompt + [to_id["bluey"]] * (length - len(prompt))
            idx = torch.tensor([prefix], dtype=torch.long)
            plain = model(idx)[0][0, -1]  # the reference

            checks = {}
            buf = KVBuf(model.cfg)
            checks["prefill"] = model(idx, buf=buf)[0][0, -1]

            buf = KVBuf(model.cfg)
            model(torch.tensor([prefix[:-1]], dtype=torch.long), buf=buf)
            checks["step"] = model(torch.tensor([[prefix[-1]]], dtype=torch.long),
                                   pos_offset=length - 1, buf=buf)[0][0, -1]

            for label, cached in checks.items():
                diff = float((cached - plain).abs().max())
                worst = max(worst, diff)
                if diff > tol:
                    print(f"  MISMATCH len={length} {label}: max abs diff {diff:.2e}")
                    return worst, False
    return worst, True


def greedy_equivalence(model, tokens, stories=3):
    to_id = {t: i for i, t in enumerate(tokens)}
    prompt = [to_id["<BOS>"]] + [to_id[t] for t in PROMPT]
    for story in range(stories):
        torch.manual_seed(100 + story)
        cached = generate_story(model, prompt, tokens, model.cfg.ctx, temperature=0.0, use_cache=True)
        torch.manual_seed(100 + story)
        plain = generate_story(model, prompt, tokens, model.cfg.ctx, temperature=0.0, use_cache=False)
        if cached != plain:
            print(f"  MISMATCH story {story}: {len(cached)} cached tokens vs {len(plain)} plain")
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint to test (default: the newest in checkpoints/)")
    ap.add_argument("--vocab", default=None,
                    help="token table (default: the one the checkpoint was trained on)")
    args = ap.parse_args()

    checkpoint = args.checkpoint or default_checkpoint()
    if checkpoint is None or not os.path.exists(checkpoint):
        print("no checkpoint found; train one first (see training/README.md)", file=sys.stderr)
        return 2
    try:
        model, tokens, generation = load(checkpoint, args.vocab)
    except TableMismatch as error:
        print(error, file=sys.stderr)
        return 2
    print(f"checkpoint {os.path.relpath(checkpoint, ROOT)}"
          f"  [{generation.name} table: {generation.vocab}, {len(tokens)} ids]")
    to_id = {t: i for i, t in enumerate(tokens)}
    prompt = [to_id["<BOS>"]] + [to_id[t] for t in PROMPT]

    worst, ok = logits_match(model, tokens)
    print(f"logits: worst max-abs diff {worst:.2e} -> {'ok' if ok else 'FAIL'}")

    same = greedy_equivalence(model, tokens)
    print(f"greedy stories identical with and without the cache: {'yes' if same else 'NO'}")

    timings = {}
    for label, use_cache in (("no cache", False), ("KV cache", True)):
        torch.manual_seed(7)
        started = time.time()
        ids = generate_story(model, prompt, tokens, model.cfg.ctx, temperature=0.8,
                             use_cache=use_cache)
        timings[label] = time.time() - started
        print(f"{label:>9}: {timings[label]:.2f}s for {len(ids)} tokens "
              f"({timings[label] / max(len(ids), 1) * 1000:.1f} ms/token)")
    print(f"speedup: {timings['no cache'] / timings['KV cache']:.1f}x")

    return 0 if (ok and same) else 1


if __name__ == "__main__":
    sys.exit(main())
