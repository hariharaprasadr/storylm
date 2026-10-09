#!/usr/bin/env python3
"""Audit a checkpoint for overfitting and memorisation of the training corpus.

    python training/memorisation.py --checkpoint checkpoints/embedded.pt \
        --corpus data/processed/train.jsonl --packed data/processed/train.npz

The validator answers "would the device show this story?"; it cannot answer
"did the model learn the distribution, or copy the corpus?". These four
measurements can:

1. Overfitting gap. Masked cross-entropy on the rows the model trained on versus
   the held-out 5 % split the training loop itself uses (same seed, same
   permutation). A model that memorised a small corpus shows a wide gap; a model
   that learned the template distribution shows a narrow one.

2. Verbatim copying, story level. For every generated story, the longest run of
   *consecutive* lines that also appears in a single training story, in the same
   order. Copying a corpus story shows up as a run near the full story length;
   independent sampling of the engine's beats shows short runs. The same number
   is computed for corpus stories against the rest of the corpus as a baseline -
   two independent engine stories already share lines, and only a rate above
   that baseline means anything.

3. Line inventory reuse. The fraction of generated lines that exist anywhere in
   the corpus, with the inventory size for context. This one is expected to be
   near 100 %: the model is *supposed* to reproduce the engine's beats word for
   word - that is the design, not a leak.

4. Variety. With only the dice changed, how many distinct stories come out for
   the same prompt? A memorising model collapses to a handful of corpus stories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from storylm.model import Config, StoryLM, decode_lines, generate_story
from storylm.vocab import (TableMismatch, default_checkpoint, series_of,
                           table_for_checkpoint)

THEMES = ["HELP", "SHARE", "BRAVE", "TRY"]  # fallback; the vocab's own list is used below


def themes_from_vocab(tokens):
    """Every <THEME=...> the table knows, so the audit covers the model's own grid."""
    return [t[len("<THEME="):-1] for t in tokens
            if t.startswith("<THEME=") and t.endswith(">")] or THEMES


def norm_line(line):
    """Letters only, lowercased - the comparison the evaluation harness uses."""
    return " ".join("".join(c if c.isalpha() else " " for c in line.lower()).split())


def load_corpus(path):
    with open(path) as handle:
        return [tuple(norm_line(line) for line in row["lines"])
                for row in map(json.loads, handle)]


def build_index(stories):
    """Line ids plus a 2-line-gram index, so runs can be extended cheaply."""
    line_id, ids = {}, []
    for story in stories:
        row = []
        for line in story:
            if line not in line_id:
                line_id[line] = len(line_id)
            row.append(line_id[line])
        ids.append(row)
    grams = defaultdict(list)
    for s, row in enumerate(ids):
        for i in range(len(row) - 1):
            grams[(row[i], row[i + 1])].append((s, i))
    return line_id, ids, grams


def longest_shared_run(row, grams, ids, skip=None):
    """Longest run of consecutive lines of `row` appearing in one other story."""
    best = 0
    for i in range(len(row) - 1):
        for s, j in grams.get((row[i], row[i + 1]), ()):
            if s == skip:
                continue
            left = 0
            while i - left - 1 >= 0 and j - left - 1 >= 0 and row[i - left - 1] == ids[s][j - left - 1]:
                left += 1
            right = 2
            while i + right < len(row) and j + right < len(ids[s]) and row[i + right] == ids[s][j + right]:
                right += 1
            best = max(best, left + right)
    return best


def masked_ce(model, ids, mask, rows, batch=16):
    """Mean masked cross-entropy over the given packed rows."""
    total, seen = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            idx = rows[start:start + batch]
            x = ids[idx]
            _, loss = model(x, targets=x, loss_mask=mask[idx])
            total += float(loss) * len(idx)
            seen += len(idx)
    return total / max(seen, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint to audit (default: the newest in checkpoints/)")
    ap.add_argument("--corpus", default="data/processed/train.jsonl",
                    help="the corpus this checkpoint was trained on")
    ap.add_argument("--packed", default=None,
                    help="packed copy of that corpus (default: the generation's train_v<N>.npz)")
    ap.add_argument("--vocab", default=None,
                    help="token table (default: the one the checkpoint was trained on)")
    ap.add_argument("--stories", type=int, default=100,
                    help="generated stories (5 worlds x themes x repeats)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
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
    packed_path = args.packed or os.path.join(
        ROOT, "data", "processed", f"train_{generation.name}.npz")

    ckpt = torch.load(checkpoint, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = StoryLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    to_id = {t: i for i, t in enumerate(tokens)}

    print(f"checkpoint {os.path.relpath(checkpoint, ROOT)}"
          f"  [{generation.name} table: {generation.vocab}]")
    print(f"corpus     {os.path.relpath(args.corpus, ROOT)}")
    print(f"packed     {os.path.relpath(packed_path, ROOT)}")

    # ---- 1. overfitting gap ------------------------------------------------
    packed = np.load(packed_path)
    ids = torch.from_numpy(packed["ids"].astype(np.int64))
    mask = torch.from_numpy(packed["mask"])
    count = len(ids)
    # exactly the split train_from_packed() builds, so val is the same rows
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    perm = torch.randperm(count)
    n_val = max(1, count // 20)
    train_idx, val_idx = perm[:-n_val], perm[-n_val:]
    ce_train = masked_ce(model, ids, mask, train_idx)
    ce_val = masked_ce(model, ids, mask, val_idx)
    print("\n1. overfitting gap (masked cross-entropy, nats)")
    print(f"   training rows   {ce_train:6.3f}   ({len(train_idx)} rows)")
    print(f"   held-out 5 %    {ce_val:6.3f}   ({len(val_idx)} rows)")
    print(f"   gap             {ce_val - ce_train:+6.3f}   "
          f"(0 = no detectable overfitting; a memorising model drifts far above train)")

    # ---- corpus index ------------------------------------------------------
    stories = load_corpus(args.corpus)
    line_id, story_ids, grams = build_index(stories)
    print(f"\n   corpus: {len(stories)} stories, {len(line_id)} distinct lines")

    # ---- generate stories --------------------------------------------------
    series = {e["id"]: e for e in series_of(generation.series_path())}
    proper = {sid: {e["story_name"].lower()} | {f.lower() for f in e["friends"]}
              for sid, e in series.items()}
    themes = themes_from_vocab(tokens)[: max(1, args.stories // (5 * args.repeats))]
    gen = []
    for sid, entry in series.items():
        for theme in themes:
            for _ in range(args.repeats):
                prompt = [to_id["<BOS>"], to_id[entry["token"]], to_id["<MODE=BEDTIME>"],
                          to_id[f"<PLACE={entry['places'][0].upper()}>"], to_id[f"<THEME={theme}>"]]
                out = generate_story(model, prompt, tokens, cfg.ctx,
                                     temperature=args.temperature, top_k=40)
                gen.append((sid, theme, decode_lines(out, tokens, proper[sid])))

    # ---- 2. verbatim copying ----------------------------------------------
    runs, line_hits, line_total = [], 0, 0
    corpus_words = set()
    for story in stories:
        for line in story:
            corpus_words.update(line.split())
    gen_words = word_hits = word_total = 0
    for _world, _theme, lines in gen:
        row = [line_id[n] for n in (norm_line(x) for x in lines) if n in line_id]
        line_hits += len(row)
        line_total += len(lines)
        for line in lines:
            for w in norm_line(line).split():
                word_total += 1
                word_hits += w in corpus_words
        runs.append(longest_shared_run(row, grams, story_ids) if len(row) > 1 else 0)
    runs_t = torch.tensor(runs, dtype=torch.float32)
    print(f"\n2. verbatim copying, story level ({len(gen)} generated stories)")
    print(f"   longest run of consecutive lines shared with one training story:")
    print(f"     max {int(runs_t.max())}  mean {float(runs_t.mean()):.2f}  "
          f"p95 {float(torch.quantile(runs_t, 0.95)):.0f}  "
          f"runs >= 6: {int((runs_t >= 6).sum())}/{len(runs)}")

    g = torch.Generator().manual_seed(args.seed)
    sample = torch.randperm(len(stories), generator=g)[:len(runs)].tolist()
    base = [longest_shared_run(story_ids[s], grams, story_ids, skip=s)
            for s in sample if len(story_ids[s]) > 1]
    base_t = torch.tensor(base, dtype=torch.float32)
    print(f"   baseline, corpus stories against the rest of the corpus:")
    print(f"     max {int(base_t.max())}  mean {float(base_t.mean()):.2f}  "
          f"p95 {float(torch.quantile(base_t, 0.95)):.0f}  "
          f"runs >= 6: {int((base_t >= 6).sum())}/{len(base)}")

    # ---- 3. line inventory reuse ------------------------------------------
    print(f"\n3. line inventory reuse")
    print(f"   {line_hits}/{line_total} generated lines "
          f"({100 * line_hits / max(line_total, 1):.1f} %) exist in the corpus "
          f"({len(line_id)} distinct lines available)")
    print(f"   {word_hits}/{word_total} generated words "
          f"({100 * word_hits / max(word_total, 1):.1f} %) are words the corpus uses "
          f"({len(corpus_words)} distinct corpus words)")

    # ---- 4. variety --------------------------------------------------------
    per_prompt = defaultdict(list)
    for world, theme, lines in gen:
        per_prompt[(world, theme)].append(" | ".join(norm_line(x) for x in lines))
    unique = sum(len(set(v)) for v in per_prompt.values())
    total = sum(len(v) for v in per_prompt.values())
    print(f"\n4. variety per prompt (repeats = {args.repeats})")
    print(f"   {unique}/{total} generated stories are distinct "
          f"({100 * unique / max(total, 1):.1f} %) across {len(per_prompt)} prompts")

    print("\nReading the numbers: a copied corpus story would show a run near the "
          "full story length (23) and a distinct-story rate near 1/repeats. A model "
          "that learned the beats but not the stories sits near the engine baseline "
          "on both, with a narrow loss gap.")


if __name__ == "__main__":
    main()
