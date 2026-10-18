#!/usr/bin/env python3
"""Check the selection tables against the closed vocabulary.

Every value `generateStory()` accepts must exist as a token in data/vocab.json, and every
prop should be sayable: a prop whose words are outside the table can never appear in a story.
Run after changing a world, a place, a prop or the token table:

    python training/check_selection_tables.py

Exit code 1 if a *selectable* value (character, mode, theme, place) is missing from the
table; unreachable props and friends are reported but do not fail, since they only shrink
what a story can mention.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from storylm import corpus_builder as cb  # noqa: E402

WORD = re.compile(r"[a-z']+")


def main() -> int:
    vocab = json.load(open(os.path.join(ROOT, "data/vocab.json")))["tokens"]
    known = set(vocab)

    def missing(tokens):
        return [t for t in tokens if t not in known]

    chars = [c["token"] for c in cb.SERIES_CONFIG]
    modes = [f"<MODE={m.upper()}>" for m in cb.MODES]
    themes = [f"<THEME={t.upper()}>" for t in cb.THEMES]
    places = [f"<PLACE={p.upper()}>" for c in cb.SERIES_CONFIG for p in c["places"]]

    hard = {"character": missing(chars), "mode": missing(modes),
            "theme": missing(themes), "place": missing(places)}
    for what, gaps in hard.items():
        print(f"{what:10} tokens missing: {gaps or 'none'}")

    # props and friends are spoken, not selected: a word outside the table can never appear
    for label, values in (("prop", [p for c in cb.SERIES_CONFIG for p in c["props"]]),
                          ("friend", [f for c in cb.SERIES_CONFIG for f in c["friends"]])):
        dead = []
        for value in values:
            words = WORD.findall(value.lower())
            if any(w not in known for w in words):
                dead.append((value, [w for w in words if w not in known]))
        print(f"{label:10} unreachable: {len(dead)}/{len(values)}"
              + (f" -> {', '.join(f'{v} [{', '.join(w)}]' for v, w in dead)}" if dead else ""))

    places_distinct = sorted({p for c in cb.SERIES_CONFIG for p in c["places"]})
    props_distinct = sorted({p for c in cb.SERIES_CONFIG for p in c["props"]})
    friends_distinct = sorted({f for c in cb.SERIES_CONFIG for f in c["friends"]})
    print(f"\n{len(cb.SERIES_CONFIG)} characters · {len(cb.MODES)} modes · {len(cb.THEMES)} themes · "
          f"{len(places_distinct)} distinct places · {len(friends_distinct)} distinct friends · "
          f"{len(props_distinct)} distinct props")
    combos = sum(len(c.get("modes") or cb.MODES) * len(c["places"]) for c in cb.SERIES_CONFIG) * len(cb.THEMES)
    print(f"prompt combinations: {combos}")

    failures = {k: v for k, v in hard.items() if v}
    if failures:
        print("\nFAIL: selectable values missing from data/vocab.json:", failures)
        return 1
    print("\nOK: every selectable value is in the token table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
