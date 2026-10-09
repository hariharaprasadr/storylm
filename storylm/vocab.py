#!/usr/bin/env python3
"""Token tables, and the generation each one belongs to.

Ids are positions in a list, so a checkpoint is only usable with the table it was
trained on: pair a 3,589-id model with a 3,509-id table and the sampler reads the
wrong word (or, once it emits an id past the end, raises `IndexError`). Every
entry point that loads a checkpoint resolves its table here, and refuses the
mismatch instead of generating noise.

Generations, all under `data/` (the id counts are the table's own `count`):

| generation | vocab | series | ids | words |
|---|---|---|---|---|
| v1 | `vocab_v1.json` | `series_v1.json` | 752 | 30 control + 722 words |
| v2 | `vocab_v2.json` | `series_v2.json` | 3,589 | 59 + 3,530 |
| v3 | `vocab_v3.json` | `series_v3.json` | 3,589 | 59 + 3,530 (detail beats) |
| v4 | `vocab.json` | `series.json` | 3,509 | 59 + 3,450 |

`data/vocab.json`/`data/series.json` are the working generation (v4) and the ones
`prepare_data.py` and the Makefile use; the `*_v2`/`*_v3` copies pin the earlier
runs, and the `*_v1` pair is the v1 generation recovered from git (`d975929`) so
the v1 checkpoint (`checkpoints/embedded_big.pt`) still runs.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, NamedTuple, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Generation(NamedTuple):
    name: str
    vocab: str  # repo-relative path
    series: str

    def vocab_path(self) -> str:
        return os.path.join(ROOT, self.vocab)

    def series_path(self) -> str:
        return os.path.join(ROOT, self.series)


# Newest first: when two generations share an id count the later one wins.
GENERATIONS: Sequence[Generation] = (
    Generation("v4", "data/vocab.json", "data/series.json"),
    Generation("v3", "data/vocab_v3.json", "data/series_v3.json"),
    Generation("v2", "data/vocab_v2.json", "data/series_v2.json"),
    Generation("v1", "data/vocab_v1.json", "data/series_v1.json"),
)

# Checkpoint search order: newest generation first, so the default is the latest
# model that exists rather than whichever file happens to sort first.
DEFAULT_CHECKPOINTS = (
    "checkpoints/embedded_v4.pt",
    "checkpoints/embedded_v3.pt",
    "checkpoints/embedded_v2_gpu.pt",
    "checkpoints/embedded_v2.pt",
    "checkpoints/embedded.pt",
    "checkpoints/embedded_big.pt",
    "artifacts/checkpoints/embedded_v4.pt",
    "artifacts/checkpoints/embedded_v3.pt",
    "artifacts/checkpoints/embedded_big.pt",
    "artifacts/checkpoints/embedded.pt",
)


class TableMismatch(SystemExit):
    """The table and the model do not agree on the vocabulary."""


def tokens_of(vocab_path: str) -> List[str]:
    with open(vocab_path) as handle:
        return json.load(handle)["tokens"]


def size_of(vocab_path: str) -> int:
    with open(vocab_path) as handle:
        return int(json.load(handle)["count"])


def series_of(path: Optional[str] = None) -> list:
    with open(path or os.path.join(ROOT, "data/series.json")) as handle:
        return json.load(handle)


def default_checkpoint() -> Optional[str]:
    for candidate in DEFAULT_CHECKPOINTS:
        path = os.path.join(ROOT, candidate)
        if os.path.exists(path):
            return path
    return None


def checkpoint_meta(checkpoint: str) -> Tuple[int, dict]:
    """(vocabulary size, provenance) from a checkpoint, with one load."""
    import torch

    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    return int(payload["config"]["vocab"]), dict(payload.get("provenance") or {})


def checkpoint_vocab_size(checkpoint: str) -> int:
    return checkpoint_meta(checkpoint)[0]


def save_checkpoint(path: str, config: dict, model_state, *,
                    vocab_path: Optional[str] = None,
                    data_path: Optional[str] = None) -> None:
    """Write a checkpoint, recording the table it was trained on.

    The pairing cannot always be recovered from the id count alone (v2 and v3
    both have 3,589 ids and different word lists), so the artifact carries its
    own table reference. Readers that predate this fall back to a size match.
    """
    import torch

    payload = {"config": config, "model": model_state}
    if vocab_path:
        payload["provenance"] = {
            "vocab": os.path.relpath(vocab_path, ROOT) if os.path.isabs(vocab_path) else vocab_path,
            "data": os.path.relpath(data_path, ROOT) if data_path and os.path.isabs(data_path) else data_path,
        }
    torch.save(payload, path)


def generations_of(vocab_size: int) -> List[Generation]:
    """Every generation whose table has this many ids (v2 and v3 both have 3,589)."""
    return [g for g in GENERATIONS
            if os.path.exists(g.vocab_path()) and size_of(g.vocab_path()) == vocab_size]


def generation_of(vocab_size: int) -> Optional[Generation]:
    """The newest generation with this many ids, for display only."""
    matches = generations_of(vocab_size)
    return matches[0] if matches else None


def generation_of_path(vocab_path: str) -> Optional[Generation]:
    """The generation a table file belongs to: by path, then by content.

    The content fallback matters because the same table ships under two names
    (`data/vocab.json` and `data/vocab_v4.json`), and a Modal run records the
    name it was given rather than the canonical one.
    """
    real = os.path.realpath(vocab_path)
    for generation in GENERATIONS:
        if os.path.exists(generation.vocab_path()) and os.path.realpath(generation.vocab_path()) == real:
            return generation
    try:
        tokens = tokens_of(vocab_path)
    except (OSError, ValueError, KeyError):
        return None
    for generation in GENERATIONS:
        if os.path.exists(generation.vocab_path()) and tokens_of(generation.vocab_path()) == tokens:
            return generation
    return None


def table_for(vocab_size: int, vocab_path: Optional[str] = None,
              context: str = "") -> Tuple[List[str], Generation]:
    """The token table a model of this vocabulary size must be paired with.

    `vocab_path` pins it explicitly (and is verified); otherwise the provenance
    recorded in the artifact wins, and failing that the size must be unique. No
    table for the size, or two tables with it, is a hard error: guessing would
    silently decode the wrong words.
    """
    if vocab_path:
        size = size_of(vocab_path)
        if size != vocab_size:
            raise TableMismatch(
                f"{context or 'checkpoint'}: vocabulary mismatch - the model has "
                f"{vocab_size} ids but {os.path.relpath(vocab_path, ROOT)} has {size}.\n"
                f"  pass the table the checkpoint was trained on, or see `python cli.py info`"
            )
        generation = generation_of_path(vocab_path)
        return tokens_of(vocab_path), (generation or Generation("?", os.path.relpath(vocab_path, ROOT),
                                                                "data/series.json"))

    matches = generations_of(vocab_size)
    if not matches:
        available = ", ".join(
            f"{g.name} ({g.vocab}, {size_of(g.vocab_path())} ids)"
            for g in GENERATIONS if os.path.exists(g.vocab_path())
        )
        raise TableMismatch(
            f"{context or 'checkpoint'}: no token table with {vocab_size} ids in data/.\n"
            f"  available: {available}\n"
            f"  a checkpoint is only usable with the table it was trained on; re-export or "
            f"re-train against the current one"
        )
    if len(matches) > 1:
        candidates = ", ".join(f"{g.name} ({g.vocab})" for g in matches)
        raise TableMismatch(
            f"{context or 'artifact'}: {vocab_size} ids matches more than one generation "
            f"and none is recorded in the artifact.\n"
            f"  candidates: {candidates}\n"
            f"  pass --vocab <file> to pin it (re-training records it automatically)"
        )
    return tokens_of(matches[0].vocab_path()), matches[0]


def table_for_checkpoint(checkpoint: str, vocab_path: Optional[str] = None) -> Tuple[List[str], Generation]:
    size, provenance = checkpoint_meta(checkpoint)
    context = os.path.relpath(checkpoint, ROOT)
    recorded = provenance.get("vocab")
    if recorded:
        recorded = os.path.join(ROOT, recorded)
        if not os.path.exists(recorded):
            recorded = None
    if not vocab_path:
        vocab_path = recorded
    elif recorded and os.path.realpath(vocab_path) != os.path.realpath(recorded):
        # Overriding is allowed, but two tables of the same width decode the same
        # ids into different words, so say so instead of silently diverging.
        print(f"warning: {context} records {os.path.relpath(recorded, ROOT)} as its table; "
              f"using {os.path.relpath(vocab_path, ROOT)} instead "
              f"({size_of(recorded)} vs {size_of(vocab_path)} ids)", file=sys.stderr)
    return table_for(size, vocab_path, context=context)


def table_for_packed(packed_path: str, vocab_path: Optional[str] = None) -> Tuple[List[str], Generation]:
    """Same, for a packed `.npz` dataset: it records its own table (`vocab_file`),
    and `meta[0]` is the id count for datasets packed before that was recorded."""
    import numpy as np

    archive = np.load(packed_path)
    size = int(archive["meta"][0])
    if not vocab_path and "vocab_file" in archive.files:
        recorded = os.path.join(ROOT, str(archive["vocab_file"]))
        if os.path.exists(recorded):
            vocab_path = recorded
    return table_for(size, vocab_path, context=os.path.relpath(packed_path, ROOT))


def control_values(tokens: Sequence[str], kind: str) -> List[str]:
    """Every `<KIND=...>` value the table carries, lowercased and de-duplicated."""
    prefix = f"<{kind}="
    seen, values = set(), []
    for token in tokens:
        if token.startswith(prefix) and token.endswith(">"):
            value = token[len(prefix):-1].lower()
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


def require(tokens: Sequence[str], names: Sequence[str], what: str) -> None:
    """Fail loudly if the prompt needs a control token this table does not have."""
    to_id = {token: i for i, token in enumerate(tokens)}
    missing = [name for name in names if name not in to_id]
    if missing:
        raise TableMismatch(
            f"{what}: this token table has no {', '.join(missing)}.\n"
            f"  it has: {', '.join(control_values(tokens, 'MODE'))[:200]}\n"
            f"  pick a value from `--list`, or run a checkpoint trained on a wider table"
        )
