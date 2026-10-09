#!/usr/bin/env python3
"""Story validation and canonicalisation against the current tables, in Python.

The firmware's `storygen` is the authority on the device, but it is built from the
*device project's* tables. Once the vocabulary changed (752 ids -> 3,509), the
binary in `build/storygen` could no longer judge a story from this repository, so
evaluation here needs the same two services against the current tables:

    validate(...)      the firmware's `storygen --check`
    canonicalize(...)  the firmware's `storygen --canonicalize`

`engine_lines()` rebuilds the per-prompt beat inventory from the same banks the
corpus builder writes stories from, so a draft line can be matched exactly,
repaired to a near neighbour, or left unmatched, exactly as the runtime does.

CLI, as a drop-in for the firmware binary on the host:

    python -m storylm.validator --check --series bluey --mode bedtime \
        --place bedroom --theme help < story.txt
    python -m storylm.validator --canonicalize --series gabby --mode funny \
        --place playroom --theme share < draft.txt

`run_check()` / `run_canonicalize()` are the helpers the training and story CLIs
call: they prefer the C++ engine when it exists *and* its vocabulary matches the
table in `data/vocab.json`, and fall back to this module otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from storylm import corpus_builder as cb

MAX_LINE_WORDS = 16
MIN_LINES = 13
MAX_EDIT_DISTANCE = 3          # the runtime repairs within three word edits
SERIES_IDS = {"one": "numberblocks", "bluey": "bluey", "gabby": "gabby",
              "amelia": "amelia", "junie": "junie"}


def _load(path: str, default):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return default
    with open(full) as handle:
        return json.load(handle)


def vocabulary(path: str = "data/vocab.json") -> Tuple[Set[str], Dict[str, int]]:
    data = _load(path, {"tokens": []})
    tokens = data.get("tokens", [])
    words = {t for t in tokens if not (t.startswith("<") and t.endswith(">"))}
    return words, {t: i for i, t in enumerate(tokens)}


def series_map(path: str = "data/series.json") -> Dict[str, dict]:
    return {entry["id"]: entry for entry in _load(path, [])}


def words_of(line: str) -> List[str]:
    """Word split used for comparison: letters only, lowercased."""
    import re
    return re.findall(r"[a-z]+", line.lower())


def normalize(line: str) -> str:
    return " ".join(words_of(line))


def engine_lines(series_id: str, mode: str, place: str, theme: str,
                 series: Optional[Dict[str, dict]] = None) -> Dict[str, str]:
    """Every line the engine can render for this prompt, keyed by its word form.

    Mirrors the runtime's `engine_lines()`: every beat of every phase the prompt
    can use, filled with every prop and friend that world allows. The detail
    beats the corpus builder adds are *not* part of the device inventory, so
    lines made of them correctly count as unmatched.
    """
    series = series or series_map()
    cfg = series[SERIES_IDS.get(series_id, series_id)]
    name = cfg["story_name"]
    out: Dict[str, str] = {}

    banks: List[str] = []
    banks += cb.OPENINGS.get((cfg["id"], mode), cb.OPENINGS[("bluey", "bedtime")])
    banks += cb.SETTLE_BEATS + cb.SKY_BEATS + cb.HELPER_BEATS + cb.QUIET_BEATS + cb.SLEEP_BEATS
    banks += cb.CHARACTER_PROBLEMS.get(cfg["id"], {}).get(theme, [])
    banks += cb.TRY_BEATS.get(theme, cb.TRY_BEATS["help"])
    banks += cb.WARM_BEATS.get(cfg["id"], cb.WARM_BEATS["bluey"])
    banks.append(cb.CLOSINGS.get((cfg["id"], mode), cb.CLOSINGS[("bluey", "bedtime")]))

    for template in banks:
        for friend in cfg["friends"]:
            for prop in cfg["props"]:
                line = template.format(name=name, friend=friend, place=place, prop=prop)
                out.setdefault(normalize(line), line)
    return out


def _edit_distance(a: Sequence[str], b: Sequence[str], cap: int = MAX_EDIT_DISTANCE) -> int:
    """Word-level Levenshtein distance, abandoned once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        best = i
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
            best = min(best, cur[j])
        if best > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def canonicalize(series_id: str, mode: str, place: str, theme: str,
                 lines: Sequence[str]) -> Tuple[List[str], str]:
    """Return (canonicalised lines, 'exact N, repaired M, unmatched K (inventory L)')."""
    inventory = engine_lines(series_id, mode, place, theme)
    canonical: List[str] = []
    exact = repaired = unmatched = 0
    for line in lines:
        key = normalize(line)
        if key in inventory:
            canonical.append(inventory[key])
            exact += 1
            continue
        draft_words = key.split()
        best, best_distance = None, MAX_EDIT_DISTANCE + 1
        for candidate_key, candidate_text in inventory.items():
            distance = _edit_distance(draft_words, candidate_key.split())
            if distance < best_distance:
                best, best_distance = candidate_text, distance
        if best is not None and best_distance <= MAX_EDIT_DISTANCE:
            canonical.append(best)
            repaired += 1
        else:
            canonical.append(line)
            unmatched += 1
    return canonical, (f"{len(canonical)} lines, exact {exact}, repaired {repaired}, "
                       f"unmatched {unmatched} (inventory {len(inventory)})")


def validate(series_id: str, mode: str, place: str, theme: str,
             lines: Sequence[str], vocab_path: str = "data/vocab.json") -> Tuple[bool, str]:
    """The firmware's `--check`: returns (ok, message) with a reject code."""
    approved, _ = vocabulary(vocab_path)
    series = series_map()
    cfg = series[SERIES_IDS.get(series_id, series_id)]

    for position, line in enumerate(lines, 1):
        for word in words_of(line):
            if word and word not in approved:
                return False, f"reject unapproved_word line {position}: word not in approved vocabulary: {word}"
        if len(line.split()) > MAX_LINE_WORDS:
            return False, f"reject line_too_long line {position}: line exceeds word budget"
    if len(lines) < MIN_LINES:
        return False, "reject too_few_lines: story is shorter than the minimum"

    seen: Set[str] = set()
    for position, line in enumerate(lines, 1):
        key = normalize(line)
        if key in seen:
            return False, f"reject repeated_line line {position}: line already used"
        seen.add(key)

    text = " ".join(normalize(line) for line in lines)
    heroes = {cfg["story_name"].lower()} | {f.lower() for f in cfg["friends"]}
    if not any(name in text.split() for name in heroes):
        return False, "reject character_mismatch: no word from the expected world"

    return True, "ok"


# ---------------------------------------------------------------------------
# CLI and the helpers the training / story CLIs call
# ---------------------------------------------------------------------------

def _read_payload(stream) -> List[str]:
    """Parse '<LINE>text</LINE>' lines from a story payload.

    The engine's protocol wraps every line; a plain story (one line per stdin
    line, as in the usage examples) is accepted too, so the CLI can be pointed at
    a draft file without wrapping it first.
    """
    raw_lines = [raw.rstrip("\n") for raw in stream]
    lines = [raw[raw.index("<LINE>") + len("<LINE>"):raw.index("</LINE>")]
             for raw in raw_lines if "<LINE>" in raw and "</LINE>" in raw]
    if lines:
        return lines
    return [raw.strip() for raw in raw_lines
            if raw.strip() and not raw.strip().startswith("#")]


def _engine_vocabulary_size() -> Optional[int]:
    """How many ids the C++ engine was built with, or None if it is not usable."""
    binary = os.path.join(ROOT, "build", "storygen")
    if not os.path.exists(binary):
        return None
    try:
        result = subprocess.run([binary, "--dump-vocab"], capture_output=True, text=True,
                                cwd=ROOT, timeout=30)
        return int(json.loads(result.stdout).get("count"))
    except Exception:
        return None


def engine_matches_table(vocab_path: str = "data/vocab.json") -> bool:
    """True when build/storygen exists *and* was built on the current table."""
    _, table = vocabulary(vocab_path)
    size = _engine_vocabulary_size()
    return size is not None and size == len(table)


def run_check(series_id: str, mode: str, place: str, theme: str,
              payload_lines: Sequence[str], lines_for_validation: Sequence[str],
              vocab_path: str = "data/vocab.json") -> str:
    """The `--check` verdict, via the C++ engine when compatible, else in Python."""
    if engine_matches_table(vocab_path):
        payload = "".join(f"<LINE>{l}</LINE>\n" for l in payload_lines) + "<END>\n"
        result = subprocess.run(
            ["./build/storygen", "--check", "--series", series_id, "--mode", mode,
             "--place", place, "--theme", theme],
            input=payload, capture_output=True, text=True, cwd=ROOT,
        )
        return result.stdout.strip()
    ok, message = validate(series_id, mode, place, theme, lines_for_validation, vocab_path)
    suffix = "" if engine_matches_table(vocab_path) else "  (python validator: no compatible storygen)"
    return f"{message} (exit {0 if ok else 2}){suffix}"


def run_canonicalize(series_id: str, mode: str, place: str, theme: str,
                     lines: Sequence[str], vocab_path: str = "data/vocab.json"
                     ) -> Tuple[List[str], str]:
    """Canonicalised lines plus the tally, via the engine when compatible."""
    if engine_matches_table(vocab_path):
        payload = "".join(f"<LINE>{l}</LINE>\n" for l in lines) + "<END>\n"
        result = subprocess.run(
            ["./build/storygen", "--canonicalize", "--series", series_id, "--mode", mode,
             "--place", place, "--theme", theme],
            input=payload, capture_output=True, text=True, cwd=ROOT,
        )
        out = [l[l.index("<LINE>") + len("<LINE>"):l.index("</LINE>")]
               for l in result.stdout.splitlines() if l.startswith("<LINE>")]
        return (out or list(lines)), result.stderr.strip().split(": ", 1)[-1]
    return canonicalize(series_id, mode, place, theme, lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Python drop-in for the firmware story validator")
    ap.add_argument("--check", action="store_true", help="validate a story from stdin")
    ap.add_argument("--canonicalize", action="store_true", help="repair a draft from stdin")
    ap.add_argument("--series", default="bluey")
    ap.add_argument("--mode", default="bedtime")
    ap.add_argument("--place", default="bedroom")
    ap.add_argument("--theme", default="help")
    ap.add_argument("--vocab", default="data/vocab.json")
    args = ap.parse_args()

    lines = _read_payload(sys.stdin)
    if args.canonicalize:
        canonical, stats = canonicalize(args.series, args.mode, args.place, args.theme, lines)
        for line in canonical:
            sys.stdout.write(f"<LINE>{line}</LINE>\n")
        sys.stderr.write(f"canonicalize: {stats}\n")
        return 0
    ok, message = validate(args.series, args.mode, args.place, args.theme, lines, args.vocab)
    print(message)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
