# Story format, vocabulary and validation rules

This document is the contract between the story engine, any language model
that replaces or extends it, and the firmware parser that guards the display
and the speaker. Everything here is enforced by
`firmware/components/story_engine/src/story_engine.cpp` and checked by
`make test`.

**Operative budget: 200-250 words per story.** This supersedes the 40-70 word
target in the engineering plan (`m5sticks3_offline_character_storyteller_project.md`
section 2). The device is a bedtime device for 6-8 year olds: one story is
about two minutes of narration and roughly three screens of text.

## 1. Prompt line

One line, no spaces, tokens in this order:

```text
<CHAR=BLUEY><MODE=BEDTIME><PLACE=BEDROOM><THEME=HELP>
```

| Token | Values |
|---|---|
| `<CHAR=..>` | `BLUEY`, `GABBY`, `ONE`, `AMELIA`, `JUNIE` |
| `<MODE=..>` | `ADVENTURE`, `BEDTIME`, `FUNNY`, `KINDNESS` |
| `<PLACE=..>` | the world's own places (see below) |
| `<THEME=..>` | `HELP`, `SHARE`, `BRAVE`, `TRY` |

Places belong to a world, so the `PLACE` value set is the union of the five
worlds' places:

| World | Places |
|---|---|
| `bluey` | `BACKYARD`, `BEDROOM`, `CUBBY` |
| `gabby` | `BEDROOM`, `PLAYROOM`, `WINDOW` |
| `numberblocks` | `PLAYROOM`, `GARDEN`, `STAIR` |
| `amelia` | `KITCHEN`, `GARDEN`, `LIVINGROOM` |
| `junie` | `BEDROOM`, `HALLWAY`, `YARD` |

Every world supports all four modes and every mode ends with the child falling
asleep - the mode colours the opening, the closing and the problem the story
solves, it never changes the fact that this is a bedtime story.

## 2. Story body

```text
<LINE>The moon was up when Bluey carried a soft teddy to the bedroom window.</LINE>
<LINE>Bluey pulled the covers up and got cosy by the bedroom window.</LINE>
...
<LINE>Bluey and Bingo slept on, cosy and warm by the bedroom window.</LINE>
<END>
```

Roughly 21-22 lines, 201-250 words, one screen page per 1-4 wrapped lines. Line
text never contains `<` or `>` and is printable ASCII only: the device can
display it and the tokenizer can encode it byte for byte.

A story is one of five worlds' **sleep spine**, not a loose pile of sentences:

```text
opening -> settle x3 -> sky x2 -> problem x2 -> help -> try x4
        -> warm x3 -> quiet x3 -> sleep x2 -> closing
```

* `opening` and `closing` come from the world (mode-flavoured).
* `problem`, `try` and `warm` come from the theme's pools: the story is always
  about the thing the `<THEME=..>` token promised.
* Beats inside one phase are interchangeable, phases are not, so the wind-down
  always comes last.
* The assembler stops adding beats once the story reaches its word target and
  skips a beat that would overshoot, so length varies a little inside the
  budget instead of being truncated mid-sentence.

Plans reference a world only through slots (`{name}`, `{friend}`, `{object}`,
`{object_the}`, `{spot}`), which is why a new world costs data rather than
code. The cost: the shared beats are tone-neutral, so `amelia` and
`numberblocks` currently read like the other worlds, and their own phase pools
(literal-minded gags, counting) are not written yet.

## 3. Budgets

| Limit | Value | Constant |
|---|---:|---|
| Words per story | 200 min, 250 max | `kMinStoryWords`, `kMaxStoryWords` |
| Lines per story | 13 min, 30 max | `kMinLines`, `kMaxLines` |
| Words per line | 16 max | `kMaxWordsPerLine` |

Measured over all 15,360 world/mode/place/theme/variant combinations in
`make test`: 201-250 words, 21-22 lines, at most 16 words on a line. The
validator's line floor is 13 because 200 words at the 16-word line limit is
13 lines; the engine promises at least 20 and the test sweep enforces that.

The limits are guards: a truncated generation, a stray `<END>` or a
half-finished story must not reach the display and the speaker as if it were
complete. Every word must also be one the narration assets can speak.

## 4. Determinism

`generate(prompt, variant)` is a pure function of the prompt, the variant and
the tables: splitmix64 on integer arithmetic only, with no time, pointer
values, locale or floating point. Beats are drawn with a deterministic
Fisher-Yates shuffle, so the same inputs produce identical bytes on macOS,
Linux and the ESP32-S3 - which is what makes the corpus, the golden test and
on-device debugging comparable.

Within one story the prop is `variant % prop_count`, the helper is
`variant % friend_count`, and every beat is drawn from the seeded stream. A
beat is never used twice in the same story, and the helper always comes from
the story's own cast.

## 5. Validation

`validate(structured_text, expected_prompt)` is applied to engine output, to
model output and to anything read back from storage. It returns the first
failure, in this order:

| Code | Trigger |
|---|---|
| `empty` | no input at all |
| `bad_byte` | byte outside printable ASCII (tab, LF, CR excepted) |
| `missing_end` | no `<END>`, or no `<LINE>` block |
| `bad_token` | text before the first `<LINE>` that is not the prompt line, text between `</LINE>` and the next `<LINE>`, text after `<END>`, or `<`/`>` inside line text |
| `unclosed_line` | `<LINE>` without `</LINE>`, or a nested `<LINE>` |
| `too_many_lines` | more than 30 lines |
| `line_too_long` | a line with more than 16 words |
| `too_many_words` | more than 250 words in the story |
| `repeated_line` | two lines equal after case folding and space collapsing |
| `unapproved_word` | a word outside the approved vocabulary |
| `character_mismatch` | the world's protagonist is never named, or no word from that world appears |
| `too_few_lines` | fewer than 13 lines |
| `too_few_words` | fewer than 200 words in the story |

Safety comes before completeness on purpose: an unsafe word or the wrong hero
is reported even when the story is also too short. Structure and safety are
checked first, the length guards last.

The approved vocabulary is derived from the same tables that produce the
stories: every spine beat, every world's opening and closing templates, plus
every prop, place marker, friend name and protagonist name. It currently holds
722 words and is exposed through `is_approved_word()` and
`approved_vocabulary_size()`.

The same tables also generate the model's token table (`make vocab`,
`data/vocab.json`): 752 ids = 30 control tokens + those 722 words, sorted and
ids stable per table revision. See [`tiny_model.md`](tiny_model.md).

`unapproved_word` is the safety-critical rule. Structure alone does not make
text child-safe or narratable: `Bluey found a knife near the bedroom window.`
is structurally perfect and is rejected only because `knife` has no entry in
the vocabulary and no narration asset. Any word the firmware cannot speak is a
word the device must not say.

`character_mismatch` needs two anchors because one world's cast is named after
ordinary words: a Numberblocks text full of "one" and "two" would satisfy a
name check by accident. A story must name its protagonist **and** contain at
least one prop or place that only exists in that world.

## 6. Corpus format

`make corpus` writes one JSON object per line, balanced across every world,
mode, place and theme:

```json
{"series": "bluey", "mode": "bedtime", "place": "bedroom", "theme": "help",
 "variant": 0, "prompt": "<CHAR=BLUEY>...",
 "lines": ["The moon was up when Bluey carried a soft teddy to the bedroom window.", "..."],
 "text": "<CHAR=BLUEY>...\n<LINE>...</LINE>\n<END>\n"}
```

Default output is 5 worlds x 4 modes x 3 places x 4 themes x 12 variants =
2,880 rows, and **every row is a distinct story** (measured; the same holds at
`--variants 240` = 57,600 rows, all distinct). Median story: 234 words, 1,612
bytes of text, about 111 seconds of narration.

Unlike the earlier short-story engine, raising `--variants` now genuinely
grows the corpus: the phase pools give 5040 x 30 x 20 x 6 x 360 x 60 x 5040 x 56
ordered combinations per (world, mode, place, theme). The corpus grows further
by adding pool lines, casts and worlds.

## 7. Adding a world or a beat

1. Add a `Series` entry (id, label, story name, token, description, rights,
   default mode, places, props, cast, opening and closing templates), or a new
   line to a spine pool (`kSettle`, `kSky`, `kProblem`, `kHelp`, `kTry`,
   `kWarm`, `kQuiet`, `kSleep`).
2. Keep templates inside the existing vocabulary where possible; a new word is
   added to the approved vocabulary automatically, but it also needs a
   narration take before the device can speak it.
3. Run `make test`. The suite fails on unknown slots, non-ASCII bytes, budget
   violations, duplicate lines in one story, a story that names nobody from its
   own world, and any output the validator would reject. The sweep covers
   15,360 stories, so a template that breaks the 200-250 word band is caught.
4. Re-run `make corpus` and `make series` so the derived files match.
