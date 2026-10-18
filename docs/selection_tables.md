# Selection tables: characters, modes, places, themes, props, friends

The complete set of values behind `generateStory()` — the Story Studio, the console call and
the Node module all resolve against these tables. Nothing here is hand-copied: every table
is extracted from the files named in §9 and re-checked against the token table.

Sources of truth:

| what | file | field |
|---|---|---|
| characters, their token, places, modes, cast | `web/story_data.js` | `window.STORY.worlds` |
| global modes, themes | `web/story_data.js` | `window.STORY.modes`, `.themes` |
| props | `storylm/corpus_builder.py` | `SERIES_CONFIG[*]["props"]` |
| friends (and the canonical place lists) | `storylm/corpus_builder.py` | `SERIES_CONFIG[*]["friends"]`, `["places"]` |
| the closed vocabulary every value must live in | `data/vocab.json` | `tokens` |

## 1. The prompt

```text
<BOS><CHAR=BLUEY><MODE=BEDTIME><PLACE=BACKYARD><THEME=HELP>
```

`world`, `mode`, `place` and `theme` are the only prompt parameters, and `seed` controls
sampling. Everything else in a story (props, friends, the lines themselves) is generated.

## 2. Characters — 5 (`world`)

| id | picker label | name in the story | token | description |
|---|---|---|---|---|
| `bluey` | Bluey | Bluey | `<CHAR=BLUEY>` | a six-year-old blue heeler puppy who loves imaginative family play |
| `gabby` | Gabby | Gabby | `<CHAR=GABBY>` | a girl who shrinks down with magical cat ears to explore her dollhouse |
| `numberblocks` | One | One | `<CHAR=ONE>` | a curious little red number block discovering mathematics and friendship |
| `amelia` | Amelia Bedelia | Amelia | `<CHAR=AMELIA>` | an earnest housekeeper who takes every idiom literally, then bakes sweet pies |
| `junie` | Junie B. Jones | Junie | `<CHAR=JUNIE>` | an outspoken first-grader who navigates school adventures with dramatic charm |

`world` accepts the id or the display name — `"Bluey"` resolves to `bluey`.

## 3. Modes — 8

`bedtime` · `adventure` · `funny` · `kindness` · `mystery` · `discovery` · `celebration` · `silly`

All five worlds list all eight. A mode colours the opening, the closing and the problem;
every mode still ends with the child asleep.

## 4. Places — 7 per world, 31 distinct (`place`)

| world | places |
|---|---|
| bluey | backyard · bedroom · cubby · creek · verandah · beach · lounge |
| gabby | bedroom · playroom · window · craftroom · kitchenette · fairygarden · musicroom |
| numberblocks | playroom · garden · stair · numberland · tower · meadow · patternpark |
| amelia | kitchen · garden · livingroom · pantry · parlor · orchard · porch |
| junie | bedroom · hallway · yard · classroom · playground · schoolbus · cafeteria |

A place is valid only for its own world: `place: "garden"` is legal for `numberblocks` and
`amelia`, not for `bluey`. (`generate_story.js` validates against the world's list and throws
with the valid values.)

## 5. Themes — 9 (`theme`)

`help` · `share` · `brave` · `try` · `honesty` · `curiosity` · `patience` · `friendship` · `creativity`

Themes are global — every world accepts every theme.

## 6. Friends — 4–9 per world, 31 distinct

The protagonist is not in this list.

| world | friends |
|---|---|
| bluey | Bingo, Bandit, Chilli, Muffin, Rusty, Mackenzie |
| gabby | Pandy Paws, Cakey, CatRat, MerCat, Pillow Cat, Baby Box |
| numberblocks | Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten |
| amelia | Mr. Rogers, Mrs. Rogers, Alcolu, Mrs. Wragge |
| junie | Lucille, Grace, Herb, Mrs., Ollie, Mr. Scary |

## 7. Props — 12 per world, 53 distinct

| world | props |
|---|---|
| bluey | a warm blanket · a night light · a bedtime storybook · a soft teddy · a woolly quilt · a cosy dressing gown · a sleepy lullaby · a red balloon · a magic featherwand ⚠ · floppy the bunny ⚠ · a bush stick · a striped beach towel ⚠ |
| gabby | a fluffy pillow · a glow star · a cosy blanket · a music box · a sparkly quilt · a tiny lantern · a tiny teapot · a sprinkle shaker · a glitter ribbon · a magical delivery box · a bubble bottle · a storybook |
| numberblocks | a soft blanket · a night light · a star chart · a cuddly teddy · a knitted quilt · a fluffy pillow · a bedtime rhyme · a magic mirror · a stepping block · a balance beam · a geometric cube · a shiny number badge ⚠ |
| amelia | a warm blanket · a hot water bottle · a bedtime book · a night light · a patchwork quilt · a lavender cushion · a mug of cocoa · a fresh lemon pie · a feather duster ⚠ · a tin of dusting powder ⚠ · a wooden rolling pin · a calico apron |
| junie | a teddy bear · a night light · a bedtime book · a fluffy blanket · a stripy quilt · a woolly jumper · a bedtime rhyme · philip johnny bob · a purple crayon · a shiny shoe · a funny pair of glasses · a pink pencil |

⚠ **Six props can never be generated.** They contain a word the closed table does not have —
`featherwand`, `floppy`, `striped`, `badge`, `duster`, `tin` — so those phrases appear in none
of the 1,277,834 corpus lines and the model cannot emit them:

* bluey — a magic featherwand, floppy the bunny, a striped beach towel
* numberblocks — a shiny number badge
* amelia — a feather duster, a tin of dusting powder

Every other prop word is in the table. Either prune these from `SERIES_CONFIG` or add the six
words to `data/vocab.json` and retrain; today they are dead entries in the builder.

## 8. Counts

| quantity | value |
|---|---|
| characters | 5 |
| modes | 8 (all worlds) |
| themes | 9 |
| places | 7 per world, 31 distinct |
| friends | 4–9 per world, 31 distinct |
| props | 12 per world, 53 distinct (47 reachable) |
| **prompt combinations** | **5 × 8 × 7 × 9 = 2,520**, every one present in the token table |

## 9. Verify or regenerate

```bash
# the tables as the builder defines them (characters, friends, places, props, modes, themes)
python - <<'PY'
import sys; sys.path.insert(0, ".")
from storylm import corpus_builder as cb
for c in cb.SERIES_CONFIG:
    print(c["id"], c["name"], c["story_name"], c["token"])
    print("  friends:", ", ".join(c["friends"]))
    print("  places :", ", ".join(c["places"]))
    print("  props  :", ", ".join(c["props"]))
print("modes :", cb.MODES)
print("themes:", cb.THEMES)
PY

# what the API will accept (the export the page and Node both read)
node -e 'globalThis.window=globalThis; require("./web/story_data.js");
  const {worlds, modes, themes} = require("./web/generate_story.js");
  console.log(worlds.map(w => [w.id, w.places.length, w.modes.length]), modes.length, themes.length)'

# every selectable value must exist in the closed table
python training/check_selection_tables.py
```

## 10. What the API validates

`web/generate_story.js`, `select()`:

* `world` — id or display name, matched case-insensitively against `STORY.worlds`;
* `mode` — the world's own list (identical to the global 8);
* `place` — the world's own list (the only world-specific selector);
* `theme` — the global nine.

Anything else throws with the valid values, e.g.
`unknown place "moon" for world "bluey": expected one of backyard, bedroom, cubby, creek, verandah, beach, lounge`.
Props and friends are not selectors; to make them selectable, extend `web/story_data.js`
(export a prop/friend token) and add them to the prompt in `generate_story.js` — which also
means retraining, because the model has never seen those tokens as prompt conditions.

## 11. This is not the device table

`docs/story_format.md` documents the **device engine** (`m5stack-sticks3/tinystories`,
`firmware/components/story_engine`): 5 worlds × 4 modes (`adventure`, `bedtime`, `funny`,
`kindness`) × 4 themes (`help`, `share`, `brave`, `try`) × 3 friends, 3 places and 4 props per
world. StoryLM's model and the tables above are the larger research set (8 modes, 9 themes,
7 places, 12 props). Keep the two apart when generating a corpus or a prompt.
