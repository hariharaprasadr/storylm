# Documentation review: gaps and improvements

Scope: every project markdown file -
[`README.md`](../README.md),
the original engineering plan (the predates-this-repo design document),
[`docs/story_format.md`](story_format.md),
[`docs/tiny_model.md`](tiny_model.md),
[`docs/how_it_works.md`](how_it_works.md),
[`training/README.md`](../training/README.md).

Every claim was checked against the code where it could be: `make test`
output, `training/model.py`, `training/configs/embedded_l8_d128.yaml`,
`story_engine.h/.cpp`, the `Makefile` and `.gitignore`.

Each item has a priority: **P0** blocks a correct V1, **P1** should be fixed
before hardware bring-up, **P2** is cleanup.

---

## Part 1 - Gaps

A gap is something missing, contradicted by the code, or contradicted by
another doc.

### 1.1 Design gaps (missing decisions)

| # | Pri | Gap | Where |
|---|---|---|---|
| G1 | P0 | **Model output has no defined path to audio.** The model writes free word sequences from 3,450 words. Narration is "~218 recorded phrases keyed to the engine's templates". There is no rule for speaking a line that matches no template (0.9 % of lines today), and no rule for splicing slot values (`{name}`, `{object}`, `{spot}`) into clips. "Text kept, line skipped in audio" (tiny_model §6) leaves a hole in a bedtime story. | tiny_model §6, §8; plan §11.1 |
| G2 | P0 | **Streaming vs. whole-story validation conflict.** The plan says to generate one line, queue it and narrate while generating, and *not* to generate the whole story first (§10.4). `validate()` only works on a whole story: it needs `<END>`, a word/line floor and a character anchor. So a streamed story could be half-narrated and then rejected. "Engine fallback for the rest" (tiny_model §6) is undefined: no doc says how an engine continuation is joined onto a partial model story. | plan §10.4; story_format §5; tiny_model §6 |
| G3 | P0 | **No evidence that the model beats the dice.** The docs claim it wins on "combination taste, phrasing, ordering", but no metric measures that. Pass rate and 99.1 % line fidelity both reward copying the engine. A model that only reproduces training stories passes every test we have. | tiny_model §1, §4, §9 |
| G4 | P0 | **Memorisation is not measured.** The run is 4,000 steps x 16 = 64 K sequences over ~2,736 training stories (~23 epochs). The 95/5 val split uses the same templates and prompts, so val loss can't detect copying. Nobody checks whether whole generated stories appear verbatim in the corpus. | tiny_model §9; training/README |
| G5 | P0 | **The quantisation is assumed, not built.** "1050 KiB INT4" is `params * 0.53125` in `model.py`. No INT4 quantiser, no FP32-vs-INT4 logit comparison and no INT4 validator pass rate exist. The plan requires all three (§8.5, §9.1, Phase C.5-6). Every headline size and quality number is for FP32. | tiny_model §3; README status |
| G6 | P1 | **The validator ignores MODE, PLACE and THEME.** `character_mismatch` checks only the world. The plan asks for a "character/mode consistency check" (§13.2). A model that ignores `<THEME=BRAVE>` or the selected place passes. | story_format §5; plan §13.2 |
| G7 | P1 | **No variant/seed policy for "regenerate".** The engine is a pure function of (prompt, variant). No doc says how the device picks the variant: hardware RNG? An NVS history to avoid repeats? Sampling seed for the model? So the child-facing "new story every night" promise is unspecified. | story_format §4; plan §2 |
| G8 | P1 | **Punctuation and case come back only for matched lines.** About 35 % of matched lines borrow punctuation from the engine text. Unmatched lines get nothing, so they display with no full stops or commas. The renderer rule is described but not specified (input, output, edge cases). | tiny_model §2, §9 |
| G9 | P1 | **Weights in PSRAM vs. weights in flash conflict.** The plan says "Never copy the entire model to PSRAM unless measured memory permits" (§9.2). tiny_model recommends copying all 1050 KiB to PSRAM (§5). The decision and its trigger condition belong in one place. | plan §9.2; tiny_model §5 |
| G10 | P1 | **Latency estimates count memory bandwidth only.** 2.7-4.5 s/story is weight bytes / PSRAM bandwidth. They leave out INT4 unpack and dequant cost, attention over a growing context, the 3,509-way softmax/sampling and the line-resample retries. | tiny_model §5; plan §6.3 |
| G11 | P1 | **Model binary format is unspecified.** The plan lists suggested header fields. tiny_model's boot check uses "magic + CRC + vocab_count + table CRC", but `data/vocab.json` has no version or hash field (`count, special_count, word_count, tokens` only). Nothing ties a checkpoint to the vocab revision it was trained on. | plan §9.2; tiny_model §5 |
| G12 | P1 | **The model has no on-device UX.** No doc covers button mapping, text font and size for 6-8 year olds on 240x135, page-turn timing synced to audio, screen brightness or dark theme at bedtime, auto-off after the story, sleep timer, volume ramp-down or low-battery behaviour. The "16 words/line" limit is called "a measured display limit", but no display has been measured yet. | plan §2, §13.1; story_format §3 |
| G13 | P1 | **The audio asset plan is not costed against the real line inventory.** "~218 phrases / ~4 min / ~2 MB" predates the expanded pools: `kProblem` 15x4, `kTry` 21x4, `kWarm` 16x4, plus settle/sky/help/quiet/sleep and 5 worlds' openings and closings. Nothing recounts the clips needed, and nothing costs a 3,450-word fallback clip set. | how_it_works §6; tiny_model §8 |
| G14 | P1 | **Acceptance criteria leave out the model.** Plan §15 has no target for validator pass rate, fallback rate, time to first line, INT4 vs FP32 drift or model-off behaviour. | plan §15 |
| G15 | P2 | **Evaluation prompts can't be "never shown during training".** The corpus covers all 2,520 prompt combinations by design, so plan §7.3's held-out prompts don't exist. Holdout has to be by variant or beat combination, and that isn't written down. | plan §7.3 |
| G16 | P2 | **Safety of word *combinations* is not argued.** The vocabulary allowlist stops bad words but not bad sentences built from approved words. With 99 % template fidelity the risk is low, but no doc argues it or bounds it. | story_format §5 |

### 1.2 Process and repository gaps

| # | Pri | Gap |
|---|---|---|
| R1 | P0 | **The project is not a git repository.** The plan and README rely on source control ("ids stable per table revision", "reproducible from a repository checkout"). There is no `.git`. |
| R2 | P0 | **`.env.local` sits in the project root and is not in `.gitignore`.** If a repo is initialised, it will be committed. |
| R3 | P1 | **The trained checkpoint depends on gitignored derived data.** `artifacts/`, `data/vocab.json` and `train.jsonl` are all ignored. A reader can't tell which vocab revision `embedded.pt` expects. |
| R4 | P1 | **No CI.** The plan lists `ci/host-tests.yml` and `firmware-build.yml`, but neither exists. `make test` and the validator pass rate run only by hand. |
| R5 | P1 | **ESP-IDF version not pinned** (plan §10.1, §20.2). Python is pinned (`requirements.txt`: torch 2.14.1, numpy 2.5.3), but ESP-IDF is not. |
| R6 | P2 | **Hardware facts are unverified.** 8 MB flash / 8 MB PSRAM / ES8311 / 240x135 / button count are stated, not cited. how_it_works says "two buttons"; the plan says "physical buttons". |
| R7 | P2 | **Licence status is unresolved.** Five franchise worlds are in the default build and the training corpus. The weights trained on those names are not covered by the "filter the world table" note. TinyStories-1M has no licence tag. |

### 1.3 Factual inconsistencies (docs vs code, or doc vs doc)

| # | Statement | Reality | Fix in |
|---|---|---|---|
| F1 | Per-layer params include "4 attention biases FP16", 384 per layer (tiny_model §3 table); "no biases except attention projections" (§3 text) | `model.py` uses `bias=False` everywhere. The remainder is 2 x 128 RMSNorm = **256** per layer, and the total of 2,024,192 only adds up without biases | tiny_model §3 |
| F2 | Norm type not stated, or "norm" | `nn.RMSNorm` | tiny_model §3, plan §6.2 |
| F3 | Spine is "settle x3" (story_format §2, how_it_works §3) | `kSettlePicks = 4`; the 5040 = 10P4 factor in story_format §6 also implies 4 | story_format, how_it_works |
| F4 | "217-250 words" over the sweep (README, story_format §3) | `make test` prints **201-250** | README, story_format |
| F5 | "Plots come from 16 shared three-beat plans" (README) | Replaced by the 8-phase sleep spine | README "five worlds" |
| F6 | "Choose ~22 beats from pools of 5-10" (tiny_model §8) | Pools hold 10-21 beats (training/README says 15-21) | tiny_model §8 |
| F7 | Worst sequence 301 tokens; output 265-297 (plan §6.2, §13.1; tiny_model §1, §3) | tiny_model §2 measures **283-305** per sequence, **278-300** per story | plan, tiny_model |
| F8 | Story time 2.7-13.3 s (tiny_model §3 table) vs 2.7-4.5 s (plan §6.3, tiny_model §5) | The table mixes both placements; label it | tiny_model §3 |
| F9 | how_it_works §8: the tiny brain "only picks the recipe steps, never writes the words"; §9: tiny brain ❌ | tiny_model chose option B (word-level generator), and it is already trained (README status) | how_it_works §8, §9 |
| F10 | Training table: 57,600 stories, batch 32, wd 0.1, 3-6 epochs (tiny_model §4) | The measured run used 2,880 stories, batch 16, ~23 epochs | Label §4 as "planned", §9 as "actual" |
| F11 | training/README: "--steps 6000 # real run" | Every measured number is from 4,000 steps | training/README |
| F12 | training/README sample log: "step 100 train 5.21 val 4.98" | Real log starts at step 250, 2.970 / 2.949 | training/README |
| F13 | Plan §6.4: special tokens get "fixed deterministic byte encodings" | Word-level ids, no bytes | plan §6.4 |
| F14 | Plan §7.3 / §20.8: "300-1,000 hand-authored stories", "create 100 original stories" | The corpus is engine-generated | plan §7.3, §20 |
| F15 | Plan §8.2 deps (transformers, datasets, tokenizers, sentencepiece...) | `requirements.txt` is torch + numpy | plan §8.2 |
| F16 | Plan §10.2 layout (`docs/architecture.md`, `characters.yaml`, `tools/`, `firmware/main/`...) | Most of it doesn't exist; `series.json`, not YAML | plan §10.2 |
| F17 | Plan §16: `training/evaluate.py`, `tools/export_embedded_model.py --tokenizer artifacts/tokenizer`, `checkpoints/best` | None exist; the checkpoint is `embedded.pt`; there is no tokenizer dir; the venv is created twice | plan §16 |
| F18 | Plan §11.1 example "Milo found a blue key in the garden" | Stale character and prop | plan §11.1 |
| F19 | Plan §13.1 timeout "20-40 s"; §13.3 "keep 20-50 fallback stories" | Generation projects at ≤13 s; the engine replaces a static fallback library | plan §13 |
| F20 | Plan §5.1 "1.0-1.4 MB INT4 model (867 KiB)" | Range and value disagree | plan §5.1 |
| F21 | Plan §10.1 `[web:97][web:102]` | Leftover citation tokens | plan §10.1 |
| F22 | Units mix "768 KB" and "768 KiB" (tiny_model §3) | 786,432 B = 768 KiB | tiny_model |
| F23 | "vocal 3,509" (tiny_model §3) | Typo: vocab | tiny_model |
| F24 | training/README line 60: an orphan copy of the heading "Runtime sampling rules..." | Duplicate line | training/README |
| F25 | Plan §12 partition table uses subtype `spiffs` for the raw model and voice blobs | Should be a custom data subtype (e.g. `0x40`) so no FS driver mounts them | plan §12 |
| F26 | `kMaxLines = 30 // engine maximum is 24` (header) vs "21-22 lines" (docs) | Pick one engine maximum and state it | story_format §3 |

---

## Part 2 - Improvements

### 2.1 Architecture and safety (fixes G1, G2, G6, G8, G16)

1. **Whole-story generation, then validation, then playback.** At ≤13 s per
   story against 111 s of narration, generating everything first costs one
   short "thinking" screen and removes the mid-story rejection problem. Change
   plan §10.4. Keep line streaming as a later optimisation once a streaming
   validator exists.
2. **Constrained decoding at line level.** Give the sampler a trie of
   permitted engine lines (after slot expansion) and mask any token that leaves
   the trie. Every line then has a clip, every line gets its punctuation, and
   `unapproved_word` can never fire. Be honest in the docs that this moves the
   model toward option A (beat selector). Record the trade-off as a decision.
3. **Fallback audio tier, if free lines are kept.** Cost a word-level clip set
   (3,450 words, 8 kHz ADPCM) against the 2.7 MB spare flash. Then the "skip the
   line in audio" behaviour goes away.
4. **Add `theme_mismatch`, `place_mismatch` and `mode_mismatch` reject codes.**
   Anchor them on theme-pool beats and place markers, the same way
   `character_mismatch` uses world-only props.
5. **Specify the renderer** (case, punctuation, cast capitalisation) as its own
   contract section in `story_format.md`, with golden tests. Move ahead with the
   planned case-preserving table (+37 ids).

### 2.2 Model evaluation (fixes G3, G4, G5, G15)

6. **Add three metrics to `sample.py`:**
   - *verbatim-story rate*: generated story == some corpus story
   - *novel-combination rate*: beat sequence not seen in training
   - *prompt adherence*: theme, place and mode beats present
7. **Raise sample size** from 20 to ≥200 stories, spread across all 240 prompts.
   Report confidence intervals, and report the line-resample count separately
   from the pass rate.
8. **Hold out by beat combination, not by random row.** Train at
   `--variants 240` (57,600 stories) to match the planned table, and record
   epochs.
9. **Build the INT4 path now on host** (quantise → dequantise → generate).
   Report FP32 vs INT4 pass rate and fidelity before any firmware work, as plan
   Phase C requires.
10. **Define a model-vs-dice blind comparison** (adult raters, or at least a
    repetition/variety score) so the "beats the dice" claim can be tested.

### 2.3 Firmware readiness (fixes G9-G12, G14)

11. **Write `docs/model_format.md`:** header, tensor table, alignment,
    endianness, vocab SHA-256, CRC. Add `version` and `sha256` to
    `data/vocab.json` and store the vocab hash in the checkpoint.
12. **Write one memory-placement decision** (PSRAM-resident by default, flash
    mmap when free PSRAM < N MB) and delete the conflicting line.
13. **Add a compute term to the latency model** (MACs/token x cycles/MAC with
    PIE SIMD). Name ms/token as the first hardware measurement.
14. **Write `docs/ux.md`:** button map, screens, font size, page timing,
    brightness, auto-off, volume, battery, and the error screens for "model
    off" and "audio missing".
15. **Extend plan §15** with model acceptance criteria: INT4 pass rate ≥ 99 %
    over 200 stories, 0 audio underruns, time to first line ≤ N s, and a
    graceful engine-only mode.
16. **Specify variant and seed selection:** hardware RNG seed, plus an NVS ring
    of the last K (prompt, variant) pairs so recent stories aren't repeated.

### 2.4 Audio assets (fixes G13)

17. **Recount the clip inventory** from the engine tables. Add a
    `storygen --dump-clips` that lists every unique sentence and slot fragment,
    then cost it in minutes and MB at 16 kHz and 8 kHz.
18. **Unblock narration with host-rendered takes** from an offline TTS with a
    licence that allows redistribution of output. Re-record with a human voice
    later. The README already allows "rendered takes".

### 2.5 Repository and process (fixes R1-R7)

19. **Run `git init`, add `.env.local` (and `.env*`) to `.gitignore`**, and
    make the first commit.
20. **Add a `make check-docs` or CI job** that runs `make test`, checks the
    numbers quoted in docs (word range, vocab size, line counts) against
    `storygen` output, and fails on drift. Most of F1-F26 are drift.
21. **Pin ESP-IDF** (e.g. in `firmware/IDF_VERSION`) and cite a datasheet for
    each hardware figure in a new `docs/hardware_notes.md`.
22. **Add a licence matrix:** world → rights holder → allowed in which build,
    plus the rule that weights trained on a world inherit its restriction.

### 2.6 Documentation structure

23. **Split the 928-line plan.** Keep it as the frozen original spec. Move live
    decisions into short ADRs (`docs/decisions/NNN-*.md`): 200-250 words,
    word-level table, build B, PSRAM placement, whole-story generation.
    Sections already superseded (§7.3, §8.2, §10.2, §11.1 example, §13, §16,
    §20) get a one-line "superseded by …" banner instead of being half-edited.
24. **Make one source for each number.** Put measured values (tokens, words,
    lines, params, sizes, pass rates) in a generated `docs/metrics.md` and link
    to it. Stop restating them in five files.
25. **Label every number "planned", "computed" or "measured"**, especially
    tiny_model §4 vs §9 and the latency tables.
26. **Bring how_it_works §8-§9 in line** with the actual build-B model and the
    current status table.
27. **Fix the typos and leftovers:** F21, F23, F24.

---

## Suggested order

1. R1, R2 (git + secrets): minutes.
2. Doc drift F1-F26 plus improvements 24-27: one pass.
3. G3, G4, G5 (evaluation + INT4 on host): decides whether the model ships.
4. G1, G2 decision (improvements 1-2): decides the narration asset list.
5. Improvement 17 (clip inventory), then the audio assets.
6. Firmware docs 11-16, then hardware bring-up per plan Phase A.
