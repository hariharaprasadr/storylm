# StoryLM: 2.02M Parameter Tiny Language Model

An edge-native, ultra-compact generative language model engineered for resource-constrained microcontrollers and embedded devices (such as the ESP32-S3), accompanied by a mathematical forward-pass tracer and an interactive web visualizer.

![Architecture Overview](web/preview.jpg) <!-- (or open web/index.html) -->

---

## Backstory: from a cloud microphone to a 2.02M model

**It started with streaming audio to the cloud.** My first experiments with
pocket-sized ESP32-S3 boards (a tiny screen, a button, a microphone and a
speaker) took the obvious shape: stream the microphone to a server, run
speech-to-text, a cloud LLM and text-to-speech there, stream the audio back.
The audio pipeline worked. But every demo exposed the same failure: without
Wi-Fi, a gateway and someone else's API, the gadget was a paperweight — and
every word spoken to it left the house.

**The requirements flipped the design.** The real goal was a bedtime device
for a 6-8 year old: the child picks a character and a mode, and the device
writes and narrates a 200-250 word story that always ends with everyone asleep.
The plan allowed *no cloud service, no LAN server, no runtime Internet after
flashing*. With the network gone, neural TTS is out too, so narration has to
be pre-recorded clips — which means every word the device can ever say must be
known in advance.

**A story engine came before the model.** The first build was a deterministic
C++ story engine: five character worlds, a fixed "sleep spine" plot skeleton,
sixteen beat tables and a validator. It solved narration and safety, but it is
dice: beats are picked from pools, and every story is assembled, never written.

**The engine became the teacher.** The same tables generate the training
corpus and the entire vocabulary: the current v4 table is 3,509 ids = 59 control
tokens + 3,450 words (v1 was 752 ids = 30 control tokens + 722 words), closed by
construction, so a model trained on it can never emit a word without a narration
clip. Two architectures were on the table: a ~50K-parameter beat selector that
only picks recipe steps, or a word-level generator that writes the text itself.
The beat selector adds little over the dice; the word-level model is the whole
point. So a decoder-only Transformer (8 layers, d=128, tied embeddings, RoPE,
RMSNorm, GELU, no biases) was trained from scratch: the v1 model (752 ids,
1.67M parameters) a few thousand steps on a laptop CPU for the baseline
checkpoint, then the expanded 57,600-story corpus on a rented T4 GPU for the big
one. That lineage now ships as v4: 2,024,192 parameters (2.02M) on the 3,509-id
table, trained on a Modal T4 via `make train-modal`.
Evaluation was never perplexity — sampled stories go back through the engine's
canonicaliser and the firmware validator, the same gate the device will use.

**That model is StoryLM.** Somewhere along the way the interesting artifact
stopped being the appliance and became the model: a complete language model —
tokenizer, training loop, checkpoints, a KV cache proven token-identical to
full recompute (2.7x faster on this laptop, §4), a forward-pass tracer and a 30-step browser
walkthrough — that fits in 1050 KiB of INT4 flash. This repo holds it.

## Key Specifications

| Engineering Metric | StoryLM (Tiny SLM) | Standard Local LLM | Frontier Cloud LLM |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | **2,024,192 (2.02M)** | 7,000,000,000 (7B) | ~1,800,000,000,000 (1.8T) |
| **Layers / Dim / Heads** | **8 layers / d=128 / 8 heads** | 32 layers / d=4096 / 32 heads | 96+ layers / d=12288+ |
| **Vocabulary** | **3,509 tokens** (closed-domain, v4) | 32,000–128,000 (BPE) | 100,000+ |
| **Weight Footprint** | **7.7 MB** (FP32) / **1050 KiB** (INT4) | ~4.2 GB (INT4) | Distributed across clusters |
| **KV-Cache RAM** | **512 KiB** (INT8, at ctx=256) | ~512 MB – 1 GB | Multi-Gigabyte VRAM |
| **Target Hardware** | **Microcontroller (ESP32-S3 / Cortex-M)** | Desktop GPU / Apple M-series | Multi-H100/TPU Datacenter |
| **Power Budget** | **0.5 W – 1.5 W** | 25 W – 150 W | 700 W per GPU |

---

## Repository Structure

```
storylm/
├── README.md                     # Project documentation & quickstart
├── cli.py                        # Unified command-line interface
├── requirements.txt              # Python dependencies (torch, numpy, pyyaml)
├── pyproject.toml                # Standard python packaging
├── Makefile                      # Convenient CLI targets (story, test, web, info, train, train-modal, audit, corpus)
│
├── storylm/                      # Core Model Library
│   ├── __init__.py               # Package exports
│   ├── model.py                  # StoryLM architecture, RoPE, Attention, KV-Cache
│   ├── vocab.py                  # Pairs a checkpoint with the token table it was trained on
│   ├── corpus_builder.py         # 3.5K-vocabulary corpus generator & word classifier
│   ├── validator.py              # validate/canonicalise a story (drop-in for storygen)
│   ├── story.py                  # Standalone prompt & story generation engine
│   └── test_cache.py             # KV-cache equivalence & speed benchmark
│
├── checkpoints/                  # Trained PyTorch Checkpoints (one per generation)
│   ├── embedded_v4.pt            # v4 table (3,509 ids) - made by `make train-modal`
│   ├── embedded_v3.pt            # v3 table (3,589 ids)
│   ├── embedded_v2_gpu.pt        # v2 table (3,589 ids), GPU run
│   ├── embedded_v2.pt            # v2 table (3,589 ids), local run
│   ├── embedded_big.pt           # v1 (752 ids) trained on the expanded corpus
│   ├── embedded.pt               # v1 (752 ids) baseline checkpoint
│   └── smoke.pt                  # 300-step local smoke run (never a shipped checkpoint)
│
├── configs/
│   └── embedded_l8_d128.yaml     # Architectural hyperparameter config
│
├── data/
│   ├── vocab.json                # 3,509-token closed dictionary (current, v4)
│   ├── series.json               # Series prompts, place IDs & theme combinations
│   ├── vocab_v1.json..vocab_v4.json    # Pinned token tables (v1 = 752 ids)
│   ├── series_v1.json..series_v4.json  # Pinned series definitions per generation
│   └── processed/
│       ├── train.jsonl           # Current v4 raw corpus, 57,960 stories / 52,183 distinct lines (~208 MB, gitignored)
│       ├── train_v2.jsonl        # v2 raw corpus
│       ├── train_v4.npz          # Pre-tokenized uint16 dataset for v4 (~3.8 MB)
│       ├── train_v3.npz          # v3 packed dataset
│       ├── train_v2.npz          # v2 packed dataset
│       └── corpus/               # The 57,960-story corpus as 5 gzipped shards (8.2 MB)
│                                 # `make corpus-join` rebuilds train.jsonl from them
│
├── training/                     # Training & Tracing Pipelines
│   ├── train_model.py            # Local PyTorch training loop (AdamW, Cosine LR)
│   ├── prepare_data.py           # Corpus tokenization & .npz dataset packing
│   ├── export_story.py           # Full fp16 weights + prompts -> web/story_data.js
│   ├── export_matrix_toy.py      # 4-dim toy block + reference numbers -> web/matrix_toy_data.js
│   ├── modal_train.py            # Scalable Cloud GPU training script (Modal)
│   ├── memorisation.py           # Overfitting / memorisation audit (`make audit`)
│   ├── check_selection_tables.py # Every selectable value vs the closed token table
│   └── sample.py                 # Evaluation & sampling harness
│
├── web/                          # Standalone Interactive Web Visualizer
│   ├── index.html                # Main page: Story Studio, Architecture, Comparison
│   ├── walkthrough.html          # 30-step forward-pass walkthrough (linked from the main page)
│   ├── matrix_walkthrough.html   # the same block at 4 dimensions, cell by cell, with an inspector
│   ├── matrix_toy_data.js        # the toy block's weights (exported from the real checkpoint)
│   ├── styles.css                # Shared page styles
│   ├── walkthrough.css           # Walkthrough-specific styles
│   ├── generate_story.js         # generateStory({world,mode,place,theme,seed}) - the model, no DOM
│   ├── story.js                  # Story Studio widget: chips + reveal, calls generateStory
│   ├── story_data.js             # Real checkpoint weights, fp16 + base64 (~4.3 MB)
│   ├── walkthrough.js            # 30-step numeric walkthrough UI (real values, real arithmetic)
│   ├── wt-trace.js               # Traces the forward pass live from the weights at page load
│   └── preview.jpg               # Full-page screenshot of the visualizer
│
└── docs/                         # Technical Specs & Architecture Notes
    ├── training.md               # End-to-end training pipeline (corpus -> pack -> train -> evaluate)
    ├── tiny_model.md             # 2.02M model architecture, memory budget & vocabulary scaling
    ├── story_format.md           # Prompt tags & story structure specs
    ├── selection_tables.md       # Every character, mode, place, theme, prop and friend
    ├── how_it_works.md           # Engineering deep-dive
    └── review_gaps_and_improvements.md # Model analysis & future roadmap
```

---

## Quickstart

### 1. Installation
Clone or navigate to the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Stories
Generate an interactive story from the pre-trained checkpoint in under 1 second:
```bash
# Default bedtime story for Bluey
python cli.py story

# Customize character, world, mode, place, and theme
python cli.py story --world bluey --mode bedtime --place backyard --theme help

# List all available characters, places, and modes
python cli.py story --list
```
The default checkpoint is the newest generation present in `checkpoints/`. Every entry
point pairs a checkpoint with the token table it was trained on and refuses a mismatch:
`python cli.py info` prints the generation each checkpoint maps to, and `--vocab` overrides
the table when you want to run a pinned generation explicitly.

### 3. Launch the Interactive Web Explainer
Explore the inner workings of the model in your browser — live at
[storylm.vercel.app](https://storylm.vercel.app/):
```bash
python cli.py web
```
This serves `web/index.html`, which features:
- **Story Studio — Live In-Browser Generation**: the page ships the real checkpoint weights (fp16, 5.4 MB, from `checkpoints/embedded_v4.pt`) and a vanilla-JS port of the model with a KV cache; pick character, mode, place and theme, and a brand-new story is generated live in the browser (measured on an idle laptop: 290-299 tokens in 0.6-0.7 s, ~450 tok/s, no server; the page prints its own token rate after every story).
- **Interactive Dataflow Diagram**: View tensor shapes through Embeddings, 8x Transformer Layers, and Output Head.
- **Component Breakdown**: Deep dives into RMSNorm, RoPE, Preallocated KV-Cache, and GELU FFN.
- **"Under the Hood" Walkthrough** (on `walkthrough.html`, linked from the main page): a 30-step slowed-down execution of a real forward pass — computed live in the browser from the same fp16 weights Story Studio runs, following the chips you selected there (the walkthrough links carry the selection; bare URLs use the defaults), and showing the actual numbers and arithmetic instead of pre-baked graphics:
  - Walk the real matrices cell by cell — and click any matrix to pop it out: the full tensor opens in a scroll-through explorer (wheel/arrow keys pan rows, shift+wheel or drag pans columns).
  - See the extra axes in 3D: any layer weight window stacked across all 8 layers, and the attention tensor as 8 head planes (drag to rotate; the plane picker brings any head to the front — nothing stays hidden at the back).
  - Follow the residual stream as a stages × 128-dim matrix, with the real q·k dot products, √d scores and softmax weights per head.
  - Watch FFN feature detectors switch on, with the real up/GELU/down arithmetic and a movement table that adds up.
  - Peek via the "logit lens" at what the model would guess after every layer.
  - Map all 3,509 output logits back onto the vocabulary — one cell per word, shaded by its actual score — then roll the seeded sampling dice and read the first line it writes.
- **Narration** (the clips, the lab page and the whole render pipeline) now lives in the device project: `m5stack-sticks3/tinystories` — `docs/narration.md` there, page at `web/narration.html`. It moved when the device project became the place the audio is shipped from; this repository keeps only the story engine, the model and the measurement of what narration costs.

- **One function for the model** (`web/generate_story.js`, no DOM, no build step): the same forward pass the Story Studio runs, callable from a page, the console or Node.

  ```js
  const story = await generateStory({
    world: "bluey",      // world id or display name: bluey, gabby, numberblocks, amelia, junie
    mode: "bedtime",     // one of the world's modes
    place: "backyard",   // one of the world's places
    theme: "help",       // one of the nine themes
    seed: 42             // same seed -> the same story, word for word
  });
  story.lines;  // ["Evening settled quietly over the house as Bluey stood in the backyard.", ...]
  story.text;   // the lines joined by "\n"   ·   story.tokens, story.ms
  ```

  Browser: `<script src="story_data.js"></script><script src="generate_story.js"></script>` then
  call it from the console (the live site has it). Node:

  ```js
  globalThis.window = globalThis;                 // story_data.js writes window.STORY
  require("./web/story_data.js");
  const { generateStory } = require("./web/generate_story.js");
  ```

  Bad inputs throw with the valid list (`unknown place "moon" for world "bluey": expected one of
  backyard, bedroom, cubby, ...`), and `StoryEngine` still exposes the low-level pieces
  (`step`, `sample`, `weights`, `config`, `vocab`) that the walkthrough pages trace.
  The full table of characters, modes, places, themes, props and friends is
  [`docs/selection_tables.md`](docs/selection_tables.md).
- **StoryLM vs. Regular LLM**: Comprehensive comparative analysis matrix and architectural pillars.

### 4. Verify KV-Cache Parity & Performance
Verify that KV-cache output is numerically identical to standard full-context recomputation:
```bash
python cli.py test
```
*Sample output (the v4 checkpoint):*
```
checkpoint checkpoints/embedded_v4.pt  [v4 table: data/vocab.json, 3509 ids]
logits: worst max-abs diff 1.91e-05 -> ok
greedy stories identical with and without the cache: yes
 no cache: 1.59s for 289 tokens (5.5 ms/token)
 KV cache: 0.60s for 289 tokens (2.1 ms/token)
speedup: 2.7x
```
The diff is float32 summation order, not cache error: both paths recompute the final
line, and greedy stories are token-identical.

---

## Training from Scratch

The full pipeline — where the corpus comes from, how it is packed to the device's
exact token stream, every hyperparameter, the Modal GPU run, and how the result is
judged — is documented in [`docs/training.md`](docs/training.md).

### Preprocess the Data
Encode the raw story corpus into tokenized `.npz` arrays (`make corpus` rebuilds the
corpus from `storylm/corpus_builder.py` and then packs it exactly this way):
```bash
python training/prepare_data.py \
  --corpus data/processed/train.jsonl \
  --vocab data/vocab.json \
  --out data/processed/train_v4.npz
```

### Train
Smoke runs are local and prove the plumbing; **real training runs on a Modal GPU**
(`training/train_model.py` refuses more than 500 CPU steps unless `--allow-cpu` is given).

```bash
# Smoke test, local, plumbing only (300 steps, ~1 min; writes checkpoints/smoke.pt)
python training/train_model.py --data data/processed/train_v4.npz --steps 300

# Real training run: Modal T4, volumes + upload + run + download
make train-modal                     # DATA=/VOCAB=/NAME=/STEPS=/BATCH= override the defaults
```

The smoke run is never a shipped checkpoint; `make train-modal` defaults to
`DATA=train_v4.npz`, `VOCAB=vocab_v4.json` and `NAME=embedded_v4`, and the checkpoint it
writes records the table it was trained on.

Setup for the GPU path: `.venv/bin/pip install modal`, then `cp .env.example .env.local`
and fill in `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.

### Check for Overfitting / Memorisation
The validator only says a story is *allowed*; it cannot see that a story was copied out of
the training corpus. This audit measures the four things that can: the train vs held-out
loss gap, story-level verbatim copying against the engine's own baseline, line-inventory
reuse, and how many distinct stories come out of one prompt.

```bash
make audit                       # newest checkpoint; corpus and packed default to its generation
python cli.py audit --checkpoint checkpoints/embedded_v4.pt \
    --corpus data/processed/train.jsonl --packed data/processed/train_v4.npz
```

The same weights drive three browser pages: Story Studio (live generation), the 30-step
walkthrough (a real forward pass at full width) and the matrix walkthrough (one block at
four dimensions, where every multiply is inspectable cell by cell — `make matrix-toy`
regenerates its slice and its reference numbers from the checkpoint).

Measured on the shipped v4 checkpoint (225 generated stories): held-out loss equal to
training loss (gap +0.001 nats), shared line-runs at the engine's own baseline (mean 11.69
against 11.78, so nothing was copied), every generated word one the corpus uses (100.0 %),
and 225/225 distinct stories for repeated prompts. The v1 baselines are flatter still
(gap −0.000 nats) because their corpora are less repetitive. Details in
[`docs/training.md`](docs/training.md) §9.

### Export Weights for the Web
After training a new checkpoint, refresh the weights both pages use:
```bash
python cli.py story --raw   # smoke-check sampling from the new checkpoint
python training/export_story.py   # defaults to the newest checkpoint; --vocab pins the table
```
`web/story_data.js` is the single source of truth: Story Studio generates from it and the
walkthrough computes every displayed number from it in the browser. The matrix walkthrough
reads `web/matrix_toy_data.js` instead — a 4-dimension slice of the same checkpoint plus the
reference numbers it must reproduce (`make matrix-toy`).

---

## License
MIT License.
