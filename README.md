# TinyLM: 1.67M Parameter Tiny Language Model

An edge-native, ultra-compact generative language model engineered for resource-constrained microcontrollers and embedded devices (such as the ESP32-S3), accompanied by a mathematical forward-pass tracer and an interactive web visualizer.

![Architecture Overview](web/preview.jpg) <!-- (or open web/index.html) -->

---

## Backstory: from a cloud microphone to a 1.67M model

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
corpus and the entire vocabulary: 752 ids = 30 control tokens + 722 words,
closed by construction, so a model trained on it can never emit a word without
a narration clip. Two architectures were on the table: a ~50K-parameter beat
selector that only picks recipe steps, or a word-level generator that writes
the text itself. The beat selector adds little over the dice; the word-level
model is the whole point. So a 1.67M-parameter decoder-only Transformer
(8 layers, d=128, tied embeddings, RoPE, RMSNorm, GELU, no biases) was trained
from scratch: a few thousand steps on a laptop CPU for the baseline checkpoint,
then the expanded 57,600-story corpus on a rented T4 GPU for the big one.
Evaluation was never perplexity — sampled stories go back through the engine's
canonicaliser and the firmware validator, the same gate the device will use.

**That model is TinyLM.** Somewhere along the way the interesting artifact
stopped being the appliance and became the model: a complete language model —
tokenizer, training loop, checkpoints, a KV cache proven token-identical to
full recompute (4.3x faster), a forward-pass tracer and a 30-step browser
walkthrough — that fits in 867 KiB of INT4 flash. This repo holds it.

## Key Specifications

| Engineering Metric | TinyLM (Tiny SLM) | Standard Local LLM | Frontier Cloud LLM |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | **1,671,296 (1.67M)** | 7,000,000,000 (7B) | ~1,800,000,000,000 (1.8T) |
| **Layers / Dim / Heads** | **8 layers / d=128 / 8 heads** | 32 layers / d=4096 / 32 heads | 96+ layers / d=12288+ |
| **Vocabulary** | **752 tokens** (closed-domain) | 32,000–128,000 (BPE) | 100,000+ |
| **Weight Footprint** | **6.7 MB** (FP32) / **~835 KB** (INT4) | ~4.2 GB (INT4) | Distributed across clusters |
| **KV-Cache RAM** | **~65 KB** (at ctx=256) | ~512 MB – 1 GB | Multi-Gigabyte VRAM |
| **Target Hardware** | **Microcontroller (ESP32-S3 / Cortex-M)** | Desktop GPU / Apple M-series | Multi-H100/TPU Datacenter |
| **Power Budget** | **0.5 W – 1.5 W** | 25 W – 150 W | 700 W per GPU |

---

## Repository Structure

```
tinylm/
├── README.md                     # Project documentation & quickstart
├── cli.py                        # Unified command-line interface
├── requirements.txt              # Python dependencies (torch, numpy, pyyaml)
├── pyproject.toml                # Standard python packaging
├── Makefile                      # Convenient CLI targets (test, story, web, trace)
│
├── tinylm/                       # Core Model Library
│   ├── __init__.py               # Package exports
│   ├── model.py                  # TinyLM architecture, RoPE, Attention, KV-Cache
│   ├── story.py                  # Standalone prompt & story generation engine
│   └── test_cache.py             # KV-cache equivalence & speed benchmark
│
├── checkpoints/                  # Trained PyTorch Checkpoints
│   ├── embedded_big.pt           # 1.67M model trained on expanded corpus (6.7 MB)
│   └── embedded.pt               # Baseline model checkpoint (6.7 MB)
│
├── configs/
│   └── embedded_l8_d128.yaml     # Architectural hyperparameter config
│
├── data/
│   ├── vocab.json                # 752-token closed dictionary
│   ├── series.json               # Series prompts, place IDs & theme combinations
│   └── processed/
│       ├── train_big.jsonl       # Full character story corpus (~178 MB)
│       ├── train_big.npz         # Pre-tokenized uint16 numpy dataset (~6.3 MB)
│       ├── train.jsonl           # Baseline corpus (~8.9 MB)
│       └── train.npz             # Tokenized baseline dataset (~437 KB)
│
├── training/                     # Training & Tracing Pipelines
│   ├── train_model.py            # Local PyTorch training loop (AdamW, Cosine LR)
│   ├── prepare_data.py           # Corpus tokenization & .npz dataset packing
│   ├── export_trace.py           # Mathematical forward pass tracer -> web/trace_data.js
│   ├── export_story.py           # Full fp16 weights + prompts -> web/story_data.js
│   ├── modal_train.py            # Scalable Cloud GPU training script (Modal)
│   └── sample.py                 # Evaluation & sampling harness
│
├── web/                          # Standalone Interactive Web Visualizer
│   ├── index.html                # Storybook design: Story Studio, Architecture, Walkthrough
│   ├── story.js                  # Vanilla-JS port of the model: live in-browser generation
│   ├── story_data.js             # Real checkpoint weights, fp16 + base64 (~4.3 MB)
│   ├── walkthrough.js            # 30-step forward pass interactive player
│   ├── trace_data.js             # Forward pass trace for the walkthrough (190 KB)
│   └── preview.jpg               # Full-page screenshot of the visualizer
│
└── docs/                         # Technical Specs & Architecture Notes
    ├── tiny_model.md             # 1.67M model architecture & memory budget
    ├── story_format.md           # Prompt tags & story structure specs
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

### 3. Launch the Interactive Web Explainer
Explore the inner workings of the model in your browser:
```bash
python cli.py web
```
This serves `web/index.html`, which features:
- **Story Studio — Live In-Browser Generation**: the page ships the real checkpoint weights (fp16, ~4.3 MB) and a vanilla-JS port of the model with a KV cache; pick character, mode, place and theme, and a brand-new story is generated live in the browser (~80-100 tok/s, no server).
- **Interactive Dataflow Diagram**: View tensor shapes through Embeddings, 8x Transformer Layers, and Output Head.
- **Component Breakdown**: Deep dives into RMSNorm, RoPE, Preallocated KV-Cache, and GELU FFN.
- **"Under the Hood" Walkthrough**: A 30-step slowed-down execution of a real forward pass:
  - Inspect 1,671,296 model weights and matrix histograms.
  - Track residual stream vectors across all 8 layers.
  - Observe multi-head attention weights and FFN feature activations.
  - Peek via the "logit lens" to see what the model would guess at intermediate layers.
  - Examine greedy next-token probability distributions and autoregressive loops.
- **TinyLM vs. Regular LLM**: Comprehensive comparative analysis matrix and architectural pillars.

### 4. Verify KV-Cache Parity & Performance
Verify that KV-cache output is numerically identical to standard full-context recomputation:
```bash
python cli.py test
```
*Sample output:*
```
logits: worst max-abs diff 0.00e+00 -> ok
greedy stories identical with and without the cache: yes
 no cache: 0.82s for 142 tokens (5.8 ms/token)
 KV cache: 0.19s for 142 tokens (1.3 ms/token)
speedup: 4.3x
```

---

## Training from Scratch

### Preprocess the Data
Encode the raw story corpus into tokenized `.npz` arrays:
```bash
python training/prepare_data.py \
  --corpus data/processed/train.jsonl \
  --vocab data/vocab.json \
  --out data/processed/train.npz
```

### Train Locally
Run training with PyTorch:
```bash
# Smoke test (300 steps)
python training/train_model.py --data data/processed/train.npz --steps 300

# Full training run
python training/train_model.py --data data/processed/train_big.npz --steps 6000
```

### Export Forward Pass Trace
To update the web visualizer's execution data after training a new checkpoint:
```bash
python cli.py trace
```
This updates `web/trace_data.js` with the real intermediate activations.

---

## License
MIT License.
