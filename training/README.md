# Training the compact model

Everything here is host-side. The model it produces is the one described in
[`docs/tiny_model.md`](../docs/tiny_model.md): 1.67 M parameters, 867 KiB at
INT4, tied embeddings, RoPE, 752-token closed vocabulary.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy
```

## Pipeline

```bash
make corpus vocab                               # stories + the 752-id table
.venv/bin/python training/prepare_data.py       # -> data/processed/train.npz
.venv/bin/python training/train_model.py --steps 300      # smoke run
.venv/bin/python training/train_model.py --steps 6000     # real run
.venv/bin/python training/train_model.py --help
```

`prepare_data.py` encodes each story with exactly the rule the firmware will
use: control tokens by exact match, words by letters-only lowercase lookup, and
**a hard error on any word outside the table**. It prints the story-token
length distribution, which is where the 384-token context comes from.

`train_model.py` is causal next-token prediction with the loss masked to the
story tokens (nothing is learned from predicting the prompt). AdamW, cosine
schedule with warmup, gradient clipping, 95/5 train/val split. At the end it
saves a checkpoint, generates one story, and hands that story to the firmware's
own validator:

```
$ .venv/bin/python training/train_model.py --steps 300
vocab 752 | total params 1,671,296 | INT4 867 KiB | KV INT8 768 KiB | ctx 384
step   100  train 5.21  val 4.98  lr 2.9e-04    12.3s
...
--- sample ---
<the generated story>
validator: ok (exit 0)
```

**Evaluation is the validator, not perplexity.** A model that writes
well-formed stories that no clip library can speak has failed, however good its
loss looks.

Measured on this repository (CPU, macOS arm64): 4,000 steps in 659 s, masked
cross-entropy train 2.970 -> 0.260 / val 2.949 -> 0.290, and **19 of 20
sampled stories passed `storygen --check`**. The log is kept at
`artifacts/train_log.txt` and the checkpoint at
`artifacts/checkpoints/embedded.pt`.

Measured fidelity (the metric a pass rate cannot see): **455/459 = 99.1 % of
generated lines match an engine line word for word**, and the residue falls
back to the engine's own beat. `sample.py` prints both numbers on every run,
and warns in the code that punctuation must not be counted as drift.

The harness runs the device's full pipeline: generate -> canonicalise (engine
renders the line) -> validate. It prints pass rate, word-level fidelity and
unmatched lines, because a pass rate alone cannot see drift:

```
20/20 stories passed the firmware validator
456/457 lines match an engine line word for word (99.8 %)
1/457 lines unmatched after canonicalisation (0.2 %) - only those need the
engine's fallback tier
```

Runtime sampling rules that make that pass rate possible

The harness runs the device's full pipeline: generate -> canonicalise (engine
renders the line) -> validate. It prints pass rate, word-level fidelity and
unmatched lines, because a pass rate alone cannot see drift:

```
20/20 stories passed the firmware validator
456/457 lines match an engine line word for word (99.8 %)
1/457 lines unmatched after canonicalisation (0.2 %) - only those need the
engine's fallback tier
```

Runtime sampling rules that make that pass rate possible (all implemented in
`model.generate_story`): stop on `<END>`, mask `<EOS>` so a story cannot be
truncated mid-line, generate one line at a time, and resample a line that
duplicates an earlier one.

For the full test recipe per layer, see [`docs/testing.md`](../docs/testing.md).
The decoder uses a KV cache; `training/test_cache.py` proves it matches the
full-recompute path (worst logit difference 1.7e-5, greedy stories identical)
and reports the speedup (2.1x at a 272-token story on CPU).

## Training on a Modal GPU

CPU works but is slow (~6 steps/s). The same `train_from_packed()` core runs on
a Modal T4, so nothing forks between the two paths.

```bash
set -a; . .env.local; set +a                      # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET

modal volume create tinystory-data
modal volume create tinystory-ckpts
modal volume put tinystory-data data/processed/train_big.npz /train_big.npz
modal volume put tinystory-data data/vocab.json /vocab.json

modal run training/modal_train.py --steps 5400 --batch 64 --name embedded_big
modal volume get tinystory-ckpts embedded_big.pt artifacts/checkpoints/embedded_big.pt
```

Design choice: **the GPU only trains.** Canonicalisation and validation need the
engine binary, so they stay on the host - one implementation of the runtime
path, not two. After downloading the checkpoint, evaluate it exactly as the
device would:

```bash
.venv/bin/python training/sample.py --checkpoint artifacts/checkpoints/embedded_big.pt --per-world 4
```

`--steps 5400 --batch 64` is about six epochs of the 57,600-story corpus. On a
T4 that is ~22 minutes of GPU time (tens of cents); the first run also builds
the image (53 s). Measured effect versus the CPU run on 100 stories each:

| | CPU 4,000 steps / 2,880 stories | Modal T4 5,400 steps / 57,600 stories |
|---|---|---|
| val loss | 0.290 | 0.267 |
| validator pass | 97/100 | **100/100** |
| word-exact engine lines | 99.3 % | **99.7 %** |
| outside the prompt's inventory | 0.88 % | **0.13 %** |

## Scaling the corpus

`make corpus` writes 2,880 stories by default. The phase pools now hold 15-21
beats each, so raising the variant count genuinely adds new stories:

```bash
./build/storygen --corpus data/processed/train.jsonl --variants 240   # 57,600 stories, ~16.0 M tokens
```

For more variety, add beats to the pools or worlds to the series table in
`firmware/components/story_engine/src/story_engine.cpp`; both grow the
vocabulary and the corpus at the same time.
