# Training the compact model

Everything here is host-side, except the training itself. The model it produces
is described in [`docs/tiny_model.md`](../docs/tiny_model.md): tied embeddings,
RoPE, a closed vocabulary (752 ids in v1, 3,589 in v2), 867 KiB–1.05 MiB at INT4.

> **Policy: real training runs on Modal, never locally.** A CPU step is several
> times slower than a T4, so anything past a smoke test belongs on the GPU:
> `make train-modal`. `training/train_model.py` refuses more than 500 steps on
> `--device cpu` unless `--allow-cpu` is passed, because that flag is what a
> mistake looks like. Local runs exist to prove the plumbing.

**Where each command runs.** This repository holds the model library, the
training loop, the corpus builder, the packed data and the checkpoints. The C++
engine that renders and validates stories (`build/storygen`) lives in the device
project (`m5stack-sticks3/tinystories`) — commands marked *device project* below
must be run there. The full pipeline, with measured results and gotchas, is in
[`docs/training.md`](../docs/training.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch numpy
.venv/bin/pip install modal            # required: training runs on Modal
cp .env.example .env.local             # then fill in MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
```

## Pipeline

```bash
make corpus vocab                                                  # stories + the token table
.venv/bin/python training/prepare_data.py                          # -> data/processed/train.npz
.venv/bin/python training/train_model.py --steps 60 --eval-every 30   # smoke: plumbing only, ~15 s
make train-modal                                                   # real run on a Modal T4
.venv/bin/python training/train_model.py --help
```

`make train-modal` creates the two Modal volumes, uploads `train_v2.npz` +
`vocab_v2.json`, runs `training/modal_train.py`, and downloads the checkpoint to
`checkpoints/`. Override the dataset or the run with make variables:

```bash
make train-modal DATA=train.npz VOCAB=vocab.json NAME=embedded_v1 STEPS=5400 BATCH=64
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
`training/train_log.txt` and the checkpoint at `checkpoints/embedded.pt`.

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

Runtime sampling rules that make that pass rate possible (all implemented in
`model.generate_story`): stop on `<END>`, mask `<EOS>` so a story cannot be
truncated mid-line, generate one line at a time, and resample a line that
duplicates an earlier one.

For the full test recipe per layer, see `docs/testing.md` in the device
project (it needs `build/storygen`). The decoder uses a KV cache;
`storylm/test_cache.py` (or `cli.py test`) proves it matches the full-recompute
path (worst logit difference 1.7e-5, greedy stories identical) and reports the
speedup (2.1x at a 272-token story on CPU).

## Is it memorising?

The validator cannot see a story copied out of the corpus — every word in it is
approved by design. `training/memorisation.py` (`make audit`) measures that
directly: the train vs held-out loss gap, the longest run of consecutive lines a
generated story shares with a single training story (against the engine's own
baseline), line-inventory reuse, and the number of distinct stories produced for
one prompt when only the dice change.

Both shipped checkpoints come out clean — held-out loss equal to training loss
(gap -0.000 nats), no six-line run shared with any training story, 100/100
distinct stories per prompt. Numbers and method: `docs/training.md` §9.

## Training on a Modal GPU (the real path)

A CPU step is several times slower than a T4 step, so this is where real runs
happen. The same `train_from_packed()` core runs on both, so nothing forks
between the paths - only the pace differs.

```bash
make train-modal                                  # volumes, upload, run, download
# equivalently, by hand (v2 dataset):
set -a; . .env.local; set +a                      # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET

modal volume create tinystory-data
modal volume create tinystory-ckpts
modal volume put tinystory-data data/processed/train_v2.npz /train_v2.npz
modal volume put tinystory-data data/vocab_v2.json /vocab_v2.json

modal run training/modal_train.py --steps 5400 --batch 64 --name embedded_v2
modal volume get tinystory-ckpts embedded_v2.pt checkpoints/embedded_v2.pt
```

For the v1 dataset substitute `train_big.npz` / `vocab.json` and
`--name embedded_big` (or pass `DATA=`/`VOCAB=`/`NAME=` to `make train-modal`).

Design choice: **the GPU only trains.** Canonicalisation and validation need the
engine binary, so they stay on the host - one implementation of the runtime
path, not two. After downloading the checkpoint, evaluate it exactly as the
device would:

```bash
.venv/bin/python training/sample.py --checkpoint checkpoints/embedded_big.pt --per-world 4
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

In the device project, `make corpus` writes 2,880 stories by default. The phase
pools now hold 15-21 beats each, so raising the variant count genuinely adds new
stories:

```bash
# device project
./build/storygen --corpus data/processed/train.jsonl --variants 240   # 57,600 stories, ~16.0 M tokens
```

For more variety, add beats to the pools or worlds to the series table in
`firmware/components/story_engine/src/story_engine.cpp`; both grow the
vocabulary and the corpus at the same time.
