#!/usr/bin/env python3
"""Train the compact model on a Modal GPU. This is the only sanctioned way to
train: local runs are for plumbing smoke tests (`--steps` up to 500), because a
CPU step is ~6x slower than a T4 and a 3.5 k-vocabulary model needs tens of
thousands of steps.

    set -a; . .env.local; set +a                 # MODAL_TOKEN_ID / MODAL_TOKEN_SECRET
    .venv/bin/modal volume create tinystory-data
    .venv/bin/modal volume create tinystory-ckpts
    .venv/bin/modal volume put tinystory-data data/processed/train_v2.npz /train_v2.npz
    .venv/bin/modal volume put tinystory-data data/vocab_v2.json /vocab_v2.json
    .venv/bin/modal run training/modal_train.py --steps 5400 --batch 64 --name embedded_v2
    .venv/bin/modal volume get tinystory-ckpts embedded_v2.pt checkpoints/embedded_v2.pt

`make train-modal` runs exactly that sequence (volumes, upload, run, download).

The GPU does training only. Canonicalisation and validation need the engine
binary, so they run on the host after the checkpoint is downloaded - that keeps
one implementation of the runtime path instead of two.
"""
from __future__ import annotations

import json

import modal

DATA_VOLUME = "tinystory-data"
CKPT_VOLUME = "tinystory-ckpts"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy==2.1.3")
    .add_local_dir("training", remote_path="/root/training")
    .add_local_dir("storylm", remote_path="/root/storylm")
)

app = modal.App("tinystory-lm")


@app.function(
    image=image,
    gpu="T4",
    timeout=4 * 3600,
    volumes={
        "/data": modal.Volume.from_name(DATA_VOLUME, create_if_missing=True),
        "/ckpt": modal.Volume.from_name(CKPT_VOLUME, create_if_missing=True),
    },
)
def train(steps: int = 5400, batch: int = 64, lr: float = 3e-4, warmup: int = 300,
          eval_every: int = 500, eval_batch: int = 64, seed: int = 0,
          name: str = "embedded_v2",
          data_file: str = "train_v2.npz", vocab_file: str = "vocab_v2.json") -> dict:
    import sys

    import torch

    # The image mirrors the repository layout, so the same imports work remotely.
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/training")
    from storylm.model import decode_lines, generate_story
    from train_model import load_packed, train_from_packed

    ids, mask, vocab, ctx, count = load_packed(f"/data/{data_file}")
    print(f"corpus {count} stories, vocab {vocab}, ctx {ctx}, device {torch.cuda.get_device_name()}")

    model, history = train_from_packed(
        ids, mask, vocab, ctx, steps=steps, batch=batch, lr=lr, warmup=warmup,
        eval_every=eval_every, eval_batch=eval_batch, seed=seed, device="cuda", log=print,
    )

    torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, f"/ckpt/{name}.pt")
    modal.Volume.from_name(CKPT_VOLUME).commit()

    # Raw sample, for a sanity check only: canonicalisation needs the engine.
    tokens = json.load(open(f"/data/{vocab_file}"))["tokens"]
    to_id = {t: i for i, t in enumerate(tokens)}
    prompt = ["<BOS>", "<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BEDROOM>", "<THEME=HELP>"]
    out = generate_story(model, [to_id[t] for t in prompt], tokens, ctx)
    return {
        "params": model.param_report(),
        "history": history,
        "sample": decode_lines(out, tokens),
        "checkpoint": f"{name}.pt",
    }


@app.local_entrypoint()
def main(steps: int = 5400, batch: int = 64, name: str = "embedded_v2",
         data_file: str = "train_v2.npz", vocab_file: str = "vocab_v2.json"):
    result = train.remote(steps=steps, batch=batch, name=name,
                          data_file=data_file, vocab_file=vocab_file)
    print(json.dumps(result["params"], indent=2))
    for entry in result["history"]:
        print(f"step {entry['step']:6d}  train {entry['train']:.3f}  val {entry['val']:.3f}"
              f"  {entry['seconds']:6.1f}s")
    print("--- raw sample (pre-canonicalisation) ---")
    for line in result["sample"]:
        print(line)
    print("checkpoint:", result["checkpoint"])
