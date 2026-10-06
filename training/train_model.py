#!/usr/bin/env python3
"""Train the compact model from scratch on the engine corpus.

    python training/prepare_data.py                       # corpus -> .npz
    python training/train_model.py --steps 300            # smoke run
    python training/train_model.py --steps 6000           # real run

`train_from_packed()` is the reusable core: the local CLI and
`training/modal_train.py` both call it, so the GPU run and the CPU run cannot
drift apart.

The loss is masked to the story tokens, so the model is trained to continue
after the prompt line and nothing else. Evaluation is the firmware's own
validator, reached through `storygen --check`, not perplexity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tinylm.model import Config, TinyLM, decode_lines, generate_story
except ImportError:
    from model import Config, TinyLM, decode_lines, generate_story

try:
    from sample import canonicalize
except ImportError:
    canonicalize = None

PROMPT = ("<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BEDROOM>", "<THEME=HELP>")


def load_packed(path):
    data = np.load(path)
    vocab, ctx, count = (int(v) for v in data["meta"])
    return data["ids"], data["mask"], vocab, ctx, count


def train_from_packed(ids_np, mask_np, vocab, ctx, *, steps, batch=16, lr=3e-4,
                      warmup=100, layers=8, d_model=128, heads=8, ffn=512,
                      eval_every=100, seed=0, device="cpu", log=print):
    """Trains and returns (model, history). Deterministic for a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = Config(vocab=vocab, ctx=ctx, layers=layers, d_model=d_model, heads=heads, ffn=ffn)
    model = TinyLM(cfg).to(device)
    report = model.param_report()
    log(f"vocab {report['vocab']} | total params {report['total']:,} "
        f"| INT4 {report['int4_bytes'] / 1024:.0f} KiB "
        f"| KV INT8 {report['kv_int8_bytes'] / 1024:.0f} KiB | ctx {ctx}")

    ids = torch.from_numpy(ids_np.astype(np.int64))
    mask = torch.from_numpy(mask_np)
    count = len(ids)
    n_val = max(1, count // 20)
    perm = torch.randperm(count)
    train_idx, val_idx = perm[:-n_val], perm[-n_val:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps,
        pct_start=min(max(warmup / max(steps, 1), 0.01), 0.9),
    )

    history = []
    model.train()
    started = time.time()
    for step in range(1, steps + 1):
        batch_idx = train_idx[torch.randint(0, len(train_idx), (batch,))]
        x, m = ids[batch_idx].to(device), mask[batch_idx].to(device)
        _, loss = model(x, targets=x, loss_mask=m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vx = ids[val_idx].to(device)
                _, val_loss = model(vx, targets=vx, loss_mask=mask[val_idx].to(device))
            elapsed = time.time() - started
            entry = {"step": step, "train": loss.item(), "val": val_loss.item(),
                     "lr": sched.get_last_lr()[0], "seconds": elapsed}
            history.append(entry)
            log(f"step {step:6d}  train {entry['train']:.3f}  val {entry['val']:.3f}  "
                f"lr {entry['lr']:.2e}  {elapsed:6.1f}s")
            model.train()

    return model.cpu(), history


def sample_story(model, tokens, ctx, temperature=0.8, top_k=40, proper_names=None):
    """One story through the device's full path: generate, canonicalise, show."""
    to_id = {token: i for i, token in enumerate(tokens)}
    prompt_ids = [to_id["<BOS>"]] + [to_id[t] for t in PROMPT]
    ids = generate_story(model, prompt_ids, tokens, ctx, temperature=temperature, top_k=top_k)
    lines = decode_lines(ids, tokens, proper_names or set())
    return canonicalize(lines, "bluey", "bedroom", "help")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train.npz")
    ap.add_argument("--vocab", default="data/vocab.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/embedded")
    args = ap.parse_args()

    ids_np, mask_np, vocab, ctx, count = load_packed(args.data)
    tokens = json.load(open(args.vocab))["tokens"]

    model, _ = train_from_packed(
        ids_np, mask_np, vocab, ctx, steps=args.steps, batch=args.batch, lr=args.lr,
        warmup=args.warmup, layers=args.layers, d_model=args.d_model, heads=args.heads,
        ffn=args.ffn, eval_every=args.eval_every, device=args.device,
    )

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "model": model.state_dict()},
               args.checkpoint + ".pt")
    print(f"saved {args.checkpoint}.pt")

    series_entry = next(e for e in json.load(open("data/series.json")) if e["id"] == "bluey")
    proper_names = {series_entry["story_name"].lower()} | {f.lower() for f in series_entry["friends"]}
    lines, stats = sample_story(model, tokens, ctx, proper_names=proper_names)
    print("--- sample (after the engine's canonicaliser) ---")
    print("    " + stats)
    for line in lines:
        print(line)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("".join(f"<LINE>{line}</LINE>\n" for line in lines) + "<END>\n")
        path = handle.name
    result = subprocess.run(
        ["./build/storygen", "--check", "--series", "bluey", "--mode", "bedtime",
         "--place", "bedroom", "--theme", "help"],
        stdin=open(path), capture_output=True, text=True,
    )
    print("validator:", result.stdout.strip(), f"(exit {result.returncode})")
    os.unlink(path)


if __name__ == "__main__":
    main()
