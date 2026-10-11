#!/usr/bin/env python3
"""Export the 4-dimension toy block that `web/matrix_walkthrough.html` walks through.

The page teaches one transformer block's arithmetic, so it needs a block small
enough to verify by hand. Rather than inventing weights, this takes four
dimensions *of the real checkpoint* — the first (block 0) of the eight blocks of
`checkpoints/embedded_v4.pt`, truncated to a 4-dimensional slice — and records
enough of the model around them for the page to stay honest about what it is
showing.

Which four dimensions? Not `0:4`: those carry the prompt's own identity and the
truncated block cannot overturn it (the toy predicts `<THEME=HELP>` at 64 %,
where the trained model predicts `<LINE>` at 99.99 %). The slice is chosen so
that the trained model's *decision* survives the truncation — one toy block on
those dimensions reproduces the model's next-token choice, and so do all eight
real blocks truncated to the same dimensions. `--dims` defaults to the slice
that does this with the cleanest margin; every candidate is printed so the
choice can be checked.

Also exported, for the page's "is this real?" callouts: the real model's own
next-token distribution, its per-layer logit lens over the toy vocabulary (the
hand-off from `<THEME=HELP>` to `<LINE>` happens at layer 6), and the 0:4
counter-example.

    .venv/bin/python training/export_matrix_toy.py            # -> web/matrix_toy_data.js
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from storylm.model import Config, StoryLM, rope_tables

# The page's 8-token toy vocabulary, with the ids they actually have in the
# shipped table (data/vocab.json) — shown in the UI, so they must be the real ones.
TOY_TOKENS = ["<PAD>", "<BOS>", "<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BEDROOM>",
              "<THEME=HELP>", "<LINE>", "the"]
PROMPT_TOKENS = ["<BOS>", "<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BEDROOM>", "<THEME=HELP>"]


def load_model(checkpoint):
    payload = torch.load(checkpoint, weights_only=False, map_location="cpu")
    model = StoryLM(Config(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def toy_weights(model, dims, block=0):
    """The 4-dim toy block, sliced out of one real block of the checkpoint."""
    sd = model.state_dict()
    idx = torch.tensor(list(dims))
    n = len(dims)
    wide = 2 * n
    qkv = sd[f"blocks.{block}.attn.qkv.weight"]

    def out_in(t, rows, cols):
        return t[rows][:, cols]

    return {
        "embed": sd["embed.weight"][:, idx],                     # [V, 4]: tokens x dims
        "qkv": {
            "q": out_in(qkv[0 * 128:1 * 128], idx, idx),
            "k": out_in(qkv[1 * 128:2 * 128], idx, idx),
            "v": out_in(qkv[2 * 128:3 * 128], idx, idx),
        },
        "proj": out_in(sd[f"blocks.{block}.attn.proj.weight"], idx, idx),
        "up": out_in(sd[f"blocks.{block}.up.weight"][:wide], slice(None), idx),
        "down": out_in(sd[f"blocks.{block}.down.weight"], idx, slice(0, wide)),
        "norm_attn": sd[f"blocks.{block}.attn.norm.weight"][idx],
        "norm_ffn": sd[f"blocks.{block}.norm.weight"][idx],
        "norm_final": sd["norm.weight"][idx],
    }


def toy_block(t, w, x, eps=1e-5):
    """One block at toy scale: 2 heads of 2 dims, per-head RoPE, 1/sqrt(head_dim) scale."""
    n = x.shape[-1]
    heads = n // 2
    gain = w["norm_attn"]

    def rms(v, g):
        return v / torch.sqrt(v.pow(2).mean(-1, keepdim=True) + eps) * g

    def rope(v, pos):
        # inv_freq = 1 / 10000^(2i/head_dim); at head_dim 2 the single pair is 1 rad/token
        out = v.clone()
        angle = pos * 1.0
        c, s = math.cos(angle), math.sin(angle)
        for h in range(heads):
            a, b = out[2 * h].clone(), out[2 * h + 1].clone()
            out[2 * h] = a * c - b * s
            out[2 * h + 1] = a * s + b * c
        return out

    xn = rms(x, gain)
    q, k, v = xn @ w["qkv"]["q"].T, xn @ w["qkv"]["k"].T, xn @ w["qkv"]["v"].T
    qr = torch.stack([rope(q[i], i) for i in range(len(x))])
    kr = torch.stack([rope(k[i], i) for i in range(len(x))])
    contexts = []
    for h in range(heads):
        sl = slice(2 * h, 2 * h + 2)
        scores = (qr[:, sl] @ kr[:, sl].T) / math.sqrt(2)
        causal = torch.triu(torch.ones(len(x), len(x), dtype=torch.bool), 1)
        weights = torch.softmax(scores.masked_fill(causal, float("-inf")), -1)
        contexts.append(weights @ v[:, sl])
    x1 = x + torch.cat(contexts, -1) @ w["proj"].T
    hidden = rms(x1, w["norm_ffn"]) @ w["up"].T
    hidden = 0.5 * hidden * (1 + torch.erf(hidden / math.sqrt(2)))   # exact GELU, as F.gelu
    return x1 + hidden @ w["down"].T


def toy_model(model, dims, blocks, eps=1e-5):
    """Truncated model: `blocks` real blocks, each sliced to `dims`."""
    idx = torch.tensor(list(dims))
    slices = [toy_weights(model, dims, block=b) for b in range(blocks)]

    def embed(ids):
        return model.state_dict()["embed.weight"][ids][:, idx]

    def forward(ids):
        x = embed(ids)
        for weights in slices:
            x = toy_block(model, weights, x, eps)
        final = x[-1] / torch.sqrt(x[-1].pow(2).mean() + eps) * weights["norm_final"]
        logits = final @ embed(torch.tensor([to_id(t) for t in TOY_TOKENS])).T
        return logits

    return embed, forward


_TOKEN_IDS: dict = {}


def to_id(token):
    return _TOKEN_IDS[token]


def real_next_token(model, prompt_ids, top=6):
    with torch.no_grad():
        logits, _ = model(torch.tensor([prompt_ids]))
    last = logits[0, -1]
    probs = torch.softmax(last, -1)
    order = torch.topk(last, top)
    return [{"token": _TOKENS[i], "id": int(i), "logit": round(float(v), 3),
             "p": float(probs[i])} for v, i in zip(order.values, order.indices)]


def logit_lens(model, prompt_ids, layer_limit=None):
    """Per-layer next-token distribution over the toy vocabulary (final norm, tied head).

    Layer 0 is the embedding itself (no block applied), then one row per block, so
    the hand-off from the prompt's own token to `<LINE>` is visible layer by layer.
    """
    idx = torch.tensor([to_id(t) for t in TOY_TOKENS])
    embed = model.state_dict()["embed.weight"]
    cos, sin = rope_tables(model.cfg.ctx, model.cfg.head_dim)

    def row(hidden, layer):
        last = model.norm(hidden)[0, -1]
        logits = last @ embed[idx].T
        probs = torch.softmax(logits, -1)
        return {
            "layer": layer,
            "logits": {t: round(float(v), 3) for t, v in zip(TOY_TOKENS, logits)},
            "probs": {t: float(v) for t, v in zip(TOY_TOKENS, probs)},
            "argmax": TOY_TOKENS[int(logits.argmax())],
        }

    rows = []
    with torch.no_grad():
        h = embed[torch.tensor(prompt_ids)].unsqueeze(0)
        rows.append(row(h, 0))
        for layer in range(model.cfg.layers):
            if layer_limit is not None and layer >= layer_limit:
                break
            h = model.blocks[layer](h, cos, sin)
            rows.append(row(h, layer + 1))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Export the 4-dim toy block used by the matrix walkthrough")
    ap.add_argument("--checkpoint", default=os.path.join(ROOT, "checkpoints/embedded_v4.pt"))
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data/vocab.json"))
    ap.add_argument("--dims", default="103:107",
                    help="dimension slice of the real block, e.g. 103:107 (default: the slice "
                         "where the model's next-token decision survives truncation)")
    ap.add_argument("--out", default=os.path.join(ROOT, "web/matrix_toy_data.js"))
    ap.add_argument("--search", action="store_true",
                    help="print every 4-dim slice with its toy argmax, then exit")
    args = ap.parse_args()

    global _TOKENS, _TOKEN_IDS
    _TOKENS = json.load(open(args.vocab))["tokens"]
    _TOKEN_IDS = {t: i for i, t in enumerate(_TOKENS)}
    model = load_model(args.checkpoint)
    prompt_ids = [to_id(t) for t in PROMPT_TOKENS]

    def slice_of(spec):
        if ":" in spec:
            lo, hi = spec.split(":")
            return list(range(int(lo), int(hi)))
        return [int(v) for v in spec.split(",")]

    if args.search:
        print(f"{'dims':>10} {'block 1':>10} {'<LINE>':>8} {'<THEME=HELP>':>13} {'8 real blocks':>14}")
        for start in range(0, model.cfg.d_model - 3):
            dims = list(range(start, start + 4))
            _, forward = toy_model(model, dims, 1)
            p1 = torch.softmax(forward(prompt_ids), -1)
            _, forward8 = toy_model(model, dims, 8)
            p8 = torch.softmax(forward8(prompt_ids), -1)
            line, help_ = TOY_TOKENS.index("<LINE>"), TOY_TOKENS.index("<THEME=HELP>")
            best1 = TOY_TOKENS[int(p1.argmax())]
            best8 = TOY_TOKENS[int(p8.argmax())]
            if best1 == "<LINE>" or best8 == "<LINE>":
                print(f"{start:>6}:{start+4:<3} {best1:>10} {float(p1[line])*100:7.1f}% "
                      f"{float(p1[help_])*100:12.1f}% {best8:>14}")
        return 0

    dims = slice_of(args.dims)
    embed, forward = toy_model(model, dims, 1)
    _, forward8 = toy_model(model, dims, 8)
    tokens8 = torch.tensor([to_id(t) for t in TOY_TOKENS])
    weights = toy_weights(model, dims)

    p1 = torch.softmax(forward(prompt_ids), -1)
    p8 = torch.softmax(forward8(prompt_ids), -1)
    _, forward_first = toy_model(model, [0, 1, 2, 3], 1)
    p_first = torch.softmax(forward_first(prompt_ids), -1)
    line, help_ = TOY_TOKENS.index("<LINE>"), TOY_TOKENS.index("<THEME=HELP>")
    if TOY_TOKENS[int(p1.argmax())] != "<LINE>":
        raise SystemExit(
            f"dims {dims} do not predict <LINE> (argmax {TOY_TOKENS[int(p1.argmax())]}, "
            f"<LINE> {float(p1[line])*100:.1f}%); run --search for slices that do")

    def arr(t):
        return [[round(float(v), 6) for v in row] for row in t.tolist()]

    def vec(t):
        return [round(float(v), 6) for v in t.tolist()]

    payload = {
        "checkpoint": os.path.relpath(args.checkpoint, ROOT),
        "block": 0,
        "dims": dims,
        "note": ("four dimensions of block 0 of the shipped checkpoint; the slice is chosen so "
                 "the model's next-token decision survives truncation"),
        "tokens": [{"token": t, "id": to_id(t)} for t in TOY_TOKENS],
        "prompt": [{"token": t, "id": to_id(t), "row": TOY_TOKENS.index(t)} for t in PROMPT_TOKENS],
        "config": {
            "toy": {"d_model": len(dims), "heads": 2, "head_dim": len(dims) // 2, "ffn": 2 * len(dims),
                    "ctx": len(PROMPT_TOKENS), "vocab": len(TOY_TOKENS), "eps": 1e-5,
                    "inv_freq": [round(1 / (10000 ** (2 * i / (len(dims) // 2))), 6)
                                 for i in range(len(dims) // 4)]},
            "real": {"d_model": model.cfg.d_model, "layers": model.cfg.layers,
                     "heads": model.cfg.heads, "head_dim": model.cfg.head_dim,
                     "ffn": model.cfg.ffn, "ctx": model.cfg.ctx, "vocab": model.cfg.vocab},
        },
        "weights": {
            "embed": arr(weights["embed"][tokens8]),
            "q": arr(weights["qkv"]["q"]), "k": arr(weights["qkv"]["k"]), "v": arr(weights["qkv"]["v"]),
            "proj": arr(weights["proj"]), "up": arr(weights["up"]), "down": arr(weights["down"]),
            "norm_attn": vec(weights["norm_attn"]), "norm_ffn": vec(weights["norm_ffn"]),
            "norm_final": vec(weights["norm_final"]),
        },
        "toy": {
            "probs_one_block": {t: float(v) for t, v in zip(TOY_TOKENS, p1)},
            "probs_eight_blocks": {t: float(v) for t, v in zip(TOY_TOKENS, p8)},
            "argmax_one_block": TOY_TOKENS[int(p1.argmax())],
            "argmax_eight_blocks": TOY_TOKENS[int(p8.argmax())],
        },
        "dims_0_4_counterexample": {
            "probs_one_block": {t: float(v) for t, v in zip(TOY_TOKENS, p_first)},
            "argmax_one_block": TOY_TOKENS[int(p_first.argmax())],
        },
        "real_model": {
            "next_token": real_next_token(model, prompt_ids),
            "logit_lens": logit_lens(model, prompt_ids),
        },
    }

    blob = json.dumps(payload, separators=(",", ":"))
    with open(args.out, "w") as handle:
        handle.write("// Generated by training/export_matrix_toy.py - do not edit by hand.\n")
        handle.write("// The 4-dim toy block on the page is this slice of the real checkpoint:\n")
        handle.write(f"//   {os.path.relpath(args.checkpoint, ROOT)} block {payload['block']} "
                     f"dims {dims}\n")
        handle.write(f"window.MATRIX_TOY = {blob};\n")

    print(f"slice {dims} of {payload['checkpoint']} block {payload['block']}")
    print(f"toy, one block  : " + "  ".join(f"{t}:{p*100:.1f}%" for t, p in
          sorted(payload['toy']['probs_one_block'].items(), key=lambda kv: -kv[1])[:3]))
    print(f"toy, eight blocks: " + "  ".join(f"{t}:{p*100:.1f}%" for t, p in
          sorted(payload['toy']['probs_eight_blocks'].items(), key=lambda kv: -kv[1])[:3]))
    print(f"real model      : " + "  ".join(f"{d['token']}:{d['p']*100:.2f}%" for d in
          payload['real_model']['next_token'][:3]))
    print(f"dims 0:4 (rejected): "
          + "  ".join(f"{t}:{p*100:.1f}%" for t, p in
                      sorted(payload['dims_0_4_counterexample']['probs_one_block'].items(),
                             key=lambda kv: -kv[1])[:2]))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.1f} KB)")


if __name__ == "__main__":
    raise SystemExit(main())
