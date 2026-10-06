#!/usr/bin/env python3
"""Export one real forward pass of the trained model for the web explainer.

    .venv/bin/python training/export_trace.py            # writes web/trace_data.js

Runs the checkpoint on the default story.py prompt, re-does the forward pass by
hand (so every intermediate can be captured), checks that the hand-computed
logits match ``TinyLM.forward``, and writes everything the page needs:

* weight statistics, histograms and small crops of real weight matrices;
* the prompt tokens and their embedding vectors;
* for every layer: the residual stream, attention weights per head, the FFN
  activations and a "logit lens" guess (what the model would predict if it
  stopped at that layer);
* the final next-token distribution, before and after temperature / top-k;
* the first few generated tokens, each with its top candidates.

The output is a ``.js`` file (``window.TRACE = {...}``) rather than JSON so the
page works when opened straight from disk (file:// blocks fetch()).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

try:
    from tinylm.model import Config, TinyLM, apply_rope
except ImportError:
    from model import Config, TinyLM, apply_rope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def r(t, nd=4):
    """Tensor -> nested list of rounded floats (keeps the file small)."""
    return [round(float(v), nd) for v in t.flatten().tolist()] if t.dim() <= 1 else \
        [r(row, nd) for row in t]


def hist(t, bins=41):
    t = t.flatten().float()
    lo, hi = float(t.min()), float(t.max())
    counts = torch.histc(t, bins=bins, min=lo, max=hi)
    return {"min": round(lo, 4), "max": round(hi, 4), "counts": [int(c) for c in counts]}


def stats(t):
    t = t.float()
    return {"min": round(float(t.min()), 4), "max": round(float(t.max()), 4),
            "mean": round(float(t.mean()), 5), "std": round(float(t.std()), 4),
            "params": t.numel(), "shape": list(t.shape)}


def topk(logits, tokens, k):
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    return [{"token": tokens[int(i)], "id": int(i), "p": round(float(v), 5),
             "logit": round(float(logits[int(i)]), 3)} for v, i in zip(vals, idx)]


@torch.no_grad()
def traced_forward(model, ids, tokens):
    """The same maths as TinyLM.forward, step by step, recording everything."""
    cfg = model.cfg
    h, d = cfg.heads, cfg.head_dim
    idx = torch.tensor([ids])
    t = idx.shape[1]
    cos, sin = model.cos[:t], model.sin[:t]

    x = model.embed(idx)  # [1, t, 128]
    trace = {"embed": r(x[0]), "layers": []}
    residual = [r(x[0])]  # residual stream per stage, all positions

    def lens(vec):
        return topk(model.head(model.norm(vec)), tokens, 5)

    for li, block in enumerate(model.blocks):
        att = block.attn
        layer = {"x_in_last": r(x[0, -1])}
        normed = att.norm(x)
        q, k, v = att.qkv(normed).split(cfg.d_model, dim=2)
        q = q.view(1, t, h, d).transpose(1, 2)
        k = k.view(1, t, h, d).transpose(1, 2)
        v = v.view(1, t, h, d).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # [1, h, t, t]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        y = (weights @ v).transpose(1, 2).reshape(1, t, h * d)
        attn_out = att.proj(y)
        layer["attn_norm_last"] = r(normed[0, -1])
        layer["q_last"] = r(q[0, :, -1, :])            # [h, d]
        layer["attn_weights"] = r(weights[0], 3)        # [h, t, t]
        layer["attn_out_last"] = r(attn_out[0, -1])
        x = x + attn_out
        layer["x_mid_last"] = r(x[0, -1])

        ffn_in = block.norm(x)
        hidden = F.gelu(block.up(ffn_in))                # [1, t, 512]
        ffn_out = block.down(hidden)
        layer["ffn_hidden_last"] = r(hidden[0, -1], 3)
        layer["ffn_active_frac"] = round(float((hidden[0, -1] > 0.05).float().mean()), 3)
        layer["ffn_out_last"] = r(ffn_out[0, -1])
        x = x + ffn_out
        layer["x_out_last"] = r(x[0, -1])
        layer["norm_out_by_pos"] = [round(float(n), 3) for n in x[0].norm(dim=-1)]
        layer["lens_top5"] = lens(x[0, -1])
        residual.append(r(x[0]))
        trace["layers"].append(layer)

    final = model.norm(x)
    logits = model.head(final)[0, -1]
    trace["final_norm_last"] = r(final[0, -1])
    trace["residual"] = residual  # [stages=9][t][128]
    return trace, logits


@torch.no_grad()
def next_token(model, ids, tokens, temperature, top_k, gen):
    logits, _ = model(torch.tensor([ids]))
    logits = logits[0, -1].clone()
    raw_top = topk(logits, tokens, 5)
    logits = logits / temperature
    logits[tokens.index("<EOS>")] = -float("inf")
    kth = torch.topk(logits, top_k).values[-1]
    logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    nxt = int(torch.multinomial(probs, 1, generator=gen))
    return nxt, raw_top, round(float(probs[nxt]), 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "web/trace_data.js"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--gen-tokens", type=int, default=16)
    args = ap.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        for cand in ("checkpoints/embedded_big.pt", "checkpoints/embedded.pt", "artifacts/checkpoints/embedded_big.pt", "artifacts/checkpoints/embedded.pt"):
            if os.path.exists(os.path.join(ROOT, cand)):
                ckpt_path = os.path.join(ROOT, cand)
                break
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = TinyLM(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg = model.cfg

    with open(os.path.join(ROOT, "data/vocab.json")) as handle:
        tokens = json.load(handle)["tokens"]
    to_id = {tok: i for i, tok in enumerate(tokens)}

    prompt = ["<BOS>", "<CHAR=BLUEY>", "<MODE=BEDTIME>", "<PLACE=BACKYARD>", "<THEME=HELP>", "<LINE>"]
    ids = [to_id[p] for p in prompt]

    trace, logits = traced_forward(model, ids, tokens)
    with torch.no_grad():
        ref, _ = model(torch.tensor([ids]))
    max_diff = float((ref[0, -1] - logits).abs().max())
    assert max_diff < 1e-3, f"hand-computed forward pass drifted from TinyLM: {max_diff}"

    # Next-token distribution: raw, then the runtime's sampling rules.
    scaled = logits / args.temperature
    scaled_masked = scaled.clone()
    scaled_masked[to_id["<EOS>"]] = -float("inf")
    kth = torch.topk(scaled_masked, args.top_k).values[-1]
    filtered = torch.where(scaled_masked < kth, torch.full_like(scaled_masked, -float("inf")), scaled_masked)
    prediction = {
        "raw_top15": topk(logits, tokens, 15),
        "temp_top15": topk(scaled, tokens, 15),
        "filtered_top15": topk(filtered, tokens, 15),
        "kept": int(torch.isfinite(filtered).sum()),
        "logit_min": round(float(logits.min()), 3),
        "logit_max": round(float(logits.max()), 3),
    }

    # The loop: feed each chosen token back in and go again.
    gen = torch.Generator().manual_seed(args.seed)
    seq = list(ids)
    steps = []
    for _ in range(args.gen_tokens):
        nxt, raw_top, p_chosen = next_token(model, seq, tokens, args.temperature, args.top_k, gen)
        steps.append({"context_len": len(seq), "chosen": tokens[nxt], "p_chosen": p_chosen, "top5": raw_top})
        seq.append(nxt)
        if tokens[nxt] == "<END>":
            break

    # Weights: stats for every tensor, histograms and crops for layer 0.
    sd = {k: v for k, v in ckpt["model"].items()}
    weights = {"tensors": [{"name": k, **stats(v)} for k, v in sd.items()]}
    weights["total_params"] = sum(p.numel() for p in model.parameters())
    weights["hist"] = {name: hist(sd[name]) for name in (
        "embed.weight", "blocks.0.attn.qkv.weight", "blocks.0.attn.proj.weight",
        "blocks.0.up.weight", "blocks.0.down.weight", "blocks.0.attn.norm.weight")}
    weights["crops"] = {
        "embed.weight": r(sd["embed.weight"][:24, :32], 3),
        "blocks.0.attn.qkv.weight": r(sd["blocks.0.attn.qkv.weight"][:24, :32], 3),
        "blocks.0.up.weight": r(sd["blocks.0.up.weight"][:24, :32], 3),
        "blocks.0.down.weight": r(sd["blocks.0.down.weight"][:24, :32], 3),
    }
    weights["crop_rows_embed"] = tokens[:24]
    weights["norm_l0"] = r(sd["blocks.0.attn.norm.weight"], 3)

    out = {
        "checkpoint": os.path.relpath(ckpt_path, ROOT),
        "config": {"vocab": cfg.vocab, "ctx": cfg.ctx, "layers": cfg.layers,
                   "d_model": cfg.d_model, "heads": cfg.heads, "head_dim": cfg.head_dim, "ffn": cfg.ffn},
        "sampling": {"temperature": args.temperature, "top_k": args.top_k, "seed": args.seed},
        "prompt": [{"token": p, "id": to_id[p]} for p in prompt],
        "check_max_logit_diff": round(max_diff, 8),
        "trace": trace,
        "prediction": prediction,
        "generation": steps,
        "weights": weights,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("// Generated by training/export_trace.py - do not edit by hand.\n")
        handle.write("window.TRACE = ")
        json.dump(out, handle, separators=(",", ":"))
        handle.write(";\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB) from {out['checkpoint']}")
    print(f"hand-computed vs TinyLM max logit diff: {max_diff:.2e}")
    print("first word candidates:", [(c["token"], c["p"]) for c in prediction["raw_top15"][:5]])
    print("generated:", " ".join(s["chosen"] for s in steps))


if __name__ == "__main__":
    main()
