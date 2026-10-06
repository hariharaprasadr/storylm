#!/usr/bin/env python3
"""The compact model from docs/tiny_model.md, in PyTorch.

Decoder-only transformer, pre-norm, RoPE, GELU FFN, tied input/output
embeddings. The shapes here are the contract the C++ runtime has to match, so
``param_report()`` is the single source of truth for the size numbers in the
docs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 752
    ctx: int = 384
    layers: int = 8
    d_model: int = 128
    heads: int = 8
    ffn: int = 512

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads


def rope_tables(ctx, head_dim, device="cpu"):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(ctx, device=device).float()
    freqs = torch.outer(pos, inv)  # [ctx, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: [batch, heads, ctx, head_dim]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.norm = nn.RMSNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, buf=None, layer=0, prefill=False):
        b, t, _ = x.shape
        cos, sin = cos[:t], sin[:t]  # absolute positions, sliced to this window
        h, d = self.cfg.heads, self.cfg.head_dim
        q, k, v = self.qkv(self.norm(x)).split(self.cfg.d_model, dim=2)
        q = q.view(b, t, h, d).transpose(1, 2)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if buf is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Append the new keys/values and attend over everything so far.
            # Usage is prefill (whole prompt, causal) then one token at a time,
            # so a mask is only needed while the buffer is empty.
            start = buf.len
            buf.k[layer][:, :, start : start + t] = k
            buf.v[layer][:, :, start : start + t] = v
            keys = buf.k[layer][:, :, : start + t]
            values = buf.v[layer][:, :, : start + t]
            y = F.scaled_dot_product_attention(q, keys, values, is_causal=(start == 0))
        return self.proj(y.transpose(1, 2).reshape(b, t, h * d))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)
        self.norm = nn.RMSNorm(cfg.d_model)
        self.up = nn.Linear(cfg.d_model, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, buf=None, layer=0):
        x = x + self.attn(x, cos, sin, buf=buf, layer=layer)
        return x + self.down(F.gelu(self.up(self.norm(x))))


class KVBuf:
    """Preallocated key/value cache, one [1, heads, ctx, head_dim] pair per layer.

    The buffer is append-only during a story; a rejected line is resampled by
    resetting `len`, which is why the storage is preallocated rather than grown.
    """

    def __init__(self, cfg, batch=1):
        self.cfg = cfg
        self.len = 0
        self.batch = batch
        self.k = [torch.zeros(batch, cfg.heads, cfg.ctx, cfg.head_dim) for _ in range(cfg.layers)]
        self.v = [torch.zeros(batch, cfg.heads, cfg.ctx, cfg.head_dim) for _ in range(cfg.layers)]

    def reset(self, length=0):
        self.len = length

    @property
    def bytes(self):
        return 2 * self.cfg.layers * self.batch * self.cfg.heads * self.cfg.ctx * self.cfg.head_dim * 4


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.head.weight = self.embed.weight  # tied, as in the spec
        cos, sin = rope_tables(cfg.ctx, cfg.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None, loss_mask=None, pos_offset=0, buf=None):
        x = self.embed(idx)
        # Keep RoPE positions inside the table: when a window has slid past the
        # context (only a degenerate model does this) fall back to the last
        # `t` positions rather than slicing a short table.
        t = idx.shape[1]
        pos_offset = max(0, min(pos_offset, self.cfg.ctx - t))
        cos = self.cos[pos_offset : pos_offset + t]
        sin = self.sin[pos_offset : pos_offset + t]
        for layer, block in enumerate(self.blocks):
            x = block(x, cos, sin, buf=buf, layer=layer)
        if buf is not None:
            buf.len += idx.shape[1]
        logits = self.head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            reduction="none",
        )
        if loss_mask is not None:
            weights = loss_mask[:, 1:].reshape(-1)
            loss = (loss * weights).sum() / weights.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return logits, loss

    def param_report(self):
        embed = self.cfg.vocab * self.cfg.d_model
        per_block = (
            4 * self.cfg.d_model**2
            + 2 * self.cfg.d_model * self.cfg.ffn
            + 2 * self.cfg.d_model
        )
        total = embed + self.cfg.layers * per_block + self.cfg.d_model
        return {
            "vocab": self.cfg.vocab,
            "embedding": embed,
            "per_block": per_block,
            "blocks_total": self.cfg.layers * per_block,
            "total": total,
            "int4_bytes": int(total * 0.53125),
            "kv_int8_bytes": self.cfg.ctx * self.cfg.d_model * 2 * self.cfg.layers,
        }


def decode_lines(ids, tokens, proper_names=frozenset(), stop_at="<EOS>"):
    """Token ids back to the story lines, the way the device renders them.

    The table is case-free and letters-only, so case and punctuation are
    presentation, not vocabulary: the renderer restores the sentence-initial
    capital, the cast names, and the full stop. ``proper_names`` is a set of
    lowercase names, normally read from data/series.json.
    """
    lines, current = [], []
    for i in ids:
        token = tokens[int(i)]
        if token in (stop_at, "<END>"):
            break
        if token == "<LINE>":
            current = []
        elif token == "</LINE>":
            if current:
                lines.append(_render(current, proper_names))
            current = []
        elif token.startswith("<") and token.endswith(">"):
            continue
        else:
            current.append(token)
    if current:
        lines.append(_render(current, proper_names))
    return lines


def _render(words, proper_names):
    out = []
    for word in words:
        out.append(word.capitalize() if word in proper_names else word)
    text = " ".join(out) + "."
    return text[:1].upper() + text[1:]


@torch.no_grad()
@torch.no_grad()
def _sample(logits, temperature, top_k, eos_id):
    """One token from the current logits; temperature <= 0 is greedy."""
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / max(temperature, 1e-6)
    logits[eos_id] = -float("inf")
    if top_k:
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))


def generate_story_cached(model, prompt_ids, tokens, ctx, max_lines=26, temperature=0.8,
                          top_k=40, resample_tries=3, max_new_tokens=80):
    """The story loop with a KV cache: prefill once, then one token per step.

    The step that samples a token also appends it to the cache, so the sequence
    is never fed twice. A rejected line (a duplicate) is undone by restoring the
    cache length and the pending logits to the state at the start of that line.
    """
    line_end = tokens.index("</LINE>")
    end_id = tokens.index("<END>")
    eos_id = tokens.index("<EOS>")
    max_new_tokens = min(max_new_tokens, 80)

    buf = KVBuf(model.cfg)
    pending = model(torch.tensor([list(prompt_ids)], dtype=torch.long), buf=buf)[0][0, -1]

    seq = list(prompt_ids)
    seen = set()
    lines_done = 0
    line_tokens = 0
    tries = 0
    mark = (len(seq), buf.len, pending)

    while lines_done < max_lines and line_tokens < max_new_tokens:
        token = _sample(pending, temperature, top_k, eos_id)
        seq.append(token)
        pending = model(torch.tensor([[token]], dtype=torch.long), pos_offset=buf.len, buf=buf)[0][0, -1]
        if token == end_id:
            return seq
        line_tokens += 1
        if token != line_end:
            continue

        words = tuple(tokens[t] for t in seq[mark[0] + 1 : -1]
                      if not (tokens[t].startswith("<") and tokens[t].endswith(">")))
        if words in seen:
            del seq[mark[0]:]          # undo the line
            buf.reset(mark[1])
            pending = mark[2]
            line_tokens = 0
            tries += 1
            if tries >= resample_tries:
                return seq
            continue
        seen.add(words)
        lines_done += 1
        line_tokens = 0
        tries = 0
        mark = (len(seq), buf.len, pending)
    return seq


def generate_story(model, prompt_ids, tokens, ctx, max_lines=26, max_new_tokens=340,
                   temperature=0.8, top_k=40, resample_tries=3, use_cache=True):
    """Generate a whole story, one line at a time, under the runtime's rules.

    * a line ends at ``</LINE>`` and ``<END>`` ends the story;
    * ``<EOS>`` is masked while sampling: it is a data detail from the training
      tail, not a runtime control, and must never truncate a story;
    * a finished line that repeats an earlier line is resampled, because the
      firmware validator rejects duplicate lines.

    Returns the token ids of the story.
    """
    if use_cache:
        return generate_story_cached(model, prompt_ids, tokens, ctx, max_lines=max_lines,
                                     temperature=temperature, top_k=top_k,
                                     resample_tries=resample_tries,
                                     max_new_tokens=max_new_tokens)
    line_end = tokens.index("</LINE>")
    end_id = tokens.index("<END>")
    eos_id = tokens.index("<EOS>")
    max_new_tokens = min(max_new_tokens, 80)  # one sentence, not a ramble

    seq = list(prompt_ids)
    seen = set()
    for _ in range(max_lines):
        accepted = False
        for _try in range(resample_tries):
            fragment = _generate_line(model, seq, ctx, line_end, end_id, eos_id,
                                      max_new_tokens, temperature, top_k)
            if not fragment:
                return seq
            tail = tokens[fragment[-1]]
            words = tuple(
                tokens[t] for t in fragment
                if not (tokens[t].startswith("<") and tokens[t].endswith(">"))
            )
            if tail == "<END>":
                return seq + fragment
            if tail == "</LINE>" and words not in seen:
                seen.add(words)
                seq.extend(fragment)
                accepted = True
                break
        if not accepted:
            return seq  # every attempt repeated a line: end the story here
    return seq


@torch.no_grad()
def _generate_line(model, seq, ctx, line_end, end_id, eos_id, max_new_tokens, temperature, top_k):
    """One line is one sentence: the budget bounds a runaway sampler."""
    """One line of new tokens. Never mutates `seq`, so a rejected line can be
    resampled from the same context."""
    out = []
    for _ in range(max_new_tokens):
        full = seq + out
        offset = max(0, len(full) - ctx)
        window = torch.tensor([full[offset:]], dtype=torch.long)
        logits, _ = model(window, pos_offset=offset)
        logits = logits[0, -1]
        if temperature <= 0:  # deterministic mode, used by the equivalence test
            out.append(int(torch.argmax(logits)))
            if out[-1] in (line_end, end_id):
                return out
            continue
        logits = logits / max(temperature, 1e-6)
        logits[eos_id] = -float("inf")
        if top_k:
            kth = torch.topk(logits, top_k).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
        nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        out.append(nxt)
        if nxt in (line_end, end_id):
            return out
    return out
