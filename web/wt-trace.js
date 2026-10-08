// Computes the walkthrough trace live in the browser, from the real weights
// shipped in story_data.js — the same checkpoint, engine and default sample
// as Story Studio (first world · bedtime · its first place · help).
// Nothing is pre-baked: every number below is calculated from the actual
// weights when the page loads, then verified against the runtime engine
// (StoryEngine.step) the studio uses to generate stories.
(() => {
  "use strict";

  const S = window.STORY;
  const E = window.StoryEngine;
  if (!S || !E) { window.WT_TRACE = null; return; }

  E.decodeAll();
  const cfg = E.config, P = E.sampling, vocab = E.vocab, W = E.weights;
  const D = cfg.d_model, H = cfg.heads, HD = cfg.head_dim, L = cfg.layers, F = cfg.ffn, V = cfg.vocab;
  const { rmsnorm, matvec, gelu } = E;
  const cosT = E.rope.cos, sinT = E.rope.sin;
  const toId = Object.fromEntries(vocab.map((t, i) => [t, i]));
  const lineEndId = toId["</LINE>"], endId = toId["<END>"], eosId = toId["<EOS>"];

  // The chips selected on Story Studio arrive as URL params
  // (?world=…&mode=…&place=…&theme=…, written by the links on the main page).
  // Anything missing or unknown falls back to the studio's default chips.
  const params = new URLSearchParams(location.search);
  const hasSelection = ["world", "mode", "place", "theme"].some((k) => params.has(k));
  const worldParam = params.get("world");
  const world = S.worlds.find((w) =>
    worldParam && (w.id === worldParam || w.name.toLowerCase() === worldParam.toLowerCase())) || S.worlds[0];
  const pickToken = (raw, tag, fallback) => {
    const token = `<${tag}=${String(raw).toUpperCase()}>`;
    return toId[token] !== undefined ? token : fallback;
  };
  const modeTok = pickToken(params.get("mode") ?? "bedtime", "MODE", "<MODE=BEDTIME>");
  const placeTok = pickToken(params.get("place") ?? world.places[0], "PLACE", `<PLACE=${world.places[0].toUpperCase()}>`);
  const themeTok = pickToken(params.get("theme") ?? "help", "THEME", "<THEME=HELP>");
  const promptTokens = ["<BOS>", world.token, modeTok, placeTok, themeTok];
  const ids = promptTokens.map((t) => toId[t]);
  const T = ids.length;

  // ---------- distributions ----------
  function softmaxTop(scores, k) {
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (scores[i] > mx) mx = scores[i];
    let sum = 0;
    const probs = new Float64Array(V);
    for (let i = 0; i < V; i++) { probs[i] = Math.exp(scores[i] - mx); sum += probs[i]; }
    for (let i = 0; i < V; i++) probs[i] /= sum;
    const idx = Array.from({ length: V }, (_, i) => i)
      .sort((a, b) => probs[b] - probs[a] || scores[b] - scores[a]).slice(0, k);
    return idx.map((i) => ({ token: vocab[i], id: i, p: probs[i], score: scores[i] }));
  }

  function applySamplingRules(logits) {
    // Exact mirror of StoryEngine.sample's ladder: temperature, <EOS> blocked, top-k.
    const scaled = new Float64Array(V);
    for (let i = 0; i < V; i++) scaled[i] = logits[i] / Math.max(P.temperature, 1e-6);
    scaled[eosId] = -Infinity;
    const topVals = Array.from(scaled).sort((a, b) => b - a);
    const kth = topVals[P.top_k - 1];
    const filtered = new Float64Array(V);
    for (let i = 0; i < V; i++) filtered[i] = scaled[i] >= kth ? scaled[i] : -Infinity;
    return { scaled, filtered, kth, kept: topVals.filter((v) => v >= kth).length };
  }

  function probsFrom(scores) {
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (scores[i] > mx) mx = scores[i];
    let sum = 0;
    const probs = new Float64Array(V);
    for (let i = 0; i < V; i++) { probs[i] = Math.exp(scores[i] - mx); sum += probs[i]; }
    for (let i = 0; i < V; i++) probs[i] /= sum;
    return probs;
  }

  // ---------- seeded sampling (deterministic for the walkthrough) ----------
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rng = mulberry32(0);
  let lastDice = null;
  const recordingRand = () => (lastDice = rng());

  function sampleStep(seqLogits) {
    lastDice = null;
    const chosenId = E.sample(seqLogits, recordingRand);
    const probs = probsFrom(applySamplingRules(seqLogits).filtered);
    return { id: chosenId, token: vocab[chosenId], p: probs[chosenId], dice: lastDice };
  }

  // ---------- forward passes (story.js math, recorded) ----------
  function embedRow(id) {
    return new Float32Array(W["embed.weight"].buffer, id * D * 4, D).slice();
  }

  function ropeAt(q, k, pos) {
    const co = cosT[pos], si = sinT[pos];
    for (let h = 0; h < H; h++) {
      const off = h * HD;
      for (let i = 0; i < HD / 2; i++) {
        const a = q[off + 2 * i], b = q[off + 2 * i + 1];
        q[off + 2 * i] = a * co[i] - b * si[i];
        q[off + 2 * i + 1] = a * si[i] + b * co[i];
        const ka = k[off + 2 * i], kb = k[off + 2 * i + 1];
        k[off + 2 * i] = ka * co[i] - kb * si[i];
        k[off + 2 * i + 1] = ka * si[i] + kb * co[i];
      }
    }
  }

  function logitsFrom(xLast) {
    return matvec(W["embed.weight"], rmsnorm(xLast, W["norm.weight"]), V, D);
  }

  function lensTop5(xLast) { return softmaxTop(logitsFrom(xLast), 5); }

  // The same forward pass as StoryEngine.step, with every intermediate kept.
  function tracedForward() {
    let x = ids.map(embedRow);
    const trace = { embed: x.map((v) => Array.from(v)), residual: [x.map((v) => Array.from(v))], layers: [] };

    for (let l = 0; l < L; l++) {
      const layer = {};
      layer.x_in_last = Array.from(x[T - 1]);

      const normed = x.map((v) => rmsnorm(v, W[`blocks.${l}.attn.norm.weight`]));
      layer.attn_norm_last = Array.from(normed[T - 1]);
      const qkv = normed.map((v) => matvec(W[`blocks.${l}.attn.qkv.weight`], v, 3 * D, D));
      const qs = [], ks = [], vs = [];
      qkv.forEach((q3, p) => {
        const q = q3.subarray(0, D).slice(), k = q3.subarray(D, 2 * D).slice();
        ropeAt(q, k, p);
        qs.push(q); ks.push(k); vs.push(q3.subarray(2 * D, 3 * D).slice());
      });
      layer.q_last = Array.from(qs[T - 1]);
      layer.k_last = Array.from(ks[T - 1]);
      layer.q_all = qs.map((v) => Array.from(v));   // [tokens][d_model], post-RoPE
      layer.k_all = ks.map((v) => Array.from(v));   // [tokens][d_model], post-RoPE
      layer.v_all = vs.map((v) => Array.from(v));   // [tokens][d_model]

      // score = q · k / sqrt(head_dim); softmax over visible tokens.
      const attn = [];
      for (let h = 0; h < H; h++) {
        const off = h * HD, head = [];
        for (let p = 0; p < T; p++) {
          const dots = [], vis = p + 1;
          for (let j = 0; j < vis; j++) {
            let s = 0;
            for (let d = 0; d < HD; d++) s += qs[p][off + d] * ks[j][off + d];
            dots.push(s);
          }
          let mx = -Infinity;
          for (let j = 0; j < vis; j++) if (dots[j] > mx) mx = dots[j];
          let sum = 0;
          const ex = dots.map((s) => { const e = Math.exp(s * 0.25 - mx * 0.25); sum += e; return e; });
          head.push(ex.map((e) => e / sum));
        }
        attn.push(head);
      }
      layer.attn_weights = attn;
      layer.attn_raw_last = [];
      for (let h = 0; h < H; h++) {
        const off = h * HD, dots = [];
        for (let j = 0; j < T; j++) {
          let s = 0;
          for (let d = 0; d < HD; d++) s += qs[T - 1][off + d] * ks[j][off + d];
          dots.push(s);
        }
        layer.attn_raw_last.push(dots);
      }

      const headsOut = x.map((_, p) => {
        const acc = new Float32Array(D);
        for (let h = 0; h < H; h++) {
          const off = h * HD;
          for (let j = 0; j <= p; j++) {
            const w = attn[h][p][j];
            for (let d = 0; d < HD; d++) acc[off + d] += w * vs[j][off + d];
          }
        }
        return acc;
      });
      const proj = headsOut.map((a) => matvec(W[`blocks.${l}.attn.proj.weight`], a, D, D));
      layer.attn_out_last = Array.from(proj[T - 1]);
      for (let p = 0; p < T; p++) for (let i = 0; i < D; i++) x[p][i] += proj[p][i];
      layer.x_mid_last = Array.from(x[T - 1]);

      const ffnIn = x.map((v) => rmsnorm(v, W[`blocks.${l}.norm.weight`]));
      const upRaw = ffnIn.map((v) => matvec(W[`blocks.${l}.up.weight`], v, F, D));
      const hidden = upRaw.map((u) => { const g = u.slice(); for (let i = 0; i < F; i++) g[i] = gelu(u[i]); return g; });
      layer.up_raw_last = Array.from(upRaw[T - 1]);
      layer.ffn_hidden_last = Array.from(hidden[T - 1]);
      layer.up_all = upRaw.map((v) => Array.from(v));       // [tokens][ffn], pre-GELU
      layer.hidden_all = hidden.map((v) => Array.from(v)); // [tokens][ffn], post-GELU
      layer.ffn_active = hidden[T - 1].filter((v) => v > 0.05).length;
      const ffnOut = hidden.map((hv) => matvec(W[`blocks.${l}.down.weight`], hv, D, F));
      layer.ffn_out_last = Array.from(ffnOut[T - 1]);
      for (let p = 0; p < T; p++) for (let i = 0; i < D; i++) x[p][i] += ffnOut[p][i];
      layer.x_out_last = Array.from(x[T - 1]);
      layer.norm_out_by_pos = x.map((v) => Math.hypot(...v));
      layer.lens_top5 = lensTop5(x[T - 1]);

      trace.layers.push(layer);
      trace.residual.push(x.map((v) => Array.from(v)));
    }

    trace.final_norm_last = Array.from(rmsnorm(x[T - 1], W["norm.weight"]));
    const logits = Array.from(logitsFrom(x[T - 1]));
    return { trace, logits };
  }

  // Lean recompute for the generation loop (no recording needed).
  function forwardLogits(seq) {
    const n = seq.length;
    let x = seq.map(embedRow);
    for (let l = 0; l < L; l++) {
      const normed = x.map((v) => rmsnorm(v, W[`blocks.${l}.attn.norm.weight`]));
      const qs = [], ks = [], vs = [];
      normed.forEach((v, p) => {
        const q3 = matvec(W[`blocks.${l}.attn.qkv.weight`], v, 3 * D, D);
        const q = q3.subarray(0, D).slice(), k = q3.subarray(D, 2 * D).slice();
        ropeAt(q, k, p);
        qs.push(q); ks.push(k); vs.push(q3.subarray(2 * D, 3 * D).slice());
      });
      const headsOut = x.map((_, p) => {
        const acc = new Float32Array(D);
        for (let h = 0; h < H; h++) {
          const off = h * HD;
          let mx = -Infinity;
          const dots = [];
          for (let j = 0; j <= p; j++) {
            let s = 0;
            for (let d = 0; d < HD; d++) s += qs[p][off + d] * ks[j][off + d];
            s *= 0.25;
            dots.push(s); if (s > mx) mx = s;
          }
          let sum = 0;
          const ex = dots.map((s) => { const e = Math.exp(s - mx); sum += e; return e; });
          for (let j = 0; j <= p; j++) {
            const w = ex[j] / sum;
            for (let d = 0; d < HD; d++) acc[off + d] += w * vs[j][off + d];
          }
        }
        return acc;
      });
      const proj = headsOut.map((a) => matvec(W[`blocks.${l}.attn.proj.weight`], a, D, D));
      for (let p = 0; p < n; p++) for (let i = 0; i < D; i++) x[p][i] += proj[p][i];
      const ffnIn = x.map((v) => rmsnorm(v, W[`blocks.${l}.norm.weight`]));
      const ffnOut = ffnIn.map((v) => {
        const u = matvec(W[`blocks.${l}.up.weight`], v, F, D);
        for (let i = 0; i < F; i++) u[i] = gelu(u[i]);
        return matvec(W[`blocks.${l}.down.weight`], u, D, F);
      });
      for (let p = 0; p < n; p++) for (let i = 0; i < D; i++) x[p][i] += ffnOut[p][i];
    }
    return logitsFrom(x[n - 1]);
  }

  // ---------- build the trace ----------
  const { trace, logits } = tracedForward();

  // Parity check: the traced hand-pass against the runtime engine the studio uses.
  const cache = E.newCache();
  let engineLogits = null;
  for (const id of ids) engineLogits = E.step(cache, id);
  let maxDiff = 0;
  for (let i = 0; i < V; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - engineLogits[i]));

  // Next-token distribution: raw, then each sampling rule in turn.
  const { scaled, filtered, kept } = applySamplingRules(logits);
  const prediction = {
    raw_top15: softmaxTop(logits, 15),
    temp_top15: softmaxTop(scaled, 15),
    filtered_top15: softmaxTop(filtered, 15),
    kept,
    logit_min: Math.min(...logits),
    logit_max: Math.max(...logits),
  };

  // The first dice roll the model makes after the Story Studio prompt —
  // then the loop that writes the first line, all with the seeded dice.
  const generation = [];
  const seq = ids.slice();
  for (let n = 0; n < 16; n++) {
    const stepLogits = n === 0 ? logits : Array.from(forwardLogits(seq));
    const pick = sampleStep(stepLogits);
    generation.push({
      context_len: seq.length,
      chosen: pick.token, chosen_id: pick.id,
      p_chosen: pick.p, dice: pick.dice,
      top5: softmaxTop(stepLogits, 5),
    });
    seq.push(pick.id);
    if (pick.id === lineEndId || pick.id === endId) break;
  }

  // ---------- weight facts (computed from the real tensors) ----------
  function shapeOf(name) {
    if (name === "embed.weight") return [V, D];
    if (name === "norm.weight") return [D];
    const part = name.replace(/^blocks\.\d+\./, "");
    return { "attn.norm.weight": [D], "attn.qkv.weight": [3 * D, D],
             "attn.proj.weight": [D, D], "norm.weight": [D],
             "up.weight": [F, D], "down.weight": [D, F] }[part];
  }
  const tensors = Object.keys(W).map((name) => {
    const a = W[name];
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    const mean = sum / a.length;
    let sq = 0;
    for (let i = 0; i < a.length; i++) { const d = a[i] - mean; sq += d * d; }
    return { name, shape: shapeOf(name), params: a.length, min: mn, max: mx, mean, std: Math.sqrt(sq / a.length) };
  });

  window.WT_TRACE = {
    checkpoint: S.checkpoint,
    config: cfg, sampling: { ...P, seed: 0 },
    sample: {
      world: world.name,
      mode: modeTok.slice(6, -1).toLowerCase(),
      place: placeTok.slice(7, -1).toLowerCase(),
      theme: themeTok.slice(7, -1).toLowerCase(),
      properNames: world.proper_names,
      from_selection: hasSelection,
    },
    prompt: promptTokens.map((t) => ({ token: t, id: toId[t] })),
    prompt_ids: ids,
    logits,
    check_max_logit_diff: maxDiff,
    trace, prediction, generation,
    weights: { tensors, total_params: tensors.reduce((a, t) => a + t.params, 0) },
    vocab, toId,
  };
})();