// Live in-browser story generation: a vanilla-JS port of storylm/model.py
// running on the real fp16 weights from web/story_data.js (window.STORY).
// No server, no library — the same forward pass the device runtime will run.
(() => {
  "use strict";

  const S = window.STORY;
  const root = document.getElementById("story-studio");
  if (!S) {
    if (root) {
      root.querySelector("#studio-output").textContent =
        "Run `python training/export_story.py` to generate web/story_data.js, then reload.";
    }
    return;
  }

  const cfg = S.config;
  const P = S.sampling;
  const vocab = S.vocab;
  const MODES = ["adventure", "bedtime", "funny", "kindness"];
  const D = cfg.d_model, H = cfg.heads, HD = cfg.head_dim, L = cfg.layers, F = cfg.ffn, V = cfg.vocab;

  // ---------- weight decoding ----------
  function decodeF16(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    if (typeof Float16Array !== "undefined") {
      return new Float32Array(new Float16Array(bytes.buffer));
    }
    // Manual half -> float decode for older engines.
    const out = new Float32Array(bytes.length / 2);
    const dv = new DataView(bytes.buffer);
    for (let i = 0; i < out.length; i++) {
      const h = dv.getUint16(i * 2, true);
      const sign = (h & 0x8000) ? -1 : 1, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
      out[i] = exp === 0 ? sign * Math.pow(2, -14) * (frac / 1024)
        : exp === 31 ? sign * Infinity
        : sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
    return out;
  }

  const W = {};
  const decodeAll = () => {
    if (W.embed) return;
    for (const [name, b64] of Object.entries(S.weights)) W[name] = decodeF16(b64);
  };

  // ---------- RoPE tables ----------
  const cosT = [], sinT = [];
  for (let pos = 0; pos < cfg.ctx; pos++) {
    const co = new Float32Array(HD / 2), si = new Float32Array(HD / 2);
    for (let i = 0; i < HD / 2; i++) {
      const f = pos / Math.pow(10000, (2 * i) / HD);
      co[i] = Math.cos(f); si[i] = Math.sin(f);
    }
    cosT.push(co); sinT.push(si);
  }

  // ---------- math helpers ----------
  const gelu = (x) => x * 0.5 * (1 + (() => {
    // Abramowitz & Stegun 7.1.26 erf approximation, |err| <= 1.5e-7
    const t = 1 / (1 + 0.3275911 * Math.abs(x / Math.SQRT2));
    const u = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const r = 1 - u * Math.exp(-(x * x) / 2);
    return x >= 0 ? r : -r;
  })());

  function rmsnorm(x, w) {
    let ms = 0;
    for (let i = 0; i < D; i++) ms += x[i] * x[i];
    ms = 1 / Math.sqrt(ms / D + 1e-6);
    const y = new Float32Array(D);
    for (let i = 0; i < D; i++) y[i] = x[i] * ms * w[i];
    return y;
  }

  // Row-major [out, in] matvec.
  function matvec(wArr, x, out, inn) {
    const y = new Float32Array(out);
    for (let o = 0; o < out; o++) {
      let s = 0;
      const row = o * inn;
      for (let i = 0; i < inn; i++) s += wArr[row + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  // ---------- KV cache (grow-only arrays, truncate on resample) ----------
  const newCache = () => ({
    len: 0,
    k: Array.from({ length: L }, () => []), // per layer: list of per-head Float32Array
    v: Array.from({ length: L }, () => []),
  });
  const truncateCache = (cache, len) => {
    cache.len = len;
    for (let l = 0; l < L; l++) { cache.k[l].length = len; cache.v[l].length = len; }
  };

  // One token through the whole model -> logits [752].
  function step(cache, tokenId) {
    const pos = cache.len;
    let x = new Float32Array(W["embed.weight"].buffer, tokenId * D * 4, D).slice();

    for (let l = 0; l < L; l++) {
      // --- attention ---
      const xa = rmsnorm(x, W[`blocks.${l}.attn.norm.weight`]);
      const qkv = matvec(W[`blocks.${l}.attn.qkv.weight`], xa, 3 * D, D);
      const q = qkv.subarray(0, D).slice();
      const k = qkv.subarray(D, 2 * D).slice();
      const v = qkv.subarray(2 * D, 3 * D).slice();
      // RoPE on q and k, interleaved even/odd pairs.
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
      cache.k[l].push(k); cache.v[l].push(v);

      const attnOut = new Float32Array(D);
      const t = pos + 1;
      for (let h = 0; h < H; h++) {
        const off = h * HD;
        const scores = new Float32Array(t);
        let mx = -Infinity;
        for (let j = 0; j < t; j++) {
          let s = 0;
          const kv = cache.k[l][j];
          for (let d = 0; d < HD; d++) s += q[off + d] * kv[off + d];
          scores[j] = s * 0.25; // 1/sqrt(16)
          if (scores[j] > mx) mx = scores[j];
        }
        let sum = 0;
        for (let j = 0; j < t; j++) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
        for (let j = 0; j < t; j++) {
          const w = scores[j] / sum;
          const vv = cache.v[l][j];
          for (let d = 0; d < HD; d++) attnOut[off + d] += w * vv[off + d];
        }
      }
      const proj = matvec(W[`blocks.${l}.attn.proj.weight`], attnOut, D, D);
      for (let i = 0; i < D; i++) x[i] += proj[i];

      // --- feed-forward ---
      const xn = rmsnorm(x, W[`blocks.${l}.norm.weight`]);
      const up = matvec(W[`blocks.${l}.up.weight`], xn, F, D);
      for (let i = 0; i < F; i++) up[i] = gelu(up[i]);
      const down = matvec(W[`blocks.${l}.down.weight`], up, D, F);
      for (let i = 0; i < D; i++) x[i] += down[i];
    }
    cache.len = pos + 1;

    // --- tied output head over embed rows ---
    const xf = rmsnorm(x, W["norm.weight"]);
    return matvec(W["embed.weight"], xf, V, D);
  }

  // ---------- sampling (port of _sample + generate_story_cached) ----------
  const lineEnd = vocab.indexOf("</LINE>");
  const endId = vocab.indexOf("<END>");
  const eosId = vocab.indexOf("<EOS>");

  function sample(logits, rand = Math.random) {
    const scaled = new Float32Array(V);
    for (let i = 0; i < V; i++) scaled[i] = logits[i] / Math.max(P.temperature, 1e-6);
    scaled[eosId] = -Infinity;
    // top-k threshold
    const topVals = Array.from(scaled).sort((a, b) => b - a);
    const kth = topVals[P.top_k - 1];
    const probs = new Float64Array(V);
    let mx = -Infinity;
    for (let i = 0; i < V; i++) if (scaled[i] >= kth && scaled[i] > mx) mx = scaled[i];
    let sum = 0;
    for (let i = 0; i < V; i++) {
      probs[i] = scaled[i] >= kth ? Math.exp(scaled[i] - mx) : 0;
      sum += probs[i];
    }
    const r = rand() * sum;
    let acc = 0;
    for (let i = 0; i < V; i++) {
      acc += probs[i];
      if (acc >= r && probs[i] > 0) return i;
    }
    return probs.findIndex((p) => p > 0);
  }

  // Exact port of generate_story_cached.
  function generate(promptIds) {
    const cache = newCache();
    let pending = null;
    for (const id of promptIds) pending = step(cache, id);

    const seq = promptIds.slice();
    const seen = new Set();
    let linesDone = 0, lineTokens = 0, tries = 0;
    let mark = { seqLen: seq.length, bufLen: cache.len, pending: pending.slice() };

    while (linesDone < P.max_lines && lineTokens < P.max_new_tokens) {
      const token = sample(pending);
      seq.push(token);
      pending = step(cache, token);
      if (token === endId) return seq;
      lineTokens++;
      if (token !== lineEnd) continue;

      const words = [];
      for (let i = mark.seqLen + 1; i < seq.length - 1; i++) {
        const t = vocab[seq[i]];
        if (!(t.startsWith("<") && t.endsWith(">"))) words.push(t);
      }
      const key = words.join(" ");
      if (seen.has(key)) {
        seq.length = mark.seqLen;
        truncateCache(cache, mark.bufLen);
        pending = mark.pending;
        lineTokens = 0; tries++;
        if (tries >= P.resample_tries) return seq;
        continue;
      }
      seen.add(key);
      linesDone++; lineTokens = 0; tries = 0;
      mark = { seqLen: seq.length, bufLen: cache.len, pending: pending.slice() };
    }
    return seq;
  }

  // ---------- line rendering (port of decode_lines/_render) ----------
  function renderLines(ids, properNames, stopAt = "<EOS>") {
    const lines = [];
    let current = [];
    for (const id of ids) {
      const token = vocab[id];
      if (token === stopAt || token === "<END>") break;
      if (token === "<LINE>") current = [];
      else if (token === "</LINE>") {
        if (current.length) lines.push(renderLine(current, properNames));
        current = [];
      } else if (token.startsWith("<") && token.endsWith(">")) continue;
      else current.push(token);
    }
    if (current.length) lines.push(renderLine(current, properNames));
    return lines;
  }
  function renderLine(words, properNames) {
    const out = words.map((w) => (properNames.includes(w) ? w[0].toUpperCase() + w.slice(1) : w));
    let text = out.join(" ") + ".";
    return text[0].toUpperCase() + text.slice(1);
  }

  // ---------- widget ----------
  if (root) {
  const $id = (id) => document.getElementById(id);
  const worldSel = $id("studio-world"), modeSel = $id("studio-mode"),
        placeSel = $id("studio-place"), themeSel = $id("studio-theme");
  const btn = $id("studio-generate");
  const out = $id("studio-output"), stats = $id("studio-stats");

  const chipGroup = (sel, items, isWorld, defaultLabel) => {
    sel.replaceChildren();
    items.forEach((label, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.i = i;
      b.className = "studio-chip" + (label === (defaultLabel ?? items[0]) ? " is-on" : "");
      b.textContent = label;
      b.addEventListener("click", () => {
        [...sel.children].forEach((c) => c.classList.remove("is-on"));
        b.classList.add("is-on");
        if (isWorld) refreshPlaces();
        syncWalkthroughLinks();
      });
      sel.append(b);
    });
  };

  const currentWorld = () =>
    S.worlds[worldSel.querySelector(".is-on")?.dataset.i | 0];

  function refreshPlaces() {
    const w = currentWorld();
    chipGroup(placeSel, w.places);
  }

  // Keep every "walkthrough" link carrying the chips selected right now,
  // so the walkthrough traces the same sample the studio would generate.
  function syncWalkthroughLinks() {
    const w = currentWorld();
    const mode = modeSel.querySelector(".is-on")?.textContent ?? "bedtime";
    const place = placeSel.querySelector(".is-on")?.textContent ?? w.places[0];
    const theme = themeSel.querySelector(".is-on")?.textContent ?? "help";
    const q = `?world=${encodeURIComponent(w.id)}&mode=${encodeURIComponent(mode)}` +
      `&place=${encodeURIComponent(place)}&theme=${encodeURIComponent(theme)}`;
    document.querySelectorAll('a[href="walkthrough.html"], a[href^="walkthrough.html?"]')
      .forEach((a) => { a.href = "walkthrough.html" + q; });
  }
  chipGroup(worldSel, S.worlds.map((w) => w.name), true);
  chipGroup(modeSel, MODES, false, "bedtime");
  chipGroup(themeSel, S.themes, false, "help");
  refreshPlaces();
  syncWalkthroughLinks();
  let busy = false;
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.disabled = true;
    btn.textContent = "Thinking…";
    out.textContent = "";
    stats.textContent = "";

    await new Promise((r) => setTimeout(r, 30)); // let the button paint
    const started = performance.now();
    decodeAll();

    const w = currentWorld();
    const mode = modeSel.querySelector(".is-on").textContent;
    const place = placeSel.querySelector(".is-on").textContent;
    const theme = themeSel.querySelector(".is-on").textContent;
    const toId = Object.fromEntries(vocab.map((t, i) => [t, i]));
    const promptTokens = [w.token, `<MODE=${mode.toUpperCase()}>`,
                          `<PLACE=${place.toUpperCase()}>`, `<THEME=${theme.toUpperCase()}>`];
    const promptIds = [toId["<BOS>"], ...promptTokens.map((t) => toId[t])];

    const ids = generate(promptIds);
    const lines = renderLines(ids, w.proper_names);
    const ms = performance.now() - started;

    // Progressive per-line reveal.
    const h = document.createElement("div");
    h.className = "studio-byline";
    h.textContent = `${w.name} · ${mode} in the ${place} · a story about ${theme}`;
    out.append(h);
    let li = 0;
    const paint = () => {
      if (li < lines.length) {
        const p = document.createElement("div");
        p.className = "studio-line is-new";
        p.textContent = lines[li++];
        out.append(p);
        setTimeout(paint, 260);
      } else {
        stats.textContent =
          `${ids.length} tokens · generated in ${(ms / 1000).toFixed(1)}s ` +
          `(~${Math.round(ids.length / (ms / 1000))} tok/s) by the real 1.67M weights, in your browser.`;
        busy = false;
        btn.disabled = false;
        btn.textContent = "Tell another story";
      }
    };
    paint();
  });
  }

  // Exposed for headless testing / debugging and the web walkthrough (wt-trace.js).
  window.StoryEngine = {
    decodeAll, generate, renderLines, sample, vocab, config: cfg, sampling: P,
    weights: W, rmsnorm, matvec, gelu, step, newCache, rope: { cos: cosT, sin: sinT },
  };
})();
