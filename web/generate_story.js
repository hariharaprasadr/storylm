// Reusable story generation: a vanilla-JS port of storylm/model.py running on the real
// fp16 weights from web/story_data.js. No server, no library, no DOM — the same forward
// pass the device runtime runs, behind one function:
//
//     const story = await generateStory({
//       world: "bluey",      // world id or display name
//       mode: "bedtime",     // one of the world's modes
//       place: "backyard",   // one of the world's places
//       theme: "help",       // one of the nine themes
//       seed: 42             // same seed -> the same story; omit for a fresh one
//     });
//     story.lines;   // ["The moon was up when Bluey found a warm blanket ...", ...]
//     story.text;    // those lines joined by "\n"
//     story.world;   // "bluey" (the id, even when you pass the display name)
//     story.tokens;  // how many tokens came out, and story.ms for the forward pass
//
// Browser
//     <script src="story_data.js"></script>
//     <script src="generate_story.js"></script>
//     then `await generateStory({ ... })` from the console — or press the Story Studio
//     button, which calls this same function.
//
// Node
//     globalThis.window = globalThis;                       // story_data.js writes window.STORY
//     require("./web/story_data.js");
//     const { generateStory } = require("./web/generate_story.js");
//
// The lower-level pieces (step, sample, weights, config) stay on `StoryEngine`, which the
// walkthrough pages use to trace this exact forward pass. They take options.model, so a
// second export can be loaded without reloading the page.
(function (root) {
  "use strict";

  // The whole model, bound to one export object — no globals, so several can coexist.
  function bindEngine(S) {
  const cfg = S.config;
  const P = S.sampling;
  const vocab = S.vocab;
  // Modes come from the exported table, so a wider vocabulary needs no code change here.
  const MODES = (S.modes && S.modes.length)
    ? S.modes
    : [...new Set((S.worlds ?? []).flatMap((w) => w.modes ?? []))];
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

  // One token through the whole model -> logits [cfg.vocab].
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
  function generate(promptIds, rand = Math.random) {
    const cache = newCache();
    let pending = null;
    for (const id of promptIds) pending = step(cache, id);

    const seq = promptIds.slice();
    const seen = new Set();
    let linesDone = 0, lineTokens = 0, tries = 0;
    let mark = { seqLen: seq.length, bufLen: cache.len, pending: pending.slice() };

    while (linesDone < P.max_lines && lineTokens < P.max_new_tokens) {
      const token = sample(pending, rand);
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

    // The vocabulary lookup the sampler needs, built once per engine.
    let ids = null;
    const tokenId = (token) => {
      if (!ids) ids = Object.fromEntries(vocab.map((t, i) => [t, i]));
      return ids[token];
    };

    return { S, decodeAll, generate, renderLines, renderLine, sample, tokenId, vocab,
             config: cfg, sampling: P, weights: W, rmsnorm, matvec, gelu, step, newCache,
             rope: { cos: cosT, sin: sinT } };
  }

  // ---------- selection, seeding, and the one function ----------

  const pick = (value, allowed, what, hint) => {
    const fallback = allowed[0];
    const want = (value === undefined || value === null || value === "")
      ? String(fallback).toLowerCase() : String(value).trim().toLowerCase();
    const hit = allowed.find((a) => String(a).toLowerCase() === want);
    if (!hit) {
      throw new Error(`unknown ${what} "${value}"${hint ? ` for ${hint}` : ""}: expected one of ` +
                      allowed.join(", "));
    }
    return hit;
  };

  const modesOf = (S, world) =>
    (world.modes && world.modes.length ? world.modes
      : (S.modes && S.modes.length ? S.modes
        : [...new Set((S.worlds || []).flatMap((w) => w.modes || []))]));

  function select(S, opts) {
    const worlds = S.worlds || [];
    if (!worlds.length) throw new Error("the model export lists no worlds");
    const want = String(opts.world === undefined ? worlds[0].id : opts.world).trim().toLowerCase();
    const world = worlds.find((w) => w.id.toLowerCase() === want || String(w.name).toLowerCase() === want);
    if (!world) {
      throw new Error(`unknown world "${opts.world}": expected one of ` + worlds.map((w) => w.id).join(", "));
    }
    return {
      world,
      mode: pick(opts.mode, modesOf(S, world), "mode", `world "${world.id}"`),
      place: pick(opts.place, world.places || [], "place", `world "${world.id}"`),
      theme: pick(opts.theme, S.themes || [], "theme"),
    };
  }

  // mulberry32 (a string seed is hashed first), so a seed reproduces a story exactly
  const hashSeed = (text) => {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  };

  function makeRng(seed) {
    let state = seed >>> 0;
    for (let i = 0; i < 4; i++) {                      // stir, so small seeds do not correlate
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      state = (t ^ (t >>> 14)) >>> 0;
    }
    return function () {
      state |= 0;
      state = (state + 0x6D2B79F5) | 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const now = () => (typeof performance !== "undefined" && performance.now
    ? performance.now() : Date.now());

  let current = null;

  function engineFor(model) {
    if (model && (!current || current.S !== model)) current = null, publish(bindEngine(model));
    if (!current) {
      if (!root.STORY) {
        throw new Error("no model: load web/story_data.js first (`python training/export_story.py`), " +
                        "or pass { model } with the same shape");
      }
      publish(bindEngine(root.STORY));
    }
    return current;
  }

  function publish(engine) {
    current = engine;
    root.StoryEngine = Object.assign(root.StoryEngine || {}, apiFor(engine));
    return engine;
  }

  /**
   * Generate one story with the exported checkpoint. Runs the model: the first call also
   * decodes the weights, and every call is a full forward pass (tens of milliseconds to a
   * couple of seconds). Pass a seed to make the story reproducible, word for word.
   */
  async function generateStory(opts = {}) {
    const engine = engineFor(opts.model);
    const S = engine.S;
    const { world, mode, place, theme } = select(S, opts);
    const promptTokens = [world.token, `<MODE=${String(mode).toUpperCase()}>`,
                          `<PLACE=${String(place).toUpperCase()}>`, `<THEME=${String(theme).toUpperCase()}>`];
    const promptIds = [engine.tokenId("<BOS>"), ...promptTokens.map((t) => {
      const id = engine.tokenId(t);
      if (id === undefined) throw new Error(`the model's table has no ${t} token`);
      return id;
    })];

    const hasSeed = opts.seed !== undefined && opts.seed !== null;
    const rand = hasSeed
      ? makeRng(typeof opts.seed === "number" ? opts.seed >>> 0 : hashSeed(String(opts.seed)))
      : Math.random;

    engine.decodeAll();                               // idempotent; ~0.3 s for 5 MB of f16
    const started = now();
    const tokens = engine.generate(promptIds, rand);
    const lines = engine.renderLines(tokens, world.proper_names || S.properNames || []);
    return {
      lines,
      text: lines.join("\n"),
      world: world.id, mode, place, theme,
      seed: hasSeed ? opts.seed : null,
      tokens: tokens.length,
      ms: now() - started,
    };
  }

  const apiFor = (engine) => ({
    generateStory, decodeAll: engine.decodeAll, generate: engine.generate,
    renderLines: engine.renderLines, sample: engine.sample, tokenId: engine.tokenId,
    select: (opts = {}) => select(engine.S, opts), makeRng,
    vocab: engine.vocab, config: engine.config, sampling: engine.sampling, weights: engine.weights,
    worlds: engine.S.worlds || [], themes: engine.S.themes || [], modes: engine.S.modes || [],
    rmsnorm: engine.rmsnorm, matvec: engine.matvec, gelu: engine.gelu, step: engine.step,
    newCache: engine.newCache, rope: engine.rope,
  });

  root.generateStory = generateStory;
  if (root.STORY) engineFor();                         // the page's export, decoded on demand

  if (typeof module !== "undefined" && module.exports) {
    // Node: same API, with everything resolved against the loaded export on first use.
    module.exports = {
      generateStory, makeRng,
      decodeAll: () => engineFor().decodeAll(),
      generate: (ids, rand) => engineFor().generate(ids, rand),
      renderLines: (ids, names, stopAt) => engineFor().renderLines(ids, names, stopAt),
      sample: (logits, rand) => engineFor().sample(logits, rand),
      tokenId: (token) => engineFor().tokenId(token),
      step: (...args) => engineFor().step(...args),
      newCache: () => engineFor().newCache(),
      get vocab() { return engineFor().vocab; },
      get config() { return engineFor().config; },
      get sampling() { return engineFor().sampling; },
      get weights() { return engineFor().weights; },
      get worlds() { return engineFor().S.worlds || []; },
      get themes() { return engineFor().S.themes || []; },
      get modes() { return engineFor().S.modes || []; },
      get rope() { return engineFor().rope; },
    };
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
