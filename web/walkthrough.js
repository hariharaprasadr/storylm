// Step-by-step walkthrough of one real forward pass of StoryLM.
// All numbers are computed live at page load by web/wt-trace.js from the
// real fp16 weights in story_data.js — the same checkpoint, engine and
// default sample as Story Studio. Every panel shows the actual values and
// the actual arithmetic; nothing here is pre-baked or color-coded.
(() => {
  "use strict";

  const T = window.WT_TRACE;
  const MX = window.WTMatrix;
  const $ = (id) => document.getElementById(id);
  const root = $("walkthrough");
  if (!root) return;
  if (!T || !MX) {
    $("wt-title").textContent = "Story weights not found";
    $("wt-plain").textContent =
      "Run `python training/export_story.py` to generate web/story_data.js, then reload.";
    return;
  }

  const cfg = T.config;
  const S = T.sample;
  const toks = T.prompt.map((p) => p.token);
  const last = toks.length - 1;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fmt = (n) => n.toLocaleString("en-US");
  const f3 = (v) => v.toFixed(3);
  const f2 = (v) => v.toFixed(2);
  const pct = (p) => (p >= 0.9995 ? "≈100%" : p < 0.0005 ? "<0.1%" : (p * 100).toFixed(p < 0.1 ? 2 : 1) + "%");
  const norm = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
  const signClass = (v) => (v > 0.0005 ? "n-pos" : v < -0.0005 ? "n-neg" : "n-zero");

  // ---------- numeric helpers ----------
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  // rows: array of arrays of {text} strings or numbers; head: array of strings
  function table(head, rows, { rowLabel = (i) => "", hot = null } = {}) {
    const t = el("table", "wt-table");
    if (head.length) {
      const tr = el("tr");
      head.forEach((h) => tr.append(el("th", "", h)));
      t.append(tr);
    }
    rows.forEach((row, i) => {
      const tr = el("tr", hot && hot(i) ? "is-hot" : "");
      if (rowLabel(i)) tr.append(el("td", "wt-table-rowlab", rowLabel(i)));
      row.forEach((cell) => {
        if (typeof cell === "number") tr.append(el("td", signClass(cell), cell.toFixed(3)));
        else tr.append(el("td", "", cell));
      });
      t.append(tr);
    });
    const wrap = el("div", "wt-table-wrap");
    wrap.append(t);
    return wrap;
  }

  function figure(caption, node) {
    const f = el("div", "wt-figure");
    f.append(el("div", "wt-fig-cap", caption), node);
    return f;
  }

  function bars(items, { highlight, showScore, nd = 1 } = {}) {
    const wrap = el("div", "wt-bars");
    const max = Math.max(...items.map((i) => i.p), 1e-9);
    items.forEach((item) => {
      const row = el("div", "wt-bar-row" + (highlight && highlight(item) ? " is-hot" : ""));
      row.append(el("span", "wt-bar-label", item.token));
      const track = el("span", "wt-bar-track");
      const fill = el("span", "wt-bar-fill");
      fill.style.width = Math.max(0.6, (item.p / max) * 100) + "%";
      track.append(fill);
      row.append(track,
        el("span", "wt-bar-val", pct(item.p) + (showScore ? `  (score ${f2(item.score)})` : "")));
      wrap.append(row);
    });
    return wrap;
  }

  // ---------- residual stream as a horizontal strip of stages ----------
  const stageNames = ["Embedding", ...Array.from({ length: cfg.layers }, (_, i) => `Layer ${i + 1}`)];
  const map = $("wt-map");
  const mapRows = [];
  const stageNorms = T.trace.residual.map((stage) => norm(stage[last]));
  (function buildMap() {
    stageNorms.forEach((n, s) => {
      const col = el("div", "wt-map-col");
      col.append(el("div", "wt-map-col-label", stageNames[s]));
      col.append(el("div", "wt-map-col-norm", f2(n)));
      col.append(el("div", "wt-map-col-delta",
        s === 0 ? "—" : (n - stageNorms[s - 1] >= 0 ? "+" : "") + f2(n - stageNorms[s - 1])));
      col.title = `${stageNames[s]} · ‖x‖ = ${f2(n)}`;
      map.append(col);
      mapRows.push(col);
    });
  })();
  function highlightMap(stage) {
    mapRows.forEach((r, i) => {
      r.classList.toggle("is-active", i === stage);
      r.classList.toggle("is-done", stage !== null && i < stage);
    });
  }

  // ---------- steps ----------
  const steps = [];
  const chapter = (name) => ({ name });

  // 0. Weights — the real tensors, actual values.
  const tensorMap = Object.fromEntries(T.weights.tensors.map((t) => [t.name, t]));
  steps.push({
    chapter: "Weights", stage: null,
    kicker: "Step 0 · What the model has learned",
    title: `The model's memory: ${fmt(T.weights.total_params)} numbers`,
    plain: `Before any story is written, the model is just a big pile of numbers called weights. Training adjusted each one, a tiny bit at a time, until the model got good at guessing the next word of the training stories. Nothing else is stored: no story text and no rules, just these numbers. Pick a tensor to read its actual values — the same fp16 weights this page runs.`,
    numbers: `Checkpoint: ${T.checkpoint} · ${T.weights.tensors.length} tensors · decoded from base64 fp16 in your browser · hand-traced pass verified against the runtime engine (max logit diff ${T.check_max_logit_diff.toExponential(1)})`,
    render(v) {
      const names = ["embed.weight", "blocks.0.attn.qkv.weight", "blocks.0.attn.proj.weight",
        "blocks.0.up.weight", "blocks.0.down.weight", "blocks.0.attn.norm.weight"];
      const describe = {
        "embed.weight": `Word table: one row of 128 numbers per token (${cfg.vocab} rows). Also reused at the very end to score every possible next word.`,
        "blocks.0.attn.qkv.weight": "Layer 1 attention: turns each token's vector into a query, a key and a value.",
        "blocks.0.attn.proj.weight": "Layer 1 attention output: mixes the 8 heads' results back into one vector.",
        "blocks.0.up.weight": "Layer 1 feed-forward 'up' matrix: 128 → 512 features.",
        "blocks.0.down.weight": "Layer 1 feed-forward 'down' matrix: 512 → 128.",
        "blocks.0.attn.norm.weight": "Layer 1 RMSNorm gains: one scale factor per dimension.",
      };
      const picker = el("div", "wt-pills");
      const body = el("div");

      const show = (name) => {
        [...picker.children].forEach((b) => b.classList.toggle("is-on", b.dataset.name === name));
        body.replaceChildren();
        const t = tensorMap[name];
        const arr = StoryEngine.weights[name];
        const shape = t.shape.length === 1 ? [1, t.shape[0]] : t.shape;
        body.append(el("p", "wt-small", describe[name] || ""));
        body.append(el("div", "wt-mono",
          `shape [${t.shape.join(" × ")}] · ${fmt(t.params)} numbers · min ${f3(t.min)} · max ${f3(t.max)} · mean ${f3(t.mean)} · std ${f3(t.std)}`));

        // The real matrix, walkable in both axes. Word-table rows are labelled
        // with their vocabulary entry, so you can find any token's row.
        const rowLabel = (r) => (name === "embed.weight" ? `${r} · ${T.vocab[r]}` : t.shape.length === 1 ? "gains" : "row " + r);
        body.append(figure(
          `The actual weight matrix, ${shape[0]} × ${shape[1]} — every cell is a checkpoint value (colour shades the same number: orange positive, blue negative)`,
          MX.matrixWithPager({
            nRows: shape[0], nCols: shape[1], getValue: (r, c) => arr[r * shape[1] + c],
            rowLabelAt: rowLabel, colLabelAt: (c) => "d" + c,
            rowWindow: 8, colWindow: 16, digits: 2,
            title: `${name} [${shape[0]} × ${shape[1]}]`,
          })));

        // 3D view: the same window through all layers makes the layer axis real.
        const suffix = name.replace(/^blocks\.\d+\./, "");
        if (name.startsWith("blocks.")) {
          const rows = Math.min(8, shape[0]), cols = Math.min(16, shape[1]);
          const planes = Array.from({ length: cfg.layers }, (_, l) => {
            const a = StoryEngine.weights[`blocks.${l}.${suffix}`];
            const s = tensorMap[`blocks.${l}.${suffix}`].shape;
            const inn = s.length === 1 ? s[0] : s[1];
            return Array.from({ length: rows }, (_, r) =>
              Array.from({ length: cols }, (_, c) => a[r * inn + c]));
          });
          body.append(figure(
            `${suffix} stacked across all ${cfg.layers} layers — ${cfg.layers} × ${rows} × ${cols} real values`,
            MX.matrixStack({
              planes, labels: planes.map((_, l) => `layer ${l + 1}`),
              rowLabels: Array.from({ length: rows }, (_, r) => "row " + r),
              colLabels: Array.from({ length: cols }, (_, c) => "d" + c), digits: 2,
              title: `${suffix} across ${cfg.layers} layers`,
            })));
        }
      };
      names.forEach((n) => {
        const b = el("button", "wt-pill", n.replace("blocks.0.", "layer1.").replace(".weight", ""));
        b.type = "button"; b.dataset.name = n;
        b.addEventListener("click", () => show(n));
        picker.append(b);
      });
      const groups = el("div", "wt-groups");
      const per = T.weights.tensors.filter((t) => t.name.startsWith("blocks.0.")).reduce((a, t) => a + t.params, 0);
      [["Word table (shared in/out)", cfg.vocab * cfg.d_model],
       [`${cfg.layers} layers × ${fmt(per)}`, per * cfg.layers],
       ["Final norm", cfg.d_model]].forEach(([k, n]) => {
        const g = el("div", "wt-group");
        g.append(el("span", "", k), el("strong", "", fmt(n)));
        groups.append(g);
      });
      v.append(groups, picker, body);
      show(names[0]);
    },
  });

  // 1. Prompt — the Story Studio default sample.
  steps.push({
    chapter: "Prompt", stage: null,
    kicker: "Step 1 · The starting prompt",
    title: "The child's choices become 5 numbers",
    plain: `The model can't read. It only sees token ids, which are numbers. ${S.from_selection ? "These are the choices you selected on the Story Studio page" : "These are the chips Story Studio starts with"} — ${S.world} · ${/^[aeiou]/.test(S.mode) ? "an" : "a"} ${S.mode} story · in the ${S.place} · about ${S.theme}. <BOS> means "a new story starts here". The model's only job is to answer one question: which token comes next?`,
    numbers: `Vocabulary: ${cfg.vocab} tokens (`
      + `${T.vocab.filter((t) => t.startsWith("<")).length} control tokens + `
      + `${T.vocab.filter((t) => !t.startsWith("<")).length} words). Context window: up to ${cfg.ctx} tokens.`,
    render(v) {
      const meaning = ["start of story", `character: ${S.world}`, `mode: ${S.mode}`,
        `place: ${S.place}`, `theme: ${S.theme}`];
      const row = el("div", "wt-prompt");
      T.prompt.forEach((p, i) => {
        const chip = el("div", "wt-token" + (i === last ? " is-last" : ""));
        chip.append(el("span", "wt-token-text", p.token), el("span", "wt-token-id", "id " + p.id),
          el("span", "wt-token-mean", meaning[i] || ""));
        row.append(chip);
      });
      const q = el("div", "wt-question");
      q.append(el("span", "", toks.join(" ")), el("span", "wt-qmark", " → ?"));
      v.append(row, q);
    },
  });

  // 2. Embedding lookup — the actual 128 numbers per token.
  steps.push({
    chapter: "Embed", stage: 0,
    kicker: "Step 2 · Look up each token",
    title: "Each id becomes a list of 128 numbers",
    plain: `Each id picks one row from the word table, a list of 128 numbers called a vector. Think of it as the token's starting "meaning", written as coordinates. These are the real values from the checkpoint — read a few: the mix of positives and negatives is what the layers will push around. Every token's vector now travels up through all ${cfg.layers} layers.`,
    numbers: `Lookup: embed.weight[id] → 128 floats per token. ${toks.length} tokens × 128 = ${toks.length * 128} numbers enter layer 1.`,
    render(v) {
      // The lookup result is itself a matrix: one row per token, 128 columns.
      v.append(figure(
        `The embedding matrix this pass looks up — ${toks.length} token rows × ${cfg.d_model} dims copied straight out of the word table`,
        MX.matrixWithPager({
          nRows: T.trace.embed.length, nCols: cfg.d_model,
          getValue: (r, c) => T.trace.embed[r][c],
          rowLabelAt: (r) => `${T.prompt_ids[r]} · ${toks[r]}`, colLabelAt: (c) => "d" + c,
          colWindow: 16, digits: 3,
          title: `embedding lookup [${toks.length} tokens × ${cfg.d_model} dims]`,
        })));
      v.append(el("div", "wt-mono",
        T.trace.embed.map((vec, i) => `${toks[i]} ‖x‖ ${f2(norm(vec))}`).join("   ·   ")));
    },
  });

  // 3. Layers
  T.trace.layers.forEach((layer, li) => {
    const n = li + 1;
    const heads = layer.attn_weights; // [h][t][t]
    const lastRow = heads.map((h) => h[last]);
    const focus = lastRow.map((row, h) => {
      const m = Math.max(...row);
      return { h, m, tok: toks[row.indexOf(m)] };
    }).sort((a, b) => b.m - a.m);
    const top = focus[0];

    steps.push({
      chapter: `L${n}`, stage: n,
      kicker: `Layer ${n} of ${cfg.layers} · Part A: attention`,
      title: `Which earlier tokens should ${toks[last]} listen to?`,
      plain: `Attention lets each token collect information from the tokens before it. We follow the last token, ${toks[last]}, because its vector is the one that will predict the next word. There are ${cfg.heads} attention heads, and each one decides independently how much to listen to each earlier token. Their weights always add up to 100%. In this layer, head ${top.h + 1} is the most focused: it puts ${pct(top.m)} of its attention on ${top.tok}.`,
      numbers: `For each head: score = (query · key) / √${cfg.head_dim}, then softmax over the ${toks.length} visible tokens. A token can't see tokens after it (causal mask). RoPE rotated queries and keys by position before the dot products below.`,
      render(v) {
        const picker = el("div", "wt-pills");
        const detail = el("div");
        const show = (h) => {
          [...picker.children].forEach((b, i) => b.classList.toggle("is-on", i === h));
          detail.replaceChildren();
          // The actual arithmetic for this head: q·k, the ÷√head_dim scale, the softmax.
          detail.append(figure(
            `Head ${h + 1}: how much ${toks[last]} listens to each earlier token — the real q·k dot products, the ÷√${cfg.head_dim} scores and the softmax weights`,
            table(["token", "q · k", `× 1/√${cfg.head_dim}`, "softmax weight"],
              layer.attn_raw_last[h].map((dot, j) =>
                [dot.toFixed(2), (dot / Math.sqrt(cfg.head_dim)).toFixed(3), pct(heads[h][last][j])]),
              { rowLabel: (j) => toks[j], hot: (j) => j === heads[h][last].indexOf(Math.max(...heads[h][last])) })));
          // Full attention matrix for this head, every token.
          detail.append(figure(
            `Head ${h + 1}, every token's attention weights in % (rows look at columns; the blocked top-right triangle is marked —)`,
            table(["query ↓ · key →", ...toks.map((t) => t.replace(/^<|>$/g, ""))],
              heads[h].map((row, p) => Array.from({ length: toks.length }, (_, j) => (j <= p ? pct(row[j]) : "—"))),
              { rowLabel: (p) => toks[p].replace(/^<|>$/g, "") })));
        };
        heads.forEach((_, h) => {
          const b = el("button", "wt-pill", `head ${h + 1}`);
          b.type = "button";
          b.addEventListener("click", () => show(h));
          picker.append(b);
        });

        // q, k and v are three real [tokens × 128] matrices this layer built.
        const qkvPick = el("div", "wt-pills");
        const qkvBody = el("div");
        const showQKV = (which) => {
          [...qkvPick.children].forEach((b) => b.classList.toggle("is-on", b.dataset.w === which));
          qkvBody.replaceChildren(MX.matrixWithPager({
            nRows: toks.length, nCols: cfg.d_model,
            getValue: (r, c) => layer[`${which}_all`][r][c],
            rowLabelAt: (r) => `${toks[r]}`, colLabelAt: (c) => "d" + c,
            colWindow: 16, digits: 2,
            title: `${which} · layer ${n} [${toks.length} tokens × ${cfg.d_model}]`,
          }));
        };
        [["q", "query x"], ["k", "key x"], ["v", "value x"]].forEach(([which, label]) => {
          const b = el("button", "wt-pill", label);
          b.type = "button"; b.dataset.w = which;
          b.addEventListener("click", () => showQKV(which));
          qkvPick.append(b);
        });
        const qkvGroup = el("div", "wt-mx-group");
        qkvGroup.append(qkvPick, qkvBody);
        v.append(figure(
          `The three matrices this layer computes from every token — ${toks.length} × ${cfg.d_model} each (queries and keys already rotated by position)`,
          qkvGroup));
        showQKV("q");

        v.append(picker, detail);
        show(top.h);

        // The attention tensor in full: one matrix per head, stacked in 3D.
        const short = (t) => t.replace(/^<|>$/g, "").slice(0, 9);
        v.append(figure(
          `All ${cfg.heads} attention heads as one ${cfg.heads} × ${toks.length} × ${toks.length} tensor (weights in %, rows look at columns, blocked cells blank)`,
          MX.matrixStack({
            planes: heads.map((head) => Array.from({ length: toks.length }, (_, p) =>
              Array.from({ length: toks.length }, (_, j) => (j <= p ? head[p][j] * 100 : null)))),
            labels: heads.map((_, h) => `head ${h + 1}`),
            rowLabels: toks.map(short), colLabels: toks.map(short), digits: 1,
            title: `attention heads · layer ${n} [${cfg.heads} × ${toks.length} × ${toks.length}]`,
          })));
      },
    });

    steps.push({
      chapter: `L${n}`, stage: n,
      kicker: `Layer ${n} of ${cfg.layers} · Part B: feed-forward`,
      title: "512 feature detectors switch on or stay quiet",
      plain: `Next, each token's vector goes through a small two-step network on its own, with no looking at other tokens. First it is widened to 512 "features": each feature is a learned pattern detector that lights up when its preferred mix of inputs is present — GELU lets positive signals through and mostly blocks negative ones. Here ${layer.ffn_active} of the 512 features are clearly active for ${toks[last]}. Then the 512 numbers are squeezed back down to 128 and added into the vector.`,
      numbers: `up = up_weight · RMSNorm(x) is 512 values; hidden = GELU(up); out = down_weight · hidden is 128 values. Every number below comes from the real weights.`,
      render(v) {
        // The FFN result is a matrix too: one 512-value row per token.
        const ffnPick = el("div", "wt-pills");
        const ffnBody = el("div");
        const showFFN = (which) => {
          [...ffnPick.children].forEach((b) => b.classList.toggle("is-on", b.dataset.w === which));
          ffnBody.replaceChildren(MX.matrixWithPager({
            nRows: toks.length, nCols: cfg.ffn,
            getValue: (r, c) => layer[which][r][c],
            rowLabelAt: (r) => toks[r], colLabelAt: (c) => "f" + c,
            colWindow: 16, digits: 2,
            title: `${which === "up_all" ? "up = up·x" : "GELU(up)"} · layer ${n} [${toks.length} × ${cfg.ffn}]`,
          }));
        };
        [["up_all", "up = up·x"], ["hidden_all", "GELU(up)"]].forEach(([which, label]) => {
          const b = el("button", "wt-pill", label);
          b.type = "button"; b.dataset.w = which;
          b.addEventListener("click", () => showFFN(which));
          ffnPick.append(b);
        });
        const ffnGroup = el("div", "wt-mx-group");
        ffnGroup.append(ffnPick, ffnBody);
        v.append(figure(
          `The feed-forward matrix for all ${toks.length} tokens — ${toks.length} × ${cfg.ffn} real values (${layer.ffn_active} of ${toks[last]}'s 512 features above 0.05)`,
          ffnGroup));
        showFFN("up_all");
        // How the vector actually moves through this layer, dimension by dimension.
        const rows = Array.from({ length: 8 }, (_, d) => [
          layer.x_in_last[d], layer.attn_out_last[d], layer.x_mid_last[d], layer.ffn_out_last[d], layer.x_out_last[d]]);
        v.append(figure(`${toks[last]}'s vector moving through layer ${n} — dims 0–7 of 128 (the full table has 128 rows)`,
          table(["dim", "x in", "attention adds", "x mid", "feed-fwd adds", "x out"],
            rows, { rowLabel: (d) => "d" + d })));
      },
    });

    const prev = li > 0 ? T.trace.layers[li - 1].lens_top5[0] : null;
    const guess = layer.lens_top5[0];
    let story;
    if (!prev) story = `After one layer, the closest guess is "${guess.token}" (${pct(guess.p)}). The model has barely started thinking.`;
    else if (prev.token === guess.token) story = `The guess stays "${guess.token}", now at ${pct(guess.p)}.`;
    else story = `The guess changes from "${prev.token}" to "${guess.token}" (${pct(guess.p)}).`;
    if (n === cfg.layers) story += " This is the real answer: the model's final belief.";

    steps.push({
      chapter: `L${n}`, stage: n,
      kicker: `Layer ${n} of ${cfg.layers} · Part C: update and peek`,
      title: n === cfg.layers ? "The final vector and the final guess" : "If we stopped here, what would it guess?",
      plain: `Each layer doesn't replace the vector. It adds to it: after attention the vector's length is ${f2(norm(layer.x_mid_last))}, after the feed-forward part ${f2(norm(layer.x_out_last))} (it started this layer at ${f2(norm(layer.x_in_last))}). This running total is called the residual stream. We can "peek" at any layer by sending the current vector straight to the output step, a trick called the logit lens — the peek is a real calculation: softmax(word_table · RMSNorm(x)). ${story}`,
      numbers: `x ← x + attention(x), then x ← x + feedforward(x). Peek scores: "${guess.token}" = row · RMSNorm(x) = ${f2(guess.score)}.`,
      render(v) {
        // The residual stream itself as a matrix: every stage × every dim.
        v.append(figure(
          `The predicting vector's journey — ${n + 1} stages × ${cfg.d_model} dims of the residual stream (rows are stages, colours shade the real values)`,
          MX.matrixWithPager({
            nRows: n + 1, nCols: cfg.d_model,
            getValue: (r, c) => T.trace.residual[r][last][c],
            rowLabelAt: (r) => stageNames[r], colLabelAt: (c) => "d" + c,
            colWindow: 16, digits: 2,
            title: `residual stream up to layer ${n} [${n + 1} stages × ${cfg.d_model}]`,
          })));
        v.append(figure(`Peek after layer ${n}: top 5 next-token guesses (logit lens)`,
          bars(layer.lens_top5, { highlight: (it) => it === layer.lens_top5[0], showScore: true })));
        const trail = el("div", "wt-trail");
        T.trace.layers.slice(0, n).forEach((l, i) => {
          const c = el("span", "wt-trail-chip" + (i === li ? " is-on" : ""), `L${i + 1}: ${l.lens_top5[0].token}`);
          trail.append(c);
        });
        v.append(figure("How the guess has evolved so far", trail));
      },
    });
  });

  // 4. Output head
  const P = T.prediction;
  const firstPick = T.generation[0];
  steps.push({
    chapter: "Predict", stage: cfg.layers,
    kicker: "Step 4 · Score every word",
    title: `128 numbers become ${cfg.vocab} scores`,
    plain: `The final vector is normalized, then compared with every row of the same word table from step 2. A word whose row "points the same way" as the vector gets a high score — the score for "${P.raw_top15[0].token}" is literally the 128-multiply dot product row · vector = ${f2(P.raw_top15[0].score)}. Softmax turns the ${cfg.vocab} scores into probabilities that add up to 100%. The top guess, "${P.raw_top15[0].token}", gets ${pct(P.raw_top15[0].p)}.${P.raw_top15[0].token === "<LINE>" ? " That makes sense: after the four setup choices, every training story starts writing its first line." : " Step 5's dice decide what actually gets written."}`,
    numbers: `scores = word_table (${cfg.vocab}×${cfg.d_model}) · RMSNorm(x) · scores range ${f2(P.logit_min)} to ${f2(P.logit_max)} · p = softmax(scores).`,
    render(v) {
      v.append(figure(`Top 15 of ${cfg.vocab} tokens (raw probabilities, real scores)`,
        bars(P.raw_top15, { highlight: (it) => it === P.raw_top15[0], showScore: true })));
      // The output head, mapped back onto the vocabulary: one cell per word.
      const headVector = T.trace.final_norm_last;
      v.append(figure(
        `Every one of the ${cfg.vocab} output scores, mapped back to its word — the same ${cfg.d_model}-number vector dotted with all ${cfg.vocab} rows of the word table`,
        MX.vocabGrid({ scores: T.logits, vocab: T.vocab, topK: 15, digits: 2 })));
      v.append(el("div", "wt-mono",
        `the vector being dotted with every row: RMSNorm(x) = [${headVector.slice(0, 8).map((v) => v.toFixed(2)).join(", ")}, …]`));
    },
  });

  // 5. Sampling
  steps.push({
    chapter: "Sample", stage: cfg.layers,
    kicker: "Step 5 · Pick one token",
    title: "Roll weighted dice, with safety rules",
    plain: `The model gives probabilities, but something still has to choose. The runtime applies three rules. Temperature ${T.sampling.temperature} sharpens the odds. <EOS> is blocked so a story can't stop before it starts. Only the top ${T.sampling.top_k} candidates stay in the running — ${P.kept} tokens here. Then it rolls weighted dice. Story Studio rolls fresh dice every time you click "Tell a story"; this walkthrough fixes the dice (seed ${T.sampling.seed}) so the numbers stay put. They land on "${firstPick.chosen}" at ${pct(firstPick.p_chosen)}${firstPick.chosen === "<LINE>" ? " — the model opens its first line" : ""}.`,
    numbers: `seed ${T.sampling.seed} · dice r = ${firstPick.dice.toFixed(4)} of 1.0 · chosen "${firstPick.chosen}" with p = ${pct(firstPick.p_chosen)} after the rules.`,
    render(v) {
      const cols = el("div", "wt-sample-cols");
      const col = (title, items) => {
        const c = el("div", "wt-sample-col");
        c.append(el("div", "wt-fig-cap", title), bars(items.slice(0, 6), { highlight: (it) => it.token === firstPick.chosen, showScore: true }));
        return c;
      };
      cols.append(col("1 · raw", P.raw_top15),
        col(`2 · temperature ${T.sampling.temperature}`, P.temp_top15),
        col(`3 · <EOS> blocked, top-${T.sampling.top_k}`, P.filtered_top15));
      const pick = el("div", "wt-picked");
      pick.append(el("span", "", "Chosen next token:"), el("strong", "", firstPick.chosen),
        el("span", "", ` (p = ${pct(firstPick.p_chosen)}, dice r = ${firstPick.dice.toFixed(4)})`));
      v.append(cols, pick);
    },
  });

  // 6. Loop
  steps.push({
    chapter: "Loop", stage: null,
    kicker: "Step 6 · Repeat",
    title: "Add the token, run everything again",
    plain: `The chosen token is added to the end of the prompt, and steps 2 to 5 run again to get the next token, then again, until the line ends. Each click below shows what the model actually considered at that point. The model is near-certain almost every time — the highlighted token is where it made a real choice, and that is where variety comes from: roll different dice there and you get a different story, exactly like Story Studio's button.`,
    numbers: `This shows the first line (${T.generation.length} dice rolls, seed ${T.sampling.seed}). On the device, a KV cache means earlier tokens aren't recomputed. Click any token to see the alternatives.`,
    render(v) {
      const line = el("div", "wt-gen");
      const info = el("div", "wt-gen-info");
      info.append(el("div", "wt-small", "Click a token to see what else it considered."));
      const showInfo = (s) => {
        info.replaceChildren(el("div", "wt-fig-cap",
          `With ${s.context_len} tokens of context → dice r = ${s.dice.toFixed(4)} → chose "${s.chosen}" (${pct(s.p_chosen)} after the rules)`),
          bars(s.top5, { highlight: (it) => it.token === s.chosen }));
      };
      const chips = T.generation.map((s) => {
        const c = el("button", "wt-gen-tok" + (s.p_chosen < 0.9 ? " is-choice" : ""), s.chosen);
        c.type = "button";
        c.title = `p = ${pct(s.p_chosen)}`;
        c.addEventListener("click", () => {
          chips.forEach((x) => x.classList.remove("is-on"));
          c.classList.add("is-on");
          showInfo(s);
        });
        return c;
      });
      const promptChip = el("span", "wt-gen-prompt", toks.join(" "));
      line.append(promptChip);
      v.append(line, info);
      if (reduceMotion) { chips.forEach((c) => line.append(c)); }
      else {
        let i = 0;
        const tick = () => {
          if (!line.isConnected || i >= chips.length) return;
          chips[i].classList.add("is-new");
          line.append(chips[i++]);
          setTimeout(tick, 550);
        };
        tick();
      }
      // The line these dice actually wrote, rendered the way the device speaks it.
      const seqIds = [...T.prompt_ids, ...T.generation.map((g) => g.chosen_id)];
      const lines = window.StoryEngine.renderLines(seqIds, S.properNames, "</LINE>");
      if (lines.length) v.append(figure("The first line these dice wrote", el("div", "wt-mono", lines[0])));
    },
  });

  // ---------- rail, controls ----------
  const rail = $("wt-rail");
  const chapters = [...new Set(steps.map((s) => s.chapter))];
  chapters.forEach((ch) => {
    const b = el("button", "wt-rail-btn", ch);
    b.type = "button";
    b.addEventListener("click", () => go(steps.findIndex((s) => s.chapter === ch)));
    rail.append(b);
  });

  let cur = 0;
  let timer = null;
  function go(i) {
    cur = Math.max(0, Math.min(steps.length - 1, i));
    const s = steps[cur];
    $("wt-kicker").textContent = s.kicker;
    $("wt-title").textContent = s.title;
    $("wt-plain").textContent = s.plain;
    $("wt-numbers").textContent = s.numbers;
    const v = $("wt-visual");
    v.replaceChildren();
    s.render(v);
    $("wt-count").textContent = `${cur + 1} / ${steps.length}`;
    $("wt-progress").style.width = ((cur + 1) / steps.length) * 100 + "%";
    $("wt-prev").disabled = cur === 0;
    $("wt-next").disabled = cur === steps.length - 1;
    [...rail.children].forEach((b) => b.classList.toggle("is-on", b.textContent === s.chapter));
    highlightMap(s.stage);
    const panel = $("wt-panel");
    panel.classList.remove("is-enter");
    void panel.offsetWidth;
    panel.classList.add("is-enter");
    if (cur === steps.length - 1) stop();
  }
  function stop() {
    clearTimeout(timer); timer = null;
    $("wt-play").textContent = "▶ Play";
    $("wt-play").setAttribute("aria-pressed", "false");
  }
  function play() {
    if (cur === steps.length - 1) go(0);
    $("wt-play").textContent = "❚❚ Pause";
    $("wt-play").setAttribute("aria-pressed", "true");
    const tick = () => {
      timer = setTimeout(() => {
        if (cur < steps.length - 1) { go(cur + 1); tick(); } else stop();
      }, Number($("wt-speed").value));
    };
    tick();
  }
  $("wt-prev").addEventListener("click", () => { stop(); go(cur - 1); });
  $("wt-next").addEventListener("click", () => { stop(); go(cur + 1); });
  $("wt-play").addEventListener("click", () => (timer ? stop() : play()));
  root.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT") return;
    if (e.key === "ArrowRight") { stop(); go(cur + 1); }
    if (e.key === "ArrowLeft") { stop(); go(cur - 1); }
  });
  go(0);
})();