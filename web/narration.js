// Narration for the web demo: the rule the device will use, executed in the browser.
//
// Three blobs of CBR MP3, each with a JSON index of (byte offset, length, seconds):
//
//   audio/lines.mp3    whole-line clips for the 100 most-used lines of the corpus
//   audio/compose.mp3  the pieces the device actually composes a line from:
//                      phrases cut out of whole recorded sentences at the slot
//                      boundaries, plus mid-sentence takes of the slot words
//   audio/words.mp3    every word spoken on its own (the fallback, and the reason
//                      the device needs the other two: see docs/narration.md)
//
// Playback rule, in order: a whole-line clip if the line is in the hot set; otherwise
// decompose the line into composed units (longest phrase first) and splice them; any
// word left over falls back to its own clip. Gaps: 30 ms between units, 140 ms after a
// comma, 320 ms at the end of a line.
//
// API
//   await Narration.load(onProgress)            -> lines + composed units (≈8 MB once)
//   await Narration.loadWords(onProgress)       -> the word library, only when asked
//   Narration.ready() / .readyWords()           -> what is loaded
//   Narration.plan(line, mode)                  -> how a line would be spoken
//   await Narration.play(lines, {mode, onLine, onDone})
//                                               mode: "compose" (default) | "words"
//   Narration.stop()
//   await Narration.playWord(word, {take})      -> {take: "mid"|"word"}
//   await Narration.playLine(text, {mode})
(() => {
  "use strict";

  const BASE = "audio/";
  const GAP_UNIT = 0.03, GAP_PUNCT = 0.14, GAP_LINE = 0.32;
  const TOKEN_RE = /[a-z0-9]+/g;

  const sets = { words: null, lines: null, compose: null };
  const cache = new Map();
  let ctx = null;
  let playing = [];
  let timers = [];
  const loads = {};

  const norm = (s) => (s.toLowerCase().match(TOKEN_RE) || []).join(" ");
  const wordsOf = (s) => s.toLowerCase().match(TOKEN_RE) || [];

  function audioContext() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  async function fetchSet(name) {
    const meta = await (await fetch(BASE + name + ".index.json")).json();
    const bytes = new Uint8Array(await (await fetch(BASE + name + ".mp3")).arrayBuffer());
    sets[name] = { meta, bytes, maxWords: maxWordsOf(meta.index) };
    return meta;
  }

  // Longest phrase in the index, so the matcher knows how far to look ahead.
  function maxWordsOf(index) {
    let most = 1;
    for (const key of Object.keys(index)) {
      const n = key.replace(/#e$/, "").split(" ").length;
      if (n > most) most = n;
    }
    return most;
  }

  function load(name, onProgress) {
    if (sets[name]) return Promise.resolve(sets[name].meta);
    if (!loads[name]) {
      onProgress && onProgress(name, 0);
      loads[name] = fetchSet(name).then((meta) => {
        onProgress && onProgress(name, 1, meta);
        return meta;
      });
    }
    return loads[name];
  }

  async function loadAll(onProgress) {
    const [lines, compose] = await Promise.all([load("lines", onProgress), load("compose", onProgress)]);
    return { lines, compose };
  }

  const ready = () => !!(sets.compose && sets.lines);
  const readyWords = () => !!sets.words;
  const loadWords = (onProgress) => load("words", onProgress).then(() => true);

  async function clipBuffer(setName, key) {
    const id = setName + "/" + key;
    if (cache.has(id)) return cache.get(id);
    const set = sets[setName];
    const entry = set && set.meta.index[key];
    if (!entry) return null;
    const slice = set.bytes.subarray(entry.o, entry.o + entry.n);
    const buf = await audioContext().decodeAudioData(slice.slice().buffer);
    cache.set(id, buf);
    return buf;
  }

  function schedule(buffer, at) {
    const c = audioContext();
    const src = c.createBufferSource();
    src.buffer = buffer;
    src.connect(c.destination);
    src.start(at);
    playing.push(src);
    return at + buffer.duration;
  }

  function stop() {
    playing.forEach((s) => { try { s.stop(); } catch (e) { /* already ended */ } });
    playing = [];
    timers.forEach((t) => clearTimeout(t));
    timers = [];
  }

  // ---- decomposition: the line -> the pieces the device would splice ----------

  function tokenize(line) {
    const out = [];
    for (const m of line.toLowerCase().matchAll(TOKEN_RE)) {
      out.push({ w: m[0], end: m.index + m[0].length });
    }
    return out;
  }

  // Does the sentence end after this token? (a full stop, not an abbreviation)
  function closesClause(line, tokens, i) {
    if (i >= tokens.length - 1) return true;
    let rest = line.slice(tokens[i].end);
    const m = rest.match(/^\s*([.!?…])/);
    if (!m) return false;
    const after = rest.slice(rest.indexOf(m[1]) + 1);
    return !/^\s*[a-z]/.test(after);        // "Mr. Scary" does not close a clause
  }

  // Longest phrase first, so "into the" wins over "into" + "the".
  function decompose(line) {
    const tokens = tokenize(line);
    const index = sets.compose.meta.index;
    const maxWords = sets.compose.maxWords || 1;
    const units = [];
    let i = 0;
    while (i < tokens.length) {
      let hit = null;
      for (let n = Math.min(maxWords, tokens.length - i); n >= 1; n--) {
        const key = tokens.slice(i, i + n).map((t) => t.w).join(" ");
        const atEnd = closesClause(line, tokens, i + n - 1);
        const key_ = atEnd && index[key + "#e"] ? key + "#e" : (index[key] ? key : null);
        if (key_) { hit = { key: key_, words: n, atEnd }; break; }
      }
      const start = i;
      if (hit) units.push(hit);
      else units.push({ key: null, words: 1, word: tokens[i].w, missing: true });
      i += units[units.length - 1].words;
      const last = units[units.length - 1];
      last.i0 = start;
      last.i1 = i;
    }
    return units;
  }

  // The clip already carries its own natural edges, so units follow each other
  // closely; only punctuation in the source text earns a real pause.
  function gapAfter(unit) {
    if (unit.endsComma) return GAP_PUNCT;
    return unit.atEnd ? GAP_LINE : GAP_UNIT;
  }

  function plan(line, mode = "compose") {
    const key = norm(line);
    const plain = { line, source: "none", key: null, units: [], missing: [] };
    if (mode !== "words" && sets.lines && sets.lines.meta.index[key]) {
      return Object.assign(plain, { source: "line", key });
    }
    if (mode === "words") {
      if (!sets.words) return plain;
      const words = wordsOf(line);
      return Object.assign(plain, {
        source: "words",
        units: words.map((w) => ({ key: sets.words.meta.index[w] ? w : null, words: 1, word: w })),
        missing: words.filter((w) => !sets.words.meta.index[w]),
      });
    }
    if (!sets.compose) return plain;
    const tokens = tokenize(line);
    const units = decompose(line).map((u) => {
      const tail = line.slice(tokens[u.i1 - 1].end);
      return Object.assign(u, { endsComma: /^\s*[,;:]/.test(tail),
                                atEnd: u.i1 >= tokens.length || closesClause(line, tokens, u.i1 - 1) });
    });
    return Object.assign(plain, { source: "compose", units,
                                  missing: units.filter((u) => !u.key).map((u) => u.word) });
  }

  // ---- playback ---------------------------------------------------------------

  async function play(list, opts = {}) {
    if (!ready()) await loadAll();
    const mode = opts.mode || "compose";
    if (mode === "words" && !sets.words) await loadWords();
    stop();
    const c = audioContext();
    const report = { lines: [], seconds: 0, lineClips: 0, composedLines: 0, wordLines: 0,
                     units: 0, missing: [] };
    let cursor = c.currentTime + 0.08;
    for (let i = 0; i < list.length; i++) {
      const line = list[i];
      const p = plan(line, mode);
      const started = cursor;
      if (p.source === "line") {
        cursor = schedule(await clipBuffer("lines", p.key), cursor);
        report.lineClips++;
      } else {
        for (const u of p.units || []) {
          let buf = null;
          if (u.key) {
            buf = await clipBuffer(mode === "words" ? "words" : "compose", u.key);
            if (buf && mode !== "words") report.units++;
          }
          if (!buf && u.word && sets.words && sets.words.meta.index[u.word]) {
            buf = await clipBuffer("words", u.word);            // last resort
          }
          if (buf) cursor = schedule(buf, cursor) + gapAfter(u);
          else if (u.word || u.key) report.missing.push(u.word || u.key);
        }
        if (p.source === "compose") report.composedLines++;
        else report.wordLines++;
      }
      report.lines.push({ line, source: p.source, start: started, end: cursor });
      if (opts.onLine) {
        const delay = Math.max(0, (started - c.currentTime) * 1000);
        timers.push(setTimeout(() => opts.onLine(i, report.lines[i]), delay));
      }
    }
    report.seconds = cursor - (c.currentTime + 0.08);
    const total = Math.max(0, (cursor - c.currentTime) * 1000);
    return await new Promise((resolve) => {
      timers.push(setTimeout(() => { if (opts.onDone) opts.onDone(report); resolve(report); }, total));
    });
  }

  async function playWord(word, opts = {}) {
    const take = opts.take === "mid" && sets.compose ? "compose" : "words";
    if (take === "compose") { if (!ready()) await loadAll(); }
    else if (!sets.words) await loadWords();
    const buf = await clipBuffer(take, norm(word));
    if (!buf) return null;
    stop();
    schedule(buf, audioContext().currentTime + 0.02);
    return { seconds: buf.duration, take };
  }

  const playLine = (text, opts = {}) => play([text], opts);

  window.Narration = {
    load: loadAll, loadWords, ready, readyWords, stop, play, plan, playWord, playLine,
    hasLine: (text) => ready() && !!sets.lines.meta.index[norm(text)],
    words: () => (sets.words ? Object.keys(sets.words.meta.index) : []),
    lines: () => (ready() ? Object.keys(sets.lines.meta.index) : []),
    units: () => (ready() ? Object.keys(sets.compose.meta.index) : []),
    meta: () => (ready() ? { compose: sets.compose.meta, lines: sets.lines.meta,
                              words: sets.words && sets.words.meta } : null),
    gaps: { unit: GAP_UNIT, punct: GAP_PUNCT, line: GAP_LINE },
  };
})();
