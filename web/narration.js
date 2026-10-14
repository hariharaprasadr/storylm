// Narration for the web demo: the same rule the device uses, executed in the browser.
//
//   * one blob of word clips (audio/words.mp3 + words.index.json, CBR MP3)
//   * one blob of whole-line ("hot line") clips (audio/lines.mp3 + lines.index.json)
//   * play a line as a single clip when it is in the hot set, otherwise assemble it
//     from word clips with the gaps the device page measured (45 ms between words,
//     140 ms after a comma, 320 ms at the end of a line)
//
// The clips are Kokoro-82M takes (Apache-2.0), rendered offline in tools/tts and
// packed by training/pack_web_audio.py. Nothing here talks to a server at runtime.
//
// API
//   await Narration.load(onProgress)                 -> fetches both blobs (≈9 MB once)
//   Narration.ready()                                -> true when loaded
//   Narration.hasLine(text)                          -> is the whole line in the hot set?
//   Narration.words() / Narration.lines()            -> available keys
//   await Narration.play(lines, {mode, onLine})      -> mode: "hybrid" | "words"
//   Narration.stop()
//   await Narration.playWord(word)
//   await Narration.playLine(text, {mode})
(() => {
  "use strict";

  const BASE = "audio/";
  const GAP_WORD = 0.045, GAP_PUNCT = 0.14, GAP_LINE = 0.32;
  const WORD_RE = /[a-z]+/g;

  const sets = { words: null, lines: null };
  const cache = new Map();
  let ctx = null;
  let playing = [];
  let timers = [];
  let loadPromise = null;

  const norm = (s) => (s.toLowerCase().match(WORD_RE) || []).join(" ");

  function audioContext() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  async function fetchSet(name) {
    const meta = await (await fetch(BASE + name + ".index.json")).json();
    const bytes = new Uint8Array(await (await fetch(BASE + name + ".mp3")).arrayBuffer());
    sets[name] = { meta, bytes };
    return meta;
  }

  async function load(onProgress) {
    if (loadPromise) return loadPromise;
    loadPromise = (async () => {
      onProgress && onProgress("words", 0);
      const w = await fetchSet("words");
      onProgress && onProgress("words", 1, w);
      onProgress && onProgress("lines", 0);
      const l = await fetchSet("lines");
      onProgress && onProgress("lines", 1, l);
      return { words: w, lines: l };
    })();
    return loadPromise;
  }

  const ready = () => !!(sets.words && sets.lines);

  async function clipBuffer(setName, key) {
    const id = setName + "/" + key;
    if (cache.has(id)) return cache.get(id);
    const set = sets[setName];
    if (!set) throw new Error("narration not loaded");
    const entry = set.meta.index[key];
    if (!entry) return null;
    const slice = set.bytes.subarray(entry.o, entry.o + entry.n);
    const buf = await audioContext().decodeAudioData(slice.slice().buffer);
    cache.set(id, buf);
    return buf;
  }

  function schedule(buffer, at, gain = 1) {
    const c = audioContext();
    const src = c.createBufferSource();
    src.buffer = buffer;
    const g = c.createGain();
    g.gain.value = gain;
    src.connect(g).connect(c.destination);
    src.start(at);
    playing.push(src);
    return at + buffer.duration;
  }

  function clearTimers() {
    timers.forEach((t) => clearTimeout(t));
    timers = [];
  }

  function stop() {
    playing.forEach((s) => { try { s.stop(); } catch (e) { /* already ended */ } });
    playing = [];
    clearTimers();
  }

  function wordsOf(line) {
    return norm(line).split(" ").filter(Boolean);
  }

  // Gaps: the device page's measured rules. A word followed by punctuation in the
  // source line gets the larger gap; the last word of a line gets the line gap.
  function gapAfter(line, index, count) {
    if (index === count - 1) return GAP_LINE;
    const tail = line.slice(line.toLowerCase().indexOf(wordsOf(line)[index]) + wordsOf(line)[index].length);
    if (/[,;:]/.test(tail.slice(0, 2))) return GAP_PUNCT;
    if (/[.!?…]/.test(tail.slice(0, 2))) return GAP_LINE * 0.6;
    return GAP_WORD;
  }

  function planLine(line, mode) {
    if (!ready()) return { source: "none", words: [], missing: [] };
    const key = norm(line);
    if (mode !== "words" && sets.lines && sets.lines.meta.index[key]) {
      return { source: "line", key };
    }
    const words = wordsOf(line).filter((w) => sets.words.meta.index[w]);
    return { source: "words", words, missing: wordsOf(line).filter((w) => !sets.words.meta.index[w]) };
  }

  // Plays a list of lines back to back. Returns when playback ends (or is stopped).
  async function play(lines, opts = {}) {
    if (!ready()) await load();          // calling play() before load() just waits
    stop();
    const mode = opts.mode || "hybrid";
    const c = audioContext();
    const report = { lines: [], seconds: 0, lineClips: 0, wordLines: 0, missing: [] };
    let cursor = c.currentTime + 0.08;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const plan = planLine(line, mode);
      const started = cursor;
      if (plan.source === "line") {
        const buf = await clipBuffer("lines", plan.key);
        cursor = schedule(buf, cursor);
        report.lineClips++;
      } else {
        let widx = 0;
        for (const w of plan.words) {
          const buf = await clipBuffer("words", w);
          if (buf) {
            cursor = schedule(buf, cursor);
            cursor += gapAfter(line, widx, plan.words.length);
          }
          widx++;
        }
        report.wordLines++;
        (plan.missing || []).forEach((m) => report.missing.push(m));
      }
      report.lines.push({ line, source: plan.source, start: started, end: cursor });
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

  async function playWord(word) {
    if (!ready()) await load();
    const buf = await clipBuffer("words", norm(word));
    if (!buf) return null;
    stop();
    const c = audioContext();
    schedule(buf, c.currentTime + 0.02);
    return { seconds: buf.duration, bytes_mb: { mp3_24k: buf.duration * 3000 / 1048576 } };
  }

  async function playLine(text, opts = {}) {
    return play([text], opts);
  }

  window.Narration = {
    load, ready, stop, play, playWord, playLine,
    hasLine: (text) => ready() && !!sets.lines.meta.index[norm(text)],
    words: () => (ready() ? Object.keys(sets.words.meta.index) : []),
    lines: () => (ready() ? Object.keys(sets.lines.meta.index) : []),
    meta: () => (ready() ? { words: sets.words.meta, lines: sets.lines.meta } : null),
    gaps: { word: GAP_WORD, punct: GAP_PUNCT, line: GAP_LINE },
  };
})();
