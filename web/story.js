// The Story Studio widget: the chips, the click, the progressive reveal.
//
// It owns no model code — generation lives in generate_story.js, which exposes
// `generateStory({world, mode, place, theme, seed})` and `window.StoryEngine`. The studio
// calls exactly the function a page script or the console would call, so the two can
// never drift apart.
(() => {
  "use strict";

  const root = document.getElementById("story-studio");
  if (!root) return;

  const $id = (id) => document.getElementById(id);
  const worldSel = $id("studio-world"), modeSel = $id("studio-mode"),
        placeSel = $id("studio-place"), themeSel = $id("studio-theme");
  const btn = $id("studio-generate");
  const out = $id("studio-output"), stats = $id("studio-stats");
  const E = window.StoryEngine;

  if (!E || !E.generateStory) {
    out.textContent = "Run `python training/export_story.py` to generate web/story_data.js, " +
                      "then reload.";
    return;
  }
  const S = { worlds: E.worlds, themes: E.themes };

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

  let busy = false;
  btn.addEventListener("click", async () => {
    if (busy) return;
    busy = true;
    btn.disabled = true;
    btn.textContent = "Thinking…";
    out.textContent = "";
    stats.textContent = "";

    await new Promise((r) => setTimeout(r, 30)); // let the button paint

    try {
      const w = currentWorld();
      const story = await E.generateStory({
        world: w.id,
        mode: modeSel.querySelector(".is-on").textContent,
        place: placeSel.querySelector(".is-on").textContent,
        theme: themeSel.querySelector(".is-on").textContent,
      });

      const h = document.createElement("div");
      h.className = "studio-byline";
      h.textContent = `${w.name} · ${story.mode} in the ${story.place} · a story about ${story.theme}`;
      out.append(h);

      // Progressive per-line reveal.
      let li = 0;
      const paint = () => {
        if (li < story.lines.length) {
          const p = document.createElement("div");
          p.className = "studio-line is-new";
          p.textContent = story.lines[li++];
          out.append(p);
          setTimeout(paint, 260);
        } else {
          stats.textContent =
            `${story.tokens} tokens · generated in ${(story.ms / 1000).toFixed(1)}s ` +
            `(~${Math.round(story.tokens / (story.ms / 1000))} tok/s) by the real fp16 checkpoint ` +
            `(${E.config.vocab} ids, ${E.config.layers} layers), in your browser.`;
          busy = false;
          btn.disabled = false;
          btn.textContent = "Tell another story";
        }
      };
      paint();
    } catch (e) {
      out.textContent = "could not generate: " + e.message;
      busy = false;
      btn.disabled = false;
      btn.textContent = "Tell a story";
    }
  });

  chipGroup(worldSel, S.worlds.map((w) => w.name), true);
  chipGroup(modeSel, E.modes, false, "bedtime");
  chipGroup(themeSel, S.themes, false, "help");
  refreshPlaces();
  syncWalkthroughLinks();
})();
