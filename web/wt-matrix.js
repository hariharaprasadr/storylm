// Matrix views for the walkthrough. Every cell is a real number taken from
// the live trace (web/wt-trace.js): 2D matrices with column/row windows,
// draggable 3D stacks of matrix planes for the layer/head axes, and a
// vocabulary grid for the output logits. Any matrix can be popped out into
// a scroll-through explorer. No dependencies, file:// safe.
(() => {
  "use strict";

  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  // Colour encodes the very number printed in the cell (sign + magnitude).
  function tint(v, scale) {
    const t = Math.min(1, Math.abs(v) / (scale || 1));
    const alpha = v === 0 ? 0 : 0.06 + t * 0.5;
    return v >= 0 ? `rgba(184, 80, 31, ${alpha.toFixed(3)})` : `rgba(49, 89, 143, ${alpha.toFixed(3)})`;
  }

  const cellText = (v, digits) => (v === null || v === undefined ? "·"
    : Math.abs(v) < Math.pow(10, -digits) / 2 ? (0).toFixed(digits) : v.toFixed(digits));

  // values: array of rows of numbers; null renders a masked cell.
  function numberMatrix({ rows, rowLabels, colLabels, digits = 3, scale = null, tintCells = true }) {
    const s = scale ?? Math.max(...rows.flat().map((v) => (v === null ? 0 : Math.abs(v))), 1e-9);
    const t = el("table", "wt-mx");
    if (colLabels && colLabels.length) {
      const tr = el("tr");
      tr.append(el("th", "wt-mx-corner", ""));
      colLabels.forEach((c) => tr.append(el("th", "", c)));
      t.append(tr);
    }
    rows.forEach((row, r) => {
      const tr = el("tr");
      tr.append(el("td", "wt-mx-rowlab", rowLabels ? rowLabels[r] : String(r)));
      row.forEach((v) => {
        const td = el("td", "wt-mx-cell", cellText(v, digits));
        if (v !== null && v !== undefined && tintCells) td.style.background = tint(v, s);
        tr.append(td);
      });
      t.append(tr);
    });
    t.title = `values ${cellText(-s, digits)} … ${cellText(s, digits)} · colour = the same number`;
    return t;
  }

  // ---------- pop-out explorer: scroll through the whole matrix ----------
  let modal = null;

  function closeExplorer() {
    if (!modal) return;
    modal.remove();
    modal = null;
    document.removeEventListener("keydown", onExplorerKey, true);
  }

  function onExplorerKey(e) {
    if (!modal) return;
    const step = 4;
    const pan = modal._pan;
    if (e.key === "Escape") { closeExplorer(); e.preventDefault(); return; }
    if (e.key === "ArrowDown") { pan(step, 0); e.preventDefault(); }
    else if (e.key === "ArrowUp") { pan(-step, 0); e.preventDefault(); }
    else if (e.key === "ArrowRight") { pan(0, step); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { pan(0, -step); e.preventDefault(); }
  }

  function openExplorer({ nRows, nCols, getValue, rowLabelAt, colLabelAt, title, digits = 2 }) {
    closeExplorer();
    const overlay = el("div", "wt-mx-modal");
    overlay.dataset.role = "matrix-explorer";
    const dialog = el("div", "wt-mx-dialog");
    const head = el("div", "wt-mx-modal-head");
    const range = el("div", "wt-mx-range");
    const closeBtn = el("button", "wt-mx-btn", "close ✕");
    closeBtn.type = "button";
    closeBtn.addEventListener("click", closeExplorer);
    head.append(el("div", "wt-mx-modal-title", title), range, closeBtn);
    const body = el("div", "wt-mx-modal-body");
    dialog.append(head, body, el("div", "wt-mx-modal-hint",
      "scroll to move through rows · shift+scroll (or drag) for columns · arrow keys work too · esc closes"));
    overlay.append(dialog);
    document.body.append(overlay);
    modal = overlay;

    // One pass over the tensor so the colours mean the same thing in every window.
    let scale = 0;
    for (let r = 0; r < nRows; r++) {
      for (let c = 0; c < nCols; c++) {
        const v = getValue(r, c);
        const a = v === null ? 0 : Math.abs(v);
        if (a > scale) scale = a;
      }
    }

    const rect = dialog.getBoundingClientRect();
    const cols = Math.max(4, Math.min(nCols, Math.floor((rect.width - 130) / 42)));
    const rows = Math.max(4, Math.min(nRows, Math.floor((rect.height - 140) / 19)));
    let r0 = 0, c0 = 0;

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const paint = () => {
      const values = Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => getValue(r0 + r, c0 + c)));
      body.replaceChildren(numberMatrix({
        rows: values,
        rowLabels: Array.from({ length: rows }, (_, r) => rowLabelAt(r0 + r)),
        colLabels: Array.from({ length: cols }, (_, c) => colLabelAt(c0 + c)),
        digits, scale,
      }));
      range.textContent = `rows ${r0 + 1}–${r0 + rows} of ${nRows} · cols ${c0 + 1}–${c0 + cols} of ${nCols}`;
    };
    const pan = (dr, dc) => {
      r0 = clamp(r0 + dr, 0, Math.max(0, nRows - rows));
      c0 = clamp(c0 + dc, 0, Math.max(0, nCols - cols));
      paint();
    };
    overlay._pan = pan;

    overlay.addEventListener("wheel", (e) => {
      e.preventDefault();
      const step = e.deltaMode === 1 ? 3 : Math.max(1, Math.round(Math.abs(e.deltaY) / 60) || 1);
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) pan(0, Math.sign(e.deltaX || e.deltaY) * step * 3);
      else pan(Math.sign(e.deltaY) * step * 3, 0);
    }, { passive: false });

    // drag inside the window pans the matrix too (touch friendly)
    let drag = null;
    body.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, r0, c0 }; body.setPointerCapture?.(e.pointerId); });
    body.addEventListener("pointermove", (e) => {
      if (!drag) return;
      pan(Math.round((drag.y - e.clientY) / 19), Math.round((drag.x - e.clientX) / 42));
      drag = { ...drag, r0, c0 };
    });
    body.addEventListener("pointerup", () => { drag = null; });
    overlay.addEventListener("click", (e) => { if (e.target === overlay) closeExplorer(); });
    document.addEventListener("keydown", onExplorerKey, true);

    paint();
  }

  // ---------- windowed matrix (click it to explore the whole thing) ----------
  function matrixWithPager({
    nRows, nCols, getValue,
    rowLabelAt = (r) => String(r), colLabelAt = (c) => "c" + String(c),
    rowWindow = null, colWindow = 16, digits = 3, fixedScale = null, title = "matrix",
  }) {
    const host = el("div", "wt-mx-win");
    const bar = el("div", "wt-mx-controls");
    const body = el("div", "wt-mx-body");
    const range = el("span", "wt-mx-range");
    const rowsShown = Math.min(rowWindow ?? nRows, nRows);
    const colsShown = Math.min(colWindow, nCols);
    let r0 = 0, c0 = 0;

    const btn = (label, cls) => {
      const b = el("button", "wt-mx-btn " + cls, label);
      b.type = "button";
      return b;
    };
    const rowPrev = btn("↑ rows", "wt-mx-rowbtn");
    const rowNext = btn("↓ rows", "wt-mx-rowbtn");
    const colPrev = btn("← cols", "wt-mx-colbtn");
    const colNext = btn("cols →", "wt-mx-colbtn");

    const paint = () => {
      const values = Array.from({ length: rowsShown }, (_, r) =>
        Array.from({ length: colsShown }, (_, c) => getValue(r0 + r, c0 + c)));
      body.replaceChildren(numberMatrix({
        rows: values,
        rowLabels: Array.from({ length: rowsShown }, (_, r) => rowLabelAt(r0 + r)),
        colLabels: Array.from({ length: colsShown }, (_, c) => colLabelAt(c0 + c)),
        digits,
        scale: fixedScale,
      }));
      range.textContent = `rows ${r0 + 1}–${r0 + rowsShown} of ${nRows} · cols ${c0 + 1}–${c0 + colsShown} of ${nCols}`;
      rowPrev.disabled = r0 === 0;
      rowNext.disabled = r0 + rowsShown >= nRows;
      colPrev.disabled = c0 === 0;
      colNext.disabled = c0 + colsShown >= nCols;
    };
    rowPrev.addEventListener("click", (e) => { e.stopPropagation(); r0 = Math.max(0, r0 - rowsShown); paint(); });
    rowNext.addEventListener("click", (e) => { e.stopPropagation(); r0 = Math.min(Math.max(0, nRows - rowsShown), r0 + rowsShown); paint(); });
    colPrev.addEventListener("click", (e) => { e.stopPropagation(); c0 = Math.max(0, c0 - colsShown); paint(); });
    colNext.addEventListener("click", (e) => { e.stopPropagation(); c0 = Math.min(Math.max(0, nCols - colsShown), c0 + colsShown); paint(); });

    // The matrix block itself is clickable: it pops out and can be scrolled through.
    const openIt = () => openExplorer({ nRows, nCols, getValue, rowLabelAt, colLabelAt, title, digits });
    body.classList.add("is-clickable");
    body.title = `${title} — click to open the full matrix and scroll through it`;
    body.addEventListener("click", openIt);
    const popBtn = btn("⤢ explore", "wt-mx-explore");
    popBtn.addEventListener("click", (e) => { e.stopPropagation(); openIt(); });

    bar.append(el("span", "wt-mx-axis", "3D slice:"), rowPrev, rowNext, colPrev, colNext, popBtn, range);
    host.append(bar, body);
    paint();
    return host;
  }

  // ---------- 3D stack ----------
  // planes[k] is a 2D matrix (numbers or null). Drag to rotate; every plane
  // can be brought to the front, so nothing is trapped at the back.
  function matrixStack({ planes, labels, rowLabels, colLabels, digits = 3, tintCells = true, title = "tensor" }) {
    const host = el("div", "wt-stack");
    const picker = el("div", "wt-stack-picker");
    const deck = el("div", "wt-stack-deck");
    const cards = [];
    const scale = Math.max(...planes.flat(2).map((v) => (v === null ? 0 : Math.abs(v))), 1e-9);

    planes.forEach((plane, i) => {
      const card = el("div", "wt-stack-card");
      card.style.setProperty("--i", i);
      card.append(el("div", "wt-stack-label", labels[i]));
      card.append(numberMatrix({ rows: plane, rowLabels, colLabels, digits, scale, tintCells }));
      card.addEventListener("click", () => setFocus(i));
      deck.append(card);
      cards.push(card);
    });

    function setFocus(i) {
      const on = !cards[i].classList.contains("is-front");
      cards.forEach((c, j) => {
        c.classList.toggle("is-front", on && j === i);
        c.classList.toggle("is-dim", on && j !== i);
      });
      [...picker.children].forEach((b, j) => b.classList.toggle("is-on", on && j === i));
    }

    labels.forEach((lab, i) => {
      const b = el("button", "wt-pill", lab);
      b.type = "button";
      b.addEventListener("click", () => setFocus(i));
      picker.append(b);
    });

    let rx = -16, ry = 26, drag = null;
    const apply = () => {
      deck.style.setProperty("--rx", rx.toFixed(1) + "deg");
      deck.style.setProperty("--ry", ry.toFixed(1) + "deg");
    };
    host.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".wt-stack-picker")) return;
      drag = { x: e.clientX, y: e.clientY, rx, ry, moved: false };
      host.dataset.dragged = "0";
      host.setPointerCapture?.(e.pointerId);
    });
    host.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; host.dataset.dragged = "1"; }
      ry = drag.ry + dx * 0.35;
      rx = Math.max(-75, Math.min(20, drag.rx - dy * 0.25));
      apply();
    });
    const endDrag = () => {
      if (drag && drag.moved) setTimeout(() => { host.dataset.dragged = "0"; }, 0);
      drag = null;
    };
    host.addEventListener("pointerup", endDrag);
    host.addEventListener("pointercancel", endDrag);
    apply();

    host.append(picker, deck, el("div", "wt-stack-hint",
      "drag to rotate · pick a plane below (or click it) to bring it to the front"));
    return host;
  }

  // ---------- vocabulary map for the output logits ----------
  function vocabGrid({ scores, vocab, topK = 15, digits = 2 }) {
    const order = Array.from({ length: vocab.length }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
    const rank = new Map(order.slice(0, topK).map((i, r) => [i, r + 1]));
    const top = Math.max(...scores, 1e-9);
    const wrap = el("div", "wt-vocab-wrap");
    const grid = el("div", "wt-vocab");
    vocab.forEach((word, i) => {
      const isTop = rank.has(i);
      const cell = el("span", "wt-vocab-cell" + (isTop ? " is-top" : ""),
        isTop ? `${word} ${scores[i].toFixed(digits)}` : word);
      cell.title = `id ${i} · ${word} · score ${scores[i].toFixed(3)}`;
      const s = Math.max(0, scores[i]) / top;
      if (s > 0) cell.style.background = tint(scores[i], top);
      grid.append(cell);
    });
    wrap.append(grid, el("div", "wt-vocab-note",
      `all ${vocab.length} vocabulary entries in id order (control tokens first, then words) · the ${topK} highest-scoring entries are labelled · hover any cell for its exact score`));
    return wrap;
  }

  window.WTMatrix = { numberMatrix, matrixWithPager, matrixStack, vocabGrid, openExplorer, closeExplorer, tint, cellText };
})();