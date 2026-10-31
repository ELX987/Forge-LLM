// Dependency-free canvas charting: XYChart (multi-series line / scatter, log axes, CI bands, Pareto frontier, box-zoom,
// pan, hover tooltips, decimation for 100k+ points), BarChart (grouped, error bars), Heatmap (autotune surfaces),
// rooflineChart, sparkline. Exports: PNG (canvas) and CSV (data). Theme aware (reads CSS variables on each draw).

import { h, clear, fmtNum, isNum, download, toCSV, uid } from './util.js';

// ------------------------------------------------------------------ theme colours
let themeCache = null;
const live = new Set();
function invalidateTheme() { themeCache = null; for (const c of live) c.invalidate(); }
try {
  new MutationObserver(invalidateTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', invalidateTheme);
} catch (_) { /* non-browser */ }

export function themeColors() {
  if (themeCache) return themeCache;
  const cs = getComputedStyle(document.documentElement);
  const g = (n, d) => (cs.getPropertyValue(n) || d).trim() || d;
  themeCache = {
    fg: g('--fg', '#222'), muted: g('--muted', '#666'), faint: g('--faint', '#999'), grid: g('--grid', '#eee'), border: g('--border', '#ccc'),
    surface: g('--surface', '#fff'), surface2: g('--surface2', '#f4f4f4'), accent: g('--accent', '#36d'), ok: g('--ok', '#2a2'), bad: g('--bad', '#c33'), warn: g('--warn', '#a60'),
    mono: g('--mono', 'monospace'), sans: g('--sans', 'sans-serif'),
    palette: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => g('--c' + i, '#36d')),
  };
  return themeCache;
}
const withAlpha = (hex, a) => {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  let s = m[1]; if (s.length === 3) s = s.split('').map((c) => c + c).join('');
  const n = parseInt(s, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

// ------------------------------------------------------------------ ticks
function niceNum(x, round) {
  const exp = Math.floor(Math.log10(x)); const f = x / 10 ** exp;
  const nf = round ? (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) : (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10);
  return nf * 10 ** exp;
}
export function linTicks(min, max, count = 6) {
  if (!(max > min)) return { ticks: [min], step: 1 };
  const range = niceNum(max - min, false);
  const step = niceNum(range / (count - 1), true);
  const t0 = Math.ceil(min / step - 1e-9) * step;
  const ticks = [];
  for (let v = t0; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
  return { ticks, step };
}
export function logTicks(minV, maxV) {
  const e0 = Math.floor(Math.log10(minV)); const e1 = Math.ceil(Math.log10(maxV));
  const mults = e1 - e0 <= 3 ? [1, 2, 5] : [1];
  const ticks = [];
  for (let e = e0; e <= e1; e++) for (const m of mults) { const v = m * 10 ** e; if (v >= minV * (1 - 1e-9) && v <= maxV * (1 + 1e-9)) ticks.push(v); }
  return ticks.length >= 2 ? ticks : [minV, maxV];
}
export function fmtTick(v, step) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e9) return trim((v / 1e9).toFixed(2)) + 'B';
  if (a >= 1e6) return trim((v / 1e6).toFixed(2)) + 'M';
  if (a >= 1e4) return trim((v / 1e3).toFixed(2)) + 'k';
  if (a < 1e-3 || a >= 1e5) return v.toExponential(0).replace('+', '');
  const dec = step ? Math.min(6, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9))) : 3;
  return trim(v.toFixed(dec));
}
function trim(s) { return s.includes('.') ? s.replace(/0+$/, '').replace(/\.$/, '') : s; }

// ------------------------------------------------------------------ base class
class BaseChart {
  constructor(container, opts = {}) {
    this.opts = { height: 260, title: '', legend: true, tools: true, empty: 'No data yet', ...opts };
    this.id = uid('chart');
    this.canvas = h('canvas', { role: 'img', 'aria-label': this.opts.title || 'chart', tabindex: '0' });
    this.tip = h('div', { class: 'chart-tip hidden' });
    this.emptyEl = h('div', { class: 'chart-empty hidden' }, this.opts.empty);
    this.body = h('div', { class: 'chart', style: { height: this.opts.height + 'px' } }, this.canvas, this.tip, this.emptyEl);
    this.legendEl = h('div', { class: 'chart-legend' });
    this.toolsEl = h('div', { class: 'chart-tools' });
    this.titleEl = h('div', { class: 'chart-title' }, h('h3', this.opts.title), this.toolsEl);
    this.root = h('div', { class: 'chart-wrap' }, (this.opts.title || this.opts.tools) ? this.titleEl : null, this.body, this.opts.legend ? this.legendEl : null);
    if (this.opts.tools) this.buildTools();
    container.appendChild(this.root);
    this.ctx = this.canvas.getContext('2d');
    this.w = 0; this.h = 0; this.dpr = 1;
    this.pending = false;
    this.destroyed = false;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.body);
    live.add(this);
    this.resize();
  }
  buildTools() {
    const btn = (label, title, fn) => h('button', { type: 'button', title, 'aria-label': title, onclick: fn }, label);
    this.extraTools();
    this.toolsEl.append(btn('PNG', 'Export as PNG', () => this.exportPNG()), btn('CSV', 'Export data as CSV', () => this.exportCSV()));
  }
  extraTools() { /* subclass hook */ }
  resize() {
    if (this.destroyed) return;
    const r = this.body.getBoundingClientRect();
    const w = Math.max(60, Math.floor(r.width)); const hh = Math.max(60, Math.floor(r.height));
    const dpr = window.devicePixelRatio || 1;
    if (w === this.w && hh === this.h && dpr === this.dpr) return;
    this.w = w; this.h = hh; this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(hh * dpr);
    this.canvas.style.height = hh + 'px';
    this.invalidate();
  }
  invalidate() {
    if (this.pending || this.destroyed) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; if (!this.destroyed) { try { this.render(); } catch (e) { console.error('chart render', e); } } });
  }
  destroy() { this.destroyed = true; live.delete(this); this.ro.disconnect(); this.root.remove(); }
  showEmpty(show) { this.emptyEl.classList.toggle('hidden', !show); }
  prep() {
    const { ctx, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    this.T = themeColors();
    ctx.font = `11px ${this.T.sans}`;
    return this.T;
  }
  exportPNG() {
    this.render();
    this.canvas.toBlob((b) => { if (b) download(`${(this.opts.title || 'chart').replace(/\W+/g, '_')}.png`, b, 'image/png'); }, 'image/png');
  }
  exportCSV() { download(`${(this.opts.title || 'chart').replace(/\W+/g, '_')}.csv`, toCSV(this.csvRows()), 'text/csv'); }
  csvRows() { return []; }
  showTip(html, x, y) {
    clear(this.tip); this.tip.append(...(Array.isArray(html) ? html : [html]));
    this.tip.classList.remove('hidden');
    const tw = this.tip.offsetWidth;
    const left = x + 12 + tw > this.w ? x - 12 - tw : x + 12;
    this.tip.style.transform = 'none';
    this.tip.style.left = Math.max(2, left) + 'px';
    this.tip.style.top = Math.max(2, Math.min(this.h - this.tip.offsetHeight - 2, y - this.tip.offsetHeight / 2)) + 'px';
  }
  hideTip() { this.tip.classList.add('hidden'); }
  localXY(e) { const r = this.canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
}

// ------------------------------------------------------------------ XY chart
function lowerBound(arr, x) { let lo = 0; let hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; } return lo; }

function normSeries(s, i) {
  const xs = []; const ys = []; const lo = []; const hi = [];
  const pts = s.points || [];
  let hasBand = false;
  for (const p of pts) {
    if (Array.isArray(p)) { xs.push(p[0]); ys.push(p[1]); if (p.length >= 4 && isNum(p[2]) && isNum(p[3])) { lo.push(p[2]); hi.push(p[3]); hasBand = true; } else { lo.push(NaN); hi.push(NaN); } }
    else if (p && typeof p === 'object') { xs.push(p.x); ys.push(p.y); const l = p.lo ?? p.ci_low; const u = p.hi ?? p.ci_high; if (isNum(l) && isNum(u)) { lo.push(l); hi.push(u); hasBand = true; } else { lo.push(NaN); hi.push(NaN); } }
  }
  const n = {
    name: s.name || `series ${i + 1}`, type: s.type || 'line', color: s.color || null, width: s.width || 1.6, dashed: !!s.dashed, step: !!s.step,
    radius: s.radius || 3.5, xs, ys, lo, hi, hasBand, hidden: !!s.hidden, labels: s.labels || null, meta: s.meta || null,
    alpha: s.alpha == null ? 1 : s.alpha, highlight: s.highlight ? new Set(s.highlight) : null, frontier: !!s.frontier, noLegend: !!s.noLegend, fillBelow: !!s.fillBelow,
  };
  if (n.type === 'line') {
    let sorted = true;
    for (let k = 1; k < xs.length; k++) if (xs[k] < xs[k - 1]) { sorted = false; break; }
    if (!sorted) {
      const idx = xs.map((_, k) => k).sort((a, b) => xs[a] - xs[b]);
      const re = (arr) => idx.map((k) => arr[k]);
      n.xs = re(xs); n.ys = re(ys); n.lo = re(lo); n.hi = re(hi);
    }
  }
  return n;
}

export class XYChart extends BaseChart {
  constructor(container, opts = {}) {
    super(container, { xLabel: '', yLabel: '', logX: false, logY: false, xFormat: null, yFormat: null, ...opts });
    this.series = [];
    this.view = null; // {x0,x1,y0,y1} in transformed space when zoomed
    this.hover = null;
    this.bindEvents();
    if (opts.series) this.setSeries(opts.series);
  }
  extraTools() {
    const btn = (label, title, fn) => h('button', { type: 'button', title, 'aria-label': title, onclick: fn }, label);
    this.logXBtn = btn('log x', 'Toggle logarithmic x axis', () => { this.opts.logX = !this.opts.logX; this.view = null; this.invalidate(); });
    this.logYBtn = btn('log y', 'Toggle logarithmic y axis', () => { this.opts.logY = !this.opts.logY; this.view = null; this.invalidate(); });
    this.toolsEl.append(btn('−', 'Zoom out', () => this.zoomBy(1.5)), btn('+', 'Zoom in', () => this.zoomBy(1 / 1.5)), btn('reset', 'Reset zoom (or double-click)', () => this.resetZoom()));
    if (!this.opts.noLogToggle) this.toolsEl.append(this.logXBtn, this.logYBtn);
  }
  setSeries(series) {
    const prevHidden = new Map(this.series.map((s) => [s.name, s.hidden]));
    this.series = series.map((s, i) => { const n = normSeries(s, i); if (prevHidden.has(n.name) && s.hidden === undefined) n.hidden = prevHidden.get(n.name); return n; });
    this.buildLegend();
    this.invalidate();
  }
  /** Append points to a named series (live streaming). pts: [[x,y], ...] */
  append(name, pts) {
    let s = this.series.find((q) => q.name === name);
    if (!s) { s = normSeries({ name, points: [] }, this.series.length); this.series.push(s); this.buildLegend(); }
    for (const p of pts) { s.xs.push(p[0]); s.ys.push(p[1]); s.lo.push(NaN); s.hi.push(NaN); }
    this.invalidate();
  }
  buildLegend() {
    clear(this.legendEl);
    if (!this.opts.legend) return;
    this.series.forEach((s, i) => {
      if (s.noLegend) return;
      const color = s.color || themeColors().palette[i % 8];
      const b = h('button', { type: 'button', class: s.hidden ? 'off' : '', 'aria-pressed': String(!s.hidden), title: 'Click to show/hide', onclick: () => { s.hidden = !s.hidden; b.classList.toggle('off', s.hidden); b.setAttribute('aria-pressed', String(!s.hidden)); this.invalidate(); } },
        h('i', { style: { background: color } }), s.name);
      this.legendEl.appendChild(b);
    });
  }
  resetZoom() { this.view = null; this.invalidate(); }
  zoomBy(f) {
    const d = this.domain(); if (!d) return;
    const cx = (d.x0 + d.x1) / 2; const cy = (d.y0 + d.y1) / 2;
    this.view = { x0: cx - (cx - d.x0) * f, x1: cx + (d.x1 - cx) * f, y0: cy - (cy - d.y0) * f, y1: cy + (d.y1 - cy) * f };
    this.invalidate();
  }
  tx(v) { return this.opts.logX ? Math.log10(v) : v; }
  ty(v) { return this.opts.logY ? Math.log10(v) : v; }
  validX(v) { return isNum(v) && (!this.opts.logX || v > 0); }
  validY(v) { return isNum(v) && (!this.opts.logY || v > 0); }

  autoDomain() {
    let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
    const o = this.opts;
    for (const s of this.series) {
      if (s.hidden) continue;
      for (let i = 0; i < s.xs.length; i++) {
        const x = s.xs[i]; const y = s.ys[i];
        if (!this.validX(x) || !this.validY(y)) continue;
        const X = this.tx(x); const Y = this.ty(y);
        if (X < x0) x0 = X; if (X > x1) x1 = X; if (Y < y0) y0 = Y; if (Y > y1) y1 = Y;
        if (s.hasBand) {
          const l = s.lo[i]; const u = s.hi[i];
          if (this.validY(l)) y0 = Math.min(y0, this.ty(l));
          if (this.validY(u)) y1 = Math.max(y1, this.ty(u));
        }
      }
    }
    if (!isFinite(x0)) return null;
    if (x1 === x0) { x0 -= 0.5; x1 += 0.5; }
    if (y1 === y0) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
    const py = (y1 - y0) * 0.06;
    y0 -= py; y1 += py;
    if (!o.logY && isNum(o.yMin)) y0 = o.yMin; else if (!o.logY && y0 < 0 && y0 + py >= 0 && this.allNonNegative) y0 = 0;
    if (!o.logY && isNum(o.yMax)) y1 = o.yMax;
    if (isNum(o.xMin)) x0 = this.tx(o.xMin);
    if (isNum(o.xMax)) x1 = this.tx(o.xMax);
    return { x0, x1, y0, y1 };
  }
  domain() { return this.view || this.autoDomain(); }

  layout(d) {
    const { ctx } = this;
    const o = this.opts;
    const yt = this.ticksFor('y', d); const xt = this.ticksFor('x', d);
    let maxw = 0;
    for (const t of yt.ticks) maxw = Math.max(maxw, ctx.measureText(this.fmtY(t, yt.step)).width);
    const L = Math.ceil(maxw) + (o.yLabel ? 30 : 16);
    const B = 22 + (o.xLabel ? 16 : 0);
    return { L, R: 14, T: 10, B, W: this.w - L - 14, H: this.h - 10 - B, xt, yt };
  }
  ticksFor(axis, d) {
    const log = axis === 'x' ? this.opts.logX : this.opts.logY;
    const a = axis === 'x' ? d.x0 : d.y0; const b = axis === 'x' ? d.x1 : d.y1;
    if (log) return { ticks: logTicks(10 ** a, 10 ** b), step: null, log: true };
    const cnt = axis === 'x' ? Math.max(3, Math.min(9, Math.floor(this.w / 90))) : Math.max(3, Math.min(8, Math.floor(this.h / 48)));
    return { ...linTicks(a, b, cnt), log: false };
  }
  fmtX(v, step) { return this.opts.xFormat ? this.opts.xFormat(v) : (this.opts.logX ? fmtLogTick(v) : fmtTick(v, step)); }
  fmtY(v, step) { return this.opts.yFormat ? this.opts.yFormat(v) : (this.opts.logY ? fmtLogTick(v) : fmtTick(v, step)); }
  color(s, i) { return s.color || this.T.palette[i % 8]; }

  render() {
    const T = this.prep(); const { ctx } = this;
    this.allNonNegative = this.series.every((s) => s.ys.every((y) => !isNum(y) || y >= 0));
    const d = this.domain();
    this.showEmpty(!d);
    if (!d) { this.geom = null; return; }
    const g = this.layout(d); this.geom = { ...g, d };
    const px = (X) => g.L + (X - d.x0) / (d.x1 - d.x0) * g.W;
    const py = (Y) => g.T + g.H - (Y - d.y0) / (d.y1 - d.y0) * g.H;
    this.px = px; this.py = py;

    // grid + axes
    ctx.lineWidth = 1; ctx.strokeStyle = T.grid; ctx.fillStyle = T.muted; ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const t of g.yt.ticks) {
      const Y = this.opts.logY ? Math.log10(t) : t; const y = Math.round(py(Y)) + 0.5;
      if (y < g.T - 1 || y > g.T + g.H + 1) continue;
      ctx.beginPath(); ctx.moveTo(g.L, y); ctx.lineTo(g.L + g.W, y); ctx.stroke();
      ctx.fillText(this.fmtY(t, g.yt.step), g.L - 6, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of g.xt.ticks) {
      const X = this.opts.logX ? Math.log10(t) : t; const x = Math.round(px(X)) + 0.5;
      if (x < g.L - 1 || x > g.L + g.W + 1) continue;
      ctx.beginPath(); ctx.moveTo(x, g.T); ctx.lineTo(x, g.T + g.H); ctx.stroke();
      ctx.fillText(this.fmtX(t, g.xt.step), x, g.T + g.H + 5);
    }
    ctx.strokeStyle = T.border;
    ctx.strokeRect(g.L + 0.5, g.T + 0.5, g.W, g.H);
    ctx.fillStyle = T.muted;
    if (this.opts.xLabel) { ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(this.opts.xLabel, g.L + g.W / 2, this.h - 2); }
    if (this.opts.yLabel) { ctx.save(); ctx.translate(11, g.T + g.H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(this.opts.yLabel, 0, 0); ctx.restore(); }

    // series
    ctx.save();
    ctx.beginPath(); ctx.rect(g.L, g.T, g.W, g.H); ctx.clip();
    this.series.forEach((s, si) => {
      if (s.hidden) return;
      const col = this.color(s, si);
      if (s.type === 'scatter') this.drawScatter(s, col, px, py, d, g);
      else this.drawLine(s, col, px, py, d, g);
    });
    ctx.restore();

    // hover crosshair / markers
    if (this.hover) this.drawHover(g, px, py);
    // selection box
    if (this.drag && this.drag.mode === 'box') {
      const { x0, y0, x1, y1 } = this.drag;
      ctx.fillStyle = withAlpha(T.accent, 0.15); ctx.strokeStyle = T.accent;
      ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      ctx.strokeRect(Math.min(x0, x1) + 0.5, Math.min(y0, y1) + 0.5, Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    if (this.view) { ctx.fillStyle = T.muted; ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.fillText('zoomed - double-click to reset', g.L + g.W - 4, g.T + 4); }
  }

  visibleRange(s, d) {
    const lo = this.opts.logX ? 10 ** d.x0 : d.x0; const hi = this.opts.logX ? 10 ** d.x1 : d.x1;
    let i0 = lowerBound(s.xs, lo); let i1 = lowerBound(s.xs, hi);
    i0 = Math.max(0, i0 - 1); i1 = Math.min(s.xs.length - 1, i1);
    return [i0, i1];
  }
  decimate(s, i0, i1, px, g) {
    const n = i1 - i0 + 1;
    const maxPts = Math.max(50, Math.floor(g.W) * 2);
    if (n <= maxPts) { const out = []; for (let i = i0; i <= i1; i++) out.push(i); return out; }
    const out = [];
    let curCol = null; let first = i0; let last = i0; let mn = -1; let mx = -1;
    const flush = () => {
      if (curCol == null) return;
      const set = [first, mn, mx, last].filter((v) => v >= 0).sort((a, b) => a - b);
      let prev = -1; for (const v of set) if (v !== prev) { out.push(v); prev = v; }
    };
    for (let i = i0; i <= i1; i++) {
      const x = s.xs[i]; const y = s.ys[i];
      if (!this.validX(x) || !this.validY(y)) continue;
      const col = Math.floor(px(this.tx(x)));
      if (col !== curCol) { flush(); curCol = col; first = i; last = i; mn = i; mx = i; }
      else { last = i; if (y < s.ys[mn]) mn = i; if (y > s.ys[mx]) mx = i; }
    }
    flush();
    return out;
  }
  drawLine(s, col, px, py, d, g) {
    const { ctx } = this;
    const [i0, i1] = this.visibleRange(s, d);
    if (i1 < i0) return;
    const idx = this.decimate(s, i0, i1, px, g);
    // CI band
    if (s.hasBand) {
      ctx.fillStyle = withAlpha(col, 0.18);
      ctx.beginPath();
      let started = false; const back = [];
      for (const i of idx) {
        const l = s.lo[i]; const u = s.hi[i];
        if (!this.validX(s.xs[i]) || !this.validY(u) || !this.validY(l)) continue;
        const x = px(this.tx(s.xs[i]));
        if (!started) { ctx.moveTo(x, py(this.ty(u))); started = true; } else ctx.lineTo(x, py(this.ty(u)));
        back.push([x, py(this.ty(l))]);
      }
      for (let k = back.length - 1; k >= 0; k--) ctx.lineTo(back[k][0], back[k][1]);
      ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = s.alpha; ctx.strokeStyle = col; ctx.lineWidth = s.width; ctx.lineJoin = 'round';
    ctx.setLineDash(s.dashed ? [6, 4] : []);
    ctx.beginPath();
    let pen = false; let lastY = 0; let cnt = 0;
    for (const i of idx) {
      const x = s.xs[i]; const y = s.ys[i];
      if (!this.validX(x) || !this.validY(y)) { pen = false; continue; }
      const X = px(this.tx(x)); const Y = py(this.ty(y));
      if (!pen) { ctx.moveTo(X, Y); pen = true; } else { if (s.step) ctx.lineTo(X, lastY); ctx.lineTo(X, Y); }
      lastY = Y; cnt++;
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    if (cnt > 0 && cnt <= Math.max(40, g.W / 8) && s.alpha === 1) { // markers when sparse
      ctx.fillStyle = col;
      for (const i of idx) {
        const x = s.xs[i]; const y = s.ys[i];
        if (!this.validX(x) || !this.validY(y)) continue;
        ctx.beginPath(); ctx.arc(px(this.tx(x)), py(this.ty(y)), 2.4, 0, 6.2832); ctx.fill();
      }
    }
  }
  drawScatter(s, col, px, py, d, g) {
    const { ctx, T } = this;
    const pts = [];
    for (let i = 0; i < s.xs.length; i++) {
      if (!this.validX(s.xs[i]) || !this.validY(s.ys[i])) continue;
      pts.push([px(this.tx(s.xs[i])), py(this.ty(s.ys[i])), i]);
    }
    if (s.frontier && s.highlight) {
      const f = pts.filter((p) => s.highlight.has(p[2])).sort((a, b) => a[0] - b[0]);
      if (f.length > 1) {
        ctx.strokeStyle = withAlpha(col, 0.8); ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]); ctx.beginPath();
        f.forEach((p, k) => { if (k === 0) ctx.moveTo(p[0], p[1]); else { ctx.lineTo(p[0], p[1]); } });
        ctx.stroke(); ctx.setLineDash([]);
      }
    }
    for (const [x, y, i] of pts) {
      const hl = s.highlight && s.highlight.has(i);
      ctx.beginPath(); ctx.arc(x, y, hl ? s.radius + 2 : s.radius, 0, 6.2832);
      ctx.fillStyle = hl ? col : withAlpha(col, 0.55); ctx.fill();
      if (hl) { ctx.lineWidth = 2; ctx.strokeStyle = T.fg; ctx.stroke(); }
      if (s.labels && s.labels[i] && (hl || pts.length <= 8)) { ctx.fillStyle = T.fg; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(String(s.labels[i]), x + 6, y - 4); }
    }
    s._pts = pts;
  }
  drawHover(g, px, py) {
    const { ctx, T } = this; const hv = this.hover;
    if (hv.kind === 'x') {
      ctx.strokeStyle = T.faint; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(hv.px + 0.5, g.T); ctx.lineTo(hv.px + 0.5, g.T + g.H); ctx.stroke(); ctx.setLineDash([]);
      for (const it of hv.items) { ctx.fillStyle = it.color; ctx.beginPath(); ctx.arc(it.sx, it.sy, 4, 0, 6.2832); ctx.fill(); ctx.strokeStyle = T.surface; ctx.lineWidth = 1.5; ctx.stroke(); }
    } else if (hv.kind === 'pt') {
      ctx.strokeStyle = T.fg; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hv.sx, hv.sy, 8, 0, 6.2832); ctx.stroke();
    }
  }

  bindEvents() {
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => this.onMove(e));
    c.addEventListener('pointerleave', () => { if (!this.drag) { this.hover = null; this.hideTip(); this.invalidate(); } });
    c.addEventListener('pointerdown', (e) => this.onDown(e));
    c.addEventListener('pointerup', (e) => this.onUp(e));
    c.addEventListener('pointercancel', () => { this.drag = null; this.invalidate(); });
    c.addEventListener('dblclick', () => this.resetZoom());
    c.addEventListener('wheel', (e) => {
      if (!(e.ctrlKey || e.metaKey || e.shiftKey) || !this.geom) return;
      e.preventDefault();
      const d = this.domain(); const g = this.geom; const [mx, my] = this.localXY(e);
      const f = e.deltaY < 0 ? 0.8 : 1.25;
      const nv = { ...d };
      if (e.shiftKey && !e.ctrlKey && !e.metaKey) { const fy = 1 - (my - g.T) / g.H; const yc = d.y0 + fy * (d.y1 - d.y0); nv.y0 = yc - (yc - d.y0) * f; nv.y1 = yc + (d.y1 - yc) * f; }
      else { const fx = (mx - g.L) / g.W; const xc = d.x0 + fx * (d.x1 - d.x0); nv.x0 = xc - (xc - d.x0) * f; nv.x1 = xc + (d.x1 - xc) * f; }
      this.view = nv; this.invalidate();
    }, { passive: false });
    c.addEventListener('keydown', (e) => {
      if (e.key === '+' || e.key === '=') this.zoomBy(1 / 1.5); else if (e.key === '-') this.zoomBy(1.5); else if (e.key === '0' || e.key === 'Escape') this.resetZoom();
    });
  }
  inPlot(x, y) { const g = this.geom; return g && x >= g.L && x <= g.L + g.W && y >= g.T && y <= g.T + g.H; }
  onDown(e) {
    if (!this.geom) return;
    const [x, y] = this.localXY(e);
    if (!this.inPlot(x, y)) return;
    this.canvas.setPointerCapture(e.pointerId);
    const pan = e.shiftKey || e.altKey || e.pointerType === 'touch' || e.button === 1;
    this.drag = { mode: pan ? 'pan' : 'box', x0: x, y0: y, x1: x, y1: y, d0: { ...this.domain() } };
    this.hideTip();
  }
  onUp(e) {
    const dr = this.drag; if (!dr) return;
    this.drag = null;
    try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
    if (dr.mode === 'box') {
      const g = this.geom; const dx = Math.abs(dr.x1 - dr.x0); const dy = Math.abs(dr.y1 - dr.y0);
      if (dx < 6 && dy < 6) { this.invalidate(); return; }
      const d = dr.d0;
      const X = (x) => d.x0 + (x - g.L) / g.W * (d.x1 - d.x0); const Y = (y) => d.y0 + (g.T + g.H - y) / g.H * (d.y1 - d.y0);
      const nv = { ...d };
      if (dx >= 6) { const a = X(Math.min(dr.x0, dr.x1)); const b = X(Math.max(dr.x0, dr.x1)); nv.x0 = a; nv.x1 = b; }
      if (dy >= 10 && dy > dx * 0.15) { const a = Y(Math.max(dr.y0, dr.y1)); const b = Y(Math.min(dr.y0, dr.y1)); nv.y0 = a; nv.y1 = b; }
      this.view = nv;
    }
    this.invalidate();
  }
  onMove(e) {
    if (!this.geom) return;
    const [x, y] = this.localXY(e);
    const dr = this.drag;
    if (dr) {
      dr.x1 = x; dr.y1 = y;
      if (dr.mode === 'pan') {
        const g = this.geom; const d = dr.d0;
        const ddx = (x - dr.x0) / g.W * (d.x1 - d.x0); const ddy = (y - dr.y0) / g.H * (d.y1 - d.y0);
        this.view = { x0: d.x0 - ddx, x1: d.x1 - ddx, y0: d.y0 + ddy, y1: d.y1 + ddy };
      }
      this.invalidate();
      return;
    }
    if (!this.inPlot(x, y)) { if (this.hover) { this.hover = null; this.hideTip(); this.invalidate(); } return; }
    const g = this.geom; const d = g.d;
    const o = this.opts;
    // scatter first
    let best = null; let bd = 14 * 14;
    this.series.forEach((s, si) => {
      if (s.hidden || s.type !== 'scatter' || !s._pts) return;
      for (const [sx, sy, i] of s._pts) { const dd = (sx - x) ** 2 + (sy - y) ** 2; if (dd < bd) { bd = dd; best = { s, si, i, sx, sy }; } }
    });
    if (best) {
      const { s, si, i, sx, sy } = best;
      this.hover = { kind: 'pt', sx, sy };
      const rows = [h('div', h('b', s.labels && s.labels[i] ? s.labels[i] : s.name))];
      rows.push(h('div', `${o.xLabel || 'x'}: ${this.fmtVal(s.xs[i], o.xFormat)}`), h('div', `${o.yLabel || 'y'}: ${this.fmtVal(s.ys[i], o.yFormat)}`));
      if (s.meta && s.meta[i]) for (const [k, v] of Object.entries(s.meta[i])) rows.push(h('div', { class: 'muted' }, `${k}: ${typeof v === 'number' ? fmtNum(v) : v}`));
      if (s.highlight && s.highlight.has(i)) rows.push(h('div', { class: 'ok' }, 'on Pareto frontier'));
      this.showTip(rows, sx, sy);
      this.invalidate();
      return;
    }
    // line series: nearest x
    const X = d.x0 + (x - g.L) / g.W * (d.x1 - d.x0);
    const xv = o.logX ? 10 ** X : X;
    const items = []; let hx = null;
    this.series.forEach((s, si) => {
      if (s.hidden || s.type === 'scatter' || !s.xs.length) return;
      let i = lowerBound(s.xs, xv);
      if (i > 0 && (i >= s.xs.length || Math.abs(this.px(this.tx(s.xs[i - 1])) - x) <= Math.abs(this.px(this.tx(s.xs[i])) - x))) i -= 1;
      const sx = this.validX(s.xs[i]) ? this.px(this.tx(s.xs[i])) : null;
      if (sx == null || !this.validY(s.ys[i]) || Math.abs(sx - x) > 48) return;
      items.push({ s, i, color: this.color(s, si), sx, sy: this.py(this.ty(s.ys[i])) });
      if (hx == null || Math.abs(sx - x) < Math.abs(hx - x)) hx = sx;
    });
    if (!items.length) { this.hover = null; this.hideTip(); this.invalidate(); return; }
    this.hover = { kind: 'x', px: hx, items };
    const first = items[0];
    const rows = [h('div', h('b', `${o.xLabel ? o.xLabel + ' ' : ''}${this.fmtVal(first.s.xs[first.i], o.xFormat)}`))];
    for (const it of items) {
      const yv = this.fmtVal(it.s.ys[it.i], o.yFormat);
      const band = it.s.hasBand && isNum(it.s.lo[it.i]) ? ` [${this.fmtVal(it.s.lo[it.i], o.yFormat)}, ${this.fmtVal(it.s.hi[it.i], o.yFormat)}]` : '';
      rows.push(h('div', h('span', { style: { color: it.color } }, '● '), `${it.s.name}: `, h('b', yv), band));
    }
    this.showTip(rows, hx, y);
    this.invalidate();
  }
  fmtVal(v, f) { return f ? f(v) : fmtNum(v, 4); }
  csvRows() {
    const rows = [['series', 'x', 'y', 'ci_low', 'ci_high']];
    for (const s of this.series) for (let i = 0; i < s.xs.length; i++) rows.push([s.name, s.xs[i], s.ys[i], isNum(s.lo[i]) ? s.lo[i] : '', isNum(s.hi[i]) ? s.hi[i] : '']);
    return rows;
  }
}
function fmtLogTick(v) {
  if (v >= 0.001 && v < 1e5) return trim(String(+v.toPrecision(3)));
  return v.toExponential(0).replace('+', '');
}

/** Pareto frontier indices. dirs: 'min' | 'max' per axis. Returns Set of indices. */
export function paretoFrontier(xs, ys, xDir = 'min', yDir = 'max') {
  const idx = xs.map((_, i) => i).filter((i) => isNum(xs[i]) && isNum(ys[i]));
  const better = (a, b, dir) => (dir === 'min' ? a <= b : a >= b);
  const strictly = (a, b, dir) => (dir === 'min' ? a < b : a > b);
  const out = new Set();
  for (const i of idx) {
    let dominated = false;
    for (const j of idx) {
      if (i === j) continue;
      if (better(xs[j], xs[i], xDir) && better(ys[j], ys[i], yDir) && (strictly(xs[j], xs[i], xDir) || strictly(ys[j], ys[i], yDir))) { dominated = true; break; }
    }
    if (!dominated) out.add(i);
  }
  return out;
}

export function lineChart(container, opts) { return new XYChart(container, opts); }
export function scatterChart(container, opts) {
  const c = new XYChart(container, { ...opts, noLogToggle: opts.noLogToggle });
  return c;
}

// ------------------------------------------------------------------ roofline
/**
 * rooflineChart(container, {peaks:[{name,tflops,color?}], bwGBs, points:[{name, ai, tflops, meta?}], title})
 * ai = FLOP/byte, tflops = achieved TFLOP/s. Log-log axes.
 */
export function rooflineChart(container, { peaks = [], bwGBs = 0, points = [], title = 'Roofline', height = 320, bwName = 'memory bandwidth' } = {}) {
  const series = [];
  const ais = points.map((p) => p.ai).filter((v) => v > 0);
  const aiMin = Math.min(0.1, ...(ais.length ? ais.map((a) => a / 2) : [0.1]));
  const maxRidge = Math.max(1, ...peaks.map((p) => (bwGBs > 0 ? p.tflops * 1000 / bwGBs : 100)));
  const aiMax = Math.max(maxRidge * 8, ...(ais.length ? ais.map((a) => a * 2) : [0]));
  peaks.forEach((p, i) => {
    const ridge = bwGBs > 0 ? p.tflops * 1000 / bwGBs : aiMax;
    const y = (ai) => Math.min(p.tflops, bwGBs * ai / 1000);
    series.push({ name: `${p.name} peak ${fmtNum(p.tflops)} TFLOP/s`, type: 'line', color: p.color, points: [[aiMin, y(aiMin)], [Math.min(ridge, aiMax), y(ridge)], [aiMax, p.tflops]], width: 2, dashed: p.provenance === 'vendor' });
  });
  if (!peaks.length && bwGBs > 0) series.push({ name: bwName, points: [[aiMin, bwGBs * aiMin / 1000], [aiMax, bwGBs * aiMax / 1000]] });
  if (points.length) {
    series.push({ name: 'kernels', type: 'scatter', points: points.map((p) => [p.ai, p.tflops]), labels: points.map((p) => p.name), meta: points.map((p) => p.meta || {}), radius: 4.5, color: themeColors().palette[3] });
  }
  return new XYChart(container, { title, height, logX: true, logY: true, xLabel: 'arithmetic intensity (FLOP/byte)', yLabel: 'TFLOP/s', series, noLogToggle: true, empty: 'No roofline data' });
}

// ------------------------------------------------------------------ bar chart
export class BarChart extends BaseChart {
  constructor(container, opts = {}) {
    super(container, { horizontal: false, stacked: false, valueFormat: null, yLabel: '', ...opts });
    this.categories = []; this.series = [];
    this.canvas.addEventListener('pointermove', (e) => this.onMove(e));
    this.canvas.addEventListener('pointerleave', () => { this.hoverBar = null; this.hideTip(); this.invalidate(); });
    this.setData(opts.categories || [], opts.series || []);
  }
  setData(categories, series) {
    this.categories = categories; this.series = series;
    clear(this.legendEl);
    if (this.opts.legend && series.length > 1) series.forEach((s, i) => this.legendEl.appendChild(h('button', { type: 'button' }, h('i', { style: { background: s.color || themeColors().palette[i % 8] } }), s.name)));
    this.invalidate();
  }
  fmt(v) { return this.opts.valueFormat ? this.opts.valueFormat(v) : fmtNum(v, 4); }
  render() {
    const T = this.prep(); const { ctx } = this;
    const cats = this.categories; const ser = this.series;
    this.showEmpty(!cats.length || !ser.length);
    this.bars = [];
    if (!cats.length || !ser.length) return;
    const horiz = this.opts.horizontal; const stacked = this.opts.stacked;
    let max = 0; let min = 0;
    for (const s of ser) s.values.forEach((v, i) => { if (!isNum(v)) return; const hi = s.ci && s.ci[i] ? s.ci[i][1] : v; max = Math.max(max, stacked ? 0 : hi); min = Math.min(min, v); });
    if (stacked) for (let i = 0; i < cats.length; i++) { let sum = 0; for (const s of ser) sum += isNum(s.values[i]) ? s.values[i] : 0; max = Math.max(max, sum); }
    if (this.opts.yMax != null) max = this.opts.yMax;
    if (max === min) max = min + 1;
    const { ticks, step } = linTicks(min, max * 1.05, 5);
    const top = ticks[ticks.length - 1] > max * 1.05 ? ticks[ticks.length - 1] : max * 1.05;
    const maxLabel = horiz ? Math.min(180, Math.max(...cats.map((c) => ctx.measureText(String(c)).width)) + 8) : 0;
    let lw = 0; for (const t of ticks) lw = Math.max(lw, ctx.measureText(fmtTick(t, step)).width);
    const L = horiz ? maxLabel + 8 : lw + (this.opts.yLabel ? 30 : 14); const R = 14; const Tp = 8; const B = horiz ? 22 : 38;
    const W = this.w - L - R; const H = this.h - Tp - B;
    const lo = Math.min(0, min);
    const val2px = (v) => (horiz ? L + (v - lo) / (top - lo) * W : Tp + H - (v - lo) / (top - lo) * H);
    ctx.lineWidth = 1; ctx.strokeStyle = T.grid; ctx.fillStyle = T.muted;
    for (const t of ticks) {
      const p = val2px(t);
      ctx.beginPath(); if (horiz) { ctx.moveTo(p + 0.5, Tp); ctx.lineTo(p + 0.5, Tp + H); } else { ctx.moveTo(L, p + 0.5); ctx.lineTo(L + W, p + 0.5); } ctx.stroke();
      if (horiz) { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(fmtTick(t, step), p, Tp + H + 4); } else { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmtTick(t, step), L - 6, p); }
    }
    const n = cats.length; const slot = (horiz ? H : W) / n;
    const groupW = slot * 0.74; const bw = stacked ? groupW : groupW / ser.length;
    cats.forEach((cat, ci) => {
      const gx = (horiz ? Tp : L) + slot * ci + (slot - groupW) / 2;
      let acc = 0;
      ser.forEach((s, si) => {
        const v = s.values[ci]; if (!isNum(v)) return;
        const col = s.color || T.palette[si % 8];
        const a = stacked ? acc : 0; const b = stacked ? acc + v : v; acc = b;
        const off = stacked ? 0 : bw * si;
        let rx; let ry; let rw; let rh;
        if (horiz) { rx = val2px(Math.min(a, b)); rw = Math.abs(val2px(b) - val2px(a)); ry = gx + off; rh = bw - 1; }
        else { ry = val2px(Math.max(a, b)); rh = Math.abs(val2px(b) - val2px(a)); rx = gx + off; rw = bw - 1; }
        const isHover = this.hoverBar && this.hoverBar.ci === ci && this.hoverBar.si === si;
        ctx.fillStyle = isHover ? col : withAlpha(col, 0.85); ctx.fillRect(rx, ry, Math.max(1, rw), Math.max(1, rh));
        this.bars.push({ rx, ry, rw, rh, ci, si, v });
        if (s.ci && s.ci[ci]) {
          const [l, u] = s.ci[ci]; ctx.strokeStyle = T.fg; ctx.lineWidth = 1.3; ctx.beginPath();
          if (horiz) { const cy = ry + rh / 2; ctx.moveTo(val2px(l), cy); ctx.lineTo(val2px(u), cy); } else { const cx = rx + rw / 2; ctx.moveTo(cx, val2px(l)); ctx.lineTo(cx, val2px(u)); ctx.moveTo(cx - 3, val2px(l)); ctx.lineTo(cx + 3, val2px(l)); ctx.moveTo(cx - 3, val2px(u)); ctx.lineTo(cx + 3, val2px(u)); }
          ctx.stroke();
        }
        if (s.highlight && s.highlight[ci]) { ctx.strokeStyle = T.fg; ctx.lineWidth = 2; ctx.strokeRect(rx, ry, Math.max(1, rw), Math.max(1, rh)); }
      });
      ctx.fillStyle = T.muted;
      if (horiz) { ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(truncate(ctx, String(cat), maxLabel), L - 6, gx + groupW / 2); }
      else { ctx.textAlign = 'center'; ctx.textBaseline = 'top'; const lbl = truncate(ctx, String(cat), slot - 2); ctx.fillText(lbl, L + slot * ci + slot / 2, Tp + H + 6); }
    });
    ctx.strokeStyle = T.border; ctx.beginPath();
    if (horiz) { ctx.moveTo(val2px(0) + 0.5, Tp); ctx.lineTo(val2px(0) + 0.5, Tp + H); } else { ctx.moveTo(L, val2px(0) + 0.5); ctx.lineTo(L + W, val2px(0) + 0.5); }
    ctx.stroke();
    if (this.opts.yLabel && !horiz) { ctx.save(); ctx.translate(11, Tp + H / 2); ctx.rotate(-Math.PI / 2); ctx.fillStyle = T.muted; ctx.textAlign = 'center'; ctx.fillText(this.opts.yLabel, 0, 0); ctx.restore(); }
  }
  onMove(e) {
    const [x, y] = this.localXY(e);
    const hit = (this.bars || []).find((b) => x >= b.rx && x <= b.rx + b.rw && y >= b.ry && y <= b.ry + b.rh);
    const prev = this.hoverBar;
    this.hoverBar = hit ? { ci: hit.ci, si: hit.si } : null;
    if (!hit) { this.hideTip(); if (prev) this.invalidate(); return; }
    const s = this.series[hit.si];
    const rows = [h('div', h('b', String(this.categories[hit.ci]))), h('div', `${s.name}: `, h('b', this.fmt(hit.v)))];
    if (s.ci && s.ci[hit.ci]) rows.push(h('div', { class: 'muted' }, `CI [${this.fmt(s.ci[hit.ci][0])}, ${this.fmt(s.ci[hit.ci][1])}]`));
    if (s.notes && s.notes[hit.ci]) rows.push(h('div', { class: 'muted' }, s.notes[hit.ci]));
    this.showTip(rows, x, y);
    this.invalidate();
  }
  csvRows() {
    const rows = [['category', ...this.series.map((s) => s.name)]];
    this.categories.forEach((c, i) => rows.push([c, ...this.series.map((s) => s.values[i])]));
    return rows;
  }
}
function truncate(ctx, s, maxW) {
  if (ctx.measureText(s).width <= maxW) return s;
  while (s.length > 1 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1);
  return s + '…';
}
export function barChart(container, opts) { return new BarChart(container, opts); }

// ------------------------------------------------------------------ heatmap
const SEQ = [[68, 1, 84], [59, 82, 139], [33, 144, 141], [93, 200, 99], [253, 231, 37]]; // viridis stops
function seqColor(t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (SEQ.length - 1); const i = Math.min(SEQ.length - 2, Math.floor(p)); const f = p - i;
  const a = SEQ[i]; const b = SEQ[i + 1];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}
export class Heatmap extends BaseChart {
  /** opts: {xLabels, yLabels, z (rows=y, cols=x; null for missing), lowerIsBetter, valueFormat, xTitle, yTitle} */
  constructor(container, opts = {}) {
    super(container, { lowerIsBetter: true, valueFormat: null, xTitle: '', yTitle: '', legend: false, ...opts });
    this.canvas.addEventListener('pointermove', (e) => this.onMove(e));
    this.canvas.addEventListener('pointerleave', () => { this.hoverCell = null; this.hideTip(); this.invalidate(); });
    this.setData(opts);
  }
  setData({ xLabels = [], yLabels = [], z = [] }) { this.xLabels = xLabels; this.yLabels = yLabels; this.z = z; this.invalidate(); }
  fmt(v) { return this.opts.valueFormat ? this.opts.valueFormat(v) : fmtNum(v, 4); }
  render() {
    const T = this.prep(); const { ctx } = this;
    const nx = this.xLabels.length; const ny = this.yLabels.length;
    this.showEmpty(!nx || !ny);
    if (!nx || !ny) return;
    let lo = Infinity; let hi = -Infinity; let best = null;
    this.z.forEach((row, j) => row.forEach((v, i) => { if (!isNum(v)) return; lo = Math.min(lo, v); hi = Math.max(hi, v); if (!best || (this.opts.lowerIsBetter ? v < best.v : v > best.v)) best = { v, i, j }; }));
    const ml = Math.max(...this.yLabels.map((l) => ctx.measureText(String(l)).width)) + 12 + (this.opts.yTitle ? 16 : 0);
    const L = ml; const R = 64; const Tp = 6; const B = 24 + (this.opts.xTitle ? 16 : 0);
    const W = this.w - L - R; const H = this.h - Tp - B;
    const cw = W / nx; const ch = H / ny;
    this.cells = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = (this.z[j] || [])[i];
      const x = L + i * cw; const y = Tp + j * ch;
      if (!isNum(v)) { ctx.fillStyle = T.surface2; ctx.fillRect(x, y, cw - 1, ch - 1); continue; }
      let t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      if (this.opts.lowerIsBetter) t = 1 - t;
      ctx.fillStyle = seqColor(t); ctx.fillRect(x, y, cw - 1, ch - 1);
      this.cells.push({ x, y, w: cw, h: ch, i, j, v });
      if (cw > 38 && ch > 16) { ctx.fillStyle = t > 0.6 ? '#111' : '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(fmtNum(v, 3), x + cw / 2, y + ch / 2); }
    }
    if (best) { ctx.strokeStyle = T.fg; ctx.lineWidth = 2.2; ctx.strokeRect(L + best.i * cw + 1, Tp + best.j * ch + 1, cw - 3, ch - 3); }
    if (this.hoverCell) { ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.strokeRect(L + this.hoverCell.i * cw + 1, Tp + this.hoverCell.j * ch + 1, cw - 3, ch - 3); }
    ctx.fillStyle = T.muted; ctx.textBaseline = 'middle'; ctx.textAlign = 'right';
    this.yLabels.forEach((l, j) => ctx.fillText(String(l), L - 6, Tp + j * ch + ch / 2));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const every = Math.ceil(nx / Math.max(1, Math.floor(W / 44)));
    this.xLabels.forEach((l, i) => { if (i % every === 0) ctx.fillText(String(l), L + i * cw + cw / 2, Tp + H + 4); });
    if (this.opts.xTitle) ctx.fillText(this.opts.xTitle, L + W / 2, this.h - 14);
    if (this.opts.yTitle) { ctx.save(); ctx.translate(10, Tp + H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(this.opts.yTitle, 0, 0); ctx.restore(); }
    // colour bar
    const bx = L + W + 14; const bh = Math.min(H, 160);
    for (let k = 0; k < bh; k++) { ctx.fillStyle = seqColor(this.opts.lowerIsBetter ? k / bh : 1 - k / bh); ctx.fillRect(bx, Tp + k, 10, 1); }
    ctx.fillStyle = T.muted; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(this.opts.lowerIsBetter ? fmtNum(lo, 3) : fmtNum(hi, 3), bx + 14, Tp - 2);
    ctx.textBaseline = 'bottom'; ctx.fillText(this.opts.lowerIsBetter ? fmtNum(hi, 3) : fmtNum(lo, 3), bx + 14, Tp + bh + 2);
    if (best) { ctx.fillStyle = T.fg; ctx.textBaseline = 'top'; ctx.fillText('best', bx, Tp + bh + 6); }
  }
  onMove(e) {
    const [x, y] = this.localXY(e);
    const c = (this.cells || []).find((q) => x >= q.x && x < q.x + q.w && y >= q.y && y < q.y + q.h);
    this.hoverCell = c ? { i: c.i, j: c.j } : null;
    if (!c) { this.hideTip(); this.invalidate(); return; }
    this.showTip([h('div', h('b', `${this.opts.xTitle || 'x'}=${this.xLabels[c.i]}, ${this.opts.yTitle || 'y'}=${this.yLabels[c.j]}`)), h('div', `${this.opts.valueLabel || 'value'}: `, h('b', this.fmt(c.v)))], x, y);
    this.invalidate();
  }
  csvRows() {
    const rows = [['y\\x', ...this.xLabels]];
    this.yLabels.forEach((l, j) => rows.push([l, ...(this.z[j] || [])]));
    return rows;
  }
}
export function heatmap(container, opts) { return new Heatmap(container, opts); }

// ------------------------------------------------------------------ sparkline (SVG, inherits currentColor)
export function sparkline(values, { width = 100, height = 26, color = 'var(--accent)', fill = true, min = null, max = null } = {}) {
  const vals = values.filter(isNum);
  const svg = h('svg:svg', { class: 'spark', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'trend' });
  if (vals.length < 2) { svg.appendChild(h('svg:line', { x1: 0, y1: height - 1, x2: width, y2: height - 1, stroke: 'var(--border)', 'stroke-width': 1 })); return svg; }
  const lo = min ?? Math.min(...vals); const hi = max ?? Math.max(...vals); const span = hi - lo || 1;
  const pts = vals.map((v, i) => [(i / (vals.length - 1)) * (width - 2) + 1, height - 2 - ((v - lo) / span) * (height - 4)]);
  const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('');
  if (fill) svg.appendChild(h('svg:path', { d: `${d}L${width - 1} ${height - 1}L1 ${height - 1}Z`, fill: color, opacity: 0.14 }));
  svg.appendChild(h('svg:path', { d, fill: 'none', stroke: color, 'stroke-width': 1.5, 'stroke-linejoin': 'round' }));
  return svg;
}
