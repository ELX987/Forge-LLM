// Small DOM + formatting helpers shared by every module. No dependencies.

const SVGNS = 'http://www.w3.org/2000/svg';
const PROP_KEYS = new Set(['value', 'checked', 'selected', 'disabled', 'indeterminate', 'open', 'hidden', 'readOnly', 'multiple', 'scrollTop', 'textContent']);

/** Create an element. h('div', {class:'x', onclick}, child, [children], 'text'). Tag prefix 'svg:' makes SVG elements. */
export function h(tag, attrs, ...children) {
  const isSvg = tag.startsWith('svg:');
  const el = isSvg ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
  if (attrs != null && (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs))) {
    children.unshift(attrs);
    attrs = null;
  }
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class' || k === 'className') el.setAttribute('class', Array.isArray(v) ? v.filter(Boolean).join(' ') : v);
      else if (k === 'style') {
        if (typeof v === 'string') el.setAttribute('style', v);
        else for (const [sk, sv] of Object.entries(v)) { if (sk.startsWith('--')) el.style.setProperty(sk, sv); else el.style[sk] = sv; }
      } else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else if (!isSvg && PROP_KEYS.has(k)) el[k] = v;
      else if (v === true) el.setAttribute(k, '');
      else el.setAttribute(k, String(v));
    }
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else if (c instanceof Node) el.appendChild(c);
    else el.appendChild(document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function mount(el, ...children) { clear(el); append(el, children); return el; }
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------------------------------------------------------------- formatting
export function isNum(x) { return typeof x === 'number' && Number.isFinite(x); }

export function fmtNum(x, digits = 3) {
  if (x == null || x === '') return '-';
  if (typeof x === 'string') return x;
  if (!isNum(x)) return String(x);
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e12) return (x / 1e12).toPrecision(3) + 'T';
  if (a >= 1e9) return (x / 1e9).toPrecision(3) + 'B';
  if (a >= 1e6) return (x / 1e6).toPrecision(3) + 'M';
  if (a >= 1e5) return (x / 1e3).toPrecision(4) + 'k';
  if (a >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 1) return Number(x.toPrecision(digits + 1)).toString();
  if (a >= 1e-3) return Number(x.toPrecision(digits)).toString();
  return x.toExponential(2);
}

export function fmtInt(x) { return isNum(x) ? Math.round(x).toLocaleString() : '-'; }
export function fmtPct(x, digits = 1) { return isNum(x) ? (x * 100).toFixed(digits) + '%' : '-'; }

export function fmtBytes(n, digits = 1) {
  if (!isNum(n)) return '-';
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let i = 0; let v = Math.abs(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (n < 0 ? '-' : '') + (i === 0 ? v.toFixed(0) : v.toFixed(digits)) + ' ' + u[i];
}

export function fmtDuration(sec) {
  if (!isNum(sec) || sec < 0) return '-';
  if (sec < 1) return Math.round(sec * 1000) + ' ms';
  if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + ' s';
  const m = Math.floor(sec / 60); const s = Math.floor(sec % 60);
  if (m < 60) return `${m}m ${String(s).padStart(2, '0')}s`;
  const hh = Math.floor(m / 60);
  if (hh < 48) return `${hh}h ${String(m % 60).padStart(2, '0')}m`;
  return `${Math.floor(hh / 24)}d ${hh % 24}h`;
}

/** epoch seconds (or ms/ISO string) -> Date */
export function toDate(t) {
  if (t == null) return null;
  if (typeof t === 'number') return new Date(t > 1e11 ? t : t * 1000);
  const d = new Date(t);
  return isNaN(d) ? null : d;
}
export function fmtTime(t) { const d = toDate(t); return d ? d.toLocaleString() : '-'; }
export function fmtRel(t) {
  const d = toDate(t); if (!d) return '-';
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < -1) return 'in ' + fmtDuration(-s);
  if (s < 5) return 'just now';
  return fmtDuration(s) + ' ago';
}
export function shortHash(x, n = 10) {
  if (!x || typeof x !== 'string') return '-';
  const i = x.indexOf(':');
  return i > 0 ? x.slice(0, i + 1) + x.slice(i + 1, i + 1 + n) : x.slice(0, n);
}

// ---------------------------------------------------------------- misc
export function clamp(x, lo, hi) { return Math.min(hi, Math.max(lo, x)); }
export function deepClone(o) { return o === undefined ? undefined : JSON.parse(JSON.stringify(o)); }
export function deepEqual(a, b) {
  if (a === b) return true;
  if (typeof a !== typeof b || a == null || b == null || typeof a !== 'object') return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a); const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const k of ka) if (!deepEqual(a[k], b[k])) return false;
  return true;
}
export function getPath(o, path, def) {
  const parts = Array.isArray(path) ? path : String(path).split('.').filter(Boolean);
  let cur = o;
  for (const p of parts) { if (cur == null) return def; cur = cur[p]; }
  return cur === undefined ? def : cur;
}
export function setPath(o, path, value) {
  const parts = Array.isArray(path) ? path : String(path).split('.').filter(Boolean);
  let cur = o;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (cur[p] == null || typeof cur[p] !== 'object') cur[p] = /^\d+$/.test(parts[i + 1]) ? [] : {};
    cur = cur[p];
  }
  if (parts.length) cur[parts[parts.length - 1]] = value;
  return o;
}
export function debounce(fn, ms = 200) {
  let t = null;
  const d = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}
export function throttleRaf(fn) {
  let queued = false; let lastArgs;
  return (...a) => { lastArgs = a; if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; fn(...lastArgs); }); };
}
let _uid = 0;
export function uid(prefix = 'u') { return `${prefix}${++_uid}`; }
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** API list responses come as [..] or {items:[..]} or {jobs:[..]} etc. */
export function listOf(r, ...keys) {
  if (Array.isArray(r)) return r;
  if (r && typeof r === 'object') {
    for (const k of [...keys, 'items', 'results', 'data']) if (Array.isArray(r[k])) return r[k];
  }
  return [];
}

export function download(filename, content, mime = 'text/plain') {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
}

export function toCSV(rows) {
  const esc = (v) => {
    if (v == null) return '';
    const s = String(v);
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  return rows.map((r) => r.map(esc).join(',')).join('\n') + '\n';
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) { /* fall through */ }
  try {
    const ta = h('textarea', { style: 'position:fixed;opacity:0' }); ta.value = text;
    document.body.appendChild(ta); ta.select(); const ok = document.execCommand('copy'); ta.remove(); return ok;
  } catch (_) { return false; }
}

export function titleCase(s) { return String(s || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()); }
export function pluralize(n, one, many) { return `${n} ${n === 1 ? one : (many || one + 's')}`; }
export function parseCsvList(s) { return String(s || '').split(',').map((x) => x.trim()).filter(Boolean); }

/** Exponential moving average (for chart smoothing). alpha in [0,1): 0 = off. */
export function ema(points, alpha) {
  if (!alpha || alpha <= 0) return points;
  const out = new Array(points.length); let m = null;
  for (let i = 0; i < points.length; i++) {
    const y = points[i][1];
    if (!isNum(y)) { out[i] = points[i]; continue; }
    m = m == null ? y : alpha * m + (1 - alpha) * y;
    out[i] = [points[i][0], m];
  }
  return out;
}

/** Normalise GPU telemetry payloads ([..] or {gpus:[..]}) into a stable shape. */
export function normGpus(d) {
  const list = Array.isArray(d) ? d : (d && (d.gpus || d.items || d.devices)) || [];
  return list.map((g, i) => ({
    index: g.index ?? i, name: g.name || `GPU ${i}`, util: g.util ?? g.utilization ?? g.gpu_util ?? null,
    mem_used_mb: g.mem_used_mb ?? g.memory_used_mb ?? null, mem_total_mb: g.mem_total_mb ?? g.memory_total_mb ?? null,
    temp_c: g.temp_c ?? g.temperature ?? null, power_w: g.power_w ?? g.power ?? null, clock_mhz: g.clock_mhz ?? null,
  }));
}
