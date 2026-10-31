// Shared UI building blocks: toasts, modals, badges, tables, tabs, notices, loading helpers, tooltips, icons.

import { h, clear, append, fmtNum, isNum, fmtTime, shortHash, copyText, uid, titleCase } from '../util.js';
import { isUnavailable } from '../api.js';

export { h };

// ------------------------------------------------------------------ icons (inline SVG, 24x24 stroke icons)
const ICONS = {
  grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  cube: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5',
  db: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6',
  chip: 'M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4',
  plus: 'M12 5v14M5 12h14',
  list: 'M8 6h13M8 12h13M8 18h13M3.5 6h.01M3.5 12h.01M3.5 18h.01',
  tune: 'M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0M14 4v4M8 10v4M16 16v4',
  flow: 'M5 4h6v5H5zM13 15h6v5h-6zM8 9v3a3 3 0 003 3h2M5 15h4v5H5z',
  scale: 'M12 4v16M5 20h14M5 8l-2 6a3 3 0 006 0zM19 8l-2 6a3 3 0 006 0zM5 8h14',
  chat: 'M4 5h16v11H9l-5 4z',
  bolt: 'M13 2L4 14h7l-1 8 9-12h-7z',
  disk: 'M4 7c0-1.7 3.6-3 8-3s8 1.3 8 3v10c0 1.7-3.6 3-8 3s-8-1.3-8-3zM4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
  check: 'M5 12l5 5L20 7',
  gear: 'M12 15a3 3 0 100-6 3 3 0 000 6zM19 12l2-1-2-3-2 .6-1.5-1L15 5h-4l-.5 2.6-1.5 1L7 8 5 11l2 1v2l-2 1 2 3 2-.6 1.5 1L11 21h4l.5-2.6 1.5-1 2 .6 2-3-2-1z',
  menu: 'M4 6h16M4 12h16M4 18h16',
  sun: 'M12 16a4 4 0 100-8 4 4 0 000 8zM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5',
  moon: 'M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z',
  x: 'M6 6l12 12M18 6L6 18',
  refresh: 'M20 11a8 8 0 10-2.3 6M20 4v7h-7',
  trash: 'M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13',
  copy: 'M9 9h10v11H9zM5 15V4h10',
  download: 'M12 4v11M7 11l5 5 5-5M5 20h14',
  play: 'M7 4l12 8-12 8z',
  stop: 'M6 6h12v12H6z',
};
export function icon(name, cls = 'ico') {
  const path = ICONS[name] || ICONS.grid;
  const svg = h('svg:svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.8', 'stroke-linecap': 'round', 'stroke-linejoin': 'round', class: cls, 'aria-hidden': 'true' });
  svg.appendChild(h('svg:path', { d: path }));
  return svg;
}

// ------------------------------------------------------------------ toasts
export function toast(message, kind = 'info', ms = 5000) {
  let host = document.getElementById('toasts');
  if (!host) { host = h('div', { id: 'toasts', 'aria-live': 'polite' }); document.body.appendChild(host); }
  const el = h('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status' },
    h('div', { class: 'msg' }, message instanceof Node ? message : String(message)),
    h('button', { class: 'btn ghost sm', 'aria-label': 'Dismiss', onclick: () => el.remove() }, '×'));
  host.appendChild(el);
  if (ms > 0) setTimeout(() => el.remove(), ms);
  return el;
}
export const toastError = (e, prefix = '') => toast((prefix ? prefix + ': ' : '') + (e && e.message ? e.message : String(e)), 'error', 8000);

// ------------------------------------------------------------------ modal
export function modal({ title = '', body = null, footer = null, wide = false, onClose = null, closable = true } = {}) {
  const dlg = h('dialog', { class: 'modal' + (wide ? ' wide' : ''), 'aria-label': title });
  const bodyEl = h('div', { class: 'modal-body' });
  append(bodyEl, [body]);
  const footEl = h('div', { class: 'modal-foot' });
  append(footEl, [footer]);
  dlg.append(
    h('div', { class: 'modal-head' }, h('h2', title), closable ? h('button', { class: 'btn ghost sm', 'aria-label': 'Close', onclick: () => api.close() }, '×') : null),
    bodyEl, footEl);
  if (!footer) footEl.classList.add('hidden');
  const api = {
    el: dlg, body: bodyEl, foot: footEl,
    close(result) { if (dlg.open) dlg.close(); dlg.remove(); if (onClose) onClose(result); },
    setBody(...n) { clear(bodyEl); append(bodyEl, n); },
    setFooter(...n) { clear(footEl); append(footEl, n); footEl.classList.toggle('hidden', !n.length); },
  };
  dlg.addEventListener('cancel', (e) => { if (!closable) e.preventDefault(); else { e.preventDefault(); api.close(); } });
  dlg.addEventListener('click', (e) => { if (e.target === dlg && closable) api.close(); });
  document.body.appendChild(dlg);
  if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
  return api;
}

export function confirmDialog({ title = 'Are you sure?', message = '', confirmLabel = 'Confirm', danger = false } = {}) {
  return new Promise((resolve) => {
    let done = false;
    const finish = (v) => { if (done) return; done = true; m.close(v); resolve(v); };
    const m = modal({
      title, body: h('div', message instanceof Node ? message : h('p', message)),
      footer: [h('button', { class: 'btn', onclick: () => finish(false) }, 'Cancel'),
        h('button', { class: 'btn ' + (danger ? 'danger' : 'primary'), onclick: () => finish(true) }, confirmLabel)],
      onClose: () => { if (!done) { done = true; resolve(false); } },
    });
  });
}

// ------------------------------------------------------------------ tooltips (delegated, for [data-tip])
let tipEl = null;
export function installTooltips() {
  const show = (t) => {
    const text = t.getAttribute('data-tip');
    if (!text) return;
    hide();
    tipEl = h('div', { class: 'tip', role: 'tooltip' }, text);
    document.body.appendChild(tipEl);
    const r = t.getBoundingClientRect(); const tr = tipEl.getBoundingClientRect();
    let x = r.left + r.width / 2 - tr.width / 2; let y = r.bottom + 6;
    x = Math.max(6, Math.min(window.innerWidth - tr.width - 6, x));
    if (y + tr.height > window.innerHeight - 6) y = r.top - tr.height - 6;
    tipEl.style.left = x + 'px'; tipEl.style.top = Math.max(6, y) + 'px';
  };
  const hide = () => { if (tipEl) { tipEl.remove(); tipEl = null; } };
  const find = (e) => (e.target && e.target.closest ? e.target.closest('[data-tip]') : null);
  document.addEventListener('mouseover', (e) => { const t = find(e); if (t) show(t); });
  document.addEventListener('mouseout', (e) => { if (find(e)) hide(); });
  document.addEventListener('focusin', (e) => { const t = find(e); if (t) show(t); });
  document.addEventListener('focusout', hide);
  document.addEventListener('scroll', hide, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide(); });
}

export function help(text) { return h('span', { class: 'help', 'data-tip': text, tabindex: '0', role: 'img', 'aria-label': text }, '?'); }

// ------------------------------------------------------------------ badges
const STATUS_KIND = { running: 'info', queued: 'dim', succeeded: 'ok', failed: 'bad', cancelled: 'warn', pass: 'ok', fail: 'bad', inconclusive: 'warn',
  reproduced: 'ok', contradicted: 'bad', partial: 'warn', untested: 'dim', trusted: 'ok', untrusted: 'warn', available: 'ok', unavailable: 'dim', ready: 'ok', starting: 'info', stopped: 'dim', created: 'dim', ok: 'ok', error: 'bad' };
export function badge(text, kind = '', opts = {}) {
  return h('span', { class: `badge ${kind}${opts.pulse ? ' pulse' : ''}`, 'data-tip': opts.tip || null }, opts.dot ? h('span', { class: 'dot' }) : null, text);
}
export function statusBadge(status) {
  const s = String(status || 'unknown');
  return badge(s, STATUS_KIND[s] || '', { dot: s === 'running' || s === 'starting', pulse: s === 'running' || s === 'starting' });
}

export const PROVENANCE = {
  measured: 'Measured by Forge on this hardware/run, with the recorded environment.',
  reproduced: 'Re-run by Forge and agreed with the paper within the stated tolerance.',
  paper_reported: 'Reported in a paper; not yet re-measured here. Treat as a hypothesis (see Claims ledger).',
  derived: 'Computed from other numbers via the cost model / roofline; inherits their uncertainty.',
  vendor: 'Vendor datasheet value (peak, not achieved).',
  target: 'A goal or threshold, not an observation.',
  untested: 'Not yet tested.',
};
export function provBadge(prov) {
  if (!prov) return null;
  const p = String(prov);
  return h('span', { class: `prov ${p}`, 'data-tip': PROVENANCE[p] || p, tabindex: '0' }, p.replace('_', '-'));
}

/** Render a MetricValue ({value, ci_low, ci_high, n, provenance, within_noise}) or a plain number. */
export function metricCell(mv, { digits = 4, showProv = true } = {}) {
  if (mv == null) return h('span', { class: 'faint' }, '-');
  if (typeof mv !== 'object') return h('span', fmtNum(mv, digits));
  const parts = [h('b', fmtNum(mv.value, digits))];
  if (isNum(mv.ci_low) && isNum(mv.ci_high)) {
    parts.push(' ', h('span', { class: 'ci', 'data-tip': `Confidence interval${mv.n != null ? ` (n=${mv.n})` : ''}` }, `[${fmtNum(mv.ci_low, digits)}, ${fmtNum(mv.ci_high, digits)}]`));
  }
  if (mv.within_noise) parts.push(' ', h('span', { class: 'noise', 'data-tip': 'Difference is within the measured noise floor; do not read this as a real gain or loss.' }, 'within noise'));
  if (showProv && mv.provenance) parts.push(' ', provBadge(mv.provenance));
  return h('span', parts);
}

export function hashLabel(hash, n = 12) {
  if (!hash) return h('span', { class: 'faint' }, '-');
  return h('span', { class: 'hash', title: hash + ' (click to copy)', onclick: async () => { if (await copyText(hash)) toast('Copied ' + shortHash(hash, 16), 'ok', 1500); } }, shortHash(hash, n));
}

// ------------------------------------------------------------------ notices / empty / loading
export function notice(kind, title, body, actions) {
  const sym = { error: '!', unavailable: '⊘', info: 'i' }[kind] || 'i';
  return h('div', { class: `notice ${kind}`, role: kind === 'error' ? 'alert' : 'status' }, h('div', { class: 'ico' }, sym),
    h('div', { class: 'body' }, title ? h('h3', title) : null, body instanceof Node ? body : (body ? h('p', body) : null), actions || null));
}
export function unavailableNotice(feature, detail) {
  return notice('unavailable', `${feature} is not available on this server`,
    detail || 'The backend module for this feature is not installed or not enabled (the API returned 404/501). The rest of the UI keeps working.');
}
export function errorNotice(err, retry) {
  if (isUnavailable(err)) return unavailableNotice('This feature');
  const extra = [];
  if (err && err.network) extra.push(h('button', { class: 'btn sm', onclick: () => { import('../state.js').then((s) => { s.setMockMode(true); window.location.reload(); }); } }, 'Use demo data'));
  if (retry) extra.push(h('button', { class: 'btn sm', onclick: retry }, 'Retry'));
  return notice('error', err && err.network ? 'Cannot reach the API' : 'Request failed', (err && err.message) || String(err), h('div', { class: 'row mt-s' }, extra));
}
export function emptyState(title, text, action) {
  return h('div', { class: 'empty' }, h('h3', title), text ? h('p', text) : null, action || null);
}
export function loadingBlock(msg = 'Loading...') { return h('div', { class: 'loading', role: 'status' }, h('span', { class: 'spinner' }), msg); }

/**
 * Load data into a container with loading + error + unavailable states.
 * `fetcher()` -> Promise<data>; `render(data, container)` returns Node(s) or fills container itself.
 * Returns reload().
 */
export function loadInto(container, fetcher, render, { loading = 'Loading...', feature = '' } = {}) {
  let seq = 0;
  async function reload(silent = false) {
    const my = ++seq;
    if (!silent) { clear(container); container.appendChild(loadingBlock(loading)); }
    try {
      const data = await fetcher();
      if (my !== seq) return;
      const out = render(data, container);
      clear(container);
      append(container, [out]);
    } catch (e) {
      if (my !== seq || (e && e.name === 'AbortError')) return;
      console.warn(e);
      clear(container);
      container.appendChild(isUnavailable(e) ? unavailableNotice(feature || 'This feature') : errorNotice(e, () => reload()));
    }
  }
  reload();
  return reload;
}

/** Periodic polling that stops on route cleanup and pauses while the tab is hidden. */
export function poll(ctx, fn, ms = 3000, { immediate = false } = {}) {
  let stopped = false; let busy = false;
  const tick = async () => {
    if (stopped || busy || document.hidden) return;
    busy = true;
    try { await fn(); } catch (e) { if (!(e && e.name === 'AbortError')) console.debug('poll error', e); } finally { busy = false; }
  };
  const t = setInterval(tick, ms);
  if (immediate) tick();
  const stop = () => { stopped = true; clearInterval(t); };
  if (ctx && ctx.onCleanup) ctx.onCleanup(stop);
  return stop;
}

// ------------------------------------------------------------------ layout helpers
export function pageHead(title, sub, ...actions) {
  return h('div', { class: 'page-head' }, h('div', { class: 'grow' }, h('h1', title), sub ? h('div', { class: 'sub' }, sub) : null), h('div', { class: 'actions' }, actions));
}
export function card(title, body, actions, cls = '') {
  return h('section', { class: `card ${cls}` }, title || actions ? h('div', { class: 'card-head' }, title ? h('h3', title) : null, actions || null) : null, body);
}
export function stat(label, value, hint, cls = '') {
  return h('div', { class: `stat ${cls}` }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), hint != null ? h('div', { class: 'hint' }, hint) : null);
}
export function kv(entries) {
  const dl = h('dl', { class: 'kv' });
  const list = Array.isArray(entries) ? entries : Object.entries(entries);
  for (const [k, v] of list) {
    if (v === undefined) continue;
    dl.append(h('dt', k), h('dd', v instanceof Node ? v : (v === null || v === '' ? h('span', { class: 'faint' }, '-') : String(v))));
  }
  return dl;
}
export function progressBar(frac, kind = '') {
  const f = isNum(frac) ? Math.max(0, Math.min(1, frac)) : 0;
  const i = h('i', { style: { width: (f * 100).toFixed(1) + '%' } });
  const el = h('div', { class: `bar ${kind}`, role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': Math.round(f * 100) }, i);
  el.set = (x, k) => { const v = Math.max(0, Math.min(1, x || 0)); i.style.width = (v * 100).toFixed(1) + '%'; el.setAttribute('aria-valuenow', Math.round(v * 100)); if (k !== undefined) el.className = `bar ${k}`; };
  return el;
}
export function fold(summary, body, open = false) {
  return h('details', { class: 'fold', open }, h('summary', summary), h('div', { class: 'fold-body' }, body));
}

/** Generic recursive key/value viewer for unknown-shaped objects (outputs, details). */
export function objectView(obj, { depth = 0, maxDepth = 4 } = {}) {
  if (obj == null) return h('span', { class: 'faint' }, '-');
  if (typeof obj !== 'object') {
    if (typeof obj === 'string' && /^(blake3|b2|sha256):[0-9a-f]{16,}$/.test(obj)) return hashLabel(obj, 16);
    return h('span', isNum(obj) ? fmtNum(obj) : String(obj));
  }
  if (Array.isArray(obj)) {
    if (!obj.length) return h('span', { class: 'faint' }, '[]');
    if (obj.every((x) => x == null || typeof x !== 'object')) return h('span', obj.map((x) => (isNum(x) ? fmtNum(x) : String(x))).join(', '));
    if (depth >= maxDepth) return h('code', JSON.stringify(obj).slice(0, 200));
    return h('div', obj.map((x, i) => h('div', { class: 'mt-s' }, h('span', { class: 'faint' }, `#${i} `), objectView(x, { depth: depth + 1, maxDepth }))));
  }
  const entries = Object.entries(obj);
  if (!entries.length) return h('span', { class: 'faint' }, '{}');
  if (depth >= maxDepth) return h('code', JSON.stringify(obj).slice(0, 200));
  if ('value' in obj && ('provenance' in obj || 'ci_low' in obj)) return metricCell(obj);
  return kv(entries.map(([k, v]) => [titleCase(k), objectView(v, { depth: depth + 1, maxDepth })]));
}

// ------------------------------------------------------------------ tabs
export function tabs({ items, active, onChange, label = 'Sections' }) {
  const id = uid('tabs');
  const el = h('div', { class: 'tabs', role: 'tablist', 'aria-label': label });
  const btns = new Map();
  let cur = active || (items[0] && items[0].id);
  const paint = () => { for (const [k, b] of btns) { b.setAttribute('aria-selected', String(k === cur)); b.tabIndex = k === cur ? 0 : -1; } };
  const set = (k, fire = true) => { cur = k; paint(); if (fire && onChange) onChange(k); };
  for (const it of items) {
    const b = h('button', { class: 'tab', role: 'tab', id: `${id}-${it.id}`, onclick: () => set(it.id) }, it.label, it.count != null ? h('span', { class: 'badge count' }, String(it.count)) : null);
    b.addEventListener('keydown', (e) => {
      const ks = items.map((x) => x.id); const i = ks.indexOf(cur);
      if (e.key === 'ArrowRight') { set(ks[(i + 1) % ks.length]); btns.get(cur).focus(); e.preventDefault(); }
      if (e.key === 'ArrowLeft') { set(ks[(i - 1 + ks.length) % ks.length]); btns.get(cur).focus(); e.preventDefault(); }
    });
    btns.set(it.id, b); el.appendChild(b);
  }
  paint();
  el.set = set; el.get = () => cur;
  return el;
}

// ------------------------------------------------------------------ table
/**
 * dataTable({columns:[{key,label,render?(row),num?,sortable?,sortValue?(row)}], rows, onRowClick, empty, sort:{key,dir}, rowClass?})
 * returns element with .setRows(rows)
 */
export function dataTable({ columns, rows = [], onRowClick, empty = 'Nothing here yet.', sort = null, rowClass = null, caption = '' }) {
  let data = rows; let sortKey = sort && sort.key; let dir = (sort && sort.dir) || 'desc';
  const wrap = h('div', { class: 'table-wrap' });
  const draw = () => {
    clear(wrap);
    if (!data.length) { wrap.appendChild(h('div', { class: 'empty', style: 'border:0' }, typeof empty === 'string' ? h('p', empty) : empty)); return; }
    let list = data.slice();
    const col = columns.find((c) => c.key === sortKey);
    if (col) {
      const val = (r) => (col.sortValue ? col.sortValue(r) : r[col.key]);
      list.sort((a, b) => {
        const x = val(a); const y = val(b);
        if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
        const c = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
        return dir === 'asc' ? c : -c;
      });
    }
    const thead = h('thead', h('tr', columns.map((c) => {
      const th = h('th', { class: (c.num ? 'num ' : '') + (c.sortable !== false && !c.noSort ? 'sortable' : ''), scope: 'col', 'aria-sort': c.key === sortKey ? (dir === 'asc' ? 'ascending' : 'descending') : null, tabindex: c.noSort ? null : '0' },
        c.label, c.key === sortKey ? (dir === 'asc' ? ' ▲' : ' ▼') : '');
      if (!c.noSort) {
        const act = () => { if (sortKey === c.key) dir = dir === 'asc' ? 'desc' : 'asc'; else { sortKey = c.key; dir = c.num ? 'desc' : 'asc'; } draw(); };
        th.addEventListener('click', act);
        th.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); } });
      }
      return th;
    })));
    const tbody = h('tbody');
    for (const r of list) {
      const tr = h('tr', { class: (onRowClick ? 'clickable ' : '') + (rowClass ? rowClass(r) || '' : ''), tabindex: onRowClick ? '0' : null });
      if (onRowClick) {
        tr.addEventListener('click', (e) => { if (!e.target.closest('button, a, input, select, .hash')) onRowClick(r); });
        tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target === tr) onRowClick(r); });
      }
      for (const c of columns) {
        const v = c.render ? c.render(r) : r[c.key];
        tr.appendChild(h('td', { class: c.num ? 'num' : '' }, v instanceof Node || Array.isArray(v) ? v : (v == null || v === '' ? h('span', { class: 'faint' }, '-') : String(v))));
      }
      tbody.appendChild(tr);
    }
    wrap.appendChild(h('table', { class: 'tbl' }, caption ? h('caption', { class: 'sr-only' }, caption) : null, thead, tbody));
  };
  draw();
  wrap.setRows = (r) => { data = r; draw(); };
  return wrap;
}

export { fmtTime };
