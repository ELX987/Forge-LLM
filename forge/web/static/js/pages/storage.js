// Storage explorer: bytes breakdown, tiers, artifacts (pin / tier), lineage DAG (SVG, layered layout), GC dry-run -> run.

import { h, clear, listOf, fmtBytes, fmtNum, fmtRel, isNum, shortHash } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, stat, tabs, dataTable, badge, hashLabel, loadInto, emptyState, notice, unavailableNotice, confirmDialog, toast, toastError, kv, provBadge, progressBar, loadingBlock, help } from '../components/ui.js';
import { BarChart } from '../charts.js';

const KIND_COLORS = { model: 'var(--c1)', checkpoint: 'var(--c2)', adapter: 'var(--c5)', dataset: 'var(--c3)', calib: 'var(--c7)', eval: 'var(--c4)', blob: 'var(--faint)' };

export function layoutDag(nodes, edges) {
  const byId = new Map(nodes.map((n) => [n.id, { ...n, parents: [], children: [] }]));
  for (const e of edges) { const c = byId.get(e.child); const p = byId.get(e.parent); if (c && p && c !== p) { c.parents.push(p.id); p.children.push(c.id); } }
  // longest-path layering (guard against cycles with a visit stack)
  const layer = new Map(); const visiting = new Set();
  const depth = (id) => {
    if (layer.has(id)) return layer.get(id);
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const n = byId.get(id); let d = 0;
    for (const p of n.parents) d = Math.max(d, depth(p) + 1);
    visiting.delete(id); layer.set(id, d); return d;
  };
  for (const id of byId.keys()) depth(id);
  const cols = new Map();
  for (const [id, l] of layer) { if (!cols.has(l)) cols.set(l, []); cols.get(l).push(id); }
  const maxL = Math.max(0, ...cols.keys());
  const W = 170; const Hh = 50; const GX = 70; const GY = 22;
  const pos = new Map();
  for (let l = 0; l <= maxL; l++) {
    const ids = cols.get(l) || [];
    if (l > 0) ids.sort((a, b) => bary(a, pos, byId) - bary(b, pos, byId));
    ids.forEach((id, i) => pos.set(id, { x: l * (W + GX) + 10, y: i * (Hh + GY) + 10, i }));
  }
  const height = Math.max(...Array.from(cols.values()).map((c) => c.length), 1) * (Hh + GY) + 10;
  return { byId, pos, width: (maxL + 1) * (W + GX) + 10, height, W, Hh };
}
function bary(id, pos, byId) { const ps = byId.get(id).parents.map((p) => (pos.get(p) || { y: 0 }).y); return ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : 0; }

function normEdges(raw) {
  return (raw || []).map((e) => ({ child: e.child ?? e.to ?? e.target ?? e.dst, parent: e.parent ?? e.from ?? e.source ?? e.src, relation: e.relation || e.kind || 'derived_from', recipe_hash: e.recipe_hash })).filter((e) => e.child && e.parent);
}

export default async function render(ctx) {
  const st = { tab: ctx.query.tab || (ctx.query.hash ? 'lineage' : 'overview'), kind: '', hash: ctx.query.hash || '', artifacts: [] };
  const content = h('div');
  const bar = tabs({ items: [{ id: 'overview', label: 'Overview' }, { id: 'artifacts', label: 'Artifacts' }, { id: 'lineage', label: 'Lineage' }, { id: 'gc', label: 'Garbage collection' }], active: st.tab, onChange: (t) => { st.tab = t; ctx.replaceQuery({ tab: t, hash: st.hash }); paint(); } });
  ctx.root.append(pageHead('Storage', 'Content-addressed, chunk-deduplicated artifact store with hot / warm / cold tiers and delta-chained checkpoints.'), bar, content);
  const charts = []; ctx.onCleanup(() => charts.forEach((c) => c.destroy()));

  clear(content); content.appendChild(loadingBlock());
  const [stats, tiers, arts] = await Promise.all([tryGet('/storage/stats'), tryGet('/storage/tiers'), tryGet('/artifacts')]);
  if (!ctx.isActive()) return;
  st.statsRes = stats; st.tiersRes = tiers; st.artsRes = arts; st.artifacts = arts.ok ? listOf(arts.data, 'artifacts') : [];
  paint();

  function paint() { charts.splice(0).forEach((c) => c.destroy()); clear(content); bar.set(st.tab, false); ({ overview, artifacts, lineage, gc })[st.tab](); }

  // ---------------------------------------------------------------- overview
  function overview() {
    if (!stats.ok) { content.appendChild(stats.unavailable ? unavailableNotice('Storage statistics') : notice('error', 'Failed', stats.error.message)); return; }
    const s = stats.data; const logical = s.logical_bytes; const uniq = s.unique_bytes ?? s.stored_bytes;
    const ratio = s.dedup_ratio ?? (logical && uniq ? logical / uniq : null); const naive = s.naive_bytes;
    content.append(h('div', { class: 'grid cols-4' },
      card('', stat('Logical size', fmtBytes(logical), 'sum of artifact sizes')), card('', stat('Unique on disk', fmtBytes(uniq), 'after chunk dedup')),
      card('', stat('Dedup ratio', isNum(ratio) ? ratio.toFixed(2) + 'x' : '-', h('span', [s.provenance ? provBadge(s.provenance) : null]))), card('', stat('vs naive copies', isNum(s.compaction_vs_naive) ? s.compaction_vs_naive.toFixed(2) + 'x' : (isNum(s.compaction_ratio) ? s.compaction_ratio.toFixed(2) + 'x' : (isNum(naive) && isNum(uniq) && uniq > 0 ? (naive / uniq).toFixed(2) + 'x' : '-')), s.algo ? 'hash: ' + s.algo : ''))));
    const kinds = s.per_kind || {};
    const names = Object.keys(kinds);
    const h1 = h('div'); const h2 = h('div');
    content.appendChild(h('div', { class: 'grid cols-2 mt' }, card('', h1), card('', h2)));
    if (names.length) charts.push(new BarChart(h1, { title: 'Bytes by artifact kind', height: 260, categories: names, valueFormat: fmtBytes, series: [{ name: 'logical', values: names.map((k) => kinds[k].logical_bytes) }, { name: 'unique on disk', values: names.map((k) => kinds[k].unique_bytes) }] }));
    else h1.appendChild(emptyState('No per-kind breakdown'));
    const tierData = s.per_tier || (tiers.ok ? Object.fromEntries(listOf(tiers.data, 'tiers').map((t) => [t.name, t.bytes])) : null);
    if (tierData && Object.keys(tierData).length) charts.push(new BarChart(h2, { title: 'Bytes by tier', height: 260, legend: false, categories: Object.keys(tierData), valueFormat: fmtBytes, series: [{ name: 'bytes', values: Object.values(tierData) }] }));
    else h2.appendChild(emptyState('No tier data'));
    if (tiers.ok) {
      const tl = listOf(tiers.data, 'tiers');
      if (tl.length) content.appendChild(h('div', { class: 'mt' }, card('Tiers', dataTable({ columns: [{ key: 'name', label: 'Tier', render: (t) => h('b', t.name) }, { key: 'path', label: 'Path', render: (t) => h('code', t.path || '-') }, { key: 'bytes', label: 'Used', num: true, render: (t) => fmtBytes(t.bytes) },
        { key: 'cap', label: 'Capacity', num: true, render: (t) => (t.capacity_bytes ? fmtBytes(t.capacity_bytes) : '-') }, { key: 'use', label: 'Usage', noSort: true, render: (t) => (t.capacity_bytes ? h('div', { style: 'min-width:120px' }, progressBar(t.bytes / t.capacity_bytes)) : '-') }], rows: tl, sort: null }))));
    }
  }

  // ---------------------------------------------------------------- artifacts
  function artifacts() {
    if (!arts.ok) { content.appendChild(arts.unavailable ? unavailableNotice('Artifacts') : notice('error', 'Failed', arts.error.message)); return; }
    const kinds = Array.from(new Set(st.artifacts.map((a) => a.kind)));
    const host = h('div');
    const kindSel = h('select', { 'aria-label': 'Kind', onchange: () => { st.kind = kindSel.value; draw(); } }, h('option', { value: '' }, 'All kinds'), ...kinds.map((k) => h('option', { value: k, selected: k === st.kind }, k)));
    const draw = () => {
      clear(host);
      const rows = st.artifacts.filter((a) => !st.kind || a.kind === st.kind);
      if (!rows.length) { host.appendChild(emptyState('No artifacts')); return; }
      host.appendChild(dataTable({ columns: [
        { key: 'name', label: 'Artifact', render: (a) => h('div', h('b', a.name || a.kind), h('div', hashLabel(a.hash, 16))) }, { key: 'kind', label: 'Kind', render: (a) => badge(a.kind, 'dim') },
        { key: 'size', label: 'Logical', num: true, render: (a) => fmtBytes(a.size) }, { key: 'stored_size', label: 'On disk', num: true, render: (a) => fmtBytes(a.stored_size) },
        { key: 'tier', label: 'Tier', render: (a) => h('select', { 'aria-label': 'Tier', onchange: (e) => setTier(a, e.target.value) }, ['hot', 'warm', 'cold'].map((t) => h('option', { value: t, selected: t === a.tier }, t))), sortValue: (a) => a.tier },
        { key: 'pinned', label: 'Pinned', render: (a) => h('button', { class: 'btn sm' + (a.pinned ? ' primary' : ''), 'aria-pressed': String(!!a.pinned), onclick: () => pin(a) }, a.pinned ? 'pinned' : 'pin') },
        { key: 'created_at', label: 'Created', render: (a) => fmtRel(a.created_at) },
        { key: 'act', label: '', noSort: true, render: (a) => h('button', { class: 'btn sm', onclick: () => { st.hash = a.hash; st.tab = 'lineage'; ctx.replaceQuery({ tab: 'lineage', hash: a.hash }); paint(); } }, 'Lineage') },
      ], rows, sort: { key: 'created_at', dir: 'desc' } }));
    };
    content.append(h('div', { class: 'toolbar' }, kindSel), host); draw();
  }
  async function pin(a) { try { await api.post(`/artifacts/${encodeURIComponent(a.hash)}/${a.pinned ? 'unpin' : 'pin'}`, a.pinned ? {} : { reason: 'pinned from UI' }); a.pinned = !a.pinned; toast(a.pinned ? 'Pinned (protected from GC)' : 'Unpinned', 'ok'); paint(); } catch (e) { toastError(e); } }
  async function setTier(a, to) { try { await api.post(`/artifacts/${encodeURIComponent(a.hash)}/tier`, { to }); a.tier = to; toast(`Moved to ${to}`, 'ok'); } catch (e) { toastError(e); paint(); } }

  // ---------------------------------------------------------------- lineage
  async function lineage() {
    const sel = h('select', { 'aria-label': 'Artifact', onchange: () => { st.hash = sel.value; ctx.replaceQuery({ tab: 'lineage', hash: st.hash }); drawGraph(); } },
      h('option', { value: '' }, 'Choose an artifact...'), ...st.artifacts.map((a) => h('option', { value: a.hash, selected: a.hash === st.hash }, `${a.name || a.kind} - ${shortHash(a.hash, 8)}`)));
    const manual = h('input', { type: 'text', placeholder: 'or paste a hash', 'aria-label': 'Artifact hash', style: 'min-width:260px', onchange: () => { st.hash = manual.value.trim(); drawGraph(); } });
    const graph = h('div'); const side = h('div');
    content.append(h('div', { class: 'toolbar' }, sel, manual, help('Edges point from parent to child (derived_from). Click a node for details.')), h('div', { class: 'grid split' }, graph, side));
    async function drawGraph() {
      clear(graph); clear(side);
      if (!st.hash) { graph.appendChild(emptyState('Pick an artifact', 'The lineage graph shows how it was derived: recipe, parents, calibration data and eval reports.')); return; }
      graph.appendChild(loadingBlock('Loading lineage...'));
      try {
        const lin = await api.get(`/artifacts/${encodeURIComponent(st.hash)}/lineage`);
        clear(graph);
        const nodes = (lin.nodes || []).map((n) => (typeof n === 'string' ? { id: n, hash: n } : { ...n, id: n.id || n.hash })); const edges = normEdges(lin.edges);
        if (!nodes.length) { graph.appendChild(emptyState('No lineage recorded', 'This artifact has no recorded parents or children.')); return; }
        const L = layoutDag(nodes, edges);
        const svg = h('svg:svg', { width: L.width, height: L.height, viewBox: `0 0 ${L.width} ${L.height}`, role: 'img', 'aria-label': 'Artifact lineage graph' });
        const defs = h('svg:defs', h('svg:marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, h('svg:path', { d: 'M0 0L10 5L0 10z' })));
        svg.appendChild(defs);
        for (const e of edges) {
          const a = L.pos.get(e.parent); const b = L.pos.get(e.child); if (!a || !b) continue;
          const x1 = a.x + L.W; const y1 = a.y + L.Hh / 2; const x2 = b.x; const y2 = b.y + L.Hh / 2; const mx = (x1 + x2) / 2;
          svg.appendChild(h('svg:path', { class: 'edge', d: `M${x1} ${y1}C${mx} ${y1} ${mx} ${y2} ${x2 - 2} ${y2}`, 'marker-end': 'url(#arrow)' }, h('svg:title', e.relation)));
        }
        let selected = null;
        for (const n of nodes) {
          const p = L.pos.get(n.id); if (!p) continue;
          const art = st.artifacts.find((a) => a.hash === n.hash || a.hash === n.id) || {};
          const g = h('svg:g', { class: 'node' + (n.id === st.hash ? ' sel' : ''), transform: `translate(${p.x},${p.y})`, tabindex: '0', role: 'button', 'aria-label': `${n.label || art.name || n.kind} ${n.kind || ''}` },
            h('svg:rect', { width: L.W, height: L.Hh, rx: 7 }), h('svg:rect', { width: 5, height: L.Hh, rx: 2, style: `fill:${KIND_COLORS[n.kind || art.kind] || 'var(--faint)'};stroke:none` }),
            h('svg:text', { x: 14, y: 20 }, trunc(n.label || art.name || shortHash(n.hash || n.id, 12), 24)), h('svg:text', { x: 14, y: 38, style: 'fill:var(--muted)' }, `${n.kind || art.kind || ''} ${shortHash(n.hash || n.id, 8)}`));
          const choose = () => { svg.querySelectorAll('.node.sel').forEach((x) => x.classList.remove('sel')); g.classList.add('sel'); selected = n; showNode(n, art, edges); };
          g.addEventListener('click', choose); g.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); choose(); } });
          svg.appendChild(g);
        }
        graph.appendChild(h('div', { class: 'lineage' }, svg));
        const root = nodes.find((n) => n.id === st.hash || n.hash === st.hash) || nodes[0]; showNode(root, st.artifacts.find((a) => a.hash === root.hash) || {}, edges); void selected;
      } catch (e) { clear(graph); graph.appendChild(e.unavailable ? unavailableNotice('Lineage') : e.status === 404 ? emptyState('Artifact not found', st.hash) : notice('error', 'Lineage failed', e.message)); }
    }
    function showNode(n, art, edges) {
      clear(side);
      const parents = edges.filter((e) => e.child === n.id).map((e) => e.parent); const children = edges.filter((e) => e.parent === n.id).map((e) => e.child);
      side.appendChild(card('Node', kv([['Name', n.label || art.name || '-'], ['Kind', n.kind || art.kind || '-'], ['Hash', hashLabel(n.hash || n.id, 20)], ['Size', art.size ? fmtBytes(art.size) : '-'], ['Tier', art.tier || '-'], ['Recipe', art.recipe_hash ? hashLabel(art.recipe_hash) : '-'],
        ['Parents', parents.length ? h('div', parents.map((p) => h('div', hashLabel(p)))) : 'none (root)'], ['Children', children.length ? String(children.length) : 'none']])));
      if (n.id !== st.hash) side.appendChild(h('button', { class: 'btn mt', onclick: () => { st.hash = n.id; ctx.replaceQuery({ tab: 'lineage', hash: st.hash }); drawGraph(); } }, 'Re-centre on this node'));
    }
    await drawGraph();
  }
  const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

  // ---------------------------------------------------------------- GC
  function gc() {
    const out = h('div'); let dryDone = null;
    const dryBtn = h('button', { class: 'btn primary', onclick: dry }, 'Dry run');
    const runBtn = h('button', { class: 'btn danger', disabled: true, onclick: real }, 'Run GC for real');
    content.append(card('Mark-and-sweep garbage collection', h('div', h('p', { class: 'muted' }, 'Reachability is computed from pinned artifacts. Delta-chain keyframes that still have live deltas are never collected. Always dry-run first; the real run is only enabled afterwards.'), h('div', { class: 'row' }, dryBtn, runBtn))), h('div', { class: 'mt' }, out));
    function report(r) {
      clear(out);
      const cands = r.candidates || r.items || r.unreachable || [];
      out.append(card(r.dry_run === false ? 'GC completed' : 'Dry-run result (nothing deleted)', h('div', h('div', { class: 'grid cols-3' }, stat('Candidates', String(cands.length || r.count || 0)), stat('Would free', fmtBytes(r.bytes_freed ?? r.freed_bytes ?? 0)), stat('Kept keyframes', String(r.kept_keyframes ?? '-'))),
        cands.length ? dataTable({ columns: [{ key: 'hash', label: 'Hash', render: (c) => hashLabel(c.hash || c) }, { key: 'kind', label: 'Kind', render: (c) => c.kind || '-' }, { key: 'size', label: 'Size', num: true, render: (c) => (c.size ? fmtBytes(c.size) : '-') }, { key: 'reason', label: 'Why' }], rows: cands.map((c) => (typeof c === 'string' ? { hash: c } : c)), sort: null }) : emptyState('Nothing to collect', 'Every stored object is reachable from a pin.'),
        (r.notes || []).map((n) => h('p', { class: 'small muted' }, n)))));
    }
    async function dry() { dryBtn.disabled = true; clear(out); out.appendChild(loadingBlock('Scanning store...')); try { dryDone = await api.post('/storage/gc', { dry_run: true }); report(dryDone); runBtn.disabled = !((dryDone.candidates || dryDone.items || []).length || dryDone.bytes_freed); } catch (e) { clear(out); out.appendChild(e.unavailable ? unavailableNotice('Garbage collection') : notice('error', 'GC dry-run failed', e.message)); } finally { dryBtn.disabled = false; } }
    async function real() {
      if (!(await confirmDialog({ title: 'Delete unreachable objects?', message: `This permanently deletes ${(dryDone.candidates || []).length} object(s) (~${fmtBytes(dryDone.bytes_freed || 0)}). Pinned artifacts are safe. This cannot be undone.`, confirmLabel: 'Run GC', danger: true }))) return;
      runBtn.disabled = true; try { const r = await api.post('/storage/gc', { dry_run: false }); report({ ...r, dry_run: false }); toast('Garbage collection finished', 'ok'); } catch (e) { toastError(e, 'GC'); }
    }
  }
}
