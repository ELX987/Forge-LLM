// Compare: eval reports side by side with confidence intervals, noise flags, tie-aware ranking and a Pareto chart.
// Statistically indistinguishable values are never ranked against each other: they share a tie group.

import { h, clear, listOf, fmtNum, fmtTime, isNum, fmtRel } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, dataTable, badge, statusBadge, metricCell, loadInto, emptyState, notice, unavailableNotice, hashLabel, toastError, help } from '../components/ui.js';
import { XYChart, paretoFrontier } from '../charts.js';

const asMV = (m) => (m && typeof m === 'object' ? m : isNum(m) ? { value: m } : null);
function defaultDir(name) { return /acc|score|tok_s|throughput|hit|pass|f1|exact|recall|accept|mfu|speedup|compression/i.test(name) ? 'max' : 'min'; }
function hasCI(m) { return m && isNum(m.ci_low) && isNum(m.ci_high); }
function overlap(a, b) { return a.ci_low <= b.ci_high && b.ci_low <= a.ci_high; }

/** Tie-aware ranking. Returns Map(rowId -> {group, tied, known}). */
export function rankWithTies(rows, metric, dir) {
  const vals = rows.map((r) => ({ id: r.id, mv: asMV(r.metrics[metric]) })).filter((x) => x.mv && isNum(x.mv.value));
  vals.sort((a, b) => (dir === 'min' ? a.mv.value - b.mv.value : b.mv.value - a.mv.value));
  const out = new Map(); let group = 0; let leader = null;
  for (const v of vals) {
    if (!leader) { group = 1; leader = v; out.set(v.id, { group, tied: false, known: hasCI(v.mv) }); continue; }
    // indistinguishable when CIs overlap the group leader's, or the server flagged within_noise
    const known = hasCI(v.mv) && hasCI(leader.mv);
    const tie = known ? overlap(v.mv, leader.mv) : false;
    if (tie) { out.set(v.id, { group, tied: true, known }); out.get(leader.id).tied = true; } else { group += 1; leader = v; out.set(v.id, { group, tied: false, known: hasCI(v.mv) }); }
  }
  return out;
}

export default async function render(ctx) {
  const host = h('div');
  ctx.root.append(pageHead('Compare', 'Evaluation reports with confidence intervals, noise floors and gate verdicts. Overlapping intervals share a rank - we do not crown a winner inside the noise.'), host);
  let evals = [];
  const sel = new Set((ctx.query.ids || '').split(',').filter(Boolean));
  let parent = ctx.query.parent || '';
  const resultHost = h('div');

  loadInto(host, async () => { const r = await api.get('/evals'); return listOf(r, 'evals', 'reports'); }, (list) => {
    evals = list;
    if (!list.length) return emptyState('No evaluation reports yet', 'Run an eval job (candidate vs parent) to populate this page.', h('a', { class: 'btn primary', href: '#/jobs/new?type=eval' }, 'New eval job'));
    if (!sel.size) list.slice(0, 8).forEach((e) => sel.add(e.id));
    if (!parent) { const base = list.find((e) => !e.parent_hash); parent = base ? base.id : ''; }
    const table = dataTable({ columns: [
      { key: 'sel', label: '', noSort: true, render: (e) => h('input', { type: 'checkbox', checked: sel.has(e.id), 'aria-label': `Include ${e.label || e.id}`, onchange: (ev) => { if (ev.target.checked) sel.add(e.id); else sel.delete(e.id); } }) },
      { key: 'label', label: 'Report', render: (e) => h('div', h('b', e.label || e.id), h('div', { class: 'small muted mono' }, e.id)) },
      { key: 'artifact_hash', label: 'Artifact', render: (e) => hashLabel(e.artifact_hash) },
      { key: 'gate', label: 'Gate', render: (e) => (e.gate ? h('span', [statusBadge(e.gate.verdict), ' ', h('span', { class: 'small muted' }, e.gate.gate_id)]) : h('span', { class: 'faint' }, '-')), sortValue: (e) => e.gate && e.gate.verdict },
      { key: 'created_at', label: 'When', render: (e) => fmtRel(e.created_at) },
    ], rows: list, sort: { key: 'created_at', dir: 'desc' } });
    const parentSel = h('select', { 'aria-label': 'Reference report', onchange: () => { parent = parentSel.value; } }, h('option', { value: '' }, '(no reference)'), ...list.map((e) => h('option', { value: e.id, selected: e.id === parent }, e.label || e.id)));
    return h('div', { class: 'stack' }, card('Select reports', h('div', table, h('div', { class: 'toolbar mt' }, h('label', 'Reference (parent) '), parentSel, help('Paired comparisons and "within noise" flags are relative to this report.'), h('button', { class: 'btn primary', onclick: () => compare() }, 'Compare selected')))), resultHost);
  }, { feature: 'Evaluation reports' });

  async function compare() {
    const ids = Array.from(sel);
    if (ids.length < 1) { toastError(new Error('Select at least one report.')); return; }
    ctx.replaceQuery({ ids: ids.join(','), parent });
    clear(resultHost); resultHost.appendChild(h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Comparing...'));
    let res = null; let clientSide = false;
    try { res = await api.post('/evals/compare', { ids, parent: parent || undefined }); } catch (e) { if (!e.unavailable) { clear(resultHost); resultHost.appendChild(notice('error', 'Compare failed', e.message)); toastError(e); return; } clientSide = true; }
    let rows;
    if (res && Array.isArray(res.rows || res.candidates || res.table)) rows = (res.rows || res.candidates || res.table).map((r) => ({ id: r.id || r.report_id, label: r.label || r.name || r.id, metrics: r.metrics || {}, gate: r.gate || null, artifact_hash: r.artifact_hash }));
    else { rows = ids.map((id) => evals.find((e) => e.id === id)).filter(Boolean).map((e) => ({ id: e.id, label: e.label || e.id, metrics: e.metrics || {}, gate: e.gate, artifact_hash: e.artifact_hash })); clientSide = true; }
    const metrics = (res && res.metrics && Array.isArray(res.metrics) ? res.metrics : Array.from(new Set(rows.flatMap((r) => Object.keys(r.metrics || {})))));
    const dirs = { ...(res && res.directions) };
    for (const m of metrics) if (!dirs[m]) dirs[m] = defaultDir(m);
    clear(resultHost);
    resultHost.append(...paintResult(rows, metrics, dirs, res && res.noise_floor, clientSide, res && res.pareto));
  }

  function paintResult(rows, metrics, dirs, noise, clientSide, serverPareto) {
    const out = [];
    if (clientSide) out.push(notice('info', 'Computed in the browser', 'The server compare endpoint is not available, so this table is built directly from the stored reports. Ranking still respects confidence intervals.'));
    const ranks = {}; const render = () => { for (const m of metrics) ranks[m] = rankWithTies(rows, m, dirs[m]); };
    render();
    const tableHost = h('div');
    const draw = () => {
      render(); clear(tableHost);
      const cols = [
        { key: 'label', label: 'Candidate', render: (r) => h('div', h('b', r.label), r.artifact_hash ? h('div', hashLabel(r.artifact_hash)) : null) },
        { key: 'gate', label: 'Gate', render: (r) => (r.gate ? h('div', statusBadge(r.gate.verdict), r.gate.reasons && r.gate.reasons.length ? h('div', { class: 'small muted', style: 'max-width:240px' }, r.gate.reasons[0]) : null) : h('span', { class: 'faint' }, '-')), noSort: true },
        ...metrics.map((m) => ({ key: m, label: m, noSort: false, sortValue: (r) => (asMV(r.metrics[m]) || {}).value, render: (r) => {
          const mv = asMV(r.metrics[m]); if (!mv) return h('span', { class: 'faint' }, '-');
          const rk = ranks[m].get(r.id);
          return h('div', metricCell(mv), rk ? h('div', { class: 'small' }, rk.known || rk.group === 1 ? h('span', { class: rk.group === 1 ? 'pareto-yes' : 'muted', 'data-tip': rk.tied ? 'Confidence intervals overlap within this group; the ordering inside it is not statistically meaningful.' : 'Distinguishable from the groups above and below.' }, `rank ${rk.group}${rk.tied ? '= (tied)' : ''}`) : h('span', { class: 'muted', 'data-tip': 'No confidence interval, so we cannot claim this differs from the leader.' }, 'rank n/a (no CI)')) : null);
        } })),
      ];
      tableHost.appendChild(dataTable({ columns: cols, rows, sort: null }));
    };
    const dirRow = h('div', { class: 'toolbar' }, h('span', { class: 'muted small' }, 'Better is: '), ...metrics.map((m) => h('label', { class: 'row tight' }, h('span', { class: 'small' }, m), h('select', { 'aria-label': `Direction for ${m}`, onchange: (e) => { dirs[m] = e.target.value; draw(); drawPareto(); } }, h('option', { value: 'min', selected: dirs[m] === 'min' }, 'lower'), h('option', { value: 'max', selected: dirs[m] === 'max' }, 'higher')))));
    draw();
    out.push(card('Comparison', h('div', dirRow, tableHost, noise && Object.keys(noise).length ? h('p', { class: 'small muted mt' }, 'Noise floors (from repeats of the parent): ', Object.entries(noise).map(([k, v]) => `${k} ±${fmtNum(v)}`).join(', ')) : null)));

    // ---- Pareto
    const paretoHost = h('div'); const list = h('div', { class: 'mt' });
    let xm = metrics.find((m) => /weight|byte|size|bits/i.test(m)) || metrics[0]; let ym = metrics.find((m) => /acc|score/i.test(m)) || metrics[1] || metrics[0];
    const mkSel = (label, getter, setter) => h('label', { class: 'row tight' }, label, h('select', { 'aria-label': label, onchange: (e) => { setter(e.target.value); drawPareto(); } }, metrics.map((m) => h('option', { value: m, selected: m === getter() }, m))));
    let chart = null;
    function drawPareto() {
      if (chart) { chart.destroy(); chart = null; } clear(paretoHost); clear(list);
      const pts = rows.map((r) => ({ r, x: (asMV(r.metrics[xm]) || {}).value, y: (asMV(r.metrics[ym]) || {}).value })).filter((p) => isNum(p.x) && isNum(p.y));
      if (pts.length < 2 || xm === ym) { paretoHost.appendChild(emptyState('Pareto needs two different metrics and at least two candidates')); return; }
      const front = paretoFrontier(pts.map((p) => p.x), pts.map((p) => p.y), dirs[xm], dirs[ym]);
      // flag points whose CI overlaps a frontier point on both axes (cannot be called dominated)
      const near = pts.map((p, i) => { if (front.has(i)) return false; return pts.some((q, j) => front.has(j) && (() => { const a = asMV(p.r.metrics[xm]); const b = asMV(q.r.metrics[xm]); const c = asMV(p.r.metrics[ym]); const d = asMV(q.r.metrics[ym]); return hasCI(a) && hasCI(b) && hasCI(c) && hasCI(d) && overlap(a, b) && overlap(c, d); })()); });
      chart = new XYChart(paretoHost, { title: `Pareto: ${ym} vs ${xm}`, height: 340, xLabel: `${xm} (${dirs[xm] === 'min' ? 'lower' : 'higher'} is better)`, yLabel: `${ym} (${dirs[ym] === 'min' ? 'lower' : 'higher'} is better)`, noLogToggle: false,
        series: [{ name: 'candidates', type: 'scatter', points: pts.map((p) => [p.x, p.y]), labels: pts.map((p) => p.r.label), highlight: Array.from(front), frontier: true, radius: 5, meta: pts.map((p, i) => ({ gate: p.r.gate ? p.r.gate.verdict : '-', note: near[i] ? 'dominated on point estimates, but CIs overlap the frontier' : '' })) }] });
      list.append(h('div', { class: 'small' }, h('b', 'Frontier: '), pts.filter((_, i) => front.has(i)).map((p) => p.r.label).join(', ') || '-'), near.some(Boolean) ? h('div', { class: 'small muted' }, 'Not dominated within noise: ', pts.filter((_, i) => near[i]).map((p) => p.r.label).join(', ')) : null);
    }
    out.push(card('Pareto frontier', h('div', h('div', { class: 'toolbar' }, mkSel('x', () => xm, (v) => { xm = v; }), mkSel('y', () => ym, (v) => { ym = v; })), paretoHost, list)));
    drawPareto();
    void serverPareto;
    return out;
  }
  void fmtTime; void tryGet; void badge; void unavailableNotice;
}
