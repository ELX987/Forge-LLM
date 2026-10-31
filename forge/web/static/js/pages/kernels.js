// Kernel lab: catalog + availability, benchmark results, autotune surface heatmap, roofline residuals, run benchmarks.

import { h, clear, listOf, fmtNum, fmtPct, isNum } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, tabs, dataTable, badge, provBadge, emptyState, notice, unavailableNotice, loadingBlock, toast, hashLabel, kv, help } from '../components/ui.js';
import { BarChart, Heatmap, rooflineChart } from '../charts.js';
import { JobConfigurator } from '../components/jobform.js';
import { fetchJobTypes } from './newjob.js';
import { navigate } from '../router.js';

export default async function render(ctx) {
  const st = { tab: ctx.query.tab || 'catalog', family: '' };
  const content = h('div');
  const bar = tabs({ items: [{ id: 'catalog', label: 'Catalog' }, { id: 'results', label: 'Bench results' }, { id: 'autotune', label: 'Autotune' }, { id: 'roofline', label: 'Roofline & residuals' }, { id: 'run', label: 'Run benchmark' }], active: st.tab, onChange: (t) => { st.tab = t; ctx.replaceQuery({ tab: t }); paint(); } });
  ctx.root.append(pageHead('Kernel lab', 'Every Triton kernel has a pure-torch oracle and a reference fallback. Triton paths are marked "not validated on hardware" until a GPU run records correctness and timing.'), bar, content);
  const charts = []; ctx.onCleanup(() => charts.forEach((c) => c.destroy()));
  const track = (c) => { charts.push(c); return c; };

  clear(content); content.appendChild(loadingBlock());
  const [cat, res, tune, desc] = await Promise.all([tryGet('/kernels'), tryGet('/kernels/results'), tryGet('/kernels/autotune-cache'), tryGet('/hardware/descriptors')]);
  if (!ctx.isActive()) return;
  const catalog = cat.ok ? listOf(cat.data, 'kernels') : []; const results = res.ok ? listOf(res.data, 'results') : [];
  paint();

  function paint() {
    charts.splice(0).forEach((c) => c.destroy());
    clear(content); bar.set(st.tab, false);
    ({ catalog: pCatalog, results: pResults, autotune: pAutotune, roofline: pRoofline, run: pRun })[st.tab]();
  }

  function pCatalog() {
    if (!cat.ok) { content.appendChild(cat.unavailable ? unavailableNotice('Kernel catalog', 'forge.kernels is not installed or GET /kernels is not exposed.') : notice('error', 'Failed', cat.error.message)); return; }
    if (!catalog.length) { content.appendChild(emptyState('No kernels registered')); return; }
    const fams = Array.from(new Set(catalog.map((k) => k.family).filter(Boolean)));
    const famSel = h('select', { 'aria-label': 'Family', onchange: () => { st.family = famSel.value; table.setRows(catalog.filter((k) => !st.family || k.family === st.family)); } }, h('option', { value: '' }, 'All families'), ...fams.map((f) => h('option', { value: f, selected: f === st.family }, f)));
    const table = dataTable({ columns: [
      { key: 'name', label: 'Kernel', render: (k) => h('b', k.name) }, { key: 'family', label: 'Family', render: (k) => badge(k.family || '-', 'dim') },
      { key: 'min_sm', label: 'Min SM', num: true, render: (k) => (k.min_sm != null ? 'sm_' + k.min_sm : '-') },
      { key: 'available', label: 'On this machine', render: (k) => (k.available ? badge('available', 'ok') : h('span', badge('fallback', 'warn', { tip: (k.reasons || []).join('; ') || 'Capability missing; the reference / dequant path is used.' }), ' ', h('span', { class: 'small muted' }, (k.reasons || [])[0] || ''))) },
      { key: 'validated_on_hardware', label: 'Hardware validation', render: (k) => (k.validated_on_hardware ? badge('validated', 'ok') : badge('not validated on hardware', 'warn', { tip: 'Implemented and CPU-tested against the oracle; no GPU run has recorded correctness and timing yet.' })), sortValue: (k) => !!k.validated_on_hardware },
      { key: 'spec', label: 'Spec', render: (k) => h('code', { class: 'small' }, typeof k.spec === 'string' ? k.spec : (k.spec && (k.spec.kernel || k.spec.name)) || '-') },
    ], rows: catalog.filter((k) => !st.family || k.family === st.family), sort: { key: 'family', dir: 'asc' } });
    content.append(h('div', { class: 'toolbar' }, famSel, h('span', { class: 'muted small' }, `${catalog.filter((k) => k.available).length}/${catalog.length} available here`)), table);
  }

  function resultsTable(rows) {
    return dataTable({ columns: [
      { key: 'kernel', label: 'Kernel', render: (r) => h('b', r.kernel) }, { key: 'shape', label: 'Shape' }, { key: 'dtype', label: 'dtype' },
      { key: 'time_us', label: 'Time (us)', num: true, render: (r) => h('span', fmtNum(r.time_us), isNum(r.iqr_us) ? h('span', { class: 'ci' }, ` ±${fmtNum(r.iqr_us)} IQR`) : null) },
      { key: 'achieved_gbs', label: 'GB/s', num: true, render: (r) => fmtNum(r.achieved_gbs) }, { key: 'pct_of_peak', label: '% of peak', num: true, render: (r) => (isNum(r.pct_of_peak) ? fmtPct(r.pct_of_peak, 0) : '-') },
      { key: 'bound', label: 'Bound', render: (r) => (r.bound ? badge(r.bound, r.bound === 'memory' ? 'info' : 'accent') : '-') },
      { key: 'residual_vs_model', label: 'Residual vs model', num: true, render: (r) => (isNum(r.residual_vs_model) ? h('span', { class: Math.abs(r.residual_vs_model) > 0.1 ? 'warn' : '' }, (r.residual_vs_model > 0 ? '+' : '') + fmtPct(r.residual_vs_model, 1)) : '-') },
      { key: 'oracle_ok', label: 'Oracle', render: (r) => (r.oracle_ok === true ? badge('pass', 'ok') : r.oracle_ok === false ? badge('FAIL', 'bad') : '-') },
      { key: 'prov', label: '', noSort: true, render: (r) => provBadge(r.provenance || 'measured') },
    ], rows, sort: null });
  }

  function pResults() {
    if (!res.ok) { content.appendChild(res.unavailable ? unavailableNotice('Kernel results') : notice('error', 'Failed', res.error.message)); return; }
    if (!results.length) { content.appendChild(emptyState('No benchmark results yet', 'Run a kernel benchmark on the GPU host. Results store achieved bandwidth, % of measured peak and the residual against the cost model.', h('button', { class: 'btn primary', onclick: () => { st.tab = 'run'; paint(); } }, 'Run benchmark'))); return; }
    const holder = h('div');
    content.append(card('Achieved % of peak bandwidth', holder), h('div', { class: 'mt' }, resultsTable(results)));
    track(new BarChart(holder, { height: 220, legend: false, categories: results.map((r) => `${r.kernel} ${(r.shape || '').split(',')[0]}`), series: [{ name: '% of peak', values: results.map((r) => (isNum(r.pct_of_peak) ? r.pct_of_peak * 100 : null)) }], yMax: 100, valueFormat: (v) => v.toFixed(0) + '%' }));
  }

  function pAutotune() {
    if (!tune.ok) { content.appendChild(tune.unavailable ? unavailableNotice('Autotune cache') : notice('error', 'Failed', tune.error.message)); return; }
    const entries = Array.isArray(tune.data) ? tune.data : (tune.data && tune.data.entries) || [];
    if (!entries.length) { content.appendChild(emptyState('Autotune cache is empty', 'Entries are written the first time a kernel is tuned for a (chip, driver, op, dtype, shape bucket, layout). Candidates that fail the oracle are never cached.')); return; }
    let ei = 0; let xk = null; let yk = null;
    const body = h('div');
    const entrySel = h('select', { 'aria-label': 'Cache entry', onchange: () => { ei = Number(entrySel.value); xk = yk = null; draw(); } }, entries.map((e, i) => h('option', { value: String(i) }, e.key || `${e.op} ${e.shape_bucket || ''}`)));
    const draw = () => {
      charts.splice(0).forEach((c) => c.destroy()); clear(body);
      const e = entries[ei]; const cands = e.candidates || [];
      const keys = Array.from(new Set(cands.flatMap((c) => Object.entries(c.config || {}).filter(([, v]) => isNum(v)).map(([k]) => k))));
      body.appendChild(card('Best config', h('div', e.best ? kv([['Config', h('code', JSON.stringify(e.best.config))], ['Median time', `${fmtNum(e.best.time_us)} us`], ['Chip', e.chip || '-'], ['Shape bucket', e.shape_bucket || '-'], ['Throttle detected', e.throttle_detected ? badge('yes - re-tune', 'warn') : 'no']]) : h('span', { class: 'muted' }, 'No best candidate recorded.'))));
      if (keys.length < 2 || !cands.length) { body.appendChild(emptyState('No tunable surface', 'Need at least two numeric config parameters with candidates to draw a heatmap.')); return; }
      xk = xk && keys.includes(xk) ? xk : keys[0]; yk = yk && keys.includes(yk) ? yk : keys[1];
      const sel = (label, get, set) => h('label', { class: 'row tight' }, label, h('select', { 'aria-label': label, onchange: (ev) => { set(ev.target.value); draw(); } }, keys.map((k) => h('option', { value: k, selected: k === get() }, k))));
      const xs = Array.from(new Set(cands.map((c) => c.config[xk]))).sort((a, b) => a - b); const ys = Array.from(new Set(cands.map((c) => c.config[yk]))).sort((a, b) => a - b);
      const z = ys.map((y) => xs.map((x) => { const m = cands.filter((c) => c.config[xk] === x && c.config[yk] === y); const ok = m.filter((c) => c.oracle_ok !== false); return ok.length ? Math.min(...ok.map((c) => c.time_us)) : null; }));
      const hol = h('div');
      body.append(h('div', { class: 'toolbar mt' }, sel('X', () => xk, (v) => { xk = v; }), sel('Y', () => yk, (v) => { yk = v; }), help('Each cell is the best time over the other parameters. Blank cells had no candidate or failed the correctness oracle (those are never selected).')), card('Autotune surface (median time, us; lower is better)', hol));
      track(new Heatmap(hol, { height: 320, xLabels: xs, yLabels: ys, z, xTitle: xk, yTitle: yk, lowerIsBetter: true, valueLabel: 'time (us)', valueFormat: (v) => fmtNum(v) + ' us' }));
      const failed = cands.filter((c) => c.oracle_ok === false).length; if (failed) body.appendChild(h('p', { class: 'small muted' }, `${failed} candidate(s) failed the oracle and were excluded.`));
    };
    content.append(h('div', { class: 'toolbar' }, h('label', 'Entry '), entrySel, h('span', { class: 'muted small' }, tune.data && tune.data.path ? tune.data.path : '')), body);
    draw();
  }

  function pRoofline() {
    const d = desc.ok ? listOf(desc.data, 'descriptors')[0] : null;
    const resid = results.filter((r) => isNum(r.residual_vs_model));
    if (d) {
      const bw = d.memory && d.memory.hbm_bw_gbs || {}; const bwV = isNum(bw.measured) ? bw.measured : bw.datasheet;
      const peaks = (d.matrix_units || []).map((m) => { const p = m.peak_tflops || {}; return { name: m.dtype, tflops: isNum(p.measured) ? p.measured : p.datasheet, provenance: isNum(p.measured) ? 'measured' : 'vendor' }; }).filter((p) => isNum(p.tflops));
      const pts = results.filter((r) => isNum(r.ai_flop_per_byte) && isNum(r.achieved_tflops)).map((r) => ({ name: `${r.kernel} ${r.shape || ''}`, ai: r.ai_flop_per_byte, tflops: r.achieved_tflops, meta: { bound: r.bound } }));
      const hol = h('div'); content.appendChild(card(`Roofline: ${d.name}`, hol)); track(rooflineChart(hol, { peaks, bwGBs: bwV || 0, points: pts, height: 340, title: 'Kernels on the roofline' }));
    } else content.appendChild(notice('info', 'No chip descriptor', 'Run the hardware probe to draw a measured roofline.'));
    if (!resid.length) { content.appendChild(emptyState('No residuals yet', 'Residual = (measured - predicted) / predicted for the cost model. Large residuals mean the model is missing a term and the planner should not be trusted for that kernel.')); return; }
    const hol2 = h('div'); content.appendChild(h('div', { class: 'mt' }, card('Cost-model residuals', hol2)));
    track(new BarChart(hol2, { height: 240, legend: false, categories: resid.map((r) => `${r.kernel} ${(r.shape || '').split(',')[0]}`), series: [{ name: 'residual', values: resid.map((r) => r.residual_vs_model * 100) }], valueFormat: (v) => v.toFixed(1) + '%', yLabel: '% vs model' }));
    content.appendChild(h('p', { class: 'small muted' }, 'Positive = slower than the cost model predicted.'));
  }

  async function pRun() {
    content.appendChild(loadingBlock());
    let types; try { types = await fetchJobTypes(); } catch (e) { clear(content); content.appendChild(e.unavailable ? unavailableNotice('Benchmarks') : notice('error', 'Failed', e.message)); return; }
    const jt = types.find((t) => t.type === 'bench');
    clear(content);
    if (!jt) { content.appendChild(unavailableNotice('Kernel benchmarks', 'No "bench" job type is registered on this server.')); return; }
    const host = h('div'); content.appendChild(host);
    const cfg = new JobConfigurator(host, { jobType: jt }); charts.push({ destroy: () => cfg.destroy() });
    content.appendChild(h('div', { class: 'row mt' }, h('button', { class: 'btn primary', onclick: async (e) => { e.target.disabled = true; try { const job = await cfg.submit(); if (job && job.id) { toast('Benchmark queued', 'ok'); navigate(`/jobs/${job.id}`); } } catch (_) { /* painted */ } finally { e.target.disabled = false; } } }, 'Run benchmark'), h('span', { class: 'muted small' }, 'GPU benchmarks must run on the NVIDIA host. On CPU they exercise the reference path only.')));
  }
  void api; void hashLabel;
}
