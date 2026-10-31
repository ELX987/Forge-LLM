// Hardware: measured chip descriptors (with datasheet-vs-measured provenance), probe, roofline plot, support matrix, known chips.

import { h, listOf, fmtNum, isNum, clear, normGpus } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, tabs, loadInto, dataTable, badge, provBadge, kv, emptyState, notice, unavailableNotice, toastError, toast, statusBadge, stat } from '../components/ui.js';
import { rooflineChart } from '../charts.js';
import { navigate } from '../router.js';

export default async function render(ctx) {
  const content = h('div');
  const state = { tab: ctx.query.tab || 'descriptor', chips: null, desc: [], kernels: [], gpus: [] };
  const tabBar = tabs({ items: [{ id: 'descriptor', label: 'Descriptor' }, { id: 'roofline', label: 'Roofline' }, { id: 'matrix', label: 'Support matrix' }, { id: 'chips', label: 'Known chips' }], active: state.tab, onChange: (t) => { state.tab = t; ctx.replaceQuery({ tab: t }); paint(); } });
  const probeBtn = h('button', { class: 'btn primary', onclick: () => probe(false) }, 'Run hardware probe');
  ctx.root.append(pageHead('Hardware', 'Chip descriptors: datasheet values side by side with what the Forge microbenchmarks actually measured. Native features are hypotheses until the probe confirms them.',
    h('button', { class: 'btn', onclick: () => probe(true), title: 'Skips the long micro-benchmarks' }, 'Quick probe'), probeBtn), tabBar, content);

  async function probe(quick) {
    try { const job = await api.post('/hardware/probe', { quick }); toast('Probe queued', 'ok'); if (job && job.id) navigate(`/jobs/${job.id}`); } catch (e) { if (e.unavailable) toast('Hardware probe is not available on this server.', 'warn'); else toastError(e, 'Probe'); }
  }

  clear(content); content.appendChild(h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Loading hardware info...'));
  const [chips, desc, kern, gpus] = await Promise.all([tryGet('/hardware/chips'), tryGet('/hardware/descriptors'), tryGet('/kernels/results'), tryGet('/system/gpus')]);
  if (!ctx.isActive()) return;
  state.chips = chips; state.descRes = desc; state.kernels = kern.ok ? listOf(kern.data, 'results') : []; state.gpus = gpus.ok ? normGpus(gpus.data) : [];
  state.desc = desc.ok ? listOf(desc.data, 'descriptors') : [];
  state.sel = state.desc[0] ? state.desc[0].id : null;
  paint();

  function paint() {
    clear(content);
    tabBar.set(state.tab, false);
    if (state.tab === 'descriptor') content.appendChild(descriptorView());
    else if (state.tab === 'roofline') content.appendChild(rooflineView());
    else if (state.tab === 'matrix') content.appendChild(matrixView());
    else content.appendChild(chipsView());
  }

  function currentDesc() { return state.desc.find((d) => d.id === state.sel) || state.desc[0] || null; }

  function descriptorView() {
    const wrap = h('div', { class: 'stack' });
    if (state.gpus.length) wrap.appendChild(card('Detected GPUs', h('div', { class: 'chips' }, state.gpus.map((g) => h('span', { class: 'chip' }, `GPU ${g.index}: ${g.name}`)))));
    if (!state.descRes.ok) { wrap.appendChild(state.descRes.unavailable ? unavailableNotice('Hardware descriptors') : notice('error', 'Could not load descriptors', state.descRes.error.message)); return wrap; }
    if (!state.desc.length) { wrap.appendChild(emptyState('No measured descriptor yet', 'Run the hardware probe on the GPU host to create one. Until then, the planner and kernel autotuner use datasheet values (tagged "vendor").', h('button', { class: 'btn primary', onclick: () => probe(false) }, 'Run hardware probe'))); return wrap; }
    const d = currentDesc();
    const sel = h('select', { 'aria-label': 'Descriptor', onchange: (e) => { state.sel = e.target.value; paint(); } }, state.desc.map((x) => h('option', { value: x.id, selected: x.id === d.id }, `${x.name} (${x.id})`)));
    const pp = d.probe_provenance || {};
    const pair = (p, key) => {
      if (!p) return '-';
      const m = p.measured; const v = p.datasheet;
      return h('span', isNum(m) ? [h('b', fmtNum(m)), ' ', provBadge(pp[key] || 'measured')] : '', isNum(m) && isNum(v) ? ' ' : '', isNum(v) ? h('span', { class: 'muted' }, [`(datasheet ${fmtNum(v)} `, provBadge('vendor'), ')']) : '');
    };
    wrap.append(
      h('div', { class: 'toolbar' }, h('label', 'Descriptor '), sel, badge('tier ' + (d.tier || '?'), d.tier === 'A' ? 'ok' : d.tier === 'B' ? 'info' : 'warn', { tip: 'A: native matrix/low-bit paths verified by probe. B: capable but partly unverified. C: portable fallback.' })),
      h('div', { class: 'grid cols-4' },
        card('', stat('SM / arch', d.sm != null ? `sm_${d.sm}` : '-', `${d.family || ''} ${d.compute_capability || ''}`)),
        card('', stat('Compute units', fmtNum(d.compute_units), d.clock_mhz ? d.clock_mhz + ' MHz' : '')),
        card('', stat('HBM', d.memory && d.memory.hbm_gb != null ? d.memory.hbm_gb + ' GB' : '-', d.memory && d.memory.l2_mb ? `L2 ${d.memory.l2_mb} MB` : '')),
        card('', stat('Launch latency', isNum(d.launch_latency_us) ? d.launch_latency_us + ' us' : '-', provBadge(pp.launch_latency_us)))),
      h('div', { class: 'grid cols-2' },
        card('Memory & on-chip', kv([
          ['HBM bandwidth (GB/s)', pair(d.memory && d.memory.hbm_bw_gbs, 'memory.hbm_bw_gbs')],
          ['Transaction bytes', d.memory && d.memory.transaction_bytes], ['Max vector load', d.memory && d.memory.max_vector_load_bytes && d.memory.max_vector_load_bytes + ' B'],
          ['Shared mem / SM', d.on_chip && d.on_chip.smem_per_cu_kib != null ? d.on_chip.smem_per_cu_kib + ' KiB' : '-'], ['SMEM banks', d.on_chip && d.on_chip.smem_banks && `${d.on_chip.smem_banks} x ${d.on_chip.bank_width_bytes} B`],
          ['Registers / SM', d.on_chip && fmtNum(d.on_chip.regs_per_cu)], ['Async copy', (d.on_chip && d.on_chip.async_copy || []).join(', ') || '-'],
          ['2:4 sparsity', d.sparsity && d.sparsity.two_four ? 'yes' : 'no'], ['Power limit', d.power_limit_w ? d.power_limit_w + ' W' : '-'],
          ['Interconnect', d.interconnect && d.interconnect.intra_gbs ? `${d.interconnect.intra_gbs} GB/s ${d.interconnect.topology || ''}` : '-']])),
        card('Ridge points (FLOP/byte)', d.derived && d.derived.ridge_flop_per_byte ? dataTable({ columns: [{ key: 'dtype', label: 'dtype' }, { key: 'v', label: 'ridge', num: true, render: (r) => fmtNum(r.v) }, { key: 'p', label: '', render: () => provBadge('derived'), noSort: true }], rows: Object.entries(d.derived.ridge_flop_per_byte).map(([dtype, v]) => ({ dtype, v })), sort: null }) : h('p', { class: 'muted' }, 'Not derived yet.'))),
      card('Matrix units', dataTable({
        columns: [{ key: 'dtype', label: 'dtype', render: (m) => h('b', m.dtype) }, { key: 'atom', label: 'MMA atom', render: (m) => (m.atom ? m.atom.join('x') : '-') },
          { key: 'peak', label: 'Peak TFLOP/s', render: (m) => pair(m.peak_tflops, `matrix_units.${m.dtype}`), noSort: true }, { key: 'bs', label: 'Block-scaled', render: (m) => (m.block_scaled ? 'yes' : 'no') }],
        rows: d.matrix_units || [], sort: null, empty: 'No matrix units recorded.' })));
    return wrap;
  }

  function rooflineView() {
    const d = currentDesc();
    if (!d) return emptyState('Roofline needs a chip descriptor', 'Run the probe, then come back. Datasheet-only rooflines from the Known chips table are shown below.', chipRoofline());
    const bw = (d.memory && d.memory.hbm_bw_gbs) || {};
    const bwV = isNum(bw.measured) ? bw.measured : bw.datasheet;
    const peaks = (d.matrix_units || []).map((m) => {
      const p = m.peak_tflops || {};
      return { name: m.dtype, tflops: isNum(p.measured) ? p.measured : p.datasheet, provenance: isNum(p.measured) ? 'measured' : 'vendor' };
    }).filter((p) => isNum(p.tflops));
    const pts = state.kernels.filter((r) => isNum(r.ai_flop_per_byte) && isNum(r.achieved_tflops)).map((r) => ({ name: `${r.kernel} ${r.shape || ''}`.trim(), ai: r.ai_flop_per_byte, tflops: r.achieved_tflops, meta: { bound: r.bound, '% of peak': isNum(r.pct_of_peak) ? (r.pct_of_peak * 100).toFixed(0) + '%' : '-' } }));
    const holder = h('div');
    const out = h('div', { class: 'stack' },
      card(`Roofline: ${d.name}`, holder),
      h('p', { class: 'muted small' }, `Memory bandwidth ${isNum(bwV) ? fmtNum(bwV) + ' GB/s' : 'unknown'} (${isNum(bw.measured) ? 'measured' : 'datasheet'}). Dashed roofs use datasheet peaks; solid use measured peaks. Points are kernel benchmark results (measured) from the Kernel Lab.`));
    setTimeout(() => rooflineChart(holder, { peaks, bwGBs: bwV || 0, points: pts, height: 380, title: 'Attainable performance vs arithmetic intensity' }), 0);
    if (!pts.length) out.appendChild(notice('info', 'No kernel results plotted', 'Run a kernel benchmark in the Kernel Lab to overlay achieved performance on the roofline.'));
    return out;
  }

  function chipRoofline() {
    const list = chipList(); const c = list.find((x) => x.tflops && x.hbm_bw_gbs); if (!c) return null;
    const holder = h('div');
    setTimeout(() => rooflineChart(holder, { peaks: Object.entries(c.tflops).filter(([, v]) => isNum(v)).map(([k, v]) => ({ name: k, tflops: v, provenance: 'vendor' })), bwGBs: c.hbm_bw_gbs, height: 300, title: `${c.name} (datasheet)` }), 0);
    return holder;
  }

  function chipList() { const c = state.chips && state.chips.ok ? state.chips.data : null; return Array.isArray(c) ? c : ((c && (c.chips || c.known)) || []); }
  function matrixList() { const c = state.chips && state.chips.ok ? state.chips.data : null; return (c && (c.support_matrix || c.matrix)) || []; }

  function matrixView() {
    if (!state.chips.ok) return state.chips.unavailable ? unavailableNotice('Hardware catalogue') : notice('error', 'Failed', state.chips.error.message);
    const rows = matrixList();
    if (!rows.length) return emptyState('No support matrix reported');
    const yn = (v) => (v === true ? badge('yes', 'ok') : v === false ? badge('no', 'dim') : badge(String(v), 'warn'));
    return h('div', { class: 'stack' }, dataTable({ columns: [
      { key: 'sm', label: 'SM', render: (r) => 'sm_' + r.sm }, { key: 'examples', label: 'Examples' }, { key: 'bf16', label: 'bf16', render: (r) => yn(r.bf16) }, { key: 'fp8', label: 'fp8 MMA', render: (r) => yn(r.fp8) },
      { key: 'fp4_block_scaled', label: 'FP4 block-scaled', render: (r) => yn(r.fp4_block_scaled) }, { key: 'cp_async', label: 'cp.async', render: (r) => yn(r.cp_async) }, { key: 'tma', label: 'TMA', render: (r) => yn(r.tma) }, { key: 'two_four', label: '2:4 sparse', render: (r) => yn(r.two_four) }, { key: 'tier', label: 'Tier', render: (r) => badge(r.tier || '?', r.tier === 'A' ? 'ok' : r.tier === 'B' ? 'info' : 'warn') },
    ], rows, sort: { key: 'sm', dir: 'asc' } }), h('p', { class: 'muted small' }, 'Everything "native" is a hypothesis until the probe confirms it (Plan HW-2). Unknown or older GPUs use the Tier C portable fallback.'));
  }

  function chipsView() {
    if (!state.chips.ok) return state.chips.unavailable ? unavailableNotice('Hardware catalogue') : notice('error', 'Failed', state.chips.error.message);
    const list = chipList();
    if (!list.length) return emptyState('No chips in the catalogue');
    return h('div', { class: 'stack' }, dataTable({ columns: [
      { key: 'name', label: 'Chip', render: (c) => h('b', c.name) }, { key: 'sm', label: 'SM', render: (c) => (c.sm != null ? 'sm_' + c.sm : '-') }, { key: 'family', label: 'Family' },
      { key: 'hbm_gb', label: 'HBM (GB)', num: true, render: (c) => fmtNum(c.hbm_gb) }, { key: 'hbm_bw_gbs', label: 'BW (GB/s)', num: true, render: (c) => fmtNum(c.hbm_bw_gbs) },
      { key: 'bf16', label: 'BF16 TFLOP/s', num: true, render: (c) => fmtNum(c.tflops && c.tflops.bf16), sortValue: (c) => c.tflops && c.tflops.bf16 }, { key: 'fp8', label: 'FP8', num: true, render: (c) => fmtNum(c.tflops && c.tflops.fp8), sortValue: (c) => c.tflops && c.tflops.fp8 },
      { key: 'fp4', label: 'FP4', num: true, render: (c) => fmtNum(c.tflops && c.tflops.fp4), sortValue: (c) => c.tflops && c.tflops.fp4 }, { key: 'prov', label: '', render: () => provBadge('vendor'), noSort: true },
    ], rows: list, sort: { key: 'sm', dir: 'asc' } }), h('p', { class: 'muted small' }, 'Datasheet peak numbers (vendor). Achieved throughput is always lower - use the probe.'));
  }
  void loadInto; void statusBadge;
}
