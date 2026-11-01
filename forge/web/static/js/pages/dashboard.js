// Dashboard: live GPUs, active/queued jobs, gates, storage compaction, claims needing attention.

import { h, clear, fmtBytes, fmtPct, listOf, isNum, normGpus } from '../util.js';
import { api, sse, tryGet } from '../api.js';
import { cudaBanner, pageHead, card, stat, loadInto, poll, statusBadge, dataTable, progressBar, emptyState, notice, unavailableNotice, badge, provBadge } from '../components/ui.js';
import { sparkline } from '../charts.js';

const cudaReport = async (refresh) => { const r = await tryGet('/system/cuda', refresh ? { refresh: 1 } : undefined); return r.ok ? r.data : null; };

export default async function render(ctx) {
  const root = ctx.root;
  const kpis = h('div', { class: 'grid auto-sm' });
  const gpuHost = h('div');
  const jobsHost = h('div');
  const gatesHost = h('div');
  const storageHost = h('div');
  const claimsHost = h('div');
  root.append(
    pageHead('Dashboard', 'Live overview of hardware, jobs, quality gates, storage and open scientific claims.',
      h('a', { class: 'btn primary', href: '#/jobs/new' }, '+ New job'), h('a', { class: 'btn', href: '#/inference' }, 'Inference')),
    cudaBanner(cudaReport),
    kpis,
    h('div', { class: 'mt' }, card('GPUs', gpuHost, h('span', { class: 'muted small', id: 'gpu-live' }, ''))),
    h('div', { class: 'grid cols-2 mt' },
      card('Active & queued jobs', jobsHost, h('a', { href: '#/jobs' }, 'all jobs')),
      card('Quality gates', gatesHost, h('a', { href: '#/compare' }, 'compare'))),
    h('div', { class: 'grid cols-2 mt' },
      card('Storage', storageHost, h('a', { href: '#/storage' }, 'details')),
      card('Claims needing attention', claimsHost, h('a', { href: '#/claims' }, 'ledger'))));

  // ---- KPIs from jobs + system info
  const kpiState = { jobs: [], info: null };
  function paintKpis() {
    const c = (s) => kpiState.jobs.filter((j) => j.status === s).length;
    clear(kpis);
    const mk = (label, value, hint, href) => h('a', { class: 'card', href: href || null, style: 'color:inherit;text-decoration:none' }, stat(label, value, hint));
    kpis.append(
      mk('Running', String(c('running')), 'jobs', '#/jobs?status=running'), mk('Queued', String(c('queued')), 'waiting for a GPU', '#/jobs?status=queued'),
      mk('Failed', String(c('failed')), 'in history', '#/jobs?status=failed'), mk('Succeeded', String(c('succeeded')), 'in history', '#/jobs?status=succeeded'),
      mk('Server', kpiState.info ? (kpiState.info.version || 'ok') : '...', kpiState.info && kpiState.info.server_time ? 'time ' + new Date(kpiState.info.server_time * 1000).toLocaleTimeString() : ''));
  }
  paintKpis();
  tryGet('/system/info').then((r) => { kpiState.info = r.data; paintKpis(); });

  // ---- GPUs (SSE with polling fallback)
  const history = new Map();
  const paintGpus = (list) => {
    const gpus = normGpus(list);
    if (!gpus.length) { clear(gpuHost); gpuHost.appendChild(emptyState('No NVIDIA GPUs detected', 'The telemetry sampler returned no devices (pynvml / nvidia-smi not available). CPU-only jobs still run.')); return; }
    clear(gpuHost);
    const grid = h('div', { class: 'grid auto' });
    for (const g of gpus) {
      const hs = history.get(g.index) || []; hs.push(g.util ?? 0); if (hs.length > 90) hs.shift(); history.set(g.index, hs);
      const memFrac = isNum(g.mem_used_mb) && isNum(g.mem_total_mb) && g.mem_total_mb ? g.mem_used_mb / g.mem_total_mb : 0;
      const hot = (g.temp_c ?? 0) > 83;
      grid.appendChild(h('div', { class: 'card', style: 'box-shadow:none' },
        h('div', { class: 'row' }, h('b', `GPU ${g.index}`), h('span', { class: 'muted grow small' }, g.name), hot ? badge('hot', 'bad') : null),
        h('div', { class: 'row mt-s' }, h('div', { class: 'grow' }, h('div', { class: 'small muted' }, `Utilisation ${g.util ?? '-'}%`), progressBar((g.util ?? 0) / 100)), sparkline(hs, { min: 0, max: 100, width: 90 })),
        h('div', { class: 'mt-s' }, h('div', { class: 'small muted' }, `Memory ${isNum(g.mem_used_mb) ? fmtBytes(g.mem_used_mb * 1048576) : '-'} / ${isNum(g.mem_total_mb) ? fmtBytes(g.mem_total_mb * 1048576) : '-'}`), progressBar(memFrac, memFrac > 0.92 ? 'warn' : '')),
        h('div', { class: 'row mt-s small muted' }, h('span', `${g.temp_c ?? '-'} °C`), h('span', `${g.power_w ?? '-'} W`), g.clock_mhz ? h('span', `${g.clock_mhz} MHz`) : null)));
    }
    gpuHost.appendChild(grid);
  };
  clear(gpuHost); gpuHost.appendChild(h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Waiting for telemetry...'));
  let pollStop = null;
  const startPolling = () => {
    if (pollStop) return;
    pollStop = poll(ctx, async () => {
      try { paintGpus(await api.get('/system/gpus')); } catch (e) {
        if (e && e.name === 'AbortError') return;
        clear(gpuHost); gpuHost.appendChild(e.unavailable ? unavailableNotice('GPU telemetry') : notice('error', 'Telemetry failed', e.message));
      }
    }, 2000, { immediate: true });
  };
  const live = document.getElementById('gpu-live');
  const stream = sse('/system/gpus/stream', {
    onEvent: (ev) => { if (ev.data && typeof ev.data === 'object') paintGpus(ev.data); },
    onStatus: (s, info) => {
      if (live) live.textContent = s === 'open' ? 'live (SSE)' : s;
      if (s === 'unavailable' || s === 'error') { startPolling(); if (live) live.textContent = 'polling'; }
      void info;
    },
  });
  ctx.onCleanup(() => stream.close());
  // if SSE does not deliver within 3s, poll once so the card is never blank
  setTimeout(() => { if (ctx.isActive() && gpuHost.querySelector('.spinner')) api.get('/system/gpus').then(paintGpus).catch((e) => { clear(gpuHost); gpuHost.appendChild(e.unavailable ? unavailableNotice('GPU telemetry') : notice('error', 'Telemetry failed', e.message)); }); }, 3000);

  // ---- jobs
  const jobsTable = dataTable({
    columns: [
      { key: 'name', label: 'Job', render: (j) => h('a', { href: `#/jobs/${j.id}` }, j.spec.name || j.id), sortValue: (j) => j.spec.name },
      { key: 'type', label: 'Type', render: (j) => j.spec.type, sortValue: (j) => j.spec.type },
      { key: 'status', label: 'Status', render: (j) => statusBadge(j.status) },
      { key: 'progress', label: 'Progress', render: (j) => h('div', { style: 'min-width:90px' }, progressBar(j.progress), h('span', { class: 'small muted' }, j.progress_msg || fmtPct(j.progress, 0))), noSort: true },
    ], rows: [], empty: h('div', h('p', 'No active jobs.'), h('a', { class: 'btn primary', href: '#/jobs/new' }, 'Start one')), sort: { key: 'status', dir: 'asc' },
  });
  clear(jobsHost); jobsHost.appendChild(jobsTable);
  const refreshJobs = async () => {
    const r = await tryGet('/jobs', { limit: 100 });
    if (!r.ok) { clear(jobsHost); jobsHost.appendChild(r.unavailable ? unavailableNotice('Jobs') : notice('error', 'Could not load jobs', r.error.message)); return; }
    kpiState.jobs = listOf(r.data, 'jobs'); paintKpis();
    if (!jobsHost.contains(jobsTable)) { clear(jobsHost); jobsHost.appendChild(jobsTable); }
    jobsTable.setRows(kpiState.jobs.filter((j) => j.status === 'running' || j.status === 'queued'));
  };
  refreshJobs();
  poll(ctx, refreshJobs, 3000);

  // ---- gates (latest verdict per gate) + recent evals
  loadInto(gatesHost, async () => {
    const [g, e] = await Promise.all([tryGet('/gates'), tryGet('/evals')]);
    if (!g.ok && !e.ok) throw g.error;
    return { gates: g.ok ? listOf(g.data, 'gates') : [], evals: e.ok ? listOf(e.data, 'evals') : [] };
  }, ({ gates, evals }) => {
    const latest = new Map();
    for (const ev of evals.slice().sort((a, b) => (a.created_at || 0) - (b.created_at || 0))) if (ev.gate) latest.set(ev.gate.gate_id, { gate: ev.gate, ev });
    const rows = (gates.length ? gates.map((g) => ({ id: g.id || g.gate_id, title: g.title || g.id, latest: g.latest ? { gate: g.latest } : Array.from(latest.values()).find((x) => x.gate.gate_id.startsWith(g.id)) })) : Array.from(latest.values()).map((x) => ({ id: x.gate.gate_id, title: x.gate.gate_id, latest: x })));
    if (!rows.length) return emptyState('No gate results yet', 'Run an eval job with a gate preset to see verdicts here.');
    return dataTable({ columns: [
      { key: 'title', label: 'Gate', render: (r) => h('b', r.title) },
      { key: 'v', label: 'Latest verdict', render: (r) => (r.latest ? statusBadge(r.latest.gate.verdict) : badge('no runs', 'dim')), sortValue: (r) => r.latest ? r.latest.gate.verdict : 'zzz' },
      { key: 'why', label: 'Reason', render: (r) => (r.latest && r.latest.gate.reasons && r.latest.gate.reasons[0]) || '', noSort: true },
    ], rows, empty: 'No gates' });
  }, { feature: 'Gates' });

  // ---- storage
  loadInto(storageHost, () => api.get('/storage/stats'), (s) => {
    const logical = s.logical_bytes; const uniq = s.unique_bytes ?? s.stored_bytes;
    const ratio = s.dedup_ratio ?? s.compaction_vs_naive ?? (logical && uniq ? logical / uniq : null);
    return h('div', { class: 'grid cols-3' },
      stat('Logical', fmtBytes(logical), 'what the artifacts would occupy'),
      stat('On disk', fmtBytes(uniq), 'unique after dedup'),
      stat('Compaction', isNum(ratio) ? ratio.toFixed(2) + 'x' : '-', s.provenance ? provBadge(s.provenance) : null));
  }, { feature: 'Storage' });

  // ---- claims
  loadInto(claimsHost, () => api.get('/claims'), (cl) => {
    const list = listOf(cl, 'claims');
    const rank = { contradicted: 0, partial: 1, untested: 2, reproduced: 3 };
    const attention = list.filter((c) => c.status !== 'reproduced' && (c.depends_rules || []).length).sort((a, b) => (rank[a.status] ?? 9) - (rank[b.status] ?? 9)).slice(0, 6);
    const counts = list.reduce((m, c) => { m[c.status] = (m[c.status] || 0) + 1; return m; }, {});
    return h('div',
      h('div', { class: 'row mb' }, ...Object.entries(counts).map(([k, v]) => h('span', [statusBadge(k), ' ', h('b', String(v))]))),
      attention.length ? h('div', attention.map((c) => h('div', { class: 'row', style: 'padding:6px 0;border-top:1px solid var(--grid)' }, statusBadge(c.status), h('div', { class: 'grow' }, h('b', c.id), ' ', c.statement), h('span', { class: 'small muted' }, `${(c.depends_rules || []).length} rule(s)`))))
        : emptyState('Nothing needs attention', 'All claims that decision rules depend on are reproduced.'));
  }, { feature: 'Claims ledger' });
}
