// Job detail / training monitor: live charts (SSE), progress, FP4 diagnostics, log tail, config, outputs, cancel / clone.

import { h, clear, listOf, fmtNum, fmtPct, fmtDuration, fmtTime, fmtRel, isNum, ema, throttleRaf, download, shortHash } from '../util.js';
import { api, sse, tryGet } from '../api.js';
import { pageHead, card, tabs, statusBadge, progressBar, badge, kv, objectView, hashLabel, confirmDialog, toast, toastError, notice, emptyState, errorNotice, unavailableNotice, loadingBlock, poll, stat, help } from '../components/ui.js';
import { XYChart, themeColors } from '../charts.js';
import { navigate } from '../router.js';
import { dump } from '../yaml.js';
import { prefs } from '../state.js';

const CHARTS = [
  { id: 'loss', title: 'Loss', keys: ['loss', 'val_loss'], smooth: true },
  { id: 'lr', title: 'Learning rate', keys: ['lr'], yFormat: (v) => v.toExponential(1) },
  { id: 'grad_norm', title: 'Gradient norm', keys: ['grad_norm'], smooth: true },
  { id: 'tps', title: 'Tokens / s', keys: ['tokens_per_s'], smooth: true },
  { id: 'step_time', title: 'Step time (ms)', keys: ['step_time_ms'], smooth: true },
  { id: 'mfu', title: 'MFU', keys: ['mfu'], smooth: true, yFormat: (v) => (v * 100).toFixed(1) + '%' },
  { id: 'mem', title: 'GPU memory (GB)', keys: ['gpu_mem_gb'] },
  { id: 'uw', title: 'Update / weight ratio', keys: ['update_weight_ratio'], logY: true, smooth: true },
];
const TERNARY_NOTES = {
  'ternary/zero_frac': 'Share of weights quantised to 0 (mean over sampled layers). Healthy about 40-52%; the run warns above 60% and aborts above 80% (zero-density collapse). Thresholds are TARGET values from the tern158 report.',
  'ternary/zero_frac_min': 'Lowest zero share among the sampled layers.',
  'ternary/zero_frac_max': 'Highest zero share among the sampled layers: the first layer to drift towards collapse.',
  'ternary/flip_rate': 'Fraction of ternary codes that changed per step (average since the previous sample). The AdamW equilibrium is about 0.59*sqrt(2*lr*wd); far above that means noise-dominated updates.',
  'ternary/flip_rate_max': 'Highest per-layer flip rate among the sampled layers.',
  'ternary/scale_mean': 'Mean group scale (recomputed every step, never learned). Large drift means the latent weights are changing scale.',
  'ternary/lam': 'Quantisation warm-up blend: 0 = full precision, 1 = fully ternary.',
  'ternary/loss_gap': 'Relative loss gap between the ternary forward and a shadow full-precision forward on the same batch.',
  'ternary/loss_ref': 'Loss of the shadow full-precision forward.',
};
// Diagnostic panels, one per metric namespace: { prefix, title, notes, tip }.
const DIAG_PANELS = [
  { prefix: 'fp4/', title: 'FP4 training diagnostics', notes: () => FP4_NOTES, tip: 'NVFP4 training is gated (Plan TR-3). These diagnostics are measured on this run but the recipe itself is unvalidated.', lead: 'Watch these to detect underflow, scale flushing and rounding bias before the loss visibly diverges.' },
  { prefix: 'ternary/', title: 'Ternary (1.58-bit) health', notes: () => TERNARY_NOTES, tip: 'Ternary QAT is experimental and validated on CPU only. Thresholds are TARGET values from the tern158 report, not Forge measurements.', lead: 'Loss alone hides collapse: watch the zero share and flip rate.' },
];
const FP4_NOTES = {
  'fp4/zero_grad_tile_frac': 'Fraction of gradient tiles that quantise to all-zero. Rising values mean underflow is eating signal.',
  'fp4/scale_flush_frac': 'Fraction of block scales flushed to zero/subnormal in E4M3.',
  'fp4/sr_bias': 'Mean signed error introduced by stochastic rounding; should hover around 0.',
  'fp4/loss_gap': 'Loss gap vs the high-precision shadow run (training-precision diagnostic).',
};
const TERMINAL = ['succeeded', 'failed', 'cancelled'];

function parseMetricEvent(d) {
  if (!d || typeof d !== 'object') return null;
  const step = d.step ?? d.s ?? d.x;
  let values = d.values || d.metrics;
  if (!values) { values = {}; for (const [k, v] of Object.entries(d)) if (!['step', 's', 'x', 'ts', 'time', 'event', 'run_id'].includes(k) && isNum(v)) values[k] = v; }
  if (!isNum(step)) return null;
  return { step, values, ts: d.ts };
}

export default async function render(ctx) {
  const id = ctx.params.id;
  const root = ctx.root;
  const head = h('div'); const progressHost = h('div'); const tabHost = h('div'); const body = h('div');
  root.append(head, progressHost, tabHost, body);
  body.appendChild(loadingBlock('Loading job...'));
  let job;
  try { job = await api.get(`/jobs/${encodeURIComponent(id)}`); } catch (e) { clear(body); body.appendChild(e.status === 404 ? emptyState('Job not found', `No job with id "${id}".`, h('a', { class: 'btn', href: '#/jobs' }, 'All jobs')) : (e.unavailable ? unavailableNotice('Jobs') : errorNotice(e, () => ctx.navigate(`/jobs/${id}`)))); return; }
  if (!ctx.isActive()) return;
  ctx.setTitle(job.spec.name || id);

  const S = {
    job, tab: ctx.query.tab || 'monitor', raw: new Map(), keys: [], logs: [], charts: new Map(), smoothing: prefs.get('chart.smoothing', 0.6), logLoss: false, streamState: 'connecting',
    lastStep: 0, extraKeys: new Set(), autoscroll: true, wrap: true, filter: '',
  };
  const bar = progressBar(job.progress);
  const msg = h('span', { class: 'muted small' });
  const live = h('span', { class: 'badge dim' }, '');

  // ------------------------------------------------------------ header
  function paintHead() {
    const j = S.job; const running = j.status === 'running' || j.status === 'queued';
    clear(head);
    head.append(h('div', { class: 'small' }, h('a', { href: '#/jobs' }, '← Jobs')),
      pageHead(j.spec.name || j.id, h('span', { class: 'row' }, statusBadge(j.status), badge(j.spec.type, 'dim'), h('span', { class: 'mono small muted' }, j.id), j.spec.gpus && j.spec.gpus.length ? badge('GPU ' + j.spec.gpus.join(','), 'info') : null, ...(j.spec.tags || []).map((t) => badge(t, 'accent')), live),
        running ? h('button', { class: 'btn danger', onclick: cancel }, 'Cancel') : null,
        h('a', { class: 'btn', href: `#/jobs/new?from=${encodeURIComponent(j.id)}` }, 'Clone'),
        !running ? h('button', { class: 'btn ghost', onclick: del }, 'Delete') : null));
    clear(progressHost);
    const elapsed = j.started_at ? (j.finished_at || Date.now() / 1000) - j.started_at : null;
    const eta = j.status === 'running' && isNum(j.progress) && j.progress > 0.01 && elapsed ? elapsed * (1 - j.progress) / j.progress : null;
    bar.set(j.progress, j.status === 'failed' ? 'bad' : j.status === 'succeeded' ? 'ok' : '');
    msg.textContent = `${fmtPct(j.progress, 1)}${j.progress_msg ? ' - ' + j.progress_msg : ''}${elapsed != null ? ' - elapsed ' + fmtDuration(elapsed) : ''}${eta ? ' - ETA ' + fmtDuration(eta) : ''}`;
    progressHost.append(h('div', { class: 'card mb' }, bar, h('div', { class: 'row mt-s' }, msg)));
    if (j.status === 'failed' && j.error) progressHost.appendChild(notice('error', 'Job failed', h('pre', { class: 'mono' }, j.error)));
  }
  async function cancel() { if (!(await confirmDialog({ title: 'Cancel job?', message: 'Sends SIGTERM (cooperative stop); the worker is killed after the grace period.', confirmLabel: 'Cancel job', danger: true }))) return; try { S.job = await api.post(`/jobs/${encodeURIComponent(id)}/cancel`, {}); toast('Cancel requested', 'ok'); paintHead(); } catch (e) { toastError(e); } }
  async function del() { if (!(await confirmDialog({ title: 'Delete job?', message: 'Removes the record, logs and metrics.', confirmLabel: 'Delete', danger: true }))) return; try { await api.del(`/jobs/${encodeURIComponent(id)}`); navigate('/jobs'); } catch (e) { toastError(e); } }
  paintHead();

  const tabBar = tabs({ items: [{ id: 'monitor', label: 'Monitor' }, { id: 'logs', label: 'Logs' }, { id: 'config', label: 'Config' }, { id: 'outputs', label: 'Outputs' }], active: S.tab, onChange: (t) => { S.tab = t; ctx.replaceQuery({ tab: t }); paintTab(); } });
  tabHost.appendChild(tabBar);

  // ------------------------------------------------------------ data
  function addPoints(key, pts) {
    let arr = S.raw.get(key); if (!arr) { arr = []; S.raw.set(key, arr); }
    for (const p of pts) arr.push(p);
    if (arr.length > 40000) { const thinned = arr.filter((_, i) => i % 2 === 0); S.raw.set(key, thinned); }
  }
  async function loadMetrics() {
    const r = await tryGet(`/jobs/${encodeURIComponent(id)}/metrics`, { max_points: 2000 });
    if (!r.ok) { S.metricsErr = r; return; }
    S.metricsErr = null;
    const d = r.data || {};
    S.keys = d.keys || Object.keys(d.points || {});
    if (isNum(d.seq)) S.seq = d.seq; // resume cursor for the SSE stream (server row id of the newest metric included here)
    S.raw.clear();
    for (const [k, pts] of Object.entries(d.points || {})) { S.raw.set(k, pts.map((p) => [p[0], p[1]])); if (pts.length) S.lastStep = Math.max(S.lastStep, pts[pts.length - 1][0]); }
  }
  async function loadLogs() {
    const r = await tryGet(`/jobs/${encodeURIComponent(id)}/logs`, { tail: 2000 });
    if (!r.ok) { S.logsErr = r; return; }
    const d = r.data;
    const lines = Array.isArray(d) ? d : Array.isArray(d && d.lines) ? d.lines : String((d && (d.text || d.content || d.log)) || (typeof d === 'string' ? d : '')).split('\n').filter((x) => x !== '');
    S.logs = lines.map((x) => (typeof x === 'string' ? x : x.line || x.msg || JSON.stringify(x)));
    if (d && isNum(d.offset)) S.logOffset = d.offset; // byte offset the stream continues from
  }
  await Promise.all([loadMetrics(), loadLogs()]);
  if (!ctx.isActive()) return;

  // ------------------------------------------------------------ stream
  let stream = null;
  // first metrics arriving while the monitor shows its empty state (or no chart exists yet) needs a full repaint
  const refreshMonitor = () => { if (S.tab !== 'monitor') return; if (S.charts.size === 0 && S.raw.size > 0) paintTab(); else updateCharts(); };
  const repaintCharts = throttleRaf(refreshMonitor);
  function startStream() {
    if (stream || TERMINAL.includes(S.job.status)) return;
    stream = sse(`/jobs/${encodeURIComponent(id)}/stream`, {
      // resume exactly where the initial REST load / the last event stopped: no duplicated points or log lines on (re)connect
      query: () => (isNum(S.seq) ? { since_seq: S.seq, log_offset: isNum(S.logOffset) ? S.logOffset : undefined, log_tail: isNum(S.logOffset) ? undefined : 0 } : { replay: 0, log_offset: isNum(S.logOffset) ? S.logOffset : undefined }),
      onStatus: (s) => { S.streamState = s; live.className = 'badge ' + (s === 'open' ? 'ok pulse' : s === 'retrying' ? 'warn' : 'dim'); live.textContent = s === 'open' ? 'live' : s === 'retrying' ? 'reconnecting...' : s === 'unavailable' ? 'polling' : s; if (s === 'unavailable' || s === 'error') startPolling(); },
      onEvent: (ev) => {
        const d = ev.data; const type = ev.event;
        if (type === 'metric' && d && d.points && typeof d.points === 'object') {
          // server shape: {points: {key: [[step, value, ts], ...]}, seq}
          for (const [k, pts] of Object.entries(d.points)) {
            const good = pts.filter((p) => isNum(p[1])).map((p) => [p[0], p[1]]);
            if (!good.length) continue;
            addPoints(k, good); if (!S.keys.includes(k)) S.keys.push(k);
            S.lastStep = Math.max(S.lastStep, ...(k.startsWith('sys/') ? [0] : good.map((p) => p[0])));
          }
          if (isNum(d.seq)) S.seq = d.seq;
          repaintCharts(); paintTiles();
        } else if (type === 'metric' || (type === 'message' && d && d.step !== undefined)) {
          const m = parseMetricEvent(d); if (!m) return;
          for (const [k, v] of Object.entries(m.values)) { if (!isNum(v)) continue; addPoints(k, [[m.step, v]]); if (!S.keys.includes(k)) S.keys.push(k); }
          S.lastStep = Math.max(S.lastStep, m.step); repaintCharts(); paintTiles();
        } else if (type === 'log') {
          if (d && Array.isArray(d.lines)) { for (const l of d.lines) pushLog(String(l)); if (isNum(d.offset)) S.logOffset = d.offset; } else {
            const line = typeof d === 'string' ? d : (d && (d.line || d.msg || d.message)) || ''; if (line) pushLog(line);
          }
        } else if (type === 'progress') {
          if (d && typeof d === 'object') { S.job.progress = d.progress ?? d.frac ?? S.job.progress; S.job.progress_msg = d.msg ?? d.message ?? S.job.progress_msg; paintProgressOnly(); }
        } else if (type === 'status') {
          const st = d && (d.status || d); refreshJob(typeof st === 'string' ? st : null);
        }
      },
    });
  }
  function paintProgressOnly() { const j = S.job; bar.set(j.progress); const elapsed = j.started_at ? (Date.now() / 1000 - j.started_at) : null; msg.textContent = `${fmtPct(j.progress, 1)}${j.progress_msg ? ' - ' + j.progress_msg : ''}${elapsed != null ? ' - elapsed ' + fmtDuration(elapsed) : ''}`; }
  async function refreshJob(newStatus) {
    try {
      S.job = await api.get(`/jobs/${encodeURIComponent(id)}`);
      paintHead();
      if (TERMINAL.includes(S.job.status)) { if (stream) { stream.close(); stream = null; } if (pollStop) pollStop(); live.className = 'badge dim'; live.textContent = 'finished'; await loadMetrics(); refreshMonitor(); if (S.tab === 'outputs') paintTab(); }
    } catch (_) { /* ignore */ }
    void newStatus;
  }
  let pollStop = null;
  function startPolling() {
    if (pollStop) return;
    pollStop = poll(ctx, async () => {
      await refreshJob(); if (TERMINAL.includes(S.job.status)) return;
      const r = await tryGet(`/jobs/${encodeURIComponent(id)}/metrics`, { since_step: S.lastStep, max_points: 500 });
      if (r.ok && r.data && r.data.points) { for (const [k, pts] of Object.entries(r.data.points)) addPoints(k, pts.filter((p) => p[0] > S.lastStep).map((p) => [p[0], p[1]])); for (const pts of Object.values(r.data.points)) if (pts.length) S.lastStep = Math.max(S.lastStep, pts[pts.length - 1][0]); repaintCharts(); paintTiles(); }
    }, 2500);
  }
  if (!TERMINAL.includes(S.job.status)) startStream();
  poll(ctx, () => (S.streamState === 'open' ? refreshProgressOnly() : Promise.resolve()), 4000);
  async function refreshProgressOnly() { if (TERMINAL.includes(S.job.status)) return; const j = await api.get(`/jobs/${encodeURIComponent(id)}`); if (j.status !== S.job.status) { S.job = j; paintHead(); refreshJob(); } else { S.job.progress = j.progress; S.job.progress_msg = j.progress_msg; paintProgressOnly(); } }
  ctx.onCleanup(() => { if (stream) stream.close(); for (const c of S.charts.values()) c.destroy(); });

  // ------------------------------------------------------------ tabs
  function paintTab() {
    for (const c of S.charts.values()) c.destroy();
    S.charts.clear(); S.logBox = null;
    clear(body);
    if (S.tab === 'monitor') paintMonitor(); else if (S.tab === 'logs') paintLogs(); else if (S.tab === 'config') paintConfig(); else paintOutputs();
  }

  // ---- monitor
  const tiles = h('div', { class: 'grid auto-sm mb' });
  function latest(k) { const a = S.raw.get(k); return a && a.length ? a[a.length - 1][1] : null; }
  function paintTiles() {
    if (S.tab !== 'monitor') return;
    clear(tiles);
    const t = (label, v, hint) => card('', stat(label, v, hint));
    tiles.append(t('Step', S.lastStep ? fmtNum(S.lastStep) : '-', S.job.total_steps ? `of ${fmtNum(S.job.total_steps)}` : ''), t('Loss', isNum(latest('loss')) ? latest('loss').toFixed(4) : '-', isNum(latest('val_loss')) ? 'val ' + latest('val_loss').toFixed(4) : ''),
      t('Tokens / s', isNum(latest('tokens_per_s')) ? fmtNum(latest('tokens_per_s')) : '-'), t('Grad norm', isNum(latest('grad_norm')) ? latest('grad_norm').toFixed(3) : '-'),
      t('MFU', isNum(latest('mfu')) ? fmtPct(latest('mfu')) : '-', 'model FLOPs utilisation'), t('GPU mem', isNum(latest('gpu_mem_gb')) ? latest('gpu_mem_gb').toFixed(1) + ' GB' : '-'));
  }
  function seriesFor(def) {
    const colors = themeColors().palette; const out = [];
    def.keys.forEach((k, i) => {
      const raw = S.raw.get(k); if (!raw || !raw.length) return;
      const alpha = def.smooth ? S.smoothing : 0;
      if (alpha > 0 && raw.length > 3) {
        out.push({ name: k + ' (raw)', points: raw, color: colors[i % 8], width: 1, alpha: 0.35 });
        out[out.length - 1].color = colors[i % 8];
        out.push({ name: k + ` (EMA ${alpha})`, points: ema(raw, alpha), color: colors[i % 8], width: 2 });
      } else out.push({ name: k, points: raw, color: colors[i % 8], width: 1.8 });
    });
    return out;
  }
  function chartCard(grid, def) {
    const holder = h('div', { class: 'card' });
    grid.appendChild(holder);
    const c = new XYChart(holder, { title: def.title, height: 230, xLabel: 'step', yFormat: def.yFormat, logY: !!def.logY, empty: 'waiting for data...', xFormat: (v) => fmtNum(v) });
    S.charts.set(def.id, c); S.chartDefs = S.chartDefs || new Map(); S.chartDefs.set(def.id, def);
    c.setSeries(seriesFor(def));
    return c;
  }
  function updateCharts() {
    paintTiles();
    for (const [cid, c] of S.charts) { const def = S.chartDefs.get(cid); if (def) c.setSeries(seriesFor(def)); }
  }
  function paintMonitor() {
    if (S.metricsErr && !S.raw.size) {
      body.appendChild(S.metricsErr.unavailable ? unavailableNotice('Metrics') : errorNotice(S.metricsErr.error)); return;
    }
    if (!S.raw.size && !['train', 'finetune', 'distill'].includes(S.job.spec.type)) {
      body.appendChild(emptyState('No step metrics for this job type', `${S.job.spec.type} jobs report progress and outputs rather than per-step training metrics. See Logs and Outputs.`)); return;
    }
    const smooth = h('input', { type: 'range', min: 0, max: 0.98, step: 0.01, value: S.smoothing, 'aria-label': 'Smoothing', oninput: () => { S.smoothing = Number(smooth.value); updateCharts(); } });
    body.append(h('div', { class: 'toolbar' }, h('label', 'Smoothing (EMA) '), smooth, help('Exponential moving average applied to noisy series. The raw series stays visible beneath.'),
      h('span', { class: 'muted small' }, 'Drag to zoom a region, shift+drag to pan, double-click to reset. Use the PNG / CSV buttons to export.')), tiles);
    paintTiles();
    const grid = h('div', { class: 'grid cols-2' });
    body.appendChild(grid);
    let any = false;
    for (const def of CHARTS) { if (def.keys.some((k) => S.raw.has(k))) { chartCard(grid, def); any = true; } }
    if (!any) grid.appendChild(emptyState('Waiting for the first metrics', 'Metrics appear as soon as the trainer reports its first logging step.'));
    // diagnostics panels (fp4/, ternary/, ...)
    for (const panel of DIAG_PANELS) {
      const dkeys = Array.from(S.raw.keys()).filter((k) => k.startsWith(panel.prefix));
      if (!dkeys.length) continue;
      const notes = panel.notes();
      const fgrid = h('div', { class: 'grid cols-2' });
      body.append(h('div', { class: 'mt' }, h('h2', panel.title + ' ', badge('experimental', 'warn', { tip: panel.tip })), h('p', { class: 'muted small' }, panel.lead)), fgrid);
      for (const k of dkeys) { chartCard(fgrid, { id: k, title: k.slice(panel.prefix.length), keys: [k], smooth: true }); fgrid.lastChild.appendChild(h('div', { class: 'small muted' }, notes[k] || '')); }
    }
    // system series
    const sys = Array.from(S.raw.keys()).filter((k) => k.startsWith('sys/'));
    if (sys.length) {
      const groups = new Map();
      for (const k of sys) { const m = /^sys\/gpu\d+\/(\w+)$/.exec(k); const g = m ? m[1] : k; if (!groups.has(g)) groups.set(g, []); groups.get(g).push(k); }
      const sgrid = h('div', { class: 'grid cols-2' }); body.append(h('h2', { class: 'mt' }, 'System'), sgrid);
      for (const [g, ks] of groups) chartCard(sgrid, { id: 'sys-' + g, title: 'GPU ' + g, keys: ks });
    }
    // other keys
    const known = new Set([...CHARTS.flatMap((c) => c.keys), 'tokens_seen']);
    const others = Array.from(S.raw.keys()).filter((k) => !known.has(k) && !DIAG_PANELS.some((p) => k.startsWith(p.prefix)) && !k.startsWith('sys/'));
    if (others.length) {
      const sel = h('select', { 'aria-label': 'Add metric chart', onchange: () => { if (sel.value) { const k = sel.value; sel.value = ''; if (!S.charts.has('x-' + k)) chartCard(ogrid, { id: 'x-' + k, title: k, keys: [k], smooth: true }); } } }, h('option', { value: '' }, 'Add chart for another metric...'), others.map((k) => h('option', { value: k }, k)));
      const ogrid = h('div', { class: 'grid cols-2 mt' });
      body.append(h('div', { class: 'toolbar mt' }, sel), ogrid);
    }
  }

  // ---- logs
  function logClass(line) { return /\b(error|fatal|traceback|exception)\b/i.test(line) ? 'l-error' : /\bwarn(ing)?\b/i.test(line) ? 'l-warn' : ''; }
  function pushLog(line) {
    S.logs.push(line); if (S.logs.length > 8000) S.logs.splice(0, S.logs.length - 8000);
    if (S.tab === 'logs' && S.logBox) {
      if (S.filter && !line.toLowerCase().includes(S.filter)) return;
      const nearBottom = S.logBox.scrollHeight - S.logBox.scrollTop - S.logBox.clientHeight < 40;
      S.logBox.appendChild(h('div', { class: logClass(line) }, line));
      if (S.autoscroll && nearBottom) S.logBox.scrollTop = S.logBox.scrollHeight;
    }
  }
  function paintLogs() {
    if (S.logsErr && !S.logs.length) { body.appendChild(S.logsErr.unavailable ? unavailableNotice('Logs') : errorNotice(S.logsErr.error)); return; }
    const box = h('div', { class: 'logbox', tabindex: '0', role: 'log', 'aria-label': 'Job log', style: S.wrap ? '' : 'white-space:pre;' });
    S.logBox = box;
    const fill = () => { clear(box); for (const l of S.logs) if (!S.filter || l.toLowerCase().includes(S.filter)) box.appendChild(h('div', { class: logClass(l) }, l)); if (!S.logs.length) box.appendChild(h('div', { class: 'l-dim' }, 'No log output yet.')); box.scrollTop = box.scrollHeight; };
    const filter = h('input', { type: 'search', placeholder: 'Filter log lines...', value: S.filter, 'aria-label': 'Filter log', oninput: () => { S.filter = filter.value.toLowerCase(); fill(); } });
    body.append(h('div', { class: 'toolbar' }, filter,
      h('label', { class: 'row tight' }, h('input', { type: 'checkbox', checked: S.autoscroll, onchange: (e) => { S.autoscroll = e.target.checked; } }), 'Auto-scroll'),
      h('label', { class: 'row tight' }, h('input', { type: 'checkbox', checked: S.wrap, onchange: (e) => { S.wrap = e.target.checked; box.style.whiteSpace = S.wrap ? 'pre-wrap' : 'pre'; } }), 'Wrap'),
      h('button', { class: 'btn sm', onclick: () => download(`${id}.log.txt`, S.logs.join('\n') + '\n') }, 'Download'),
      h('span', { class: 'muted small' }, `${S.logs.length} lines`)), box);
    fill();
  }

  // ---- config
  function paintConfig() {
    const j = S.job;
    const yaml = dump({ type: j.spec.type, name: j.spec.name, gpus: j.spec.gpus, tags: j.spec.tags, params: j.spec.params });
    body.append(h('div', { class: 'grid split' },
      card('Parameters (YAML)', h('pre', { class: 'mono', style: 'max-height:560px;overflow:auto' }, yaml), h('button', { class: 'btn sm', onclick: () => { navigator.clipboard && navigator.clipboard.writeText(yaml).then(() => toast('Copied', 'ok', 1200)); } }, 'Copy')),
      card('Run info', kv([['Id', j.id], ['Type', j.spec.type], ['Status', statusBadge(j.status)], ['Created', fmtTime(j.created_at)], ['Started', j.started_at ? fmtTime(j.started_at) : '-'], ['Finished', j.finished_at ? fmtTime(j.finished_at) : '-'],
        ['PID', j.pid ?? '-'], ['Exit code', j.exit_code ?? '-'], ['Run directory', j.run_dir ? h('code', j.run_dir) : '-']]))));
  }

  // ---- outputs
  function paintOutputs() {
    const o = S.job.outputs || {};
    const entries = Object.entries(o);
    if (!entries.length) { body.appendChild(emptyState(TERMINAL.includes(S.job.status) ? 'No outputs recorded' : 'No outputs yet', 'Jobs publish results (artifact hashes, report ids, summaries) through ctx.set_output while they run.')); return; }
    const hashes = entries.filter(([, v]) => typeof v === 'string' && /^(blake3|b2|sha256):[0-9a-f]{16,}$/.test(v));
    body.append(card('Outputs', objectView(o)));
    if (hashes.length) body.appendChild(h('div', { class: 'card mt' }, h('h3', 'Artifacts'), h('div', hashes.map(([k, v]) => h('div', { class: 'row', style: 'padding:4px 0' }, h('b', k), hashLabel(v, 20), h('a', { class: 'btn sm', href: `#/storage?hash=${encodeURIComponent(v)}` }, 'Lineage'))))));
    void shortHash; void fmtRel; void listOf;
  }

  paintTab();
}
