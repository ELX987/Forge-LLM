// Inference: deployments, chat playground (streaming) with live engine stats, and the serving simulator / capacity curve.

import { h, clear, listOf, fmtNum, fmtPct, fmtRel, isNum, fmtDuration, deepClone } from '../util.js';
import { api, sse, tryGet } from '../api.js';
import { pageHead, card, tabs, dataTable, statusBadge, badge, provBadge, loadInto, emptyState, notice, unavailableNotice, confirmDialog, modal, toast, toastError, progressBar, stat, poll, kv, help, loadingBlock } from '../components/ui.js';
import { SchemaForm } from '../forms.js';
import { XYChart, BarChart, sparkline } from '../charts.js';
import { fetchJobTypes } from './newjob.js';

const pick = (o, names, def = null) => { if (!o) return def; for (const n of names) if (o[n] !== undefined && o[n] !== null) return o[n]; return def; };
function pctl(o, base, p) { return pick(o, [`${base}_${p}`, `${base}_${p}_ms`, `${base}_ms_${p}`]) ?? (o && o[base] && typeof o[base] === 'object' ? o[base][p] : (o && o[`${base}_ms`] && typeof o[`${base}_ms`] === 'object' ? o[`${base}_ms`][p] : null)); }

export default async function render(ctx) {
  const st = { tab: ctx.query.tab || 'deployments', dep: ctx.query.dep || '', deployments: [] };
  const content = h('div');
  const tabBar = tabs({ items: [{ id: 'deployments', label: 'Deployments' }, { id: 'playground', label: 'Chat playground' }, { id: 'simulator', label: 'Simulator & capacity' }], active: st.tab, onChange: (t) => { st.tab = t; ctx.replaceQuery({ tab: t, dep: st.dep }); paint(); } });
  ctx.root.append(pageHead('Inference', 'Run the serving engine (paged KV, prefix cache, speculative decoding), chat with it, and plan capacity before renting more GPUs.', h('button', { class: 'btn primary', onclick: () => startDialog() }, '+ Start deployment')), tabBar, content);
  let cleanup = [];
  ctx.onCleanup(() => { cleanup.forEach((f) => f()); cleanup = []; });

  async function loadDeps() {
    const r = await tryGet('/deployments');
    st.depsRes = r; st.deployments = r.ok ? listOf(r.data, 'deployments') : [];
  }
  await loadDeps();
  if (!ctx.isActive()) return;
  paint();

  function paint() {
    cleanup.forEach((f) => f()); cleanup = [];
    clear(content); tabBar.set(st.tab, false);
    if (st.tab === 'deployments') paintDeployments(); else if (st.tab === 'playground') paintPlayground(); else paintSimulator();
  }

  // ------------------------------------------------------------------ deployments
  function paintDeployments() {
    const host = h('div'); content.appendChild(host);
    const draw = () => {
      clear(host);
      if (!st.depsRes.ok) { host.appendChild(st.depsRes.unavailable ? unavailableNotice('Deployments', 'The serving module is not available on this server (GET /deployments returned 404/501). The simulator tab still works if the planner is installed.') : notice('error', 'Could not load deployments', st.depsRes.error.message)); return; }
      if (!st.deployments.length) { host.appendChild(emptyState('No deployments', 'Start the serving engine on a model or artifact. It runs as a serve job and exposes an OpenAI-compatible API.', h('button', { class: 'btn primary', onclick: () => startDialog() }, 'Start deployment'))); return; }
      host.appendChild(dataTable({ columns: [
        { key: 'name', label: 'Deployment', render: (d) => h('div', h('b', d.name || d.id), h('div', { class: 'small muted mono' }, d.id)) },
        { key: 'artifact_ref', label: 'Model / artifact', render: (d) => h('code', String(d.artifact_ref || '-').slice(0, 40)) },
        { key: 'status', label: 'Status', render: (d) => statusBadge(d.status) },
        { key: 'cfg', label: 'Config', noSort: true, render: (d) => h('span', { class: 'small muted' }, ['kv_dtype', 'max_model_len', 'max_num_seqs'].filter((k) => d.serve_config && d.serve_config[k] !== undefined).map((k) => `${k}=${d.serve_config[k]}`).join(' ') || '-') },
        { key: 'port', label: 'Port', num: true, render: (d) => d.port ?? '-' },
        { key: 'created_at', label: 'Started', render: (d) => fmtRel(d.created_at) },
        { key: 'act', label: '', noSort: true, render: (d) => h('div', { class: 'row tight' },
          h('button', { class: 'btn sm primary', disabled: d.status !== 'running', onclick: () => { st.dep = d.id; st.tab = 'playground'; ctx.replaceQuery({ tab: 'playground', dep: d.id }); paint(); } }, 'Chat'),
          d.job_id ? h('a', { class: 'btn sm', href: `#/jobs/${d.job_id}` }, 'Job') : null,
          h('button', { class: 'btn sm danger', onclick: () => stop(d) }, 'Stop')) },
      ], rows: st.deployments, sort: { key: 'created_at', dir: 'desc' } }));
    };
    draw();
    const p = poll(ctx, async () => { await loadDeps(); if (st.tab === 'deployments') draw(); }, 4000);
    cleanup.push(p);
  }
  async function stop(d) {
    if (!(await confirmDialog({ title: 'Stop deployment?', message: `Stops the serve job for "${d.name}" and frees the GPU memory.`, confirmLabel: 'Stop', danger: true }))) return;
    try { await api.del(`/deployments/${encodeURIComponent(d.id)}`); toast('Deployment stopped', 'ok'); await loadDeps(); paint(); } catch (e) { toastError(e); }
  }

  async function startDialog() {
    const [models, arts, types] = await Promise.all([tryGet('/models'), tryGet('/artifacts', { kind: 'model' }), fetchJobTypes().catch(() => [])]);
    const refs = [];
    if (models.ok) for (const m of listOf(models.data, 'models')) refs.push({ value: m.id, label: `${m.name || m.id} (model)` });
    if (arts.ok) for (const a of listOf(arts.data, 'artifacts')) refs.push({ value: a.hash, label: `${a.name || a.hash.slice(0, 18)} (artifact)` });
    const serve = types.find((t) => t.type === 'serve');
    const nameEl = h('input', { type: 'text', placeholder: 'my-deployment', style: 'width:100%', 'aria-label': 'Deployment name' });
    const refSel = h('select', { 'aria-label': 'Model or artifact', style: 'width:100%' }, refs.length ? refs.map((r) => h('option', { value: r.value }, r.label)) : [h('option', { value: '' }, '(no models registered)')]);
    const refTxt = h('input', { type: 'text', placeholder: 'or type a model id / artifact hash / path', style: 'width:100%', 'aria-label': 'Model reference' });
    const formHost = h('div'); let form = null; let rawEl = null;
    if (serve) {
      form = new SchemaForm(formHost, { schema: serve.schema, defaults: serve.defaults, value: serve.defaults });
    } else { rawEl = h('textarea', { rows: 8, 'aria-label': 'ServeConfig JSON', placeholder: '{"max_model_len": 8192}' }); formHost.append(h('p', { class: 'muted small' }, 'Serve schema unavailable - enter ServeConfig as JSON.'), rawEl); }
    const err = h('div', { class: 'sf-err', role: 'alert' });
    const go = h('button', { class: 'btn primary', onclick: async () => {
      let cfg;
      try { cfg = form ? form.getValue() : (rawEl.value.trim() ? JSON.parse(rawEl.value) : {}); } catch (e) { err.textContent = 'Invalid JSON: ' + e.message; return; }
      const artifact_ref = refTxt.value.trim() || refSel.value;
      if (!artifact_ref) { err.textContent = 'Choose a model or artifact.'; return; }
      go.disabled = true;
      try { const d = await api.post('/deployments', { name: nameEl.value.trim() || `serve-${Date.now() % 100000}`, artifact_ref, serve_config: cfg }); toast('Deployment starting...', 'ok'); m.close(); st.dep = d.id; await loadDeps(); paint(); } catch (e) { go.disabled = false; if (form && e.fieldErrors) form.setErrors(e.fieldErrors); err.textContent = e.message; }
    } }, 'Start');
    const m = modal({ title: 'Start deployment', wide: true, body: h('div', { class: 'stack' }, h('div', { class: 'grid cols-2' }, h('div', h('label', { class: 'small muted' }, 'Name'), nameEl), h('div', h('label', { class: 'small muted' }, 'Model / artifact'), refSel, refTxt)), h('h3', 'Serve configuration'), formHost, err), footer: [h('button', { class: 'btn', onclick: () => m.close() }, 'Cancel'), go] });
  }

  // ------------------------------------------------------------------ playground
  function paintPlayground() {
    const running = st.deployments.filter((d) => d.status === 'running' || d.status === 'starting');
    if (!st.depsRes.ok) { content.appendChild(st.depsRes.unavailable ? unavailableNotice('Chat playground', 'No serving backend on this server.') : notice('error', 'Failed', st.depsRes.error.message)); return; }
    if (!st.deployments.length) { content.appendChild(emptyState('Nothing to chat with', 'Start a deployment first.', h('button', { class: 'btn primary', onclick: () => startDialog() }, 'Start deployment'))); return; }
    if (!st.dep || !st.deployments.find((d) => d.id === st.dep)) st.dep = (running[0] || st.deployments[0]).id;
    const depSel = h('select', { 'aria-label': 'Deployment', onchange: () => { st.dep = depSel.value; ctx.replaceQuery({ tab: 'playground', dep: st.dep }); paint(); } }, st.deployments.map((d) => h('option', { value: d.id, selected: d.id === st.dep }, `${d.name || d.id} (${d.status})`)));
    const dep = st.deployments.find((d) => d.id === st.dep);

    // sampling controls
    const S = { temperature: 0.7, top_p: 0.95, top_k: 0, min_p: 0, max_tokens: 256, seed: '', repetition_penalty: 1, frequency_penalty: 0, presence_penalty: 0, stop: '' };
    const sys = h('textarea', { rows: 3, placeholder: 'You are a helpful assistant.', 'aria-label': 'System prompt', style: 'font-family:var(--sans)' });
    const slider = (key, label, min, max, step, hint) => {
      const out = h('span', { class: 'mono small' }, String(S[key]));
      const inp = h('input', { type: 'range', min, max, step, value: S[key], 'aria-label': label, oninput: () => { S[key] = Number(inp.value); out.textContent = String(S[key]); } });
      return h('div', h('div', { class: 'row' }, h('label', { class: 'small' }, label), hint ? help(hint) : null, h('span', { class: 'grow' }), out), inp);
    };
    const num = (key, label, hint) => h('div', h('div', { class: 'row' }, h('label', { class: 'small' }, label), hint ? help(hint) : null), h('input', { type: 'number', value: S[key], style: 'width:100%', 'aria-label': label, oninput: (e) => { S[key] = e.target.value === '' ? '' : Number(e.target.value); } }));
    const stopIn = h('input', { type: 'text', placeholder: 'stop strings, comma separated', 'aria-label': 'Stop strings', style: 'width:100%', oninput: (e) => { S.stop = e.target.value; } });
    const controls = h('div', { class: 'stack' }, h('div', h('label', { class: 'small muted' }, 'System prompt'), sys),
      slider('temperature', 'Temperature', 0, 2, 0.05, '0 = greedy.'), slider('top_p', 'Top-p', 0, 1, 0.01), slider('min_p', 'Min-p', 0, 1, 0.01, 'Drop tokens with probability below min_p x top token.'), num('top_k', 'Top-k (0 = off)'), num('max_tokens', 'Max new tokens'), num('seed', 'Seed (blank = random)'),
      slider('repetition_penalty', 'Repetition penalty', 0.8, 2, 0.01), slider('frequency_penalty', 'Frequency penalty', -2, 2, 0.05), slider('presence_penalty', 'Presence penalty', -2, 2, 0.05), stopIn);

    // chat
    const msgs = []; const chat = h('div', { class: 'chat', role: 'log', 'aria-live': 'polite', 'aria-label': 'Conversation' });
    const input = h('textarea', { rows: 2, placeholder: 'Message... (Enter to send, Shift+Enter for newline)', 'aria-label': 'Message', style: 'font-family:var(--sans)' });
    const sendBtn = h('button', { class: 'btn primary' }, 'Send'); const stopBtn = h('button', { class: 'btn danger hidden' }, 'Stop');
    const info = h('div', { class: 'small muted', role: 'status' });
    const clearBtn = h('button', { class: 'btn', onclick: () => { msgs.length = 0; clear(chat); info.textContent = ''; } }, 'Clear');
    let streamH = null;
    const add = (role, text) => { const el = h('div', { class: `msg ${role}` }, text); chat.appendChild(el); chat.scrollTop = chat.scrollHeight; return el; };
    if (dep.status !== 'running') chat.appendChild(h('div', { class: 'msg system' }, `Deployment is "${dep.status}". Chat becomes available when it is running.`));
    const finish = () => { streamH = null; sendBtn.disabled = false; stopBtn.classList.add('hidden'); };
    function send() {
      const text = input.value.trim(); if (!text || streamH) return;
      input.value = '';
      const messages = []; if (sys.value.trim()) messages.push({ role: 'system', content: sys.value.trim() });
      messages.push(...msgs, { role: 'user', content: text });
      msgs.push({ role: 'user', content: text }); add('user', text);
      const out = add('assistant', ''); out.classList.add('cursor');
      let acc = ''; let first = null; let count = 0; const t0 = performance.now(); let usage = null;
      const body = { messages, stream: true, temperature: S.temperature, top_p: S.top_p, max_tokens: S.max_tokens, repetition_penalty: S.repetition_penalty, frequency_penalty: S.frequency_penalty, presence_penalty: S.presence_penalty };
      if (S.top_k) body.top_k = S.top_k; if (S.min_p) body.min_p = S.min_p; if (S.seed !== '') body.seed = S.seed;
      if (S.stop.trim()) body.stop = S.stop.split(',').map((x) => x.trim()).filter(Boolean);
      sendBtn.disabled = true; stopBtn.classList.remove('hidden'); info.textContent = 'waiting for first token...';
      const done = () => {
        out.classList.remove('cursor'); if (acc) msgs.push({ role: 'assistant', content: acc }); else if (!out.textContent) out.textContent = '(empty response)';
        const dt = (performance.now() - t0) / 1000;
        info.textContent = `client-measured: ${first != null ? `TTFT ${Math.round(first)} ms, ` : ''}${count} chunks in ${dt.toFixed(1)} s${count > 1 && first != null ? ` (~${(count / Math.max(0.001, dt - first / 1000)).toFixed(1)} chunks/s)` : ''}${usage ? `, ${usage.completion_tokens ?? '?'} completion tokens` : ''}`;
        finish();
      };
      streamH = sse(`/deployments/${encodeURIComponent(dep.id)}/chat`, { method: 'POST', body, reconnect: false,
        onStatus: (s, e) => { if (s === 'unavailable') { out.classList.remove('cursor'); out.textContent = 'Chat proxy is not available on this server (404/501).'; finish(); } else if (s === 'error') { out.classList.remove('cursor'); out.textContent = acc || `Error: ${(e && e.message) || 'stream failed'}`; finish(); } else if (s === 'closed' && streamH) done(); },
        onEvent: (ev) => {
          const d = ev.data;
          if (d === '[DONE]' || ev.raw === '[DONE]') { streamH && streamH.close(); done(); return; }
          if (ev.event === 'error') { out.textContent = acc + '\n[error] ' + (d && d.message || JSON.stringify(d)); return; }
          let tok = '';
          if (d && typeof d === 'object') {
            tok = pick(d.choices && d.choices[0] && d.choices[0].delta, ['content'], '') || (d.choices && d.choices[0] && d.choices[0].text) || d.token || d.text || d.content || '';
            if (d.usage) usage = d.usage;
          } else if (typeof d === 'string') tok = d;
          if (tok) { if (first == null) first = performance.now() - t0; count += 1; acc += tok; out.textContent = acc; chat.scrollTop = chat.scrollHeight; info.textContent = 'streaming...'; }
        } });
    }
    sendBtn.addEventListener('click', send);
    stopBtn.addEventListener('click', () => { if (streamH) { streamH.close(); toast('Generation stopped (client side)', 'info', 1500); } });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
    cleanup.push(() => { if (streamH) streamH.close(); });

    // stats side panel
    const statsHost = h('div'); const hist = { tps: [], run: [], kv: [] }; let specChart = null;
    const paintStats = (s) => {
      const running = pick(s, ['running', 'num_running', 'running_seqs']); const waiting = pick(s, ['waiting', 'num_waiting', 'waiting_seqs']);
      const kvu = pick(s, ['kv_usage', 'kv_cache_usage', 'kv_usage_frac']); const tps = pick(s, ['tokens_per_s', 'tok_s', 'throughput_tok_s']); const phr = pick(s, ['prefix_hit_rate', 'prefix_cache_hit_rate']);
      const spec = pick(s, ['spec_acceptance', 'spec_acceptance_per_position', 'acceptance_per_position'], []);
      for (const [k, v] of [['tps', tps], ['run', running], ['kv', kvu]]) { if (isNum(v)) { hist[k].push(v); if (hist[k].length > 90) hist[k].shift(); } }
      clear(statsHost);
      statsHost.append(
        h('div', { class: 'grid cols-2' }, card('', stat('Running', isNum(running) ? String(running) : '-', 'sequences')), card('', stat('Waiting', isNum(waiting) ? String(waiting) : '-', 'queued'))),
        h('div', { class: 'mt-s' }, h('div', { class: 'row' }, h('span', { class: 'small muted' }, 'KV cache usage'), h('span', { class: 'grow' }), h('b', isNum(kvu) ? fmtPct(kvu, 0) : '-')), progressBar(isNum(kvu) ? kvu : 0, isNum(kvu) && kvu > 0.9 ? 'bad' : isNum(kvu) && kvu > 0.75 ? 'warn' : '')),
        h('div', { class: 'row mt-s' }, h('div', { class: 'grow' }, h('div', { class: 'small muted' }, 'Tokens / s'), h('b', { style: 'font-size:1.3rem' }, isNum(tps) ? fmtNum(tps) : '-')), sparkline(hist.tps, { width: 110, height: 34 })),
        kv([['TTFT p50 / p99', `${fmtNum(pctl(s, 'ttft', 'p50'))} / ${fmtNum(pctl(s, 'ttft', 'p99'))} ms`], ['ITL p50 / p99', `${fmtNum(pctl(s, 'itl', 'p50'))} / ${fmtNum(pctl(s, 'itl', 'p99'))} ms`], ['Prefix hit rate', isNum(phr) ? fmtPct(phr, 0) : '-'], ['Spec mode', pick(s, ['spec_mode', 'speculation'], '-')], ['Requests', pick(s, ['requests_total'], '-')], ['Uptime', isNum(s.uptime_s) ? fmtDuration(s.uptime_s) : '-']]));
      if (Array.isArray(spec) && spec.length) {
        const holder = h('div', { class: 'mt-s' }); statsHost.appendChild(holder);
        if (specChart) specChart.destroy();
        specChart = new BarChart(holder, { title: 'Speculative acceptance by position', height: 150, legend: false, categories: spec.map((_, i) => `d${i + 1}`), series: [{ name: 'acceptance', values: spec }], yMax: 1, valueFormat: (v) => fmtPct(v, 0) });
      }
    };
    const pollStats = async () => { try { paintStats(await api.get(`/deployments/${encodeURIComponent(dep.id)}/stats`)); } catch (e) { clear(statsHost); statsHost.appendChild(e.unavailable ? unavailableNotice('Engine stats') : notice('error', 'Stats unavailable', e.message)); } };
    statsHost.appendChild(loadingBlock('Reading engine stats...'));
    cleanup.push(poll(ctx, pollStats, 1000, { immediate: true }));
    cleanup.push(() => { if (specChart) specChart.destroy(); });

    content.append(h('div', { class: 'toolbar' }, h('label', 'Deployment '), depSel, statusBadge(dep.status), dep.port ? badge(':' + dep.port, 'dim') : null),
      h('div', { class: 'grid pg-grid' },
        card('Sampling', controls), card('Chat', h('div', chat, h('div', { class: 'mt-s' }, input), h('div', { class: 'row mt-s' }, sendBtn, stopBtn, clearBtn, info))), card('Engine stats (live)', statsHost, h('span', { class: 'small muted' }, 'polling 1 s'))));
  }

  // ------------------------------------------------------------------ simulator
  async function paintSimulator() {
    const [models, chips] = await Promise.all([tryGet('/models'), tryGet('/hardware/chips')]);
    if (st.tab !== 'simulator') return;
    const modelList = models.ok ? listOf(models.data, 'models') : []; const cd = chips.ok ? chips.data : null;
    const chipList = Array.isArray(cd) ? cd : (cd && (cd.chips || cd.known)) || [];
    const f = { model: modelList[0] ? modelList[0].id : '', chip: (chipList.find((c) => /h100/i.test(c.name)) || chipList[0] || {}).name || '', quant: 'w4a16', kv_dtype: 'fp8_e4m3', ctxLen: 8192, prefix: 0 };
    const sel = (label, key, opts) => h('div', { class: 'stack' }, h('label', { class: 'small muted' }, label), h('select', { 'aria-label': label, onchange: (e) => { f[key] = e.target.value; } }, opts.map(([v, l]) => h('option', { value: v, selected: v === f[key] }, l))));
    const out = h('div');
    const run = async () => {
      clear(out); out.appendChild(loadingBlock('Simulating...'));
      try {
        const res = await api.post('/serve/simulate', { model: f.model, chip: f.chip, quant: f.quant, weight_bits: ({ bf16: 16, fp8: 8, w8a16: 8, w4a16: 4, nvfp4: 4, mxfp4: 4 })[f.quant], kv_dtype: f.kv_dtype, workload: { context_lengths: [f.ctxLen], batch_sizes: [1, 8, 32, 128], prefix_reuse: f.prefix } });
        clear(out);
        const cap = res.capacity || res.fit || {}; const ceil = res.ceiling || {};
        const curve = res.curve || res.capacity_curve || res.points || [];
        const bb = res.byte_breakdown || null;
        out.append(h('div', { class: 'grid cols-4' },
          card('', stat('Weights', isNum(cap.weights_gb) ? fmtNum(cap.weights_gb) + ' GB' : '-', provBadge(cap.provenance || 'derived'))),
          card('', stat('KV per sequence', isNum(cap.kv_gb_per_seq) ? fmtNum(cap.kv_gb_per_seq) + ' GB' : '-', `at ${fmtNum(f.ctxLen)} tokens`)),
          card('', stat('Free for KV', isNum(cap.free_gb) ? fmtNum(cap.free_gb) + ' GB' : '-', cap.fits === false ? badge('does not fit', 'bad') : null)),
          card('', stat('Max concurrency', isNum(cap.max_concurrency) ? fmtNum(cap.max_concurrency) : '-', isNum(ceil.decode_tok_s_batch1) ? `ideal ${fmtNum(ceil.decode_tok_s_batch1)} tok/s @ b=1` : ''))),
        h('div', { class: 'grid cols-2 mt' }, card('', h('div', { id: 'sim-c1' })), card('', h('div', { id: 'sim-c2' }))));
        const c1 = out.querySelector('#sim-c1'); const c2 = out.querySelector('#sim-c2');
        if (curve.length) {
          const feasible = curve.filter((p) => p.feasible !== false);
          const infeasible = curve.filter((p) => p.feasible === false);
          const ch1 = new XYChart(c1, { title: 'Capacity curve: throughput vs concurrency', height: 260, xLabel: 'concurrent sequences', yLabel: 'decode tok/s (derived)', logX: true, series: [{ name: 'feasible', points: feasible.map((p) => [p.concurrency, p.tok_s]) }, ...(infeasible.length ? [{ name: 'exceeds KV memory', points: infeasible.map((p) => [p.concurrency, p.tok_s]), dashed: true }] : [])] });
          const ch2 = new XYChart(c2, { title: 'Inter-token latency vs concurrency', height: 260, xLabel: 'concurrent sequences', yLabel: 'ITL (ms, derived)', logX: true, series: [{ name: 'ITL', points: curve.filter((p) => isNum(p.itl_ms)).map((p) => [p.concurrency, p.itl_ms]) }] });
          cleanup.push(() => { ch1.destroy(); ch2.destroy(); });
        } else { c1.appendChild(emptyState('No capacity curve returned')); }
        if (bb) { const holder = h('div'); out.appendChild(h('div', { class: 'mt' }, card('', holder))); const b = new BarChart(holder, { title: 'Memory byte breakdown (GB)', height: 160, horizontal: true, legend: false, categories: Object.keys(bb).map((k) => k.replace('_gb', '')), series: [{ name: 'GB', values: Object.values(bb) }] }); cleanup.push(() => b.destroy()); }
        out.appendChild(h('p', { class: 'small muted mt' }, res.note || 'All figures are derived from the cost model (bandwidth roofline), not measured on hardware. Compare with the live engine stats to calibrate.'));
      } catch (e) { clear(out); out.appendChild(e.unavailable ? unavailableNotice('Serving simulator', 'POST /serve/simulate is not available (forge.serve / planner not installed).') : notice('error', 'Simulation failed', e.message)); }
    };
    content.append(card('Serving simulator', h('div', { class: 'stack' }, h('div', { class: 'grid cols-4' },
      sel('Model', 'model', modelList.length ? modelList.map((m) => [m.id, m.name || m.id]) : [['', '(none)']]), sel('Chip', 'chip', chipList.length ? chipList.map((c) => [c.name, c.name]) : [['', '(none)']]),
      sel('Weights', 'quant', [['bf16', 'bf16'], ['fp8', 'fp8'], ['w8a16', 'w8a16'], ['w4a16', 'w4a16'], ['nvfp4', 'nvfp4'], ['mxfp4', 'mxfp4']]), sel('KV dtype', 'kv_dtype', [['bf16', 'bf16'], ['fp8_e4m3', 'fp8'], ['int8', 'int8'], ['int4', 'int4']]),
      h('div', { class: 'stack' }, h('label', { class: 'small muted' }, 'Context length'), h('input', { type: 'number', value: f.ctxLen, step: 256, min: 128, 'aria-label': 'Context length', oninput: (e) => { f.ctxLen = Number(e.target.value) || 4096; } })),
      h('div', { class: 'stack' }, h('label', { class: 'small muted' }, 'Prefix reuse (0-1)'), h('input', { type: 'number', value: f.prefix, step: 0.1, min: 0, max: 1, 'aria-label': 'Prefix reuse', oninput: (e) => { f.prefix = Number(e.target.value) || 0; } }))), h('div', h('button', { class: 'btn primary', onclick: run }, 'Simulate')))), h('div', { class: 'mt' }, out));
    void deepClone;
  }
}
