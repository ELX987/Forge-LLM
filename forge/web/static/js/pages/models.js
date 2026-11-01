// Models: registry of local / Hugging Face models, inspect, download, delete. Licence + distillation gate shown.

import { h, listOf, fmtNum, fmtRel, fmtBytes, isNum } from '../util.js';
import { api } from '../api.js';
import { pageHead, loadInto, dataTable, badge, statusBadge, hashLabel, confirmDialog, modal, objectView, toast, toastError, emptyState, kv, help } from '../components/ui.js';
import { formDialog, runAction } from '../components/dialogs.js';
import { navigate } from '../router.js';

export default async function render(ctx) {
  const host = h('div');
  let search = '';
  let all = [];
  let tableEl = null;

  const searchEl = h('input', { type: 'search', placeholder: 'Search models...', 'aria-label': 'Search models', oninput: (e) => { search = e.target.value.toLowerCase(); paint(); } });
  ctx.root.append(
    pageHead('Models', 'Registered base models, checkpoints and compressed variants available to jobs.',
      h('button', { class: 'btn', title: 'Register compress outputs (runs/*/artifact) as selectable models', onclick: () => importArtifacts() }, 'Import artifacts'),
      h('button', { class: 'btn', onclick: () => registerLocal() }, 'Register local path'),
      h('button', { class: 'btn primary', onclick: () => addHf() }, '+ Hugging Face model')),
    h('div', { class: 'toolbar' }, searchEl), host);

  function distillBadge(m) {
    if (m.distill_allowed === true) return badge('distillation OK', 'ok', { tip: 'Licence permits using this model as a distillation teacher.' });
    if (m.distill_allowed === false) return badge('no distillation', 'bad', { tip: 'Licence forbids distillation; distill jobs with this teacher will be blocked (LicenceBlocked).' });
    return badge('licence unknown', 'warn', { tip: 'Distillation permission is unknown; verify the licence before using as a teacher.' });
  }

  function paint() {
    const list = all.filter((m) => !search || `${m.name} ${m.id} ${m.path_or_repo} ${m.licence}`.toLowerCase().includes(search));
    if (tableEl) tableEl.setRows(list);
  }

  const reload = loadInto(host, () => api.get('/models'), (data) => {
    all = listOf(data, 'models');
    if (!all.length) return emptyState('No models registered', 'Add a Hugging Face repo or register a local directory containing safetensors + config.json.', h('div', { class: 'row', style: 'justify-content:center' }, h('button', { class: 'btn primary', onclick: () => addHf() }, '+ Hugging Face model'), h('button', { class: 'btn', onclick: () => registerLocal() }, 'Register local path')));
    tableEl = dataTable({
      columns: [
        { key: 'name', label: 'Name', render: (m) => h('div', h('b', m.name || m.id), h('div', { class: 'small muted mono' }, m.path_or_repo || '')) },
        { key: 'source', label: 'Source', render: (m) => h('div', badge(m.source || '-', 'dim'), m.quantization ? badge(m.quantization, 'ok', { tip: `Packed weights: ${m.quant_summary && m.quant_summary.n_quantized_layers} layers${m.base_model ? ', compressed from ' + m.base_model : ''}` }) : null, isNum(m.weight_gb) && m.weight_gb > 0 ? badge(fmtNum(m.weight_gb, 3) + ' GB', 'dim') : null) },
        { key: 'params_b', label: 'Params', num: true, render: (m) => (m.params_b != null ? fmtNum(m.params_b) + 'B' : '-') },
        { key: 'licence', label: 'Licence', render: (m) => h('div', h('div', m.licence || '-'), distillBadge(m)) },
        { key: 'hash', label: 'Weights hash', render: (m) => hashLabel(m.hash) },
        { key: 'status', label: 'Status', render: (m) => h('div', statusBadge(m.status || 'ready'),
          m.info && m.info.forge_supported === false ? badge('unsupported arch', 'bad', { tip: m.info.unsupported_reason || 'Forge cannot run this architecture.' }) : null,
          m.status === 'failed' && m.info && m.info.error ? h('div', { class: 'small muted' }, m.info.error) : null) },
        { key: 'created_at', label: 'Added', render: (m) => fmtRel(m.created_at) },
        { key: 'act', label: '', noSort: true, render: (m) => h('div', { class: 'row tight' },
          h('button', { class: 'btn sm', onclick: () => inspect(m) }, 'Inspect'),
          m.source === 'hf' && m.status !== 'ready' && m.status !== 'downloading' ? h('button', { class: 'btn sm', onclick: () => download(m) }, 'Download') : null,
          m.servable ? h('a', { class: 'btn sm', href: `#/inference` }, 'Serve') : null,
          h('a', { class: 'btn sm', href: `#/finetune?model=${encodeURIComponent(m.id)}` }, 'Fine-tune'),
          h('button', { class: 'btn sm danger', 'aria-label': `Delete ${m.name}`, onclick: () => del(m) }, 'Delete')) },
      ], rows: all, sort: { key: 'created_at', dir: 'desc' },
    });
    return tableEl;
  }, { feature: 'Models' });

  async function inspect(m) {
    const dlg = modal({ title: `Inspect ${m.name || m.id}`, wide: true, body: h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Reading config...') });
    try {
      const r = await api.get(`/models/${encodeURIComponent(m.id)}/inspect`);
      const kvb = r.kv_bytes_per_token;
      dlg.setBody(h('div', { class: 'stack' },
        kv([['Name', m.name], ['Id', m.id], ['Licence', r.licence || m.licence || '-'], ['Params', r.params_b != null ? fmtNum(r.params_b) + 'B' : '-']]),
        r.forge_supported === false ? h('div', { class: 'notice error' }, r.unsupported_reason || 'This architecture is not supported by Forge.') : null,
        (r.warnings || []).length ? h('div', { class: 'notice warn' }, (r.warnings || []).map((w) => h('div', w))) : null,
        r.config ? h('div', h('h3', 'Architecture'), objectView(r.config)) : null,
        kvb ? h('div', h('h3', 'KV bytes per token ', help('Per-token KV cache footprint at different KV precisions. Derived from the architecture, not measured.')),
          typeof kvb === 'object' ? kv(Object.entries(kvb).map(([bits, b]) => [`${bits}-bit KV`, fmtBytes(Number(b))])) : h('span', fmtBytes(kvb))) : null,
        h('details', { class: 'fold' }, h('summary', 'Raw response'), h('div', { class: 'fold-body' }, h('pre', JSON.stringify(r, null, 2))))));
    } catch (e) { dlg.setBody(h('div', { class: 'notice error' }, e.unavailable ? 'Model inspection is not available on this server.' : e.message)); }
  }

  async function download(m) {
    const job = await runAction(() => api.post(`/models/${encodeURIComponent(m.id)}/download`, {}), 'Download queued');
    if (job && job.id) navigate(`/jobs/${job.id}`);
  }
  async function del(m) {
    if (!(await confirmDialog({ title: 'Remove model?', message: `Remove "${m.name}" from the registry? Files in the content-addressed store are kept until GC.`, confirmLabel: 'Remove', danger: true }))) return;
    await runAction(() => api.del(`/models/${encodeURIComponent(m.id)}`), 'Model removed'); reload();
  }
  function addHf() {
    formDialog({ title: 'Add Hugging Face model', submitLabel: 'Add',
      intro: 'Registers the repo. Use the full Hugging Face id as org/name (e.g. google/gemma-4-E4B - a bare "gemma-4-E4B" is not a repo). Tick "download now" to queue a download job. Gated repos (Gemma, Llama) need the licence accepted on huggingface.co and HF_TOKEN exported on the server before it starts.',
      fields: [
        { name: 'path_or_repo', label: 'Repo id (org/name)', placeholder: 'google/gemma-4-E4B', required: true },
        { name: 'name', label: 'Display name', placeholder: '(defaults to the repo name)' },
        { name: 'revision', label: 'Revision', placeholder: 'main' },
        { name: 'download', label: 'Download now', type: 'checkbox', value: true },
      ],
      onSubmit: async (v) => {
        if (!/^[A-Za-z0-9][\w.-]*\/[\w.-]+$/.test((v.path_or_repo || '').trim())) throw new Error('Repo id must look like org/name, e.g. google/gemma-4-E4B');
        const m = await api.post('/models', { source: 'hf', path_or_repo: v.path_or_repo, name: v.name || v.path_or_repo.split('/').pop(), revision: v.revision || null });
        toast('Model added', 'ok'); reload();
        if (v.download && m && m.id) { try { const job = await api.post(`/models/${encodeURIComponent(m.id)}/download`, {}); if (job && job.id) navigate(`/jobs/${job.id}`); } catch (e) { toastError(e, 'Download'); } }
      } });
  }
  async function importArtifacts() {
    try {
      const r = await api.post('/models/import-artifacts', {});
      toast(`${r.imported.length} artifact(s) imported, ${r.updated} refreshed${r.skipped.length ? `, ${r.skipped.length} skipped` : ''}`, r.skipped.length && !r.imported.length ? 'info' : 'ok');
      reload();
    } catch (e) { toastError(e, 'Import artifacts'); }
  }
  function registerLocal() {
    formDialog({ title: 'Register local model', submitLabel: 'Register',
      intro: 'Path on the server host to a directory with config.json and *.safetensors.',
      fields: [{ name: 'path', label: 'Directory', placeholder: '/workspace/models/my-model', required: true }],
      onSubmit: async (v) => { await api.post('/models/register-local', { path: v.path }); toast('Model registered', 'ok'); reload(); } });
  }
}
