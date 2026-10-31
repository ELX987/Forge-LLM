// Datasets: upload, ingest (tokenise + shard), preview, manifests.

import { h, listOf, fmtNum, fmtRel, fmtBytes, fmtInt } from '../util.js';
import { api } from '../api.js';
import { pageHead, loadInto, dataTable, badge, statusBadge, confirmDialog, modal, objectView, toast, toastError, emptyState, kv, progressBar, card, hashLabel } from '../components/ui.js';
import { runAction } from '../components/dialogs.js';
import { navigate } from '../router.js';
import { BarChart } from '../charts.js';

export default async function render(ctx) {
  const host = h('div');
  const uploadHost = h('div');
  ctx.root.append(
    pageHead('Datasets', 'Token shards with manifests: tokenizer hash, mixture, dedup report and licence. Upload raw files, then ingest.', h('a', { class: 'btn', href: '#/jobs/new?type=dataset_ingest' }, 'Ingest from path / HF')),
    card('Upload', uploadHost), h('div', { class: 'mt' }, host));

  // ---- upload
  const bar = progressBar(0); bar.classList.add('hidden');
  const status = h('div', { class: 'small muted' });
  const fileInput = h('input', { type: 'file', multiple: true, 'aria-label': 'Choose files', class: 'sr-only', id: 'ds-file', onchange: () => doUpload(Array.from(fileInput.files)) });
  const drop = h('div', { class: 'drop', ondragover: (e) => { e.preventDefault(); drop.classList.add('over'); }, ondragleave: () => drop.classList.remove('over'),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove('over'); doUpload(Array.from(e.dataTransfer.files)); } },
  h('p', 'Drop .txt / .jsonl / .parquet files here, or'), h('label', { class: 'btn', for: 'ds-file' }, 'Choose files'), fileInput, h('div', { class: 'mt-s' }, bar, status));
  uploadHost.append(drop);
  async function doUpload(files) {
    if (!files.length) return;
    bar.classList.remove('hidden');
    for (const f of files) {
      status.textContent = `Uploading ${f.name} (${fmtBytes(f.size)})...`;
      const fd = new FormData(); fd.append('file', f, f.name); fd.append('name', f.name);
      try { await api.upload('/datasets/upload', fd, (fr) => bar.set(fr)); toast(`Uploaded ${f.name}`, 'ok'); } catch (e) { toastError(e, 'Upload'); }
    }
    bar.classList.add('hidden'); status.textContent = ''; fileInput.value = ''; reload();
  }

  const reload = loadInto(host, () => api.get('/datasets'), (data) => {
    const rows = listOf(data, 'datasets');
    if (!rows.length) return emptyState('No datasets yet', 'Upload files above, or run a dataset_ingest job against a path or Hugging Face dataset.');
    return dataTable({
      columns: [
        { key: 'name', label: 'Name', render: (d) => h('div', h('b', d.name || d.id), h('div', { class: 'small muted mono' }, d.id)) },
        { key: 'total_tokens', label: 'Tokens', num: true, render: (d) => (d.total_tokens ? fmtNum(d.total_tokens) : '-') },
        { key: 'shards', label: 'Shards', num: true, render: (d) => (d.shards ? d.shards.length : '-'), sortValue: (d) => (d.shards || []).length },
        { key: 'mixture', label: 'Mixture', render: (d) => Object.entries(d.mixture || {}).map(([k, v]) => `${k} ${typeof v === 'number' && v <= 1 ? Math.round(v * 100) + '%' : fmtNum(v)}`).join(', ') || '-', noSort: true },
        { key: 'licence', label: 'Licence', render: (d) => d.licence || h('span', { class: 'faint' }, 'unknown') },
        { key: 'status', label: 'Status', render: (d) => statusBadge(d.status || (d.shards && d.shards.length ? 'ready' : 'uploaded')) },
        { key: 'created_at', label: 'Added', render: (d) => fmtRel(d.created_at) },
        { key: 'act', label: '', noSort: true, render: (d) => h('div', { class: 'row tight' },
          h('button', { class: 'btn sm', onclick: () => details(d) }, 'Details'), h('button', { class: 'btn sm', onclick: () => preview(d) }, 'Preview'),
          !(d.shards && d.shards.length) ? h('button', { class: 'btn sm primary', onclick: () => ingest(d) }, 'Ingest') : null,
          h('button', { class: 'btn sm danger', 'aria-label': `Delete ${d.name}`, onclick: () => del(d) }, 'Delete')) },
      ], rows, sort: { key: 'created_at', dir: 'desc' },
    });
  }, { feature: 'Datasets' });

  async function ingest(d) {
    const job = await runAction(() => api.post(`/datasets/${encodeURIComponent(d.id)}/ingest`, {}), 'Ingest job queued');
    if (job && job.id) navigate(`/jobs/${job.id}`);
  }
  async function del(d) {
    if (!(await confirmDialog({ title: 'Delete dataset?', message: `Delete "${d.name}"? Shards stay in the store until GC if other artifacts reference them.`, confirmLabel: 'Delete', danger: true }))) return;
    await runAction(() => api.del(`/datasets/${encodeURIComponent(d.id)}`), 'Dataset deleted'); reload();
  }
  async function preview(d) {
    const dlg = modal({ title: `Preview ${d.name || d.id}`, wide: true, body: h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Loading sample...') });
    try {
      const r = await api.get(`/datasets/${encodeURIComponent(d.id)}/preview`);
      const rows = r.records || r.rows || r.samples || r.documents || [];
      const extra = Object.fromEntries(Object.entries(r).filter(([k]) => !['records', 'rows', 'samples', 'documents', 'id'].includes(k)));
      dlg.setBody(h('div', { class: 'stack' }, Object.keys(extra).length ? kv(Object.entries(extra).map(([k, v]) => [k, objectView(v)])) : null,
        rows.length ? rows.map((t, i) => h('div', { class: 'card', style: 'box-shadow:none' }, h('div', { class: 'small muted' }, `sample ${i + 1}`), h('pre', { class: 'mono' }, typeof t === 'string' ? t : JSON.stringify(t, null, 2)))) : emptyState('No preview rows', 'The server returned no samples for this dataset.')));
    } catch (e) { dlg.setBody(h('div', { class: 'notice error' }, e.unavailable ? 'Preview is not available on this server.' : e.message)); }
  }
  function details(d) {
    const body = h('div', { class: 'stack' },
      kv([['Id', d.id], ['Tokenizer hash', hashLabel(d.tokenizer_hash)], ['Total tokens', fmtInt(d.total_tokens)], ['Filtering version', d.filtering_version || '-'], ['Licence', d.licence || '-']]),
      d.dedup_report && Object.keys(d.dedup_report).length ? h('div', h('h3', 'Dedup report'), objectView(d.dedup_report)) : null);
    if (d.shards && d.shards.length) {
      const cDiv = h('div');
      body.append(h('h3', `Shards (${d.shards.length})`), cDiv);
      const m = modal({ title: d.name || d.id, wide: true, body });
      new BarChart(cDiv, { title: 'Tokens per shard', height: 200, legend: false, categories: d.shards.map((_, i) => `#${i}`), series: [{ name: 'tokens', values: d.shards.map((s) => s.tokens) }], valueFormat: fmtNum });
      void m; return;
    }
    modal({ title: d.name || d.id, wide: true, body });
  }
}
