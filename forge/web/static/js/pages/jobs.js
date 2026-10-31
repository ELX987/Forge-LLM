// Jobs list: filter, live refresh, cancel / clone / delete.

import { h, clear, listOf, fmtRel, fmtDuration, fmtPct } from '../util.js';
import { api } from '../api.js';
import { pageHead, dataTable, statusBadge, progressBar, confirmDialog, emptyState, errorNotice, unavailableNotice, loadingBlock, poll, toast, toastError, badge } from '../components/ui.js';
import { navigate } from '../router.js';

const STATUSES = ['running', 'queued', 'succeeded', 'failed', 'cancelled'];

export default async function render(ctx) {
  const st = { status: ctx.query.status || '', type: ctx.query.type || '', q: ctx.query.q || '', jobs: [], loaded: false };
  const host = h('div');
  const typeSel = h('select', { 'aria-label': 'Filter by type', onchange: () => { st.type = typeSel.value; sync(); } }, h('option', { value: '' }, 'All types'));
  const search = h('input', { type: 'search', placeholder: 'Search name, id, tag...', value: st.q, 'aria-label': 'Search jobs', oninput: () => { st.q = search.value; sync(); } });
  const chips = h('div', { class: 'chips' });
  const paintChips = () => {
    clear(chips);
    for (const s of ['', ...STATUSES]) {
      const n = s ? st.jobs.filter((j) => j.status === s).length : st.jobs.length;
      chips.appendChild(h('button', { type: 'button', class: 'chip' + (st.status === s ? ' on' : ''), 'aria-pressed': String(st.status === s), onclick: () => { st.status = s; sync(); } }, s || 'all', ' ', h('b', String(n))));
    }
  };
  ctx.root.append(pageHead('Jobs', 'All training, fine-tuning, compression, eval and serving jobs.', h('a', { class: 'btn primary', href: '#/jobs/new' }, '+ New job')), h('div', { class: 'toolbar' }, chips, typeSel, search), host);

  let table = null;
  const filtered = () => st.jobs.filter((j) => (!st.status || j.status === st.status) && (!st.type || j.spec.type === st.type)
    && (!st.q || `${j.spec.name} ${j.id} ${(j.spec.tags || []).join(' ')} ${j.spec.type}`.toLowerCase().includes(st.q.toLowerCase())));
  function sync() {
    ctx.replaceQuery({ status: st.status, type: st.type, q: st.q });
    paintChips();
    if (table) table.setRows(filtered());
  }

  function build() {
    const types = Array.from(new Set(st.jobs.map((j) => j.spec.type))).sort();
    const cur = typeSel.value;
    clear(typeSel); typeSel.appendChild(h('option', { value: '' }, 'All types'));
    for (const t of types) typeSel.appendChild(h('option', { value: t }, t));
    typeSel.value = types.includes(st.type) ? st.type : (types.includes(cur) ? cur : '');
    table = dataTable({
      columns: [
        { key: 'name', label: 'Job', render: (j) => h('div', h('a', { href: `#/jobs/${j.id}` }, h('b', j.spec.name || j.id)), h('div', { class: 'small muted mono' }, j.id)), sortValue: (j) => j.spec.name },
        { key: 'type', label: 'Type', render: (j) => badge(j.spec.type, 'dim'), sortValue: (j) => j.spec.type },
        { key: 'status', label: 'Status', render: (j) => statusBadge(j.status) },
        { key: 'progress', label: 'Progress', render: (j) => h('div', { style: 'min-width:120px' }, progressBar(j.progress, j.status === 'failed' ? 'bad' : j.status === 'succeeded' ? 'ok' : ''), h('span', { class: 'small muted' }, j.progress_msg || fmtPct(j.progress, 0))) },
        { key: 'gpus', label: 'GPUs', render: (j) => (j.spec.gpus && j.spec.gpus.length ? j.spec.gpus.join(',') : '-'), noSort: true },
        { key: 'created_at', label: 'Created', render: (j) => fmtRel(j.created_at) },
        { key: 'dur', label: 'Duration', render: (j) => (j.started_at ? fmtDuration((j.finished_at || Date.now() / 1000) - j.started_at) : '-'), sortValue: (j) => (j.started_at ? (j.finished_at || Date.now() / 1000) - j.started_at : 0), num: true },
        { key: 'act', label: '', noSort: true, render: (j) => h('div', { class: 'row tight' },
          ['running', 'queued'].includes(j.status) ? h('button', { class: 'btn sm danger', onclick: () => cancel(j) }, 'Cancel') : null,
          h('a', { class: 'btn sm', href: `#/jobs/new?from=${encodeURIComponent(j.id)}` }, 'Clone'),
          !['running', 'queued'].includes(j.status) ? h('button', { class: 'btn sm ghost', 'aria-label': `Delete ${j.spec.name}`, onclick: () => del(j) }, 'Delete') : null) },
      ], rows: filtered(), sort: { key: 'created_at', dir: 'desc' }, onRowClick: (j) => navigate(`/jobs/${j.id}`),
      empty: h('div', h('p', 'No jobs match.'), h('a', { class: 'btn primary', href: '#/jobs/new' }, 'Start a job')),
    });
    clear(host); host.appendChild(table); paintChips();
  }

  async function load(first) {
    try {
      st.jobs = listOf(await api.get('/jobs', { limit: 500 }), 'jobs');
      if (first || !table) build(); else { paintChips(); table.setRows(filtered()); }
    } catch (e) {
      if (first) { clear(host); host.appendChild(e.unavailable ? unavailableNotice('Jobs') : errorNotice(e, () => load(true))); }
    }
  }
  host.appendChild(loadingBlock());
  await load(true);
  poll(ctx, () => load(false), 3000);

  async function cancel(j) { if (!(await confirmDialog({ title: 'Cancel job?', message: `Send a cooperative stop to "${j.spec.name}" (SIGTERM, then SIGKILL after the grace period).`, confirmLabel: 'Cancel job', danger: true }))) return; try { await api.post(`/jobs/${encodeURIComponent(j.id)}/cancel`, {}); toast('Cancel requested', 'ok'); load(false); } catch (e) { toastError(e); } }
  async function del(j) { if (!(await confirmDialog({ title: 'Delete job record?', message: `Delete "${j.spec.name}" and its logs/metrics from the database?`, confirmLabel: 'Delete', danger: true }))) return; try { await api.del(`/jobs/${encodeURIComponent(j.id)}`); toast('Deleted', 'ok'); load(false); } catch (e) { toastError(e); } }
  void emptyState;
}
