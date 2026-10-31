// Claims ledger: every paper claim the system relies on, with status, prior trust, linked experiments and dependent rules.

import { h, clear, listOf, parseCsvList } from '../util.js';
import { api } from '../api.js';
import { pageHead, card, dataTable, statusBadge, badge, loadInto, emptyState, modal, toast, toastError, kv, help } from '../components/ui.js';

const STATUSES = ['untested', 'reproduced', 'contradicted', 'partial'];

export default async function render(ctx) {
  const st = { q: ctx.query.q || '', status: ctx.query.status || '', paper: ctx.query.paper || '', attention: ctx.query.attention === '1', claims: [] };
  const host = h('div');
  const chips = h('div', { class: 'chips' });
  const paperSel = h('select', { 'aria-label': 'Paper', onchange: () => { st.paper = paperSel.value; sync(); } }, h('option', { value: '' }, 'All papers'));
  const search = h('input', { type: 'search', placeholder: 'Search claims...', value: st.q, 'aria-label': 'Search claims', oninput: () => { st.q = search.value; sync(); } });
  const att = h('label', { class: 'row tight' }, h('input', { type: 'checkbox', checked: st.attention, onchange: (e) => { st.attention = e.target.checked; sync(); } }), 'Needs attention only', help('Not yet reproduced AND at least one decision rule depends on it.'));
  ctx.root.append(pageHead('Claims ledger', 'Numbers from papers are hypotheses. Each claim here is tested by experiments before the planner\'s rules may rely on it.'), h('div', { class: 'toolbar' }, chips, paperSel, search, att), host);
  let table = null;
  const filtered = () => st.claims.filter((c) => (!st.status || c.status === st.status) && (!st.paper || c.source_paper === st.paper)
    && (!st.q || `${c.id} ${c.statement} ${c.notes} ${c.location}`.toLowerCase().includes(st.q.toLowerCase())) && (!st.attention || (c.status !== 'reproduced' && (c.depends_rules || []).length)));
  function sync() { ctx.replaceQuery({ q: st.q, status: st.status, paper: st.paper, attention: st.attention ? '1' : '' }); paintChips(); if (table) table.setRows(filtered()); }
  function paintChips() {
    clear(chips);
    for (const s of ['', ...STATUSES]) { const n = s ? st.claims.filter((c) => c.status === s).length : st.claims.length; chips.appendChild(h('button', { type: 'button', class: 'chip' + (st.status === s ? ' on' : ''), 'aria-pressed': String(st.status === s), onclick: () => { st.status = s; sync(); } }, s || 'all', ' ', h('b', String(n)))); }
  }

  const reload = loadInto(host, () => api.get('/claims'), (data) => {
    st.claims = listOf(data, 'claims');
    if (!st.claims.length) return emptyState('Ledger is empty', 'The planner seeds the ledger with claims from the supporting papers on first start (forge.planner.claims.seed_claims).');
    const papers = Array.from(new Set(st.claims.map((c) => c.source_paper))).sort();
    clear(paperSel); paperSel.appendChild(h('option', { value: '' }, 'All papers')); for (const p of papers) paperSel.appendChild(h('option', { value: p, selected: p === st.paper }, p));
    paintChips();
    table = dataTable({ columns: [
      { key: 'id', label: 'Claim', render: (c) => h('div', h('b', c.id), h('div', { class: 'small muted' }, `${c.source_paper}${c.location ? ' - ' + c.location : ''}`)) },
      { key: 'statement', label: 'Statement', render: (c) => h('div', { style: 'max-width:520px' }, c.statement, c.notes ? h('div', { class: 'small muted mt-s' }, c.notes) : null), noSort: true },
      { key: 'status', label: 'Status', render: (c) => statusBadge(c.status), sortValue: (c) => STATUSES.indexOf(c.status) },
      { key: 'prior', label: 'Prior', render: (c) => statusBadge(c.prior) },
      { key: 'rules', label: 'Rules', num: true, render: (c) => (c.depends_rules || []).length, sortValue: (c) => (c.depends_rules || []).length },
      { key: 'exp', label: 'Experiments', num: true, render: (c) => (c.linked_experiments || []).length, sortValue: (c) => (c.linked_experiments || []).length },
    ], rows: filtered(), sort: { key: 'status', dir: 'desc' }, onRowClick: (c) => detail(c), caption: 'Claims' });
    if (ctx.query.id) { const c = st.claims.find((x) => x.id === ctx.query.id); if (c) setTimeout(() => detail(c), 50); }
    return table;
  }, { feature: 'Claims ledger' });

  function detail(c) {
    const statusSel = h('select', { 'aria-label': 'Status' }, STATUSES.map((s) => h('option', { value: s, selected: s === c.status }, s)));
    const priorSel = h('select', { 'aria-label': 'Prior' }, ['trusted', 'untrusted'].map((s) => h('option', { value: s, selected: s === c.prior }, s)));
    const notes = h('textarea', { rows: 4, 'aria-label': 'Notes', style: 'font-family:var(--sans)' }); notes.value = c.notes || '';
    const exps = h('input', { type: 'text', value: (c.linked_experiments || []).join(', '), 'aria-label': 'Linked experiments', style: 'width:100%', placeholder: 'job or eval ids, comma separated' });
    const err = h('div', { class: 'sf-err', role: 'alert' });
    const save = h('button', { class: 'btn primary', onclick: async () => {
      save.disabled = true;
      const updated = { ...c, status: statusSel.value, prior: priorSel.value, notes: notes.value, linked_experiments: parseCsvList(exps.value) };
      try {
        try { await api.put('/claims', updated); } catch (e) { if (e.status === 404 || e.status === 405) await api.put(`/claims/${encodeURIComponent(c.id)}`, updated); else throw e; }
        Object.assign(c, updated); toast('Claim updated', 'ok'); m.close(); sync(); reload(true);
      } catch (e) { save.disabled = false; err.textContent = e.unavailable ? 'Updating claims is not available on this server.' : e.message; toastError(e); }
    } }, 'Save');
    const m = modal({ title: c.id, wide: true,
      body: h('div', { class: 'stack' }, h('p', { style: 'font-size:1.05rem' }, c.statement), kv([['Source', `${c.source_paper}${c.location ? ' - ' + c.location : ''}`], ['Depends-on rules', (c.depends_rules || []).length ? h('div', { class: 'chips' }, c.depends_rules.map((r) => badge(r, 'accent'))) : 'none'],
        ['Linked experiments', (c.linked_experiments || []).length ? h('div', { class: 'chips' }, c.linked_experiments.map((x) => (/^job_/.test(x) ? h('a', { class: 'chip', href: `#/jobs/${x}` }, x) : (/^ev/.test(x) ? h('a', { class: 'chip', href: `#/compare?ids=${x}` }, x) : badge(x))))) : 'none']]),
        h('div', { class: 'grid cols-2' }, h('div', h('label', { class: 'small muted' }, 'Status'), statusSel), h('div', h('label', { class: 'small muted' }, 'Prior trust'), priorSel)),
        h('div', h('label', { class: 'small muted' }, 'Linked experiments'), exps), h('div', h('label', { class: 'small muted' }, 'Notes'), notes), err,
        h('p', { class: 'small muted' }, 'Mark a claim "reproduced" only when a linked experiment (measured, with CI) supports it. "paper-reported" numbers elsewhere in the UI stay tagged until then.')),
      footer: [h('button', { class: 'btn', onclick: () => m.close() }, 'Close'), save] });
  }
  void card;
}
