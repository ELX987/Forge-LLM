// New Job wizard: 1) pick a job type  2) configure (schema-driven form)  3) review & submit.

import { h, clear, listOf, titleCase } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, emptyState, loadingBlock, notice, unavailableNotice, toast, kv, badge, objectView } from '../components/ui.js';
import { JobConfigurator } from '../components/jobform.js';
import { navigate } from '../router.js';
import { dump } from '../yaml.js';

export async function fetchJobTypes() {
  const r = await tryGet('/schemas/job-types');
  if (!r.ok) throw r.error;
  const d = r.data;
  const list = Array.isArray(d) ? d : (d && (d.job_types || d.types || d.items)) || (d && typeof d === 'object' ? Object.entries(d).map(([type, v]) => ({ type, ...v })) : []);
  return list.map((t) => ({ ...t, title: t.title || titleCase(t.type), schema: t.schema || {}, defaults: t.defaults || {}, templates: (t.templates || []).map((x) => (typeof x === 'string' ? x : x.name)) }));
}

export default async function render(ctx) {
  const body = h('div');
  ctx.root.append(pageHead('New job', 'Every job type is configured from its pydantic schema - new config fields appear here automatically.'), body);
  body.appendChild(loadingBlock('Loading job types...'));
  let types;
  try { types = await fetchJobTypes(); } catch (e) { clear(body); body.appendChild(e.unavailable ? unavailableNotice('Job types', 'GET /api/v1/schemas/job-types is not available - the jobs/API backend is not installed on this server.') : notice('error', 'Could not load job types', e.message)); return; }
  if (!ctx.isActive()) return;

  const st = { step: 1, type: ctx.query.type || null, prefill: {}, configurator: null };
  // clone prefill
  if (ctx.query.from) {
    const r = await tryGet(`/jobs/${encodeURIComponent(ctx.query.from)}`);
    if (r.ok && r.data && r.data.spec) { st.type = r.data.spec.type; st.prefill = { params: r.data.spec.params, name: (r.data.spec.name || '') + ' (copy)', gpus: r.data.spec.gpus, tags: r.data.spec.tags }; toast('Loaded configuration from the previous job', 'info', 2500); }
  }
  if (st.type && types.find((t) => t.type === st.type)) st.step = 2; else st.type = null;

  const steps = h('div', { class: 'steps' });
  const stage = h('div');
  const nav = h('div', { class: 'row mt', style: 'justify-content:space-between' });
  body.replaceChildren(steps, stage, nav);

  function paintSteps() {
    clear(steps);
    ['Choose type', 'Configure', 'Review & submit'].forEach((l, i) => steps.appendChild(h('div', { class: 'step' + (st.step === i + 1 ? ' active' : st.step > i + 1 ? ' done' : '') }, `${i + 1}. ${l}`)));
  }

  function go(step) {
    st.step = step;
    if (step < 3 || !st.configurator) { /* keep configurator across steps */ }
    paint();
  }

  function paint() {
    paintSteps(); clear(stage); clear(nav);
    const jt = types.find((t) => t.type === st.type);
    if (st.step === 1) {
      const grid = h('div', { class: 'grid auto' });
      for (const t of types) {
        grid.appendChild(h('button', { type: 'button', class: 'type-card' + (st.type === t.type ? ' sel' : ''), onclick: () => { if (st.type !== t.type) { st.configurator = null; st.prefill = {}; } st.type = t.type; go(2); } },
          h('div', { class: 't' }, t.title, ' ', badge(t.type, 'dim')), h('div', { class: 'd' }, t.description || '')));
      }
      stage.appendChild(types.length ? grid : emptyState('No job types registered'));
      return;
    }
    if (!jt) { go(1); return; }
    if (st.step === 2) {
      if (st.configurator && st.configurator.type === jt.type && st.holder) stage.appendChild(st.holder);
      else {
        if (st.configurator) st.configurator.destroy();
        const host = h('div'); stage.appendChild(host);
        st.configurator = new JobConfigurator(host, { jobType: jt, prefill: st.prefill });
        st.holder = host;
      }
      nav.append(h('button', { class: 'btn', onclick: () => go(1) }, '← Change type'), h('button', { class: 'btn primary', onclick: () => { go(3); } }, 'Review →'));
      return;
    }
    if (st.step === 3) {
      const spec = st.configurator ? st.configurator.getSpec() : null;
      if (!spec) { go(2); return; }
      const yaml = dump(spec.params);
      const errs = st.configurator.form.validateNow(false);
      stage.append(h('div', { class: 'grid cols-2' },
        card('Summary', h('div', kv([['Type', `${jt.title} (${jt.type})`], ['Name', spec.name], ['GPUs', spec.gpus ? spec.gpus.join(', ') : 'auto / CPU'], ['Tags', spec.tags.join(', ') || '-']]),
          errs.length ? notice('error', `${errs.length} validation issue(s)`, h('ul', { class: 'errs' }, errs.slice(0, 10).map((e) => h('li', h('code', e.path || '(root)'), ': ', e.message)))) : h('p', { class: 'ok mt' }, 'Config is valid against the schema.'))),
        card('Parameters (YAML)', h('pre', { class: 'mono', style: 'max-height:420px;overflow:auto' }, yaml))));
      const submit = h('button', { class: 'btn primary', onclick: async () => {
        submit.disabled = true;
        try { const job = await st.configurator.submit({ force: false }); if (job && job.id) { toast('Job submitted', 'ok'); navigate(`/jobs/${job.id}`); } else submit.disabled = false; } catch (_) { submit.disabled = false; go(2); }
      } }, 'Submit job');
      nav.append(h('button', { class: 'btn', onclick: () => go(2) }, '← Edit'), submit);
    }
  }
  paint();
  ctx.onCleanup(() => { if (st.configurator) st.configurator.destroy(); });
  void listOf; void api; void objectView;
}
