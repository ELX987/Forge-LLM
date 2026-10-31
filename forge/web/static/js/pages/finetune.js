// Fine-tune: shortcut wizard over the `finetune` job type. Quick pickers (model, dataset, method, LR, steps, LoRA rank)
// write into whatever schema fields exist; the full schema-driven form stays available below.

import { h, clear, listOf, getPath, setPath, deepClone, titleCase } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, notice, unavailableNotice, loadingBlock, toast, toastError, badge, emptyState } from '../components/ui.js';
import { JobConfigurator } from '../components/jobform.js';
import { fetchJobTypes } from './newjob.js';
import { navigate } from '../router.js';

function propAt(schema, dotted) {
  let s = schema;
  for (const p of dotted.split('.')) {
    if (!s || !s.properties || !s.properties[p]) return null;
    s = s.properties[p];
    if (Array.isArray(s.anyOf)) { const nn = s.anyOf.find((b) => b.type !== 'null'); if (nn) s = { ...nn, ...s, anyOf: undefined }; }
  }
  return s;
}
function findProp(schema, candidates, typePred) {
  for (const c of candidates) { const p = propAt(schema, c); if (p && (!typePred || typePred(p))) return { path: c, schema: p }; }
  return null;
}

export default async function render(ctx) {
  const body = h('div');
  ctx.root.append(pageHead('Fine-tune', 'Full, LoRA or QLoRA fine-tuning of a registered model on an SFT dataset. Quick pickers on top, every other option below.'), body);
  body.appendChild(loadingBlock());
  let types;
  try { types = await fetchJobTypes(); } catch (e) { clear(body); body.appendChild(e.unavailable ? unavailableNotice('Fine-tuning', 'The jobs backend (job-type schemas) is not available on this server.') : notice('error', 'Could not load job types', e.message)); return; }
  const jt = types.find((t) => t.type === 'finetune');
  if (!jt) { clear(body); body.appendChild(unavailableNotice('Fine-tune job type', 'This server does not register a "finetune" job type (forge.train not installed).')); return; }
  const [models, datasets] = await Promise.all([tryGet('/models'), tryGet('/datasets')]);
  if (!ctx.isActive()) return;
  const modelList = models.ok ? listOf(models.data, 'models') : [];
  const dsList = datasets.ok ? listOf(datasets.data, 'datasets') : [];
  clear(body);

  const quickHost = h('div');
  const formHost = h('div');
  body.append(quickHost, formHost);
  const cfg = new JobConfigurator(formHost, { jobType: jt, prefill: ctx.query.model ? { params: {} } : {} });
  ctx.onCleanup(() => cfg.destroy());

  const S = jt.schema;
  const F = {
    model: findProp(S, ['model', 'base_model', 'model_id', 'model_path', 'pretrained_model'], (p) => p.type === 'string'),
    dataset: findProp(S, ['data.dataset', 'dataset', 'dataset_id', 'train_dataset', 'data.train_dataset'], (p) => p.type === 'string'),
    method: findProp(S, ['method', 'mode', 'finetune_type', 'ft_method'], (p) => Array.isArray(p.enum)),
    lr: findProp(S, ['optimizer.lr', 'lr', 'learning_rate'], (p) => p.type === 'number'),
    steps: findProp(S, ['max_steps', 'steps', 'epochs', 'num_epochs', 'schedule.max_steps'], (p) => p.type === 'integer' || p.type === 'number'),
    rank: findProp(S, ['lora.r', 'lora.rank', 'lora_r', 'lora_rank', 'rank'], (p) => p.type === 'integer'),
  };
  const setField = (f, value) => {
    const v = cfg.form.getValue();
    setPath(v, f.path, value);
    if (f.path === 'data.dataset' || f.path === 'dataset') { const alt = F.dataset.path === 'dataset' ? 'data.dataset' : 'dataset'; if (propAt(S, alt) && propAt(S, alt).type === 'string') setPath(v, alt, value); }
    cfg.form.setValue(v);
  };
  const cur = (f, def = '') => { const x = getPath(cfg.form.getValue(), f.path); return x === undefined || x === null ? def : x; };

  const field = (label, control, hint) => h('div', { class: 'stack', style: 'min-width:200px' }, h('label', { class: 'small muted' }, label), control, hint ? h('div', { class: 'small faint' }, hint) : null);
  const quick = h('div', { class: 'grid auto' });
  if (F.model) {
    const sel = h('select', { 'aria-label': 'Base model' }, h('option', { value: '' }, 'Choose registered model...'), ...modelList.map((m) => h('option', { value: m.id }, `${m.name || m.id}${m.params_b ? ` (${m.params_b}B)` : ''}`)));
    sel.addEventListener('change', () => { if (sel.value) setField(F.model, sel.value); });
    const txt = h('input', { type: 'text', value: cur(F.model), placeholder: 'or type a model id / path', 'aria-label': 'Model id or path', onchange: () => setField(F.model, txt.value) });
    if (ctx.query.model) { setField(F.model, ctx.query.model); txt.value = ctx.query.model; }
    quick.appendChild(field('Base model', h('div', { class: 'stack' }, sel, txt), modelList.length ? '' : 'No registered models - type an id or path.'));
  }
  if (F.dataset) {
    const sel = h('select', { 'aria-label': 'Dataset' }, h('option', { value: '' }, 'Choose dataset...'), ...dsList.map((d) => h('option', { value: d.id }, `${d.name || d.id}${d.total_tokens ? ` (${Math.round(d.total_tokens / 1e6)}M tok)` : ''}`)));
    sel.addEventListener('change', () => { if (sel.value) setField(F.dataset, sel.value); });
    quick.appendChild(field('SFT dataset', sel, dsList.length ? '' : 'No datasets yet - upload one on the Datasets page.'));
  }
  if (F.method) {
    const sel = h('select', { 'aria-label': 'Method' }, F.method.schema.enum.map((m) => h('option', { value: m }, String(m).toUpperCase() === String(m) ? m : titleCase(String(m)))));
    sel.value = cur(F.method, F.method.schema.default ?? F.method.schema.enum[0]);
    sel.addEventListener('change', () => setField(F.method, sel.value));
    quick.appendChild(field('Method', sel, 'LoRA trains a small adapter; full updates every weight (needs far more memory).'));
  }
  const num = (f, label, hint, step) => {
    const inp = h('input', { type: 'number', step: step || 'any', value: cur(f), 'aria-label': label });
    inp.addEventListener('change', () => { if (inp.value !== '') setField(f, Number(inp.value)); });
    quick.appendChild(field(label, inp, hint));
  };
  if (F.lr) num(F.lr, 'Learning rate', 'Typical: 1e-4 - 3e-4 for LoRA, 1e-5 - 3e-5 for full.');
  if (F.steps) num(F.steps, titleCase(F.steps.path.split('.').pop()), '', 1);
  if (F.rank) num(F.rank, 'LoRA rank', 'Higher = more capacity and memory.', 1);
  const note = Object.values(F).every((x) => !x) ? notice('info', 'Quick pickers unavailable', 'The finetune schema uses field names this shortcut does not recognise; use the full form below.') : null;

  const submitRow = h('div', { class: 'row mt' }, h('button', { class: 'btn primary', onclick: async (e) => {
    e.target.disabled = true;
    try { const job = await cfg.submit(); if (job && job.id) { toast('Fine-tune job submitted', 'ok'); navigate(`/jobs/${job.id}`); } } catch (_) { /* painted */ } finally { e.target.disabled = false; }
  } }, 'Start fine-tuning'), h('span', { class: 'muted small' }, 'You can also save the configuration as a reusable preset in the form below.'));
  quickHost.append(card('Quick setup', h('div', note || quick), null, 'mb'), submitRow, h('h2', { class: 'mt' }, 'Full configuration'));
  void emptyState; void badge; void api; void toastError; void deepClone;
}
