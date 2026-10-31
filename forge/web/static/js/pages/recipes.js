// Recipes: list + builder (stage palette, live YAML, validation, planner fit / bound / ceiling panel and suggestions).

import { h, clear, listOf, fmtNum, fmtRel, debounce, deepClone, isNum, titleCase } from '../util.js';
import { api, tryGet } from '../api.js';
import { pageHead, card, loadInto, dataTable, badge, provBadge, confirmDialog, toast, toastError, notice, unavailableNotice, emptyState, loadingBlock, objectView, statusBadge, help, kv } from '../components/ui.js';
import { formDialog } from '../components/dialogs.js';
import { dump, parseAny } from '../yaml.js';
import { navigate } from '../router.js';

const PALETTE = [
  { name: 'calibrate', desc: 'Collect Hessian / channel-scale / router statistics on a calibration set.', params: { samples: 128, seq_len: 2048 } },
  { name: 'ptq', desc: 'Post-training quantisation (rtn, gptq, awq, smoothquant).', params: { method: 'gptq', bits: 4, group_size: 128, act_order: true } },
  { name: 'pack', desc: 'Pack quantised weights into the hardware-derived kernel layout.', params: { layout: 'auto' } },
  { name: 'kv', desc: 'KV-cache quantisation (FP8, KIVI-style per-channel keys).', params: { dtype: 'fp8_e4m3', residual_window: 128 } },
  { name: 'prune', desc: 'Structured pruning: GQA-aware width, heads, depth, experts.', params: { mode: 'width', ratio: 0.1 } },
  { name: 'distill', desc: 'Recover quality by distilling from the parent (licence gated).', params: { loss: 'forward_kl', steps: 1000 } },
  { name: 'lowrank', desc: 'Whitened-SVD low-rank factorisation with break-even guard.', params: { rank_fraction: 0.5, whiten: true } },
  { name: 'delta', desc: 'Low-rank or 1-bit-sign delta from a base model.', params: { mode: 'lowrank', rank: 16 } },
  { name: 'lossless', desc: 'Artifact-level lossless entropy coding (bit-exact).', params: { codec: 'exp_split_zstd' } },
  { name: 'finetune', desc: 'Short fine-tune / healing between stages.', params: { method: 'lora', steps: 500 } },
  { name: 'eval', desc: 'Evaluate against the parent with noise-aware statistics.', params: { suites: ['distribution'] } },
  { name: 'gate', desc: 'Apply a pass / fail / inconclusive gate to the eval report.', params: { gate: 'G-W4' } },
];

const EMPTY_RECIPE = () => ({ name: 'new-recipe', inputs: { model: '' }, stages: [{ name: 'calibrate', params: { samples: 128, seq_len: 2048 } }, { name: 'ptq', params: { method: 'gptq', bits: 4, group_size: 128, act_order: true } }, { name: 'eval', params: { suites: ['distribution'] } }], baselines: [], budget: {}, policy: { licence_check: true }, seeds: [0] });

export default async function render(ctx) {
  if (ctx.params.id) return editor(ctx);
  return list(ctx);
}

// ------------------------------------------------------------------------------------------ list
async function list(ctx) {
  const host = h('div');
  ctx.root.append(pageHead('Recipes', 'Declarative pipelines of cached, hash-addressed stages. Change one stage and only downstream stages re-run.', h('a', { class: 'btn primary', href: '#/recipes/new' }, '+ New recipe')), host);
  const reload = loadInto(host, () => api.get('/recipes'), (data) => {
    const rows = listOf(data, 'recipes');
    if (!rows.length) return emptyState('No recipes yet', 'Build one from the stage palette and let the planner estimate fit before spending GPU hours.', h('a', { class: 'btn primary', href: '#/recipes/new' }, 'Create a recipe'));
    return dataTable({ columns: [
      { key: 'name', label: 'Recipe', render: (r) => h('a', { href: `#/recipes/${encodeURIComponent(r.id)}` }, h('b', r.name || r.id)) },
      { key: 'n_stages', label: 'Stages', num: true, render: (r) => r.n_stages ?? (r.stages ? r.stages.length : '-') },
      { key: 'created_at', label: 'Created', render: (r) => fmtRel(r.created_at) },
      { key: 'act', label: '', noSort: true, render: (r) => h('div', { class: 'row tight' }, h('a', { class: 'btn sm', href: `#/recipes/${encodeURIComponent(r.id)}` }, 'Open'),
        h('button', { class: 'btn sm primary', onclick: () => run(r) }, 'Run'), h('button', { class: 'btn sm danger', onclick: () => del(r) }, 'Delete')) },
    ], rows, sort: { key: 'created_at', dir: 'desc' }, onRowClick: (r) => navigate(`/recipes/${r.id}`) });
  }, { feature: 'Recipes' });
  async function run(r) { try { const j = await api.post(`/recipes/${encodeURIComponent(r.id)}/run`, {}); toast('Recipe job queued', 'ok'); if (j && j.id) navigate(`/jobs/${j.id}`); } catch (e) { toastError(e); } }
  async function del(r) { if (!(await confirmDialog({ title: 'Delete recipe?', message: r.name, confirmLabel: 'Delete', danger: true }))) return; try { await api.del(`/recipes/${encodeURIComponent(r.id)}`); reload(); } catch (e) { toastError(e); } }
}

// ------------------------------------------------------------------------------------------ editor
function deriveQuant(rec) {
  let quant = 'bf16'; let kv = 'bf16';
  for (const s of rec.stages || []) {
    const p = s.params || {};
    if (s.name === 'ptq') {
      const fmt = String(p.fmt || p.format || p.dtype || '');
      if (/nvfp4/i.test(fmt)) quant = 'nvfp4'; else if (/mxfp4/i.test(fmt)) quant = 'mxfp4'; else if (/fp8/i.test(fmt)) quant = 'fp8';
      else if (Number(p.bits) === 4) quant = 'w4a16'; else if (Number(p.bits) === 8) quant = 'w8a16';
    }
    if (s.name === 'kv' && p.dtype) kv = String(p.dtype);
  }
  return { quant, kv };
}

async function editor(ctx) {
  const rid = ctx.params.id; const isNew = rid === 'new';
  const root = ctx.root;
  root.appendChild(loadingBlock());
  let saved = null;
  let rec = EMPTY_RECIPE();
  if (!isNew) {
    try {
      saved = await api.get(`/recipes/${encodeURIComponent(rid)}`);
      const y = saved.yaml || saved.content || (saved.recipe ? null : null);
      if (y) rec = parseAny(y); else if (saved.recipe) rec = saved.recipe; else if (saved.stages) rec = saved;
    } catch (e) { clear(root); root.append(e.status === 404 ? emptyState('Recipe not found', rid, h('a', { class: 'btn', href: '#/recipes' }, 'All recipes')) : (e.unavailable ? unavailableNotice('Recipes') : notice('error', 'Failed to load recipe', e.message))); return; }
  }
  const [models, chips] = await Promise.all([tryGet('/models'), tryGet('/hardware/chips')]);
  if (!ctx.isActive()) return;
  clear(root);
  const modelList = models.ok ? listOf(models.data, 'models') : [];
  const chipData = chips.ok ? chips.data : null;
  const chipList = Array.isArray(chipData) ? chipData : (chipData && (chipData.chips || chipData.known)) || [];

  const S = { rec, id: isNew ? null : rid, dirty: false, planner: { model: modelList[0] ? modelList[0].id : '', chip: chipList.find((c) => /h100/i.test(c.name)) ? chipList.find((c) => /h100/i.test(c.name)).name : (chipList[0] ? chipList[0].name : ''), ctxLen: 4096, batch: 8, quantOverride: '', kvOverride: '' } };
  rec.stages = rec.stages || []; rec.inputs = rec.inputs || {}; rec.baselines = rec.baselines || []; rec.budget = rec.budget || {}; rec.policy = rec.policy || {}; rec.seeds = rec.seeds || [0];

  const stagesHost = h('div', { class: 'stack' });
  const yamlEl = h('textarea', { class: 'sf-raw', rows: 22, spellcheck: 'false', 'aria-label': 'Recipe YAML', style: 'min-height:320px' });
  const yamlErr = h('div', { class: 'sf-err', role: 'alert' });
  const valHost = h('div');
  const nameEl = h('input', { type: 'text', value: rec.name || '', 'aria-label': 'Recipe name', style: 'min-width:240px', oninput: () => { S.rec.name = nameEl.value; syncYaml(); } });

  function syncYaml() { S.dirty = true; if (document.activeElement !== yamlEl) yamlEl.value = dump(S.rec); validateSoon(); estimateSoon(); }

  // ---- stage list
  function paintStages() {
    clear(stagesHost);
    if (!S.rec.stages.length) stagesHost.appendChild(emptyState('No stages', 'Click a stage in the palette to add it.'));
    S.rec.stages.forEach((st, i) => {
      const pal = PALETTE.find((p) => p.name === st.name);
      const paramsEl = h('textarea', { rows: Math.min(8, Math.max(2, Object.keys(st.params || {}).length + 1)), spellcheck: 'false', 'aria-label': `Parameters for stage ${i + 1}`, style: 'font-size:.85rem' });
      paramsEl.value = Object.keys(st.params || {}).length ? dump(st.params) : '';
      const err = h('div', { class: 'sf-err' });
      paramsEl.addEventListener('input', () => { try { const v = paramsEl.value.trim() ? parseAny(paramsEl.value) : {}; if (v && typeof v === 'object' && !Array.isArray(v)) { st.params = v; err.textContent = ''; paramsEl.classList.remove('invalid'); syncYaml(); } else throw new Error('must be a mapping'); } catch (e) { err.textContent = e.message; paramsEl.classList.add('invalid'); } });
      const move = (d) => { const j = i + d; if (j < 0 || j >= S.rec.stages.length) return; const a = S.rec.stages; [a[i], a[j]] = [a[j], a[i]]; paintStages(); syncYaml(); };
      const nameSel = h('select', { 'aria-label': 'Stage type', onchange: () => { st.name = nameSel.value; const p = PALETTE.find((x) => x.name === st.name); if (p && !Object.keys(st.params || {}).length) st.params = deepClone(p.params); paintStages(); syncYaml(); } }, PALETTE.map((p) => h('option', { value: p.name, selected: p.name === st.name }, p.name)), PALETTE.some((p) => p.name === st.name) ? null : h('option', { value: st.name, selected: true }, st.name));
      stagesHost.appendChild(h('div', { class: 'card', style: 'box-shadow:none' },
        h('div', { class: 'row' }, h('b', `${i + 1}.`), nameSel, pal ? help(pal.desc) : badge('unknown stage', 'warn'), h('span', { class: 'grow' }),
          h('button', { class: 'btn sm ghost', 'aria-label': 'Move up', disabled: i === 0, onclick: () => move(-1) }, '↑'), h('button', { class: 'btn sm ghost', 'aria-label': 'Move down', disabled: i === S.rec.stages.length - 1, onclick: () => move(1) }, '↓'),
          h('button', { class: 'btn sm ghost', title: 'Duplicate', onclick: () => { S.rec.stages.splice(i + 1, 0, deepClone(st)); paintStages(); syncYaml(); } }, 'Copy'),
          h('button', { class: 'btn sm ghost', 'aria-label': 'Remove stage', onclick: () => { S.rec.stages.splice(i, 1); paintStages(); syncYaml(); } }, '×')),
        h('div', { class: 'mt-s' }, paramsEl, err)));
    });
  }

  // ---- other recipe sections as YAML textareas
  function sectionEditor(key, label, hint, asList) {
    const ta = h('textarea', { rows: 3, spellcheck: 'false', 'aria-label': label, style: 'font-size:.85rem' });
    const show = () => { const v = S.rec[key]; ta.value = (v == null || (Array.isArray(v) ? !v.length : !Object.keys(v).length)) ? '' : dump(v); };
    show();
    const err = h('div', { class: 'sf-err' });
    ta.addEventListener('input', () => { try { const v = ta.value.trim() ? parseAny(ta.value) : (asList ? [] : {}); if (asList ? !Array.isArray(v) : (typeof v !== 'object' || Array.isArray(v) || v === null)) throw new Error(asList ? 'must be a list' : 'must be a mapping'); S.rec[key] = v; err.textContent = ''; ta.classList.remove('invalid'); syncYaml(); } catch (e) { err.textContent = e.message; ta.classList.add('invalid'); } });
    return h('div', h('label', { class: 'small muted' }, label, ' ', hint ? help(hint) : null), ta, err);
  }

  // ---- palette
  const palette = h('div', { class: 'chips' }, PALETTE.map((p) => h('button', { type: 'button', class: 'chip', 'data-tip': p.desc, onclick: () => { S.rec.stages.push({ name: p.name, params: deepClone(p.params) }); paintStages(); syncYaml(); } }, '+ ', p.name)));

  // ---- YAML two-way
  yamlEl.value = dump(S.rec);
  yamlEl.addEventListener('input', debounce(() => {
    try { const v = parseAny(yamlEl.value); if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('top level must be a mapping'); S.rec = Object.assign({ name: '', inputs: {}, stages: [], baselines: [], budget: {}, policy: {}, seeds: [0] }, v); yamlErr.textContent = ''; yamlEl.classList.remove('invalid'); nameEl.value = S.rec.name || ''; S.dirty = true; paintStages(); validateSoon(); estimateSoon(); } catch (e) { yamlErr.textContent = e.message; yamlEl.classList.add('invalid'); }
  }, 300));

  // ---- validation (server)
  const validateSoon = debounce(async () => {
    try {
      const res = await api.post('/recipes/validate', { yaml: dump(S.rec) });
      clear(valHost);
      const errs = res.errors || []; const warns = res.warnings || [];
      valHost.append(res.valid === false || errs.length ? notice('error', 'Recipe is invalid', h('ul', { class: 'errs' }, errs.map((e) => h('li', typeof e === 'string' ? e : (e.message || JSON.stringify(e)))))) : h('div', { class: 'ok small' }, 'Recipe is valid', res.hash ? h('span', { class: 'muted' }, ` - hash ${String(res.hash).slice(0, 18)}`) : ''),
        ...warns.map((w) => notice('info', 'Warning', typeof w === 'string' ? w : w.message)));
    } catch (e) { clear(valHost); valHost.append(e.unavailable ? h('div', { class: 'muted small' }, 'Server-side validation unavailable.') : h('div', { class: 'bad small' }, e.message)); }
  }, 500);

  // ---- planner panel
  const planHost = h('div'); const suggestHost = h('div');
  const ctl = (label, el) => h('div', { class: 'stack' }, h('label', { class: 'small muted' }, label), el);
  const modelSel = h('select', { 'aria-label': 'Model', onchange: () => { S.planner.model = modelSel.value; estimateSoon(); } }, modelList.length ? modelList.map((m) => h('option', { value: m.id, selected: m.id === S.planner.model }, m.name || m.id)) : [h('option', { value: '' }, '(no models registered)')]);
  const chipSel = h('select', { 'aria-label': 'Chip', onchange: () => { S.planner.chip = chipSel.value; estimateSoon(); } }, chipList.length ? chipList.map((c) => h('option', { value: c.name, selected: c.name === S.planner.chip }, c.name)) : [h('option', { value: '' }, '(no chips)')]);
  const quantSel = h('select', { 'aria-label': 'Quantisation override', onchange: () => { S.planner.quantOverride = quantSel.value; estimateSoon(); } }, [['', 'from recipe'], ['bf16', 'bf16'], ['fp8', 'fp8'], ['w8a16', 'w8a16'], ['w4a16', 'w4a16'], ['nvfp4', 'nvfp4 (Blackwell)'], ['mxfp4', 'mxfp4']].map(([v, l]) => h('option', { value: v }, l)));
  const kvSel = h('select', { 'aria-label': 'KV dtype override', onchange: () => { S.planner.kvOverride = kvSel.value; estimateSoon(); } }, [['', 'from recipe'], ['bf16', 'bf16'], ['fp8_e4m3', 'fp8'], ['int8', 'int8'], ['int4', 'int4']].map(([v, l]) => h('option', { value: v }, l)));
  const ctxIn = h('input', { type: 'number', value: S.planner.ctxLen, min: 128, step: 128, 'aria-label': 'Context length', style: 'width:100%', oninput: () => { S.planner.ctxLen = Number(ctxIn.value) || 4096; estimateSoon(); } });
  const batchIn = h('input', { type: 'number', value: S.planner.batch, min: 1, 'aria-label': 'Batch size', style: 'width:100%', oninput: () => { S.planner.batch = Number(batchIn.value) || 1; estimateSoon(); } });

  function plannerBody() {
    const d = deriveQuant(S.rec);
    return {
      model: S.planner.model, chip: S.planner.chip, quant: S.planner.quantOverride || d.quant, kv_dtype: S.planner.kvOverride || d.kv,
      workload: { context_lengths: [S.planner.ctxLen], batch_sizes: [S.planner.batch] },
    };
  }
  let estSeq = 0;
  const estimateNow = async () => {
    const my = ++estSeq; const body = plannerBody();
    clear(planHost); planHost.appendChild(loadingBlock('Estimating...'));
    try {
      const est = await api.post('/planner/estimate', body);
      if (my !== estSeq) return;
      clear(planHost); planHost.append(renderEstimate(est, body));
    } catch (e) { if (my !== estSeq) return; clear(planHost); planHost.appendChild(e.unavailable ? unavailableNotice('Planner', 'POST /planner/estimate is not available (forge.planner not installed).') : notice('error', 'Estimate failed', e.message)); }
  };
  const estimateSoon = debounce(estimateNow, 600);
  async function suggest() {
    clear(suggestHost); suggestHost.appendChild(loadingBlock('Searching recipes...'));
    try {
      const b = plannerBody();
      const res = await api.post('/planner/plan', { model: b.model, chip: b.chip, workload: b.workload, budget: S.rec.budget || {} });
      clear(suggestHost);
      const cands = res.candidates || res.recipes || listOf(res);
      if (!cands.length) { suggestHost.appendChild(emptyState('No suggestions', 'The planner returned no ranked recipes for this model / chip / workload.')); return; }
      suggestHost.append(...cands.map((c, i) => h('div', { class: 'card', style: 'box-shadow:none;margin-bottom:8px' },
        h('div', { class: 'row' }, h('b', `${i + 1}. ${c.name || 'candidate'}`), c.experimental ? badge('experimental', 'warn') : null, h('span', { class: 'grow' }), (c.recipe || c.yaml) ? h('button', { class: 'btn sm', onclick: () => useRecipe(c) }, 'Use this recipe') : null),
        c.rationale ? h('div', { class: 'small muted' }, c.rationale) : null,
        c.est ? renderEstimate(c.est, null, true) : null,
        (c.claims || []).length ? h('div', { class: 'small mt-s' }, 'Depends on claims: ', ...(c.claims || []).map((x) => h('a', { href: '#/claims?id=' + encodeURIComponent(x), style: 'margin-right:6px' }, x))) : null)),
      res.note ? h('p', { class: 'small muted' }, res.note) : null);
    } catch (e) { clear(suggestHost); suggestHost.appendChild(e.unavailable ? unavailableNotice('Planner search') : notice('error', 'Planner failed', e.message)); }
  }
  function useRecipe(c) {
    try { const v = typeof c.recipe === 'string' ? parseAny(c.recipe) : (c.recipe || parseAny(c.yaml)); S.rec = Object.assign({ inputs: {}, baselines: [], budget: {}, policy: {}, seeds: [0] }, v, { inputs: { ...(v.inputs || {}), ...(S.rec.inputs || {}) } }); nameEl.value = S.rec.name; yamlEl.value = dump(S.rec); S.dirty = true; paintStages(); validateSoon(); estimateSoon(); toast('Recipe loaded into the builder', 'ok'); } catch (e) { toastError(e); }
  }

  // ---- save / run
  async function save() {
    if (!S.rec.name) { toast('Give the recipe a name first', 'warn'); return null; }
    try {
      const body = { name: S.rec.name, yaml: dump(S.rec) };
      let r;
      if (S.id) r = await api.put(`/recipes/${encodeURIComponent(S.id)}`, body); else { r = await api.post('/recipes', body); }
      S.dirty = false; toast('Recipe saved', 'ok');
      if (!S.id && r && r.id) { S.id = r.id; ctx.navigate(`/recipes/${r.id}`); }
      return r;
    } catch (e) { if (e.fieldErrors && Object.keys(e.fieldErrors).length) toast(Object.entries(e.fieldErrors).map(([k, v]) => `${k}: ${v}`).join('; '), 'error', 8000); else toastError(e, 'Save'); return null; }
  }
  async function runIt() {
    const r = S.dirty || !S.id ? await save() : { id: S.id };
    const id = (r && r.id) || S.id; if (!id) return;
    try { const j = await api.post(`/recipes/${encodeURIComponent(id)}/run`, {}); toast('Recipe job queued', 'ok'); if (j && j.id) navigate(`/jobs/${j.id}`); } catch (e) { toastError(e, 'Run'); }
  }

  root.append(
    pageHead(isNew ? 'New recipe' : (S.rec.name || 'Recipe'), 'Stages run in order; each stage output is cached by hash of (stage, params, input hashes, code version).',
      h('a', { class: 'btn', href: '#/recipes' }, 'All recipes'), h('button', { class: 'btn', onclick: save }, 'Save'), h('button', { class: 'btn primary', onclick: runIt }, 'Save & run')),
    h('div', { class: 'grid split' },
      h('div', { class: 'stack' },
        card('Recipe', h('div', { class: 'stack' }, h('div', { class: 'row' }, h('label', 'Name'), nameEl), sectionEditor('inputs', 'Inputs', 'Named inputs: model id, artifact hashes, dataset ids.'),
          h('div', { class: 'grid cols-2' }, sectionEditor('baselines', 'Baselines', 'Baselines to compare against (list).', true), sectionEditor('seeds', 'Seeds', 'Seeds for stochastic stages / repeated evals (list).', true)),
          h('div', { class: 'grid cols-2' }, sectionEditor('budget', 'Budget', 'e.g. gpu_hours, max_wall_hours'), sectionEditor('policy', 'Policy', 'e.g. licence_check: true')))),
        card('Stage palette', palette), card('Stages', stagesHost),
        card('YAML (live, editable)', h('div', yamlEl, yamlErr, valHost))),
      h('div', { class: 'stack' },
        card('Planner: will it fit?', h('div', { class: 'stack' }, h('div', { class: 'grid cols-2' }, ctl('Model', modelSel), ctl('Chip', chipSel), ctl('Quantisation', quantSel), ctl('KV dtype', kvSel), ctl('Context length', ctxIn), ctl('Concurrent seqs', batchIn)), planHost,
          h('p', { class: 'small muted' }, 'Numbers are derived from the cost model (weights + KV bytes against chip bandwidth), not measured. Tags show provenance.'))),
        card('Planner suggestions', h('div', h('button', { class: 'btn', onclick: suggest }, 'Suggest recipes for this model + chip'), h('div', { class: 'mt' }, suggestHost))))));
  paintStages();
  validateSoon(); estimateNow();
  const warnLeave = (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', warnLeave);
  ctx.onCleanup(() => window.removeEventListener('beforeunload', warnLeave));
  void titleCase; void statusBadge; void kv; void fmtNum; void isNum; void objectView;
}

function numFmt(v) { return isNum(v) ? fmtNum(v) : String(v); }

/** Render planner output: fit / bound / ceiling cards (compact when nested in a suggestion). */
function renderEstimate(est, _body, compact = false) {
  if (!est || typeof est !== 'object') return h('div', { class: 'muted' }, 'No estimate returned.');
  const sections = [['fit', 'Fit'], ['bound', 'Bound'], ['ceiling', 'Ceiling']].filter(([k]) => est[k] != null);
  if (!sections.length) return card('Estimate', objectView(est));
  const out = h('div', { class: compact ? 'grid cols-3' : 'stack' });
  for (const [k, label] of sections) {
    const v = est[k];
    let headline; let sub = '';
    if (k === 'fit') { headline = v.fits === false ? badge('does not fit', 'bad') : v.fits === true ? badge('fits', 'ok') : badge('unknown', 'dim'); sub = isNum(v.max_concurrency) ? `max concurrency ~${fmtNum(v.max_concurrency)}` : ''; }
    else if (k === 'bound') { headline = badge(String(v.kind || v.type || v.bound || '?'), 'info'); sub = isNum(v.batch_crossover) ? `compute-bound above batch ~${fmtNum(v.batch_crossover)}` : ''; }
    else { const tok = v.decode_tok_s_batch1 ?? v.decode_tok_s ?? v.tok_s; headline = h('b', isNum(tok) ? `${fmtNum(tok)} tok/s` : '-'); sub = 'decode ceiling at batch 1'; }
    const rows = Object.entries(v).filter(([key, val]) => !['provenance', 'note', 'fits', 'kind', 'type', 'bound'].includes(key) && val !== null && typeof val !== 'object').map(([key, val]) => [titleCase(key), numFmt(val)]);
    out.appendChild(h('div', { class: compact ? '' : 'card', style: 'box-shadow:none' },
      h('div', { class: 'row' }, h('span', { class: 'small muted', style: 'text-transform:uppercase;letter-spacing:.05em' }, label), v.provenance ? provBadge(v.provenance) : null), h('div', { style: 'font-size:1.2rem;margin:2px 0' }, headline), sub ? h('div', { class: 'small muted' }, sub) : null,
      compact ? null : kv(rows), v.note && !compact ? h('div', { class: 'small muted mt-s' }, v.note) : null));
  }
  if (est.chip && !compact) out.appendChild(h('div', { class: 'small muted' }, `Chip: ${est.chip}`));
  return out;
}
