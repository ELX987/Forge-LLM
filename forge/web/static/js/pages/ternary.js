// Ternary Lab (EXPERIMENTAL): shortcut wizard over the `ternary_train` job type + live training simulator + equilibrium flip-rate calculator.
// Ternary QAT in Forge is validated on CPU only: it uses fake-quantised bf16 compute (no ternary speed-up in training) and the resulting 2-bit model is the product.
// The full schema-driven form stays available below the quick setup.

import { h, clear, listOf, getPath, setPath, fmtPct } from '../util.js';
import { tryGet } from '../api.js';
import { pageHead, card, notice, unavailableNotice, loadingBlock, toast, badge } from '../components/ui.js';
import { JobConfigurator } from '../components/jobform.js';
import { trainSimPanel } from '../components/trainsim.js';
import { fetchJobTypes } from './newjob.js';
import { navigate } from '../router.js';

// Size presets from the tern158 research code (hidden, layers, heads, kv_heads, ffn). `tiny` is the CPU smoke model.
export const PRESETS = [
  { id: 'tiny', label: 'Tiny CPU smoke test (a few hundred K params)', tiny: true, group: 32, seq: 64 },
  { id: '125m', label: '125M  (768 x 12, ffn 2048)', hidden: 768, layers: 12, heads: 12, kv: 12, ffn: 2048, vocab: 32000, group: 128, seq: 2048 },
  { id: '350m', label: '350M  (1024 x 24, ffn 2816)', hidden: 1024, layers: 24, heads: 16, kv: 16, ffn: 2816, vocab: 32000, group: 128, seq: 2048 },
  { id: '1b', label: '1B  (2048 x 22, GQA 16/4, ffn 5632)', hidden: 2048, layers: 22, heads: 16, kv: 4, ffn: 5632, vocab: 32000, group: 128, seq: 2048 },
  { id: '2b', label: '2B  (2560 x 30, GQA 20/5, ffn 6912)', hidden: 2560, layers: 30, heads: 20, kv: 5, ffn: 6912, vocab: 32000, group: 128, seq: 2048 },
];
const SCHEMES = [['ternary_qat', 'Ternary QAT as Forge runs it today (fake-quant bf16 compute)'], ['bf16', 'bf16 mixed precision (reference)'], ['ternary_native', 'tern158 design: int8 forward, sharded latents (NOT available in Forge yet)']];

/** Predicted equilibrium per-step code-flip rate of AdamW on ternary latents (tern158 report; TARGET until re-measured in Forge). */
export function flipRate(lr, wd) { return 0.59 * Math.sqrt(Math.max(0, 2 * lr * wd)); }

export default async function render(ctx) {
  const body = h('div');
  ctx.root.append(pageHead('Ternary Lab', 'Train a model whose weights are {-1, 0, +1} (1.58 bit) with int8 activations. EXPERIMENTAL: validated on CPU only.'), body);
  body.appendChild(loadingBlock());
  let types;
  try { types = await fetchJobTypes(); } catch (e) { clear(body); body.appendChild(e.unavailable ? unavailableNotice('Ternary Lab', 'The jobs backend (job-type schemas) is not available on this server.') : notice('error', 'Could not load job types', e.message)); return; }
  const jt = types.find((t) => t.type === 'ternary_train');
  if (!jt) { clear(body); body.appendChild(unavailableNotice('Ternary training', 'This server does not register a "ternary_train" job type (forge.ternary not installed). The generic New Job form still works for other methods.')); return; }
  const [datasets, chips] = await Promise.all([tryGet('/datasets'), tryGet('/hardware/chips')]);
  if (!ctx.isActive()) return;
  const dsList = datasets.ok ? listOf(datasets.data, 'datasets') : [];
  const cd = chips.ok ? chips.data : null;
  const chipList = (Array.isArray(cd) ? cd : (cd && (cd.chips || cd.known)) || []).map((c) => c.name).filter(Boolean);
  clear(body);

  const st = { preset: PRESETS[0], scheme: 'ternary_qat', chip: chipList.find((n) => /pro-6000|h100/i.test(n)) || chipList[0] || 'h100-sxm-80g', nGpu: 1, price: 0 };
  const sim = trainSimPanel();
  const flipOut = h('div', { class: 'small' });
  const formHost = h('div');
  const quickHost = h('div');
  let cfg;

  const val = () => (cfg && cfg.form ? cfg.form.getValue() : {});
  const patch = (pairs) => { const v = val(); for (const [p, x] of pairs) setPath(v, p, x); cfg.form.setValue(v); refresh(); };

  function refresh() {
    const v = val();
    const lr = Number(getPath(v, 'optim.lr', 3e-4)), wd = Number(getPath(v, 'optim.weight_decay', 0));
    const fr = flipRate(lr, wd);
    clear(flipOut);
    flipOut.append(h('b', `Equilibrium flip rate ~ ${fmtPct(fr, 2)} per step`), h('span', { class: 'muted' }, `  (0.59 x sqrt(2 x lr x wd) = 0.59 x sqrt(2 x ${lr} x ${wd}); TARGET value from the tern158 report, not measured in Forge)`));
    if (wd === 0) flipOut.append(h('div', { class: 'faint' }, 'Weight decay 0 gives no equilibrium: flips only follow the learning-rate schedule. That is the recommended setting when converting a pretrained model.'));
    if (fr > 0.01) flipOut.append(notice('warn', 'High flip rate', 'More than 1% of codes would flip per step at equilibrium: updates are likely noise-dominated. Lower the learning rate or weight decay.'));
    const p = st.preset;
    if (p.tiny) { sim.show(notice('info', 'No estimate for the tiny model', 'Pick a size preset (125M and up) to estimate memory, step time and cost.')); return; }
    const seq = Number(getPath(v, 'data.seq_len', p.seq)) || p.seq;
    const mb = Number(getPath(v, 'micro_batch_size', 4)), acc = Number(getPath(v, 'grad_accum_steps', 8));
    sim.update({ config: { scheme: st.scheme, hidden: p.hidden, layers: p.layers, heads: p.heads, kv_heads: p.kv, ffn: p.ffn, vocab: p.vocab, seq_len: seq, global_batch_tokens: mb * acc * seq * st.nGpu,
      max_tokens: Number(getPath(v, 'budget.max_tokens', 2e8)) || 2e8, grad_checkpoint: !!getPath(v, 'activation_checkpointing', false), shard_state: st.nGpu > 1, group_size: Number(getPath(v, 'precision.ternary.group_size', p.group)) || p.group },
      chip: st.chip, n_gpu: st.nGpu, price_per_gpu_hour: st.price });
  }
  const applyPreset = (id) => {
    const p = PRESETS.find((x) => x.id === id) || PRESETS[0];
    st.preset = p;
    const pairs = [['precision.ternary.group_size', p.group], ['data.seq_len', p.seq]];
    if (p.tiny) pairs.push(['model.source', 'tiny'], ['model.tokenizer', 'byte'], ['model.arch_overrides', {}]);
    else pairs.push(['model.source', 'scratch'], ['model.arch_overrides', { vocab_size: p.vocab, hidden_size: p.hidden, intermediate_size: p.ffn, num_layers: p.layers, num_heads: p.heads, num_kv_heads: p.kv, max_position_embeddings: p.seq, tie_word_embeddings: true }]);
    patch(pairs);
  };

  cfg = new JobConfigurator(formHost, { jobType: jt, prefill: {}, onChange: () => refresh() });
  ctx.onCleanup(() => cfg.destroy());

  const field = (label, control, hint) => h('div', { class: 'stack', style: 'min-width:200px' }, h('label', { class: 'small muted' }, label), control, hint ? h('div', { class: 'small faint' }, hint) : null);
  const num = (label, path, hint, step) => { const inp = h('input', { type: 'number', step: step || 'any', value: getPath(val(), path, ''), 'aria-label': label }); inp.addEventListener('change', () => { if (inp.value !== '') patch([[path, Number(inp.value)]]); }); return field(label, inp, hint); };

  const presetSel = h('select', { 'aria-label': 'Size preset', onchange: (e) => applyPreset(e.target.value) }, PRESETS.map((p) => h('option', { value: p.id }, p.label)));
  const dsSel = h('select', { 'aria-label': 'Dataset', onchange: (e) => { if (e.target.value) patch([['data.dataset', [e.target.value]]]); } }, h('option', { value: '' }, 'Choose dataset...'), ...dsList.map((d) => h('option', { value: d.id }, `${d.name || d.id}${d.total_tokens ? ` (${Math.round(d.total_tokens / 1e6)}M tok)` : ''}`)));
  const ruleSel = h('select', { 'aria-label': 'Quantiser rule', onchange: (e) => patch([['precision.ternary.rule', e.target.value]]) }, ['lloyd', 'absmean', 'twn'].map((r) => h('option', { value: r }, r)));
  const schemeSel = h('select', { 'aria-label': 'Simulator scheme', onchange: (e) => { st.scheme = e.target.value; refresh(); } }, SCHEMES.map(([v, l]) => h('option', { value: v }, l)));
  const chipSel = h('select', { 'aria-label': 'Simulator chip', onchange: (e) => { st.chip = e.target.value; refresh(); } }, (chipList.length ? chipList : [st.chip]).map((n) => h('option', { value: n, selected: n === st.chip }, n)));
  const gpuInp = h('input', { type: 'number', min: 1, max: 4096, value: 1, 'aria-label': 'GPUs', onchange: (e) => { st.nGpu = Math.max(1, Number(e.target.value) || 1); refresh(); } });
  const priceInp = h('input', { type: 'number', min: 0, step: 'any', value: 0, 'aria-label': 'USD per GPU-hour', onchange: (e) => { st.price = Math.max(0, Number(e.target.value) || 0); refresh(); } });

  const quick = h('div', { class: 'grid auto' },
    field('Size preset', presetSel, 'Sizes from the tern158 research presets. The tiny preset runs on CPU in about a minute.'),
    field('Dataset', dsSel, dsList.length ? 'Tokenised dataset from the Datasets page.' : 'No datasets yet: ingest one first (tokenizer must match the model).'),
    num('Learning rate', 'optim.lr', 'Scratch ~3e-4; converting a pretrained model ~5e-5.'),
    num('Weight decay', 'optim.weight_decay', '0.1 for scratch; 0 when converting.'),
    num('Scale group (K)', 'precision.ternary.group_size', '128 for converted models; -1 = per row; 0 = per tensor.', 1),
    field('Quantiser rule', ruleSel, 'Lloyd recovers most of the gap to the optimal threshold (tern158 report, unreproduced here).'),
    num('Warm-up steps', 'precision.ternary.warmup_steps', 'Blend from full precision to ternary over this many steps (0 for scratch, ~500 for conversion).', 1));
  const simControls = h('div', { class: 'grid auto' }, field('Simulate scheme', schemeSel), field('Chip', chipSel), field('GPUs', gpuInp), field('USD / GPU-hour', priceInp));

  const submitRow = h('div', { class: 'row mt' }, h('button', { class: 'btn primary', onclick: async (e) => {
    e.target.disabled = true;
    try { const job = await cfg.submit(); if (job && job.id) { toast('Ternary job submitted', 'ok'); navigate(`/jobs/${job.id}`); } } catch (_) { /* painted by the form */ } finally { e.target.disabled = false; }
  } }, 'Start ternary training'), h('span', { class: 'muted small' }, 'Health (zero density, flip rate) appears live on the job page under "Ternary (1.58-bit) health".'));

  quickHost.append(
    notice('warn', 'Experimental', 'Ternary QAT here is validated on CPU only. Training uses fake-quantised bf16 compute, so it is not faster than bf16 training; the product is the 2-bit model. Thresholds shown are TARGET values from the tern158 report.'),
    card('Quick setup', quick, badge('EXPERIMENTAL', 'warn'), 'mb'),
    card('Flip-rate calculator', flipOut, null, 'mb'),
    card('Simulator inputs', simControls, null, 'mb'), sim.el, submitRow, h('h2', { class: 'mt' }, 'Full configuration'));
  body.append(quickHost, formHost);
  applyPreset('tiny');
}
