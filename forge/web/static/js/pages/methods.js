// Method picker: every training / fine-tuning / quantization / compression method Forge knows, with its honest status.
// Metadata only (GET /methods). "planned" cards are reserved slots that are NOT implemented. Nothing here is GPU-validated unless the card says so.

import { h, clear, listOf, titleCase } from '../util.js';
import { tryGet } from '../api.js';
import { pageHead, card, badge, notice, unavailableNotice, loadingBlock, emptyState } from '../components/ui.js';
import { navigate } from '../router.js';

const FAMILIES = [['pretrain', 'Pre-training and full fine-tuning'], ['low_precision_training', 'Low-precision training'], ['peft', 'Parameter-efficient fine-tuning'], ['distillation', 'Distillation'],
  ['ptq', 'Post-training quantization'], ['qat', 'Quantization-aware training'], ['compression', 'Compression'], ['ternary', 'Ternary (1.58-bit)']];
const STATUS = { supported: ['supported', 'ok', 'Implemented and CPU-tested. GPU validation is a separate flag.'], experimental: ['experimental', 'warn', 'Implemented behind a gate or emulated; not validated on hardware.'], planned: ['planned', 'dim', 'Reserved slot: not implemented.'] };

function startTarget(m) {
  if (m.id === 'ternary-train') return '/ternary';
  if (!m.job_type) return null;
  return `/jobs/new?type=${encodeURIComponent(m.job_type)}`;
}

export default async function render(ctx) {
  const body = h('div');
  ctx.root.append(pageHead('Methods', 'Training, fine-tuning, quantization and compression methods in one place, each labelled with what is actually implemented and validated.'), body);
  body.appendChild(loadingBlock());
  const res = await tryGet('/methods');
  if (!ctx.isActive()) return;
  clear(body);
  if (!res.ok) { body.appendChild(res.unavailable ? unavailableNotice('Method catalog', 'GET /methods is not exposed by this server.') : notice('error', 'Could not load methods', res.error.message)); return; }
  const methods = listOf(res.data, 'methods');
  if (!methods.length) { body.appendChild(emptyState('No methods registered')); return; }
  let status = ctx.query.status || '';
  const chips = h('div', { class: 'row tight mb' });
  const list = h('div');
  const paintChips = () => { clear(chips); for (const s of ['', 'supported', 'experimental', 'planned']) chips.appendChild(h('button', { class: `btn sm ${status === s ? 'primary' : 'ghost'}`, onclick: () => { status = s; paintChips(); paint(); } }, s ? titleCase(s) : 'All')); };
  function paint() {
    clear(list);
    for (const [fam, title] of FAMILIES) {
      const ms = methods.filter((m) => m.family === fam && (!status || m.status === status));
      if (!ms.length) continue;
      list.appendChild(h('h2', { class: 'mt' }, title));
      list.appendChild(h('div', { class: 'grid auto' }, ms.map((m) => {
        const [label, kind, tip] = STATUS[m.status] || [m.status, 'dim', ''];
        const target = startTarget(m);
        return card(null, h('div', { class: 'stack' },
          h('div', { class: 'row tight' }, h('b', m.title), badge(label, kind, { tip }), m.gpu_validated ? badge('GPU validated', 'ok') : badge('not GPU-validated', 'warn', { tip: 'No recorded hardware run (see docs/KERNELS.md validation tables).' })),
          h('div', { class: 'small' }, m.summary),
          (m.needs || []).length ? h('div', { class: 'small muted' }, 'Needs: ' + m.needs.join(', ')) : null,
          h('div', { class: 'small muted' }, [m.job_type ? `job: ${m.job_type}` : null, m.produces ? `produces: ${m.produces}` : null, m.diagnostics_prefix ? `metrics: ${m.diagnostics_prefix}/*` : null, m.planner_scheme ? `planner scheme: ${m.planner_scheme}` : null].filter(Boolean).join('  |  ')),
          h('div', { class: 'row tight' }, target ? h('button', { class: 'btn sm primary', onclick: () => navigate(target) }, m.id === 'ternary-train' ? 'Open Ternary Lab' : 'Configure a job') : h('button', { class: 'btn sm', disabled: true }, 'Not implemented'),
            m.docs ? h('span', { class: 'small faint' }, m.docs) : null)), null, 'mb');
      })));
    }
    if (!list.children.length) list.appendChild(emptyState('Nothing matches this filter'));
  }
  paintChips(); paint();
  body.append(chips, list);
}
