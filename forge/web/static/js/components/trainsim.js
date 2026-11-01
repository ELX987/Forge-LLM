// Training simulator panel: memory per GPU, step-time breakdown, throughput, cost for a training scheme (POST /planner/train-sim).
// First-order and UNCALIBRATED; the panel says so. Used by the Ternary Lab; reusable by any page that can build a TrainSimConfig.

import { h, clear, fmtNum, fmtPct, debounce, isNum } from '../util.js';
import { api } from '../api.js';
import { card, notice, badge, loadingBlock, unavailableNotice } from './ui.js';

function bars(entries, unit, total) {
  const max = Math.max(1e-12, total || Math.max(...entries.map(([, v]) => v), 0));
  return h('div', { class: 'stack' }, entries.filter(([, v]) => v > 0).map(([k, v]) => h('div', { class: 'row tight', style: 'align-items:center' },
    h('div', { class: 'small', style: 'width:230px' }, k),
    h('div', { class: 'bar', style: 'flex:1', role: 'img', 'aria-label': `${k}: ${fmtNum(v, 3)} ${unit}` }, h('i', { style: { width: Math.min(100, (v / max) * 100).toFixed(1) + '%' } })),
    h('div', { class: 'small mono', style: 'width:90px;text-align:right' }, `${fmtNum(v, 3)} ${unit}`))));
}

/** @returns {{el: HTMLElement, update: Function}}; call update({config, chip, n_gpu, price_per_gpu_hour}) whenever inputs change. */
export function trainSimPanel() {
  const out = h('div');
  let seq = 0;
  async function run(req) {
    const my = ++seq;
    clear(out); out.appendChild(loadingBlock('Estimating...'));
    try {
      const r = await api.post('/planner/train-sim', req);
      if (my !== seq) return;
      clear(out);
      const mem = Object.entries(r.mem_gb || {});
      const total = mem.reduce((a, [, v]) => a + v, 0);
      out.append(
        h('div', { class: 'row tight mb' }, r.ok ? badge('fits', 'ok') : badge('does not fit', 'err'), badge('ESTIMATE', 'warn', { tip: 'First-order roofline with guessed efficiency factors, uncalibrated. tern158 compared it with one run only; Forge has none. Not a measurement.' }),
          badge(r.scheme, 'dim')),
        h('div', { class: 'small muted' }, `micro-batch ${r.micro_batch} x accumulation ${r.accum}  |  ${isNum(r.tokens_per_s) ? fmtNum(r.tokens_per_s, 4) : '-'} tok/s  |  MFU ~${fmtPct(r.mfu)} (vs bf16 peak)  |  ${fmtNum(r.total_hours, 3)} h${r.cost_usd ? `, $${fmtNum(r.cost_usd, 3)}` : ''}  |  ${fmtNum(r.energy_kwh, 3)} kWh`),
        h('h4', { class: 'mt' }, `Memory per GPU: ${fmtNum(total, 3)} of ${fmtNum(r.vram_gb * 0.93 * 1e9 / 1024 ** 3, 3)} GiB usable`), bars(mem, 'GiB', total),
        h('h4', { class: 'mt' }, `Step time ${fmtNum(r.step_s, 3)} s`), bars(Object.entries(r.breakdown_s || {}), 's', null),
        ...(r.notes || []).map((n) => notice(/ESTIMATE|does NOT fit/.test(n) ? 'warn' : 'info', 'Note', n)),
        ...(r.suggestions || []).map((s) => h('div', { class: 'small' }, 'Tip: ' + s)));
    } catch (e) {
      if (my !== seq) return;
      clear(out);
      out.appendChild(e.unavailable ? unavailableNotice('Training simulator', 'POST /planner/train-sim is not available on this server.') : notice('error', 'Simulation failed', e.message || String(e)));
    }
  }
  /** Replace the panel content with a message (cancels any pending / in-flight estimate). */
  let hold = false;
  const show = (node) => { hold = true; seq += 1; clear(out); out.appendChild(node); };
  const debounced = debounce((r) => { if (!hold) run(r); }, 450);
  const update = (r) => { hold = false; debounced(r); };
  return { el: card('Simulate (updates as you edit)', out), update, show };
}
