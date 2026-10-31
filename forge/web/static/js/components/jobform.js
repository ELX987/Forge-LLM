// JobConfigurator: the schema-driven configuration panel for one job type
// (template / saved-config selector, name, tags, GPU picker, SchemaForm with YAML/JSON toggle, validation, submit).

import { h, clear, listOf, deepClone, parseCsvList, fmtBytes, isNum, normGpus } from '../util.js';
import { api, tryGet } from '../api.js';
import { SchemaForm } from '../forms.js';
import { toast, toastError, confirmDialog, notice, badge, modal, objectView } from './ui.js';
import { formDialog } from './dialogs.js';

export class JobConfigurator {
  /**
   * @param {HTMLElement} container
   * @param {{jobType:object, prefill?:{params?:object,name?:string,gpus?:number[],tags?:string[]}, onChange?:Function}} opts
   */
  constructor(container, opts) {
    this.container = container;
    this.jt = opts.jobType;
    this.prefill = opts.prefill || {};
    this.onChange = opts.onChange || (() => {});
    this.errors = [];
    this.gpus = [];
    this.selectedGpus = new Set(this.prefill.gpus || []);
    this.render();
  }

  get type() { return this.jt.type; }

  render() {
    const jt = this.jt;
    const defaults = deepClone(jt.defaults || {});
    const initial = this.prefill.params ? this.merge(defaults, this.prefill.params) : defaults;
    this.nameEl = h('input', { type: 'text', value: this.prefill.name || '', placeholder: `${jt.type}-${new Date().toISOString().slice(0, 10)}`, 'aria-label': 'Job name', style: 'min-width:240px' });
    this.tagsEl = h('input', { type: 'text', value: (this.prefill.tags || []).join(', '), placeholder: 'tag1, tag2', 'aria-label': 'Tags' });
    this.templateSel = h('select', { 'aria-label': 'Template', onchange: () => this.applyTemplate() }, h('option', { value: '' }, '(template...)'));
    this.savedSel = h('select', { 'aria-label': 'Saved config', onchange: () => this.applySaved() }, h('option', { value: '' }, '(saved config...)'));
    this.gpuHost = h('div', { class: 'chips' });
    this.formHost = h('div');
    this.statusEl = h('div', { class: 'small', role: 'status' });
    this.saveBtn = h('button', { class: 'btn sm', type: 'button', onclick: () => this.saveConfig() }, 'Save config');
    this.delBtn = h('button', { class: 'btn sm danger', type: 'button', onclick: () => this.deleteSaved() }, 'Delete saved');
    this.delBtn.classList.add('hidden');
    this.checkBtn = this.type === 'dataset_ingest' ? h('button', { class: 'btn sm', type: 'button', title: 'Resolve each source (local path or Hugging Face id), list configs/splits/columns and show a few documents', onclick: () => this.checkSources() }, 'Check source') : null;
    this.container.append(
      h('div', { class: 'card mb' },
        h('div', { class: 'toolbar', style: 'margin-bottom:6px' }, h('label', 'Name'), this.nameEl, h('label', 'Tags'), this.tagsEl),
        h('div', { class: 'toolbar', style: 'margin-bottom:6px' }, this.templateSel, this.savedSel, this.saveBtn, this.delBtn, this.checkBtn, h('span', { class: 'muted small' }, 'Templates are read-only seeds; saved configs are yours.')),
        h('div', { class: 'row' }, h('span', { class: 'sf-label' }, 'GPUs'), this.gpuHost)),
      h('div', { class: 'card' }, this.formHost, this.statusEl));
    this.form = new SchemaForm(this.formHost, { schema: jt.schema, defaults: jt.defaults || {}, value: initial, onChange: (v, errs) => { this.errors = errs || []; this.paintStatus(); this.onChange(v, this.errors); } });
    this.errors = this.form.errors;
    this.paintStatus();
    this.loadSide();
  }

  merge(base, over) {
    if (over && typeof over === 'object' && !Array.isArray(over) && base && typeof base === 'object' && !Array.isArray(base)) {
      const out = { ...base };
      for (const [k, v] of Object.entries(over)) out[k] = k in base ? this.merge(base[k], v) : v;
      return out;
    }
    return over === undefined ? base : over;
  }

  async loadSide() {
    const [tpl, saved, gpus] = await Promise.all([tryGet('/configs/templates', { type: this.type }), tryGet('/configs', { job_type: this.type }), tryGet('/system/gpus')]);
    this.templates = tpl.ok ? listOf(tpl.data, 'templates') : [];
    for (const t of this.templates) {
      const name = typeof t === 'string' ? t : t.name;
      this.templateSel.appendChild(h('option', { value: name }, name));
    }
    if (!this.templates.length && (this.jt.templates || []).length) for (const n of this.jt.templates) this.templateSel.appendChild(h('option', { value: n }, n));
    this.saved = saved.ok ? listOf(saved.data, 'configs') : [];
    for (const c of this.saved) this.savedSel.appendChild(h('option', { value: c.id }, c.name));
    if (!this.templateSel.options || this.templateSel.options.length <= 1) this.templateSel.classList.add('hidden');
    if (!this.saved.length) this.savedSel.classList.add('hidden');
    this.gpus = gpus.ok ? normGpus(gpus.data) : [];
    this.paintGpus();
  }

  paintGpus() {
    clear(this.gpuHost);
    if (!this.gpus.length) { this.gpuHost.appendChild(h('span', { class: 'muted small' }, 'No GPUs reported - the scheduler will run this on CPU / assign automatically.')); return; }
    const auto = h('button', { type: 'button', class: 'chip' + (this.selectedGpus.size === 0 ? ' on' : ''), 'aria-pressed': String(this.selectedGpus.size === 0), onclick: () => { this.selectedGpus.clear(); this.paintGpus(); } }, 'auto');
    this.gpuHost.appendChild(auto);
    for (const g of this.gpus) {
      const on = this.selectedGpus.has(g.index);
      const busy = isNum(g.util) && g.util > 60;
      this.gpuHost.appendChild(h('button', { type: 'button', class: 'chip' + (on ? ' on' : ''), 'aria-pressed': String(on), title: `${g.name}${isNum(g.mem_used_mb) ? ` - ${fmtBytes(g.mem_used_mb * 1048576)} used` : ''}`, onclick: () => { if (on) this.selectedGpus.delete(g.index); else this.selectedGpus.add(g.index); this.paintGpus(); } },
        `GPU ${g.index}`, busy ? badge('busy', 'warn') : null));
    }
  }

  paintStatus() {
    clear(this.statusEl);
    if (this.errors.length) this.statusEl.appendChild(h('span', { class: 'bad' }, `${this.errors.length} validation issue(s) - see highlighted fields.`));
    else this.statusEl.appendChild(h('span', { class: 'ok' }, 'Config is valid against the schema.'));
  }

  async applyTemplate() {
    const name = this.templateSel.value; if (!name) return;
    let t = (this.templates || []).find((x) => (typeof x === 'string' ? x : x.name) === name);
    let params = t && typeof t === 'object' ? (t.params || t.params_json || t.config || t.values || t.content) : null;
    if (!params) { const r = await tryGet('/configs/templates', { type: this.type, name }); const l = r.ok ? listOf(r.data, 'templates') : []; const f = l.find((x) => typeof x === 'object' && x.name === name) || (r.ok && !Array.isArray(r.data) ? r.data : null); params = f && (f.params || f.params_json || f.config || f.values || f.content); }
    if (typeof params === 'string') { try { params = JSON.parse(params); } catch (_) { params = null; } }
    if (!params || typeof params !== 'object') { toast('The server did not return the template contents.', 'warn'); this.templateSel.value = ''; return; }
    this.form.setValue(this.merge(deepClone(this.jt.defaults || {}), params));
    if (!this.nameEl.value) this.nameEl.value = name;
    toast(`Loaded template "${name}"`, 'ok', 1800);
  }
  applySaved() {
    const id = this.savedSel.value; this.delBtn.classList.toggle('hidden', !id);
    const c = (this.saved || []).find((x) => x.id === id); if (!c) return;
    let params = c.params_json || c.params;
    if (typeof params === 'string') { try { params = JSON.parse(params); } catch (_) { params = {}; } }
    this.form.setValue(this.merge(deepClone(this.jt.defaults || {}), params || {}));
    this.nameEl.value = this.nameEl.value || c.name;
    toast(`Loaded "${c.name}"`, 'ok', 1800);
  }
  saveConfig() {
    formDialog({ title: 'Save config', submitLabel: 'Save', fields: [{ name: 'name', label: 'Name', required: true, value: this.nameEl.value }],
      onSubmit: async (v) => {
        const c = await api.post('/configs', { job_type: this.type, name: v.name, params: this.form.getValue() });
        toast('Config saved', 'ok');
        if (c && c.id) { this.saved = [...(this.saved || []), c]; this.savedSel.appendChild(h('option', { value: c.id }, c.name)); this.savedSel.classList.remove('hidden'); this.savedSel.value = c.id; this.delBtn.classList.remove('hidden'); }
      } });
  }
  async deleteSaved() {
    const id = this.savedSel.value; if (!id) return;
    if (!(await confirmDialog({ title: 'Delete saved config?', message: 'This cannot be undone.', confirmLabel: 'Delete', danger: true }))) return;
    try { await api.del(`/configs/${encodeURIComponent(id)}`); const opt = Array.from(this.savedSel.options).find((o) => o.value === id); if (opt) opt.remove(); this.savedSel.value = ''; this.delBtn.classList.add('hidden'); toast('Deleted', 'ok'); } catch (e) { toastError(e); }
  }

  /** dataset_ingest only: POST /datasets/source-preview for every source and show what the server resolved. */
  async checkSources() {
    const sources = (this.form.getValue() || {}).sources || [];
    if (!sources.length) { toast('Add a source first.', 'warn'); return; }
    const body = h('div', { class: 'stack' }, h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Checking...'));
    const dlg = modal({ title: 'Check source', wide: true, body });
    const parts = [];
    for (const [i, src] of sources.entries()) {
      try { parts.push(h('div', { class: 'card', style: 'box-shadow:none' }, h('h3', `Source ${i + 1}: ${src.path || '(empty)'}`), objectView(await api.post('/datasets/source-preview', { source: src, n: 3 })))); }
      catch (e) { parts.push(h('div', { class: 'notice error' }, `Source ${i + 1}: ${e.message}`)); }
    }
    dlg.setBody(h('div', { class: 'stack' }, parts));
  }

  getSpec() {
    return { type: this.type, name: this.nameEl.value.trim() || this.nameEl.placeholder, params: this.form.getValue(), gpus: this.selectedGpus.size ? Array.from(this.selectedGpus).sort((a, b) => a - b) : null, tags: parseCsvList(this.tagsEl.value) };
  }

  /** POST /jobs. Resolves to the job record; field errors from the API are painted into the form. */
  async submit({ force = false } = {}) {
    const errs = this.form.validateNow(false);
    if (errs.length && !force) {
      const ok = await confirmDialog({ title: 'Config has validation issues', message: h('div', h('p', 'The form reports problems that the server will probably reject:'), h('ul', { class: 'errs' }, errs.slice(0, 8).map((e) => h('li', h('code', e.path || '(root)'), ': ', e.message)))), confirmLabel: 'Submit anyway' });
      if (!ok) { this.form.setErrors(Object.fromEntries(errs.map((e) => [e.path, e.message]))); return null; }
    }
    try {
      return await api.post('/jobs', this.getSpec());
    } catch (e) {
      if (e.fieldErrors && Object.keys(e.fieldErrors).length) { this.form.setErrors(e.fieldErrors); toast(h('span', [h('b', 'Server rejected the config: '), e.message]), 'error', 8000); } else toastError(e, 'Submit failed');
      throw e;
    }
  }
  destroy() { if (this.form) this.form.destroy(); }
}

export function unavailableTypes() { return notice('unavailable', 'Job types unavailable', 'GET /schemas/job-types failed - the jobs/API module is not available on this server.'); }
