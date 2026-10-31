// Small reusable dialogs built on ui.modal: simple field forms and JSON viewers.

import { h, copyText } from '../util.js';
import { modal, toast, toastError } from './ui.js';

/**
 * formDialog({title, intro, fields:[{name,label,type:'text'|'number'|'select'|'checkbox'|'textarea',options,placeholder,help,required,value}], submitLabel, onSubmit(values) -> Promise})
 * Shows field errors returned in ApiError.fieldErrors. Closes on success.
 */
export function formDialog({ title, intro, fields, submitLabel = 'Submit', onSubmit, wide = false }) {
  const inputs = {};
  const errs = {};
  const rows = fields.map((f) => {
    let input;
    if (f.type === 'select') {
      input = h('select', { id: `fd-${f.name}`, 'aria-label': f.label }, (f.options || []).map((o) => (typeof o === 'object' ? h('option', { value: o.value }, o.label) : h('option', { value: o }, o))));
      if (f.value !== undefined) input.value = f.value;
    } else if (f.type === 'checkbox') {
      input = h('input', { type: 'checkbox', id: `fd-${f.name}`, checked: !!f.value });
    } else if (f.type === 'textarea') {
      input = h('textarea', { id: `fd-${f.name}`, rows: f.rows || 4, placeholder: f.placeholder || '' }); input.value = f.value || '';
    } else {
      input = h('input', { type: f.type || 'text', id: `fd-${f.name}`, placeholder: f.placeholder || '', style: 'width:100%' }); if (f.value !== undefined) input.value = f.value;
      if (f.required) input.required = true;
    }
    inputs[f.name] = input;
    const err = h('div', { class: 'sf-err', role: 'alert' });
    errs[f.name] = err;
    return h('div', { class: 'sf-field', style: 'border:0' },
      h('div', { class: 'sf-labelcol' }, h('div', { class: 'sf-label' }, h('label', { for: `fd-${f.name}` }, f.label), f.required ? h('span', { class: 'req' }, '*') : null), f.help ? h('div', { class: 'sf-desc' }, f.help) : null),
      h('div', { class: 'sf-control' }, input, err));
  });
  const general = h('div', { class: 'sf-err', role: 'alert' });
  const submit = h('button', { class: 'btn primary', type: 'submit' }, submitLabel);
  const form = h('form', { novalidate: true }, intro ? h('p', { class: 'muted' }, intro) : null, rows, general);
  const m = modal({ title, body: form, wide, footer: [h('button', { class: 'btn', type: 'button', onclick: () => m.close() }, 'Cancel'), submit] });
  const values = () => {
    const v = {};
    for (const f of fields) {
      const el = inputs[f.name];
      v[f.name] = f.type === 'checkbox' ? el.checked : (f.type === 'number' ? (el.value === '' ? undefined : Number(el.value)) : el.value.trim());
    }
    return v;
  };
  const go = async (e) => {
    if (e) e.preventDefault();
    Object.values(errs).forEach((x) => { x.textContent = ''; }); general.textContent = '';
    const v = values();
    for (const f of fields) if (f.required && (v[f.name] === '' || v[f.name] === undefined)) { errs[f.name].textContent = 'Required'; inputs[f.name].focus(); return; }
    submit.disabled = true;
    try {
      await onSubmit(v);
      m.close(true);
    } catch (err) {
      submit.disabled = false;
      const fe = (err && err.fieldErrors) || {};
      let any = false;
      for (const [k, msg] of Object.entries(fe)) if (errs[k]) { errs[k].textContent = msg; any = true; }
      if (!any) general.textContent = (err && err.message) || String(err);
    }
  };
  form.addEventListener('submit', go);
  submit.addEventListener('click', go);
  const first = Object.values(inputs)[0]; if (first) setTimeout(() => first.focus(), 30);
  return m;
}

export function jsonDialog(title, value) {
  const text = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return modal({ title, wide: true, body: h('pre', { class: 'mono', style: 'max-height:60vh;overflow:auto' }, text),
    footer: [h('button', { class: 'btn', onclick: async () => { if (await copyText(text)) toast('Copied', 'ok', 1200); } }, 'Copy')] });
}

export async function runAction(fn, successMsg) {
  try { const r = await fn(); if (successMsg) toast(successMsg, 'ok'); return r; } catch (e) { toastError(e); return undefined; }
}
