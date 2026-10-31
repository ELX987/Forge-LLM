// JSON-Schema -> form generator with live validation, dirty markers, reset-to-default, YAML/JSON raw editor.
// Supports: string / number / integer (slider + input), boolean (switch), enum (select), arrays (string lists, enum
// multi-select, object rows), nested objects (collapsible groups), additionalProperties maps (key/value editor),
// nullable ("unset" toggle), anyOf/oneOf unions, x-group tabs, x-advanced sections, x-unit, descriptions as help text.
// The schema is the one served by /api/v1/schemas/job-types (refs already inlined by forge.core.jsonschema.schema_for).

import { h, clear, deepClone, deepEqual, getPath, setPath, debounce, titleCase, isNum, uid, fmtNum } from './util.js';
import { dump as yamlDump, parseAny as yamlParse } from './yaml.js';
import { prefs } from './state.js';

// ------------------------------------------------------------------ schema helpers
const isObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);

/** Returns {schema, nullable} with type-array / anyOf-null / x-nullable normalised. */
export function resolve(schema) {
  let s = schema || {};
  let nullable = !!s['x-nullable'];
  if (Array.isArray(s.type)) {
    const nn = s.type.filter((t) => t !== 'null');
    if (nn.length !== s.type.length) nullable = true;
    s = { ...s, type: nn.length === 1 ? nn[0] : nn };
  }
  for (const key of ['anyOf', 'oneOf']) {
    if (Array.isArray(s[key])) {
      const nn = s[key].filter((b) => !(isObj(b) && b.type === 'null'));
      if (nn.length !== s[key].length) {
        nullable = true;
        if (nn.length === 1) { const { [key]: _drop, ...rest } = s; s = { ...nn[0], ...rest }; } else s = { ...s, [key]: nn };
      }
    }
  }
  if (Array.isArray(s.allOf) && s.allOf.length) {
    const merged = {};
    for (const part of s.allOf) Object.assign(merged, part);
    const { allOf: _a, ...rest } = s; s = { ...merged, ...rest };
  }
  return { schema: s, nullable };
}

function typeMatches(t, v) {
  switch (t) {
    case 'string': return typeof v === 'string';
    case 'integer': return typeof v === 'number' && Number.isInteger(v);
    case 'number': return typeof v === 'number' && Number.isFinite(v);
    case 'boolean': return typeof v === 'boolean';
    case 'array': return Array.isArray(v);
    case 'object': return isObj(v);
    case 'null': return v === null;
    default: return true;
  }
}

/** Validate a value against a (resolved-ref) JSON schema subset. Returns [{path, message}]. */
export function validate(schema, value, path = '') {
  const errs = [];
  const add = (message, p = path) => errs.push({ path: p, message });
  const { schema: s, nullable } = resolve(schema);
  if (value === undefined) return errs;
  if (value === null) { if (!nullable && s.type && !(Array.isArray(s.type) && s.type.includes('null'))) add('must not be null'); return errs; }
  const union = s.anyOf || s.oneOf;
  if (Array.isArray(union) && union.length) {
    let best = null;
    for (const br of union) {
      const e = validate({ ...br, ...(s.description && !br.description ? {} : {}) }, value, path);
      if (!e.length) { best = null; break; }
      if (!best || e.length < best.length) best = e;
    }
    if (best) { if (union.length > 1 && best.length > 1 && best[0].path === path) add('does not match any allowed form'); else errs.push(...best); }
    return errs;
  }
  if (s.const !== undefined && !deepEqual(s.const, value)) add(`must equal ${JSON.stringify(s.const)}`);
  if (Array.isArray(s.enum) && !s.enum.some((e) => deepEqual(e, value))) { add(`must be one of: ${s.enum.map((e) => JSON.stringify(e)).join(', ')}`); return errs; }
  if (s.type) {
    const types = Array.isArray(s.type) ? s.type : [s.type];
    if (!types.some((t) => typeMatches(t, value))) { add(`must be ${types.join(' or ')}`); return errs; }
  }
  if (typeof value === 'string') {
    if (isNum(s.minLength) && value.length < s.minLength) add(`must have at least ${s.minLength} characters`);
    if (isNum(s.maxLength) && value.length > s.maxLength) add(`must have at most ${s.maxLength} characters`);
    if (s.pattern) { try { if (!new RegExp(s.pattern).test(value)) add(`must match pattern ${s.pattern}`); } catch (_) { /* python-only regex */ } }
  } else if (typeof value === 'number') {
    if (isNum(s.minimum) && value < s.minimum) add(`must be >= ${s.minimum}`);
    if (isNum(s.maximum) && value > s.maximum) add(`must be <= ${s.maximum}`);
    if (isNum(s.exclusiveMinimum) && value <= s.exclusiveMinimum) add(`must be > ${s.exclusiveMinimum}`);
    if (isNum(s.exclusiveMaximum) && value >= s.exclusiveMaximum) add(`must be < ${s.exclusiveMaximum}`);
    if (isNum(s.multipleOf) && s.multipleOf > 0) { const q = value / s.multipleOf; if (Math.abs(q - Math.round(q)) > 1e-9) add(`must be a multiple of ${s.multipleOf}`); }
  } else if (Array.isArray(value)) {
    if (isNum(s.minItems) && value.length < s.minItems) add(`needs at least ${s.minItems} item(s)`);
    if (isNum(s.maxItems) && value.length > s.maxItems) add(`allows at most ${s.maxItems} item(s)`);
    if (s.uniqueItems && new Set(value.map((v) => JSON.stringify(v))).size !== value.length) add('items must be unique');
    if (s.items) value.forEach((v, i) => errs.push(...validate(s.items, v, path ? `${path}.${i}` : String(i))));
  } else if (isObj(value)) {
    const props = s.properties || {};
    for (const r of s.required || []) if (value[r] === undefined) add('is required', path ? `${path}.${r}` : r);
    for (const [k, v] of Object.entries(value)) {
      const p = path ? `${path}.${k}` : k;
      if (props[k]) errs.push(...validate(props[k], v, p));
      else if (s.additionalProperties === false) add('unknown field', p);
      else if (isObj(s.additionalProperties)) errs.push(...validate(s.additionalProperties, v, p));
    }
  }
  return errs;
}

/** Sensible default for a (sub)schema. */
export function defaultFor(schema, depth = 0) {
  const { schema: s, nullable } = resolve(schema);
  if (s.default !== undefined) return deepClone(s.default);
  if (s.const !== undefined) return s.const;
  if (Array.isArray(s.enum) && s.enum.length) return s.enum[0];
  const union = s.anyOf || s.oneOf;
  if (Array.isArray(union) && union.length) return defaultFor(union[0], depth + 1);
  if (nullable) return null;
  const t = Array.isArray(s.type) ? s.type[0] : s.type;
  switch (t) {
    case 'string': return '';
    case 'integer': case 'number': return isNum(s.minimum) ? s.minimum : (isNum(s.exclusiveMinimum) ? s.exclusiveMinimum + (t === 'integer' ? 1 : 0.1) : 0);
    case 'boolean': return false;
    case 'array': return [];
    case 'object': {
      const o = {};
      if (depth < 6) for (const r of s.required || []) if (s.properties && s.properties[r]) o[r] = defaultFor(s.properties[r], depth + 1);
      return o;
    }
    default: return null;
  }
}

function groupsOf(schema) {
  const props = schema.properties || {};
  const order = []; const map = new Map();
  const named = schema['x-groups'] || [];
  for (const g of named) { map.set(g, []); order.push(g); }
  for (const [k, p] of Object.entries(props)) {
    const g = (p && p['x-group']) || 'General';
    if (!map.has(g)) { map.set(g, []); order.push(g); }
    map.get(g).push(k);
  }
  const names = order.filter((g) => map.get(g).length);
  // General first
  names.sort((a, b) => (a === 'General' ? -1 : b === 'General' ? 1 : 0));
  return { names, map };
}

const kindOf = (s) => {
  if (Array.isArray(s.enum)) return 'enum';
  if (s.const !== undefined) return 'const';
  const u = s.anyOf || s.oneOf;
  if (Array.isArray(u) && u.length > 1) return 'union';
  const t = Array.isArray(s.type) ? s.type.find((x) => x !== 'null') : s.type;
  if (t === 'boolean') return 'boolean';
  if (t === 'integer' || t === 'number') return 'number';
  if (t === 'string') return 'string';
  if (t === 'array') return 'array';
  if (t === 'object' || s.properties) {
    if (s.properties && Object.keys(s.properties).length) return 'group';
    if (s.additionalProperties === false) return 'group';
    return 'map';
  }
  if (Array.isArray(u) && u.length === 1) return 'union';
  return 'json';
};

const isComplexKind = (k) => k === 'group' || k === 'array' || k === 'map' || k === 'json' || k === 'union';

// ------------------------------------------------------------------ the form
export class SchemaForm {
  /**
   * @param {HTMLElement} container
   * @param {{schema:object, value?:object, defaults?:object, onChange?:Function, allowRaw?:boolean, title?:string, hideFilter?:boolean}} opts
   */
  constructor(container, opts) {
    this.container = container;
    this.schema = resolve(opts.schema).schema;
    this.defaults = opts.defaults !== undefined ? deepClone(opts.defaults) : (isObj(this.schema.default) ? deepClone(this.schema.default) : {});
    this.data = opts.value !== undefined ? deepClone(opts.value) : deepClone(this.defaults);
    if (!isObj(this.data)) this.data = {};
    this.onChange = opts.onChange || (() => {});
    this.opts = opts;
    this.mode = 'form';
    this.records = new Map(); // pathStr -> {el, errEl, text, noDirty, rebuild}
    this.serverErrors = {};
    this.errors = [];
    this.showAdvanced = !!prefs.get('form.advanced', false);
    this.activeTab = null;
    this.rootEl = h('div', { class: 'sf' });
    container.appendChild(this.rootEl);
    this.runValidate = debounce(() => this.validateNow(true), 120);
    this.build();
  }

  // ---------------------------------------------------------- public API
  getValue() { return JSON.parse(JSON.stringify(this.data)); }
  setValue(v) { this.data = isObj(v) ? deepClone(v) : {}; this.serverErrors = {}; this.build(); this.validateNow(false); }
  /** Replace defaults (used by dirty markers / reset). */
  setDefaults(d) { this.defaults = deepClone(d || {}); this.refreshDirty(); }
  resetAll() { this.data = deepClone(this.defaults); this.serverErrors = {}; this.build(); this.changed(''); }
  validateNow(notify = true) {
    this.errors = validate(this.schema, this.data, '');
    this.paintErrors();
    if (notify) this.onChange(this.getValue(), this.errors);
    return this.errors;
  }
  /** errors: {path: message} (e.g. from ApiError.fieldErrors) */
  setErrors(map) {
    this.serverErrors = { ...(map || {}) };
    this.paintErrors(true);
  }
  clearServerErrors() { this.serverErrors = {}; this.paintErrors(); }
  destroy() { this.runValidate.cancel(); this.rootEl.remove(); }

  // ---------------------------------------------------------- build
  build() {
    this.runValidate.cancel();
    this.records.clear();
    clear(this.rootEl);
    const tools = this.buildTools();
    this.bodyEl = h('div', { class: 'sf-body' });
    this.summaryEl = h('div', { class: 'sf-summary' });
    this.rootEl.append(tools, this.bodyEl, this.summaryEl);
    if (this.mode === 'form') this.buildForm(); else this.buildRaw();
  }

  buildTools() {
    const el = h('div', { class: 'sf-tools' });
    const seg = (id, label) => h('button', { type: 'button', class: 'btn sm' + (this.mode === id ? ' primary' : ''), 'aria-pressed': String(this.mode === id), onclick: () => this.switchMode(id) }, label);
    el.append(h('div', { class: 'row tight', role: 'group', 'aria-label': 'Editor mode' }, seg('form', 'Form'), seg('yaml', 'YAML'), seg('json', 'JSON')));
    if (this.mode === 'form') {
      this.filterEl = h('input', { type: 'search', placeholder: 'Filter fields...', 'aria-label': 'Filter fields', oninput: () => this.applyFilter() });
      const adv = h('label', { class: 'row tight' }, h('input', { type: 'checkbox', checked: this.showAdvanced, onchange: (e) => { this.showAdvanced = e.target.checked; prefs.set('form.advanced', this.showAdvanced); this.setAdvancedOpen(this.showAdvanced); } }), 'Show advanced');
      el.append(this.filterEl, adv, h('button', { type: 'button', class: 'btn sm ghost', title: 'Reset every field to its default', onclick: () => this.resetAll() }, 'Reset all'));
    }
    return el;
  }

  switchMode(mode) {
    if (mode === this.mode) return;
    if (this.mode !== 'form' && this.rawEl) { // commit raw text
      try { this.data = this.parseRaw(this.rawEl.value); } catch (e) { this.rawErr.textContent = 'Fix the syntax error before switching: ' + e.message; return; }
    }
    this.mode = mode;
    this.build();
    this.validateNow(true);
  }

  parseRaw(text) {
    const v = this.mode === 'json' ? JSON.parse(text || '{}') : yamlParse(text);
    return isObj(v) ? v : (v == null ? {} : (() => { throw new Error('top level must be a mapping/object'); })());
  }

  buildRaw() {
    const text = this.mode === 'json' ? JSON.stringify(this.data, null, 2) : yamlDump(this.data);
    this.rawErr = h('div', { class: 'sf-err', role: 'alert' });
    this.rawEl = h('textarea', { class: 'sf-raw', spellcheck: 'false', 'aria-label': this.mode.toUpperCase() + ' editor', rows: 24 });
    this.rawEl.value = text;
    this.rawEl.addEventListener('input', debounce(() => {
      try {
        this.data = this.parseRaw(this.rawEl.value);
        this.rawErr.textContent = ''; this.rawEl.classList.remove('invalid');
        this.errors = validate(this.schema, this.data, '');
        this.paintRawErrors();
        this.onChange(this.getValue(), this.errors);
      } catch (e) {
        this.rawErr.textContent = (this.mode === 'json' ? 'JSON' : '') + ' ' + e.message; this.rawEl.classList.add('invalid');
        this.onChange(null, [{ path: '', message: e.message }]);
      }
    }, 250));
    this.bodyEl.append(this.rawEl, this.rawErr);
    this.paintRawErrors();
  }
  paintRawErrors() {
    if (this.mode === 'form') return;
    clear(this.summaryEl);
    const all = [...this.errors, ...Object.entries(this.serverErrors).map(([path, message]) => ({ path, message }))];
    if (all.length) this.summaryEl.append(h('ul', { class: 'errs' }, all.slice(0, 20).map((e) => h('li', h('code', e.path || '(root)'), ': ', e.message))));
  }

  buildForm() {
    const s = this.schema;
    const { names, map } = groupsOf(s);
    const props = s.properties || {};
    const req = new Set(s.required || []);
    const mkPane = (keys) => this.renderProps(s, keys, [], req, 0);
    if (names.length <= 1) {
      this.bodyEl.append(mkPane(names.length ? map.get(names[0]) : Object.keys(props)));
    } else {
      const tabsEl = h('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Config groups' });
      const panes = h('div', { class: 'sf-panes' });
      this.tabButtons = new Map(); this.panes = new Map();
      names.forEach((g, i) => {
        const id = uid('sf-pane');
        const btn = h('button', { type: 'button', class: 'tab', role: 'tab', id: id + '-tab', 'aria-controls': id, onclick: () => this.activateTab(g) }, g);
        const pane = h('div', { class: 'sf-pane', id, role: 'tabpanel', 'aria-labelledby': id + '-tab', 'data-tab': g }, h('h3', { class: 'sf-pane-title' }, g), mkPane(map.get(g)));
        this.tabButtons.set(g, btn); this.panes.set(g, pane);
        tabsEl.appendChild(btn); panes.appendChild(pane);
        void i;
      });
      this.bodyEl.append(tabsEl, panes);
      this.activateTab(this.activeTab && this.panes.has(this.activeTab) ? this.activeTab : names[0]);
    }
    this.refreshDirty();
    this.setAdvancedOpen(this.showAdvanced);
    this.validateNow(false);
  }

  activateTab(name) {
    if (!this.panes) return;
    this.activeTab = name;
    for (const [g, pane] of this.panes) { pane.classList.toggle('hidden', g !== name && !this.rootEl.classList.contains('filtering')); this.tabButtons.get(g).setAttribute('aria-selected', String(g === name)); }
  }

  setAdvancedOpen(open) {
    this.rootEl.querySelectorAll('details.sf-adv').forEach((d) => { d.open = open; });
  }

  /** Render a list of property keys of an object schema at `path` (array of segments). */
  renderProps(schema, keys, path, requiredSet, depth) {
    const wrap = h('div', { class: 'sf-fields' });
    const props = schema.properties || {};
    const normal = []; const adv = [];
    for (const k of keys) {
      const p = props[k]; if (!p) continue;
      (resolve(p).schema['x-advanced'] || p['x-advanced'] ? adv : normal).push(k);
    }
    for (const k of normal) wrap.appendChild(this.makeField(k, props[k], path.concat(k), { required: requiredSet.has(k), depth }));
    if (adv.length) {
      const d = h('details', { class: 'fold sf-adv' }, h('summary', `Advanced (${adv.length})`), h('div', { class: 'fold-body' }, adv.map((k) => this.makeField(k, props[k], path.concat(k), { required: requiredSet.has(k), depth }))));
      wrap.appendChild(d);
    }
    return wrap;
  }

  // ---------------------------------------------------------- fields
  key(path) { return path.join('.'); }
  get(path) { return getPath(this.data, path); }
  set(path, value) {
    if (!path.length) { this.data = value; this.changed(''); return; }
    if (value === undefined) {
      const parent = path.length > 1 ? getPath(this.data, path.slice(0, -1)) : this.data;
      if (parent != null) { if (Array.isArray(parent)) parent.splice(Number(path[path.length - 1]), 1); else delete parent[path[path.length - 1]]; }
    } else setPath(this.data, path, value);
    this.changed(this.key(path));
  }
  defaultAt(path) { return getPath(this.defaults, path); }
  changed(_keyStr) {
    this.refreshDirty();
    this.runValidate();
    if (Object.keys(this.serverErrors).length) { this.serverErrors = {}; }
  }
  refreshDirty() {
    for (const [k, rec] of this.records) {
      if (rec.noDirty || !rec.el) continue;
      const path = k.split('.').filter((x) => x !== '');
      const cur = getPath(this.data, path); const def = getPath(this.defaults, path);
      const dirty = !deepEqual(cur === undefined ? null : cur, def === undefined ? null : def);
      rec.el.classList.toggle('is-dirty', dirty);
    }
  }

  /** Create a labelled field. opts: {required, depth, noDirty, label, inList} */
  makeField(name, rawSchema, path, opts = {}) {
    const { schema, nullable } = resolve(rawSchema);
    const kind = kindOf(schema);
    const keyStr = this.key(path);
    const complex = isComplexKind(kind);
    const label = opts.label !== undefined ? opts.label : titleCase(name);

    if (kind === 'group') return this.makeGroup(name, schema, path, opts, nullable, label);

    const field = h('div', { class: 'sf-field' + (complex ? ' block' : '') });
    const resetBtn = opts.noDirty ? null : h('button', { type: 'button', class: 'btn ghost sm sf-reset', title: 'Reset to default', 'aria-label': `Reset ${name} to default`, onclick: () => { this.resetField(path); } }, '↺');
    const labelEl = h('div', { class: 'sf-label' }, h('label', { for: undefined }, label), opts.required ? h('span', { class: 'req', title: 'required' }, '*') : null,
      h('span', { class: 'dirty', title: 'Changed from default' }), h('span', { class: 'path' }, name), resetBtn);
    const descEl = schema.description ? h('div', { class: 'sf-desc' }, schema.description) : null;
    const errEl = h('div', { class: 'sf-err', role: 'alert' });
    const ctlHost = h('div', { class: 'sf-control' });
    field.append(h('div', { class: 'sf-labelcol' }, labelEl, descEl), ctlHost);
    const build = () => {
      clear(ctlHost);
      const ctl = this.makeControl(schema, path, { nullable, required: opts.required, noDirty: opts.noDirty });
      ctlHost.append(ctl, errEl);
    };
    build();
    this.records.set(keyStr, { el: field, errEl, noDirty: !!opts.noDirty, text: `${name} ${label} ${schema.description || ''}`.toLowerCase(), rebuild: build });
    return field;
  }

  resetField(path) {
    const def = this.defaultAt(path);
    this.set(path, def === undefined ? undefined : deepClone(def));
    this.rebuildPath(path);
    this.refreshDirty();
  }
  rebuildPath(path) {
    const rec = this.records.get(this.key(path));
    if (rec && rec.rebuild) { this.dropPrefix(this.key(path) + '.'); rec.rebuild(); }
  }
  dropPrefix(prefix) { for (const k of Array.from(this.records.keys())) if (k.startsWith(prefix)) this.records.delete(k); }

  makeGroup(name, schema, path, opts, nullable, label) {
    const keyStr = this.key(path);
    const cur = this.get(path);
    const det = h('details', { class: 'sf-group', open: (opts.depth || 0) < 1 || (cur != null && Object.keys(cur || {}).length < 12 && (opts.depth || 0) < 2) });
    const errEl = h('div', { class: 'sf-err', role: 'alert' });
    const resetBtn = h('button', { type: 'button', class: 'btn ghost sm sf-reset', title: 'Reset group to defaults', onclick: (e) => { e.preventDefault(); e.stopPropagation(); this.resetField(path); } }, '↺');
    const dirtyDot = h('span', { class: 'dirty', title: 'Changed from default' });
    const sum = h('summary', label, h('span', { class: 'path' }, name), dirtyDot, resetBtn);
    det.appendChild(sum);
    const body = h('div', { class: 'sf-fields' });
    det.appendChild(body);
    const field = h('div', { class: 'sf-field block' }, det);
    const build = () => {
      clear(body);
      if (schema.description) body.appendChild(h('div', { class: 'sf-desc', style: 'padding:6px 0' }, schema.description));
      body.appendChild(errEl);
      let isNull = nullable && this.get(path) === null;
      const inner = h('div');
      const fill = () => {
        clear(inner);
        if (nullable && this.get(path) == null) { inner.appendChild(h('div', { class: 'muted small' }, 'Unset (null). Tick "set" to configure.')); return; }
        if (this.get(path) == null) this.setSilent(path, defaultFor(schema));
        const keys = Object.keys(schema.properties || {});
        const g = groupsOf(schema);
        void g;
        inner.appendChild(this.renderProps(schema, keys, path, new Set(schema.required || []), (opts.depth || 0) + 1));
        if (!keys.length) inner.appendChild(h('div', { class: 'muted' }, 'No configurable fields.'));
      };
      if (nullable) {
        const cb = h('input', { type: 'checkbox', checked: !isNull, 'aria-label': `Set ${name}`, onchange: (e) => {
          if (e.target.checked) this.set(path, deepClone(this.defaultAt(path) ?? defaultFor({ ...schema, default: undefined }))); else this.set(path, null);
          this.dropPrefix(keyStr + '.'); fill(); this.setAdvancedOpen(this.showAdvanced);
        } });
        body.appendChild(h('label', { class: 'sf-nullable', style: 'margin:6px 0' }, cb, 'set'));
        if (!isNull && this.get(path) == null) this.setSilent(path, null);
      }
      body.appendChild(inner);
      fill();
    };
    build();
    this.records.set(keyStr, { el: field, errEl, noDirty: !!opts.noDirty, text: `${name} ${label} ${schema.description || ''}`.toLowerCase(), rebuild: build, group: det });
    // dirty indicator for group head
    const origRefresh = () => { const rec = this.records.get(keyStr); if (rec) { rec.dirtyDot = dirtyDot; } };
    origRefresh();
    return field;
  }
  setSilent(path, value) { if (!path.length) this.data = value; else setPath(this.data, path, value); }

  // ---------------------------------------------------------- controls
  /** Returns a DOM node editing the value at `path`. */
  makeControl(schema, path, opts = {}) {
    const kind = kindOf(schema);
    const nullable = !!opts.nullable && kind !== 'group';
    const host = h('div', { class: 'sf-ctl-host' });
    let inner;
    const cur = this.get(path);
    const nullCb = nullable ? h('label', { class: 'sf-nullable' }, h('input', { type: 'checkbox', checked: cur === null || cur === undefined && schema.default === null, 'aria-label': 'Unset (null)', onchange: (e) => {
      if (e.target.checked) { this.set(path, null); setDisabled(true); } else {
        const d = this.defaultAt(path);
        this.set(path, d != null ? deepClone(d) : defaultFor({ ...schema, default: undefined }));
        this.dropPrefix(this.key(path) + '.'); mountInner(); setDisabled(false);
      }
    } }), 'unset') : null;
    const mountInner = () => { clear(host); inner = this.rawControl(kind, schema, path, opts); host.append(inner); if (nullCb) host.append(nullCb); };
    const setDisabled = (d) => { host.classList.toggle('is-null', d); inner.querySelectorAll('input,select,textarea,button').forEach((el) => { el.disabled = d; }); };
    mountInner();
    if (nullable && (cur === null || (cur === undefined && schema.default === null))) setDisabled(true);
    return host;
  }

  rawControl(kind, schema, path, opts) {
    switch (kind) {
      case 'boolean': return this.boolControl(schema, path);
      case 'number': return this.numberControl(schema, path, opts);
      case 'string': return this.stringControl(schema, path, opts);
      case 'enum': return this.enumControl(schema, path, opts);
      case 'const': return h('code', JSON.stringify(schema.const));
      case 'array': return this.arrayControl(schema, path);
      case 'map': return this.mapControl(schema, path);
      case 'union': return this.unionControl(schema, path);
      case 'group': { const w = h('div'); w.appendChild(this.renderProps(schema, Object.keys(schema.properties || {}), path, new Set(schema.required || []), 2)); return w; }
      default: return this.jsonControl(schema, path);
    }
  }

  boolControl(schema, path) {
    const cur = this.get(path);
    const input = h('input', { type: 'checkbox', checked: cur === true, role: 'switch', onchange: (e) => this.set(path, e.target.checked) });
    return h('label', { class: 'sf-switch' }, input, h('span'));
  }

  numberControl(schema, path, opts) {
    const isInt = (Array.isArray(schema.type) ? schema.type.includes('integer') && !schema.type.includes('number') : schema.type === 'integer');
    const lo = isNum(schema.minimum) ? schema.minimum : (isNum(schema.exclusiveMinimum) ? schema.exclusiveMinimum : null);
    const hi = isNum(schema.maximum) ? schema.maximum : (isNum(schema.exclusiveMaximum) ? schema.exclusiveMaximum : null);
    const cur = this.get(path);
    const input = h('input', { type: 'number', inputmode: 'decimal', 'aria-label': 'value', step: schema.multipleOf || (isInt ? 1 : 'any') });
    if (lo != null) input.min = lo; if (hi != null) input.max = hi;
    input.value = isNum(cur) ? String(cur) : '';
    const logScale = lo != null && hi != null && lo > 0 && hi / lo >= 1000;
    let slider = null;
    const toSl = (v) => Math.round(1000 * (logScale ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo)));
    const fromSl = (t) => {
      const f = t / 1000;
      let v = logScale ? Math.exp(Math.log(lo) + f * (Math.log(hi) - Math.log(lo))) : lo + f * (hi - lo);
      v = isInt ? Math.round(v) : Number(v.toPrecision(3));
      return Math.min(hi, Math.max(lo, v));
    };
    if (lo != null && hi != null && hi > lo && (isInt ? hi - lo > 1 : true) && (logScale || hi - lo <= 1e6)) {
      slider = h('input', { type: 'range', min: 0, max: 1000, step: 1, 'aria-label': 'slider' + (logScale ? ' (log scale)' : ''), title: logScale ? 'log scale' : '' });
      slider.value = isNum(cur) ? Math.max(0, Math.min(1000, toSl(cur))) : 0;
      slider.addEventListener('input', () => { const v = fromSl(Number(slider.value)); input.value = String(v); this.set(path, v); });
    }
    input.addEventListener('input', () => {
      const raw = input.value.trim();
      if (raw === '') { this.set(path, undefined); return; }
      const v = Number(raw);
      if (!Number.isFinite(v)) return;
      this.set(path, isInt ? (Number.isInteger(v) ? v : v) : v);
      if (slider) slider.value = Math.max(0, Math.min(1000, toSl(Math.min(hi, Math.max(lo, v)))));
    });
    return h('div', { class: 'inline' }, slider, input, schema['x-unit'] ? h('span', { class: 'sf-unit' }, schema['x-unit']) : null,
      lo != null || hi != null ? h('span', { class: 'faint small' }, `${lo != null ? fmtNum(lo) : ''}…${hi != null ? fmtNum(hi) : ''}`) : null);
  }

  stringControl(schema, path, opts) {
    const cur = this.get(path);
    const multiline = schema.format === 'textarea' || schema['x-multiline'] || (isNum(schema.maxLength) && schema.maxLength > 200);
    const def = this.defaultAt(path);
    const commit = (val) => this.set(path, val === '' && def === undefined && !opts.required ? undefined : val);
    let input;
    if (multiline) {
      input = h('textarea', { rows: 4, 'aria-label': 'value' }); input.value = typeof cur === 'string' ? cur : '';
      input.addEventListener('input', () => commit(input.value));
      return h('div', input);
    }
    input = h('input', { type: schema.format === 'password' ? 'password' : 'text', 'aria-label': 'value', autocomplete: 'off', spellcheck: 'false' });
    input.value = typeof cur === 'string' ? cur : '';
    if (schema.examples && schema.examples.length) {
      const id = uid('dl'); input.setAttribute('list', id);
      input.addEventListener('input', () => commit(input.value));
      return h('div', input, h('datalist', { id }, schema.examples.map((e) => h('option', { value: String(e) }))));
    }
    input.addEventListener('input', () => commit(input.value));
    return h('div', { class: 'inline' }, input, schema['x-unit'] ? h('span', { class: 'sf-unit' }, schema['x-unit']) : null);
  }

  enumControl(schema, path, opts) {
    const vals = schema.enum;
    const cur = this.get(path);
    const sel = h('select', { 'aria-label': 'value' });
    let idx = vals.findIndex((v) => deepEqual(v, cur));
    if (idx < 0 && cur !== undefined && cur !== null) sel.appendChild(h('option', { value: '__invalid' }, `${JSON.stringify(cur)} (not allowed)`));
    if (cur === undefined && !opts.required) sel.appendChild(h('option', { value: '__unset' }, '(unset)'));
    vals.forEach((v, i) => sel.appendChild(h('option', { value: String(i) }, v === null ? '(none)' : String(v))));
    if (idx >= 0) sel.value = String(idx); else if (cur === undefined) sel.value = '__unset'; else sel.value = '__invalid';
    sel.addEventListener('change', () => {
      if (sel.value === '__unset') this.set(path, undefined); else if (sel.value !== '__invalid') this.set(path, vals[Number(sel.value)]);
    });
    return sel;
  }

  arrayControl(schema, path) {
    const itemsSchema = resolve(schema.items || {}).schema;
    const itemKind = kindOf(itemsSchema);
    const host = h('div', { class: 'sf-list' });
    const keyStr = this.key(path);
    const arr = () => { let a = this.get(path); if (!Array.isArray(a)) { a = []; this.setSilent(path, a); } return a; };
    // enum multi-select
    if (Array.isArray(itemsSchema.enum)) {
      const sel = new Set(arr().map((v) => JSON.stringify(v)));
      const chips = h('div', { class: 'chips' });
      itemsSchema.enum.forEach((v) => {
        const on = sel.has(JSON.stringify(v));
        const b = h('button', { type: 'button', class: 'chip' + (on ? ' on' : ''), 'aria-pressed': String(on), onclick: () => {
          const a = arr(); const i = a.findIndex((x) => deepEqual(x, v));
          if (i >= 0) a.splice(i, 1); else a.push(v);
          b.classList.toggle('on', i < 0); b.setAttribute('aria-pressed', String(i < 0));
          this.changed(keyStr);
        } }, String(v));
        chips.appendChild(b);
      });
      return chips;
    }
    const rows = h('div', { class: 'sf-list' });
    const addBtn = h('button', { type: 'button', class: 'btn sm', onclick: () => {
      const a = arr(); if (isNum(schema.maxItems) && a.length >= schema.maxItems) return;
      a.push(defaultFor(itemsSchema)); this.changed(keyStr); draw();
    } }, '+ Add');
    const draw = () => {
      this.dropPrefix(keyStr + '.');
      clear(rows);
      const a = arr();
      a.forEach((_, i) => {
        const p = path.concat(String(i));
        let ctl;
        if (itemKind === 'group') {
          ctl = h('div', { class: 'sf-item-card' }, this.renderProps(itemsSchema, Object.keys(itemsSchema.properties || {}), p, new Set(itemsSchema.required || []), 2));
        } else ctl = this.makeControl(itemsSchema, p, { noDirty: true });
        const rec = { el: null, errEl: h('div', { class: 'sf-err', role: 'alert' }), noDirty: true, text: '' };
        this.records.set(this.key(p), rec);
        const row = h('div', { class: 'sf-list-row' }, h('div', { class: 'grow' }, ctl, rec.errEl),
          h('div', { class: 'row tight' },
            i > 0 ? h('button', { type: 'button', class: 'btn ghost sm', title: 'Move up', 'aria-label': 'Move up', onclick: () => { const x = arr(); [x[i - 1], x[i]] = [x[i], x[i - 1]]; this.changed(keyStr); draw(); } }, '↑') : null,
            i < a.length - 1 ? h('button', { type: 'button', class: 'btn ghost sm', title: 'Move down', 'aria-label': 'Move down', onclick: () => { const x = arr(); [x[i + 1], x[i]] = [x[i], x[i + 1]]; this.changed(keyStr); draw(); } }, '↓') : null,
            h('button', { type: 'button', class: 'btn ghost sm', title: 'Remove', 'aria-label': 'Remove item', onclick: () => { if (isNum(schema.minItems) && arr().length <= schema.minItems) return; arr().splice(i, 1); this.changed(keyStr); draw(); } }, '×')));
        rows.appendChild(row);
      });
      if (!a.length) rows.appendChild(h('div', { class: 'faint small' }, 'Empty list'));
      this.paintErrors();
    };
    draw();
    host.append(rows, h('div', addBtn));
    return host;
  }

  mapControl(schema, path) {
    const valSchema = isObj(schema.additionalProperties) ? resolve(schema.additionalProperties).schema : null;
    const keyStr = this.key(path);
    const host = h('div', { class: 'sf-list' });
    const obj = () => { let o = this.get(path); if (!isObj(o)) { o = {}; this.setSilent(path, o); } return o; };
    const rows = h('div', { class: 'sf-list' });
    const keyIn = h('input', { type: 'text', placeholder: 'new key', 'aria-label': 'New key' });
    const add = () => {
      const k = keyIn.value.trim(); if (!k) return;
      const o = obj(); if (k in o) { keyIn.classList.add('invalid'); return; }
      keyIn.classList.remove('invalid');
      o[k] = valSchema ? defaultFor(valSchema) : '';
      keyIn.value = ''; this.changed(keyStr); draw();
    };
    keyIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } });
    const draw = () => {
      this.dropPrefix(keyStr + '.');
      clear(rows);
      const o = obj();
      for (const k of Object.keys(o)) {
        const p = path.concat(k);
        const rec = { el: null, errEl: h('div', { class: 'sf-err', role: 'alert' }), noDirty: true, text: '' };
        this.records.set(this.key(p), rec);
        const ctl = valSchema && kindOf(valSchema) !== 'json' ? this.makeControl(valSchema, p, { noDirty: true }) : this.jsonControl({}, p);
        const kEl = h('input', { type: 'text', value: k, 'aria-label': 'Key', onchange: (e) => {
          const nk = e.target.value.trim(); if (!nk || nk === k || nk in obj()) { e.target.value = k; return; }
          const oo = obj(); const entries = Object.entries(oo); for (const [kk] of entries) delete oo[kk];
          for (const [kk, vv] of entries) oo[kk === k ? nk : kk] = vv;
          this.changed(keyStr); draw();
        } });
        rows.appendChild(h('div', { class: 'sf-kv-row' }, kEl, h('div', ctl, rec.errEl), h('button', { type: 'button', class: 'btn ghost sm', 'aria-label': `Remove ${k}`, title: 'Remove', onclick: () => { delete obj()[k]; this.changed(keyStr); draw(); } }, '×')));
      }
      if (!Object.keys(o).length) rows.appendChild(h('div', { class: 'faint small' }, 'No entries'));
      this.paintErrors();
    };
    draw();
    host.append(rows, h('div', { class: 'row' }, keyIn, h('button', { type: 'button', class: 'btn sm', onclick: add }, '+ Add entry')));
    return host;
  }

  unionControl(schema, path) {
    const branches = (schema.anyOf || schema.oneOf || []).map((b) => resolve(b).schema);
    const label = (b, i) => b.title || (b.const !== undefined ? String(b.const) : '') || (Array.isArray(b.enum) ? 'one of ' + b.enum.slice(0, 3).join('/') : '') || (b.type === 'array' ? `list of ${(resolve(b.items || {}).schema.type) || 'any'}` : b.type) || `option ${i + 1}`;
    const cur = this.get(path);
    let bi = branches.findIndex((b) => cur !== undefined && !validate(b, cur, '').length);
    if (bi < 0) bi = 0;
    const host = h('div', { class: 'sf-list' });
    const slot = h('div');
    const draw = () => {
      this.dropPrefix(this.key(path) + '.');
      clear(slot);
      const b = branches[bi];
      slot.appendChild(this.rawControl(kindOf(b), { ...b, description: undefined }, path, { required: true }));
    };
    if (branches.length > 1) {
      const sel = h('select', { 'aria-label': 'Value type' }, branches.map((b, i) => h('option', { value: String(i) }, label(b, i))));
      sel.value = String(bi);
      sel.addEventListener('change', () => { bi = Number(sel.value); this.set(path, defaultFor(branches[bi])); draw(); });
      host.append(sel);
    }
    draw();
    host.append(slot);
    return host;
  }

  jsonControl(schema, path) {
    const cur = this.get(path);
    const ta = h('textarea', { rows: 3, spellcheck: 'false', 'aria-label': 'JSON value', placeholder: 'JSON value' });
    ta.value = cur === undefined ? '' : (typeof cur === 'string' ? JSON.stringify(cur) : JSON.stringify(cur, null, isObj(cur) || Array.isArray(cur) ? 2 : 0));
    ta.addEventListener('input', () => {
      const t = ta.value.trim();
      if (t === '') { ta.classList.remove('invalid'); this.set(path, undefined); return; }
      try { const v = JSON.parse(t); ta.classList.remove('invalid'); this.set(path, v); } catch (_) {
        // allow bare words as strings
        if (/^[^{}\[\]"]*$/.test(t) && !/^[-\d.]/.test(t) && !['true', 'false', 'null'].includes(t)) { ta.classList.remove('invalid'); this.set(path, t); } else ta.classList.add('invalid');
      }
    });
    return ta;
  }

  // ---------------------------------------------------------- errors, filter
  paintErrors(reveal = false) {
    if (this.mode !== 'form') { this.paintRawErrors(); return; }
    const all = new Map();
    for (const e of this.errors) all.set(e.path, e.message);
    for (const [p, m] of Object.entries(this.serverErrors)) all.set(p, m);
    for (const rec of this.records.values()) { if (rec.errEl) rec.errEl.textContent = ''; if (rec.el) rec.el.classList.remove('field-invalid'); }
    clear(this.summaryEl);
    const leftovers = [];
    let first = null;
    for (const [p, m] of all) {
      let rec = this.records.get(p); let shown = p;
      if (!rec) { // nearest ancestor
        const parts = p.split('.');
        while (parts.length > 1 && !rec) { parts.pop(); rec = this.records.get(parts.join('.')); }
        shown = p;
      }
      if (rec && rec.errEl) {
        rec.errEl.textContent = rec.errEl.textContent ? rec.errEl.textContent + '; ' : '';
        rec.errEl.textContent += (shown !== this.recKey(rec) ? `${shown}: ` : '') + m;
        if (rec.el) rec.el.classList.add('field-invalid');
        if (!first) first = rec;
      } else leftovers.push([p, m]);
    }
    if (leftovers.length) this.summaryEl.append(h('ul', { class: 'errs' }, leftovers.map(([p, m]) => h('li', h('code', p || '(root)'), ': ', m))));
    if (reveal && first) this.revealRecord(first);
  }
  recKey(rec) { for (const [k, r] of this.records) if (r === rec) return k; return ''; }
  revealRecord(rec) {
    if (!rec.el) return;
    let n = rec.el;
    while (n && n !== this.rootEl) {
      if (n.tagName === 'DETAILS') n.open = true;
      if (n.classList && n.classList.contains('sf-pane') && n.dataset.tab) this.activateTab(n.dataset.tab);
      n = n.parentElement;
    }
    rec.el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  applyFilter() {
    const q = (this.filterEl.value || '').trim().toLowerCase();
    this.rootEl.classList.toggle('filtering', !!q);
    if (this.panes) for (const [g, pane] of this.panes) pane.classList.toggle('hidden', !q && g !== this.activeTab);
    for (const [, rec] of this.records) { if (rec.el && rec.text) rec.el.classList.toggle('sf-hidden-by-filter', !!q && !rec.text.includes(q) && !(rec.group)); }
    if (q) {
      this.rootEl.querySelectorAll('details').forEach((d) => { d.open = true; });
      // hide groups without any visible leaf
      const groups = Array.from(this.rootEl.querySelectorAll('.sf-field')).filter((f) => f.querySelector(':scope > details.sf-group')).reverse();
      for (const g of groups) {
        const anyVisible = Array.from(g.querySelectorAll('.sf-field')).some((f) => !f.classList.contains('sf-hidden-by-filter') && !f.querySelector(':scope > details.sf-group'));
        g.classList.toggle('sf-hidden-by-filter', !anyVisible);
      }
    } else {
      this.rootEl.querySelectorAll('.sf-hidden-by-filter').forEach((f) => f.classList.remove('sf-hidden-by-filter'));
      this.setAdvancedOpen(this.showAdvanced);
    }
  }
}

export function createSchemaForm(container, opts) { return new SchemaForm(container, opts); }
