// Hash router. Route table lives in routes.js. A page module exports `default async function render(ctx)`.
//   ctx = { root, params, query, route, navigate(path, query), onCleanup(fn), setTitle(t), isActive() }

import { ROUTES, DEFAULT_ROUTE } from './routes.js';
import { h, clear } from './util.js';
import { emit } from './state.js';

export function parseHash(hash = window.location.hash) {
  let s = hash.replace(/^#/, '');
  if (!s.startsWith('/')) s = '/' + s;
  const qi = s.indexOf('?');
  const path = (qi >= 0 ? s.slice(0, qi) : s).replace(/\/+$/, '') || '/';
  const query = {};
  if (qi >= 0) for (const [k, v] of new URLSearchParams(s.slice(qi + 1))) query[k] = v;
  return { path, query };
}

export function buildHash(path, query) {
  let s = '#' + path;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) if (v != null && v !== '') qs.set(k, v);
    const q = qs.toString();
    if (q) s += '?' + q;
  }
  return s;
}

export function matchRoute(path, routes = ROUTES) {
  const segs = path.split('/').filter(Boolean);
  for (const r of routes) {
    const rs = r.path.split('/').filter(Boolean);
    if (rs.length !== segs.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < rs.length; i++) {
      if (rs[i].startsWith(':')) params[rs[i].slice(1)] = decodeURIComponent(segs[i]);
      else if (rs[i] !== segs[i]) { ok = false; break; }
    }
    if (ok) return { route: r, params };
  }
  return null;
}

export function navigate(path, query) {
  const target = buildHash(path, query);
  if (window.location.hash === target) window.dispatchEvent(new HashChangeEvent('hashchange'));
  else window.location.hash = target;
}

/** Replace the query string without adding history entries or re-rendering. */
export function replaceQuery(query) {
  const { path } = parseHash();
  const t = buildHash(path, query);
  try { history.replaceState(null, '', window.location.pathname + window.location.search + t); } catch (_) { /* ignore */ }
}

export class Router {
  constructor(outlet) {
    this.outlet = outlet;
    this.cleanups = [];
    this.token = 0;
    this.current = null;
    window.addEventListener('hashchange', () => this.render());
  }

  start() {
    if (!window.location.hash || window.location.hash === '#') window.location.hash = '#' + DEFAULT_ROUTE;
    else this.render();
  }

  runCleanups() {
    const list = this.cleanups; this.cleanups = [];
    for (const fn of list) { try { fn(); } catch (e) { console.error('cleanup failed', e); } }
  }

  async render() {
    const { path, query } = parseHash();
    if (path === '/') { window.location.replace('#' + DEFAULT_ROUTE); return; } // '#/' is the home route, not a 404
    const my = ++this.token;
    this.runCleanups();
    const m = matchRoute(path);
    clear(this.outlet);
    if (!m) {
      this.outlet.appendChild(h('div', { class: 'page' },
        h('div', { class: 'empty' }, h('h3', 'Page not found'), h('p', `No page at "${path}".`), h('a', { class: 'btn primary', href: '#' + DEFAULT_ROUTE }, 'Go to dashboard'))));
      emit('route', { route: null, path, query });
      return;
    }
    const { route, params } = m;
    const root = h('div', { class: 'page', 'data-page': route.id });
    this.outlet.appendChild(root);
    root.appendChild(h('div', { class: 'loading' }, h('span', { class: 'spinner' }), 'Loading...'));
    document.title = `${route.title} - Forge LLM`;
    const ctx = {
      root, params, query, route,
      navigate, replaceQuery,
      onCleanup: (fn) => { if (my === this.token) this.cleanups.push(fn); else { try { fn(); } catch (_) { /* ignore */ } } },
      setTitle: (t) => { if (my === this.token) document.title = `${t} - Forge LLM`; },
      isActive: () => my === this.token,
    };
    this.current = { route, params, query };
    emit('route', { route, path, query, params });
    try {
      const mod = await import(route.module);
      if (my !== this.token) return;
      clear(root);
      await (mod.default || mod.render)(ctx);
    } catch (e) {
      if (my !== this.token) return;
      console.error(e);
      clear(root);
      root.appendChild(h('div', { class: 'notice error' }, h('div', { class: 'ico' }, '!'),
        h('div', { class: 'body' }, h('h3', 'This page failed to load'), h('p', { class: 'mono' }, String(e && e.message || e)),
          h('button', { class: 'btn', onclick: () => this.render() }, 'Retry'))));
    }
  }
}
