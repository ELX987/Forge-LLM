// Tiny persistent preference store + app-wide event bus. localStorage is always wrapped in try/catch.

const PREFIX = 'forge.';
const mem = new Map();

function lsGet(key) { try { return window.localStorage.getItem(PREFIX + key); } catch (_) { return mem.has(key) ? mem.get(key) : null; } }
function lsSet(key, val) { try { window.localStorage.setItem(PREFIX + key, val); } catch (_) { mem.set(key, val); } }
function lsDel(key) { try { window.localStorage.removeItem(PREFIX + key); } catch (_) { /* ignore */ } mem.delete(key); }

const listeners = new Map();

export const prefs = {
  get(key, def = null) {
    const raw = lsGet(key);
    if (raw == null) return def;
    try { return JSON.parse(raw); } catch (_) { return raw; }
  },
  set(key, value) {
    if (value === undefined || value === null) lsDel(key); else lsSet(key, JSON.stringify(value));
    emit('pref:' + key, value);
  },
  remove(key) { lsDel(key); emit('pref:' + key, null); },
};

export function on(evt, fn) {
  if (!listeners.has(evt)) listeners.set(evt, new Set());
  listeners.get(evt).add(fn);
  return () => listeners.get(evt)?.delete(fn);
}
export function emit(evt, payload) {
  for (const fn of Array.from(listeners.get(evt) || [])) { try { fn(payload); } catch (e) { console.error(e); } }
}

// ---- theme: 'auto' | 'light' | 'dark'
export function getTheme() { return prefs.get('theme', 'auto'); }
export function applyTheme(theme = getTheme()) {
  const root = document.documentElement;
  if (theme === 'light' || theme === 'dark') root.setAttribute('data-theme', theme); else root.removeAttribute('data-theme');
  emit('theme', theme);
}
export function setTheme(theme) { prefs.set('theme', theme); applyTheme(theme); }
export function effectiveTheme() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// ---- mock (demo) mode
export function isMockMode() {
  try { if (new URLSearchParams(window.location.search).get('mock') === '1') return true; } catch (_) { /* ignore */ }
  try { return window.sessionStorage.getItem('forge.mock') === '1'; } catch (_) { return false; }
}
export function setMockMode(on) {
  try { if (on) window.sessionStorage.setItem('forge.mock', '1'); else window.sessionStorage.removeItem('forge.mock'); } catch (_) { /* ignore */ }
}

// ---- API token
export function getToken() { return prefs.get('token', '') || ''; }
export function setToken(t) { if (t) prefs.set('token', t); else prefs.remove('token'); }
