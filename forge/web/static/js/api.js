// API client: fetch wrapper with bearer-token support, normalised errors, fetch-based SSE with auto-reconnect.
// In mock mode (?mock=1) every call is routed to the in-browser fake backend (js/mock.js).

import { getToken, isMockMode, emit } from './state.js';

export const API_BASE = '/api/v1';
let mockModule = null;

async function mock() {
  if (!mockModule) mockModule = await import('./mock.js');
  return mockModule;
}

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, fieldErrors = {}, network = false, path = '' } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.fieldErrors = fieldErrors;
    this.network = network;
    this.path = path;
  }
  /** Backend module / route missing: 404, 405, 501, 503 (feature not available). */
  get unavailable() { return [404, 405, 501, 503].includes(this.status); }
  get unauthorized() { return this.status === 401 || this.status === 403; }
}

export function isUnavailable(err) { return !!(err && err.unavailable); }

/** Extract {"dotted.path": "message"} from FastAPI/pydantic style or custom error bodies. */
export function extractFieldErrors(body) {
  const out = {};
  const put = (loc, msg) => {
    let parts = Array.isArray(loc) ? loc.map(String) : String(loc || '').split('.').filter(Boolean);
    while (parts.length && ['body', 'params', 'query', 'config'].includes(parts[0]) && parts.length > 1) parts = parts.slice(1);
    const key = parts.join('.');
    out[key] = out[key] ? out[key] + '; ' + msg : msg;
  };
  if (!body || typeof body !== 'object') return out;
  const det = body.detail !== undefined ? body.detail : body;
  if (Array.isArray(det)) {
    for (const e of det) if (e && typeof e === 'object') put(e.loc || e.path || e.field || '', e.msg || e.message || 'invalid');
  }
  const errs = body.errors || body.field_errors || body.fieldErrors || (det && det.errors) || (det && det.fields);
  if (Array.isArray(errs)) for (const e of errs) put(e.loc || e.path || e.field || '', e.msg || e.message || 'invalid');
  else if (errs && typeof errs === 'object') for (const [k, v] of Object.entries(errs)) put(k, Array.isArray(v) ? v.join('; ') : String(v));
  return out;
}

function errorMessage(body, status, statusText) {
  if (body && typeof body === 'object') {
    const det = body.detail !== undefined ? body.detail : body.error || body.message;
    if (typeof det === 'string') return det;
    if (det && typeof det === 'object' && !Array.isArray(det) && (det.message || det.msg)) return det.message || det.msg;
    if (Array.isArray(det) && det.length) {
      const first = det[0];
      return typeof first === 'string' ? first : `${(first.loc || []).slice(1).join('.') || 'request'}: ${first.msg || first.message || 'invalid'}`;
    }
    if (body.error && body.error.message) return body.error.message;
  }
  if (typeof body === 'string' && body && body.length < 300) return body;
  return `${status} ${statusText || 'request failed'}`.trim();
}

function buildUrl(path, query) {
  let url = path.startsWith('http') ? path : API_BASE + path;
  if (query) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(query)) {
      if (v == null || v === '') continue;
      if (Array.isArray(v)) qs.set(k, v.join(',')); else qs.set(k, String(v));
    }
    const s = qs.toString();
    if (s) url += (url.includes('?') ? '&' : '?') + s;
  }
  return url;
}

function authHeaders(extra = {}) {
  const h = { ...extra };
  const tok = getToken();
  if (tok) h.Authorization = 'Bearer ' + tok;
  return h;
}

export async function request(method, path, { query, body, signal, form, timeoutMs = 60000 } = {}) {
  if (isMockMode()) {
    const m = await mock();
    return m.mockRequest(method, path, { query, body, form });
  }
  const url = buildUrl(path, query);
  const init = { method, headers: authHeaders({ Accept: 'application/json' }), signal };
  if (form) init.body = form;
  else if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  let resp;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  if (signal) signal.addEventListener('abort', () => ctrl.abort(), { once: true });
  init.signal = ctrl.signal;
  try {
    resp = await fetch(url, init);
  } catch (e) {
    clearTimeout(timer);
    if (e && e.name === 'AbortError' && signal && signal.aborted) throw e;
    const err = new ApiError(e && e.name === 'AbortError' ? 'Request timed out' : 'API unreachable (network error)', { network: true, path });
    emit('api:unreachable', err);
    throw err;
  }
  clearTimeout(timer);
  const ctype = resp.headers.get('content-type') || '';
  let data = null;
  try {
    data = ctype.includes('json') ? await resp.json() : await resp.text();
  } catch (_) { data = null; }
  if (!resp.ok) {
    const err = new ApiError(errorMessage(data, resp.status, resp.statusText), { status: resp.status, detail: data, fieldErrors: extractFieldErrors(data), path });
    if (err.unauthorized) emit('api:unauthorized', err);
    throw err;
  }
  emit('api:ok', null);
  return data;
}

export const api = {
  get: (path, query, opts) => request('GET', path, { query, ...opts }),
  post: (path, body, opts) => request('POST', path, { body: body === undefined ? {} : body, ...opts }),
  put: (path, body, opts) => request('PUT', path, { body, ...opts }),
  patch: (path, body, opts) => request('PATCH', path, { body, ...opts }),
  del: (path, query, opts) => request('DELETE', path, { query, ...opts }),
  /** multipart upload; onProgress(frac) via XHR (not available in mock -> simulated). */
  upload(path, formData, onProgress) {
    if (isMockMode()) return mock().then((m) => m.mockUpload(path, formData, onProgress));
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', buildUrl(path));
      const tok = getToken();
      if (tok) xhr.setRequestHeader('Authorization', 'Bearer ' + tok);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
      xhr.onerror = () => reject(new ApiError('Upload failed (network error)', { network: true, path }));
      xhr.onload = () => {
        let data = null;
        try { data = JSON.parse(xhr.responseText); } catch (_) { data = xhr.responseText; }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new ApiError(errorMessage(data, xhr.status, xhr.statusText), { status: xhr.status, detail: data, fieldErrors: extractFieldErrors(data), path }));
      };
      xhr.send(formData);
    });
  },
};

/**
 * Safe wrapper for optional features: resolves {ok, data, unavailable, error}. Never throws except for abort.
 */
export async function tryGet(path, query, opts) {
  try {
    return { ok: true, data: await api.get(path, query, opts), unavailable: false, error: null };
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    return { ok: false, data: null, unavailable: isUnavailable(e), error: e };
  }
}

// ------------------------------------------------------------------ SSE (fetch streaming; supports Authorization + POST)

/**
 * sse(path, {query (object, or a function returning one - re-evaluated on every reconnect so resume cursors stay current), method, body, onEvent(ev), onStatus(status), reconnect, maxRetries, signal}) -> {close()}
 * ev = {event: string, data: any (parsed JSON when possible), raw: string, id}
 * Status: 'connecting' | 'open' | 'retrying' | 'closed' | 'unavailable' | 'error'
 */
export function sse(path, opts = {}) {
  if (isMockMode()) {
    let handle = null; let closed = false;
    mock().then((m) => { if (!closed) handle = m.mockStream(path, opts); });
    return { close() { closed = true; if (handle) handle.close(); } };
  }
  const { query, method = 'GET', body, onEvent, onStatus, reconnect = true, maxRetries = Infinity } = opts;
  let closed = false; let ctrl = null; let retries = 0; let lastId = null;
  const status = (s, info) => { try { onStatus && onStatus(s, info); } catch (e) { console.error(e); } };

  async function run() {
    while (!closed) {
      ctrl = new AbortController();
      status(retries ? 'retrying' : 'connecting');
      let gotData = false;
      try {
        const headers = authHeaders({ Accept: 'text/event-stream' });
        if (body !== undefined) headers['Content-Type'] = 'application/json';
        if (lastId) headers['Last-Event-ID'] = lastId;
        const resp = await fetch(buildUrl(path, typeof query === 'function' ? query() : query), { method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal: ctrl.signal });
        if (!resp.ok) {
          let data = null; try { data = await resp.json(); } catch (_) { /* ignore */ }
          const err = new ApiError(errorMessage(data, resp.status, resp.statusText), { status: resp.status, detail: data, path });
          status(err.unavailable ? 'unavailable' : 'error', err);
          if (err.unavailable || err.unauthorized || resp.status < 500) { closed = true; return; }
          throw err;
        }
        status('open');
        retries = 0;
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = ''; let ev = { event: 'message', data: [], id: null };
        const flush = () => {
          if (!ev.data.length && ev.event === 'message') return;
          const raw = ev.data.join('\n');
          let data = raw;
          try { data = JSON.parse(raw); } catch (_) { /* keep raw text */ }
          if (ev.id) lastId = ev.id;
          gotData = true;
          try { onEvent && onEvent({ event: ev.event, data, raw, id: ev.id }); } catch (e) { console.error(e); }
          ev = { event: 'message', data: [], id: null };
        };
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.search(/\r\n|\n|\r/)) >= 0) {
            const line = buf.slice(0, idx);
            const nl = buf.slice(idx).startsWith('\r\n') ? 2 : 1;
            buf = buf.slice(idx + nl);
            if (line === '') flush();
            else if (line.startsWith(':')) continue;
            else {
              const c = line.indexOf(':');
              const field = c < 0 ? line : line.slice(0, c);
              let val = c < 0 ? '' : line.slice(c + 1);
              if (val.startsWith(' ')) val = val.slice(1);
              if (field === 'event') ev.event = val; else if (field === 'data') ev.data.push(val); else if (field === 'id') ev.id = val;
            }
          }
        }
        if (ev.data.length) flush();
        if (!reconnect) { closed = true; status('closed'); return; }
      } catch (e) {
        if (closed || (e && e.name === 'AbortError')) { status('closed'); return; }
        if (!reconnect) { status('error', e); closed = true; return; }
      }
      if (closed) break;
      retries += 1;
      if (retries > maxRetries) { status('error'); closed = true; return; }
      const delay = Math.min(10000, 500 * 2 ** Math.min(retries, 5)) * (gotData ? 0.5 : 1);
      await new Promise((r) => setTimeout(r, delay));
    }
    status('closed');
  }
  run();
  if (opts.signal) opts.signal.addEventListener('abort', () => { closed = true; if (ctrl) ctrl.abort(); }, { once: true });
  return { close() { closed = true; if (ctrl) ctrl.abort(); } };
}
