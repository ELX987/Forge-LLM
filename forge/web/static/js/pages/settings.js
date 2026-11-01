// Settings: API token, theme, demo mode, defaults, server info.

import { h, clear } from '../util.js';
import { api, tryGet } from '../api.js';
import { cudaBanner, pageHead, card, kv, toast, toastError, objectView, notice, badge, confirmDialog } from '../components/ui.js';
import { getToken, setToken, getTheme, setTheme, prefs, isMockMode, setMockMode } from '../state.js';

export default async function render(ctx) {
  const tokenEl = h('input', { type: 'password', value: getToken(), placeholder: 'FORGE_API_TOKEN (sent as Bearer)', autocomplete: 'off', 'aria-label': 'API token', style: 'min-width:320px' });
  const show = h('label', { class: 'row tight' }, h('input', { type: 'checkbox', onchange: (e) => { tokenEl.type = e.target.checked ? 'text' : 'password'; } }), 'Show');
  const tokenStatus = h('div', { class: 'small', role: 'status' });
  const test = async () => {
    setToken(tokenEl.value.trim());
    tokenStatus.className = 'small muted'; tokenStatus.textContent = 'Testing...';
    try { await api.get('/system/info'); tokenStatus.className = 'small ok'; tokenStatus.textContent = 'Connected - token accepted (or none required).'; } catch (e) { tokenStatus.className = 'small bad'; tokenStatus.textContent = e.unauthorized ? 'Rejected (401/403): wrong or missing token.' : e.network ? 'API unreachable.' : e.message; }
  };

  const themeSel = h('select', { 'aria-label': 'Theme', onchange: () => { setTheme(themeSel.value); } }, ['auto', 'light', 'dark'].map((t) => h('option', { value: t, selected: t === getTheme() }, t === 'auto' ? 'auto (follow system)' : t)));
  const smoothing = h('input', { type: 'range', min: 0, max: 0.98, step: 0.01, value: prefs.get('chart.smoothing', 0.6), 'aria-label': 'Default smoothing', oninput: () => { prefs.set('chart.smoothing', Number(smoothing.value)); sm.textContent = smoothing.value; } });
  const sm = h('span', { class: 'mono small' }, String(prefs.get('chart.smoothing', 0.6)));
  const adv = h('input', { type: 'checkbox', checked: !!prefs.get('form.advanced', false), onchange: (e) => prefs.set('form.advanced', e.target.checked) });
  const infoHost = h('div');

  ctx.root.append(pageHead('Settings', 'Stored in this browser only. Nothing here is sent anywhere except the API token to your own Forge server.'),
    h('div', { class: 'grid cols-2' },
      h('div', { class: 'stack' },
        card('API access', h('div', { class: 'stack' }, h('p', { class: 'muted' }, 'If the server was started with FORGE_API_TOKEN, paste it here. It is kept in this browser and sent as an Authorization: Bearer header on every request (including streams).'),
          h('div', { class: 'row' }, tokenEl, show), h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: test }, 'Save & test'), h('button', { class: 'btn', onclick: () => { tokenEl.value = ''; setToken(''); tokenStatus.textContent = 'Token cleared.'; } }, 'Clear')), tokenStatus)),
        card('Appearance & defaults', h('div', { class: 'stack' }, h('div', { class: 'row' }, h('label', 'Theme'), themeSel), h('div', { class: 'row' }, h('label', 'Default chart smoothing'), smoothing, sm),
          h('label', { class: 'row' }, adv, 'Show "advanced" config fields by default'))),
        card('Demo data', h('div', { class: 'stack' }, h('p', { class: 'muted' }, 'Demo mode replaces the API with a small in-browser fake backend so every page can be reviewed without the server. All numbers are synthetic and labelled DEMO DATA.'),
          isMockMode() ? h('button', { class: 'btn', onclick: () => { setMockMode(false); const u = new URL(window.location.href); u.searchParams.delete('mock'); window.location.href = u.toString(); } }, 'Exit demo mode') : h('button', { class: 'btn', onclick: () => { setMockMode(true); window.location.reload(); } }, 'Enter demo mode'))),
        card('Local data', h('button', { class: 'btn danger', onclick: async () => { if (await confirmDialog({ title: 'Clear local settings?', message: 'Removes token, theme and form preferences from this browser.', confirmLabel: 'Clear', danger: true })) { try { Object.keys(localStorage).filter((k) => k.startsWith('forge.')).forEach((k) => localStorage.removeItem(k)); } catch (_) { /* ignore */ } toast('Cleared', 'ok'); window.location.reload(); } } }, 'Clear all local settings'))),
      card('Server', infoHost)));

  const r = await tryGet('/system/info');
  if (!ctx.isActive()) return;
  clear(infoHost);
  if (!r.ok) infoHost.appendChild(r.unavailable ? notice('unavailable', 'No /system/info endpoint', 'The server did not expose system information.') : notice('error', 'Server unreachable', r.error.message));
  else infoHost.append(cudaBanner(async (refresh, probe) => { const c = await tryGet('/system/cuda', { probe: probe ? 1 : 0, ...(refresh ? { refresh: 1 } : {}) }); return c.ok ? c.data : null; }), objectView(r.data));
  void kv; void badge; void toastError;
}
