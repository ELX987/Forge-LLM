// App bootstrap: theme, nav, router, connection status, demo-data banner.

import { Router } from './router.js';
import { ROUTES } from './routes.js';
import { h, clear } from './util.js';
import { applyTheme, effectiveTheme, setTheme, getTheme, isMockMode, setMockMode, on } from './state.js';
import { icon, installTooltips, toast } from './components/ui.js';
import { api } from './api.js';

applyTheme();
installTooltips();

// ---- nav icons + active state
const nav = document.getElementById('nav');
for (const a of nav.querySelectorAll('a[data-icon]')) a.prepend(icon(a.dataset.icon));
function markActive(route) {
  const id = route && (route.nav ? route.id : route.parent);
  for (const a of nav.querySelectorAll('a')) {
    const target = ROUTES.find((r) => '#' + r.path === a.getAttribute('href'));
    if (target && target.id === id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  }
}
on('route', ({ route }) => {
  markActive(route);
  document.getElementById('page-title').textContent = route ? route.title : 'Forge LLM';
  document.body.classList.remove('nav-open');
  menuBtn.setAttribute('aria-expanded', 'false');
  const v = document.getElementById('view');
  window.scrollTo(0, 0);
  void v;
});

// ---- mobile nav
const menuBtn = document.getElementById('menu-btn');
menuBtn.appendChild(icon('menu'));
menuBtn.addEventListener('click', () => {
  const open = document.body.classList.toggle('nav-open');
  menuBtn.setAttribute('aria-expanded', String(open));
});
document.getElementById('scrim').addEventListener('click', () => { document.body.classList.remove('nav-open'); menuBtn.setAttribute('aria-expanded', 'false'); });

// ---- theme toggle (auto -> light -> dark)
const themeBtn = document.getElementById('theme-btn');
function paintThemeBtn() {
  clear(themeBtn);
  const t = getTheme();
  themeBtn.appendChild(icon(effectiveTheme() === 'dark' ? 'moon' : 'sun'));
  themeBtn.title = `Theme: ${t} (click to change)`;
}
themeBtn.addEventListener('click', () => {
  const order = ['auto', 'light', 'dark'];
  const next = order[(order.indexOf(getTheme()) + 1) % order.length];
  setTheme(next); paintThemeBtn(); toast(`Theme: ${next}`, 'info', 1200);
});
on('theme', paintThemeBtn);
paintThemeBtn();
try { window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paintThemeBtn); } catch (_) { /* ignore */ }

// ---- banners: demo data + unreachable API + auth
const banners = document.getElementById('banners');
function paintBanners(extra) {
  clear(banners);
  if (isMockMode()) {
    banners.appendChild(h('div', { class: 'banner demo', role: 'status' }, 'DEMO DATA - this UI is talking to a small in-browser fake backend. Nothing is real, nothing is saved to a server.',
      h('button', { class: 'btn sm', onclick: () => { setMockMode(false); const u = new URL(window.location.href); u.searchParams.delete('mock'); window.location.href = u.toString(); } }, 'Exit demo')));
  }
  if (extra) banners.appendChild(extra);
}
paintBanners();
let unreachableShown = false;
on('api:unreachable', () => {
  if (unreachableShown || isMockMode()) return;
  unreachableShown = true;
  setConn('offline');
  paintBanners(h('div', { class: 'banner err', role: 'alert' }, 'Cannot reach the Forge API at /api/v1. Is the server running?',
    h('button', { class: 'btn sm', onclick: () => { setMockMode(true); window.location.reload(); } }, 'Demo data'),
    h('button', { class: 'btn sm', onclick: () => { unreachableShown = false; paintBanners(); checkConn(); router.render(); } }, 'Retry')));
});
on('api:unauthorized', () => {
  toast(h('span', 'The API rejected the request (401/403). Set your API token in ', h('a', { href: '#/settings' }, 'Settings'), '.'), 'warn', 8000);
});
on('api:ok', () => { if (unreachableShown) { unreachableShown = false; paintBanners(); } setConn('online'); });

const connEl = document.getElementById('conn-status');
function setConn(state, info) {
  connEl.className = 'badge ' + (state === 'online' ? 'ok' : state === 'offline' ? 'bad' : (state === 'demo' || state === 'partial') ? 'warn' : 'dim');
  connEl.textContent = state === 'online' ? ('API online' + (info ? ' v' + info : '')) : state === 'demo' ? 'DEMO DATA' : state === 'partial' ? 'API: no /system/info' : state === 'offline' ? 'API offline' : 'connecting...';
}
async function checkConn() {
  if (isMockMode()) { setConn('demo'); return; }
  try { const info = await api.get('/system/info'); setConn('online', info && info.version); } catch (e) { setConn(e && e.network ? 'offline' : (e && e.unavailable ? 'partial' : 'online')); }
}

const router = new Router(document.getElementById('view'));
checkConn();
router.start();
setInterval(() => { if (!document.hidden) checkConn(); }, 30000);

window.addEventListener('unhandledrejection', (e) => { console.error(e.reason); });
