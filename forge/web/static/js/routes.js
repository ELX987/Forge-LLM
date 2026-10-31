// Single source of truth for the page list. index.html's <nav> must link to every entry with nav: true
// (checked by tests/test_web_static.py). Order matters: static paths before parameterised ones.

export const ROUTES = [
  { id: 'dashboard', path: '/dashboard', title: 'Dashboard', nav: true, group: 'Overview', icon: 'grid', module: './pages/dashboard.js' },
  { id: 'models', path: '/models', title: 'Models', nav: true, group: 'Assets', icon: 'cube', module: './pages/models.js' },
  { id: 'datasets', path: '/datasets', title: 'Datasets', nav: true, group: 'Assets', icon: 'db', module: './pages/datasets.js' },
  { id: 'hardware', path: '/hardware', title: 'Hardware', nav: true, group: 'Assets', icon: 'chip', module: './pages/hardware.js' },
  { id: 'new-job', path: '/jobs/new', title: 'New Job', nav: true, group: 'Run', icon: 'plus', module: './pages/newjob.js' },
  { id: 'jobs', path: '/jobs', title: 'Jobs', nav: true, group: 'Run', icon: 'list', module: './pages/jobs.js' },
  { id: 'job-detail', path: '/jobs/:id', title: 'Job', nav: false, parent: 'jobs', module: './pages/jobdetail.js' },
  { id: 'finetune', path: '/finetune', title: 'Fine-tune', nav: true, group: 'Run', icon: 'tune', module: './pages/finetune.js' },
  { id: 'recipes', path: '/recipes', title: 'Recipes', nav: true, group: 'Run', icon: 'flow', module: './pages/recipes.js' },
  { id: 'recipe-edit', path: '/recipes/:id', title: 'Recipe', nav: false, parent: 'recipes', module: './pages/recipes.js' },
  { id: 'compare', path: '/compare', title: 'Compare', nav: true, group: 'Analyse', icon: 'scale', module: './pages/compare.js' },
  { id: 'inference', path: '/inference', title: 'Inference', nav: true, group: 'Serve', icon: 'chat', module: './pages/inference.js' },
  { id: 'kernels', path: '/kernels', title: 'Kernel Lab', nav: true, group: 'Analyse', icon: 'bolt', module: './pages/kernels.js' },
  { id: 'storage', path: '/storage', title: 'Storage', nav: true, group: 'Analyse', icon: 'disk', module: './pages/storage.js' },
  { id: 'claims', path: '/claims', title: 'Claims Ledger', nav: true, group: 'Analyse', icon: 'check', module: './pages/claims.js' },
  { id: 'settings', path: '/settings', title: 'Settings', nav: true, group: 'System', icon: 'gear', module: './pages/settings.js' },
];

export const DEFAULT_ROUTE = '/dashboard';
