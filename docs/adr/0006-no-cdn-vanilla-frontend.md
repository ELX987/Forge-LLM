# ADR 0006: No-CDN, no-build vanilla front end

- Status: accepted
- Date: 2026-10-30
- Related Plan items: UI-1, UI-3, C-070, C-072, C-073

## Context

The web UI is used on rented machines that may have restricted egress, are reached through SSH tunnels, and are
destroyed after use. A UI that loads scripts, fonts or styles from a CDN fails offline and leaks usage to third
parties; a UI that needs a Node build adds a toolchain and a build step to every instance.

## Decision

* The front end is plain **ES modules, CSS and HTML under `forge/web/static/`**, served by the FastAPI process at
  `/`. No bundler, no framework, no `npm install`, no external URLs (fonts, icons, scripts or styles).
* Small in-repo helpers replace libraries: `api.js` (fetch + SSE), `forms.js` (JSON Schema to DOM), `charts.js`
  (canvas charts with zoom and large-series decimation), `router.js` (hash routes).
* **Forms are generated from JSON Schema** of the pydantic config models (`forge.core.jsonschema.schema_for`),
  so adding a config field requires no UI code.
* Static assets are packaged via `package-data` so `pip install` ships the UI.

## Consequences

* Works offline and behind tunnels; instant start; trivially auditable.
* Less polish than a component framework, and charts are bespoke. Acceptable for an engineering tool.
* A contributor-facing rule: **never add a remote URL to the front end**. CI greps for `http://` and `https://`
  in static JS/HTML/CSS except documentation links that are plain text (see `.github/workflows/ci.yml`).
