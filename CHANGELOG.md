# Changelog

All notable changes are recorded here. Format: Keep a Changelog; versions follow `forge/__version__.py`.

## [Unreleased]

### Added
- Repository scaffold, `forge.core` (schemas, hashing, db, registry, run context, recipe engine).
- Operations tooling: `scripts/vast_setup.sh`, `run_server.sh`, `gpu_smoke.sh`, `run_gpu_tests.sh`,
  `gpu_report.py`, CPU dev setup scripts (bash + PowerShell).
- Docker: `docker/Dockerfile.cuda`, `docker/docker-compose.yml`, `.dockerignore`.
- CI: CPU tests on Python 3.10 / 3.12, lint, shellcheck, docker build check, manual GPU job for self-hosted runners.
- Documentation: VAST_AI, USER_GUIDE, KERNELS, API, CONFIG_REFERENCE, CLAIMS, ROADMAP_STATUS, runbooks, ADRs 0001-0006.
- Config templates under `configs/` validated by `tests/test_configs_templates.py`.

### Known limitations
- No GPU code path has been executed on hardware yet; see `docs/KERNELS.md` for the validation table.
