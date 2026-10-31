# Security notes

Forge LLM is an engineering tool that runs training/serving jobs as subprocesses and can read and write files
under `FORGE_HOME`. Treat the API as **remote code execution for whoever can reach it with a valid token**.

## Authentication

* Set `FORGE_API_TOKEN` (the scripts `scripts/run_server.sh` and the Docker entrypoint generate one and print it).
  Clients send `Authorization: Bearer <token>` (the API also accepts `X-Forge-Token` and a `?token=` query parameter for browser
  `EventSource`; query tokens end up in URLs and logs, so avoid them where headers work). `GET /api/v1/system/health` is public.
  If the variable is empty, **authentication is off**.
* The token is stored in `$FORGE_HOME/api_token` (mode 600) by `run_server.sh`; the UI keeps it in the browser's
  local storage. Rotate by deleting that file and restarting.
* There is a single shared token: no users, roles or audit identities yet (Plan C-073 / C-120 are not done).

## Network exposure

* **Do not expose the API publicly without a tunnel or TLS.** The server speaks plain HTTP; tokens and chat
  content travel unencrypted on an exposed port.
* Recommended on vast.ai: bind to `127.0.0.1` and use `ssh -N -L 8765:127.0.0.1:8765 ...`
  (see `docs/VAST_AI.md`). If you must publish a port, put it behind a TLS reverse proxy and keep the token.
* Deployments (serve jobs) listen on their own ports (`FORGE_SERVE_BASE_PORT`, default 8800+) with their own
  `api_key` setting; tunnel those too.

## Secrets

* `HF_TOKEN` is read from the environment and never written to job records or logs by design. Check logs before
  sharing them anyway.
* Never commit tokens. `.gitignore` excludes `.env` and `*.token`.
* `gpu_report/` contains hardware and version information, not secrets, but review it before posting.

## Data and artifacts

* Weights, datasets and checkpoints are stored unencrypted under `FORGE_HOME`. Destroy the rented instance's
  volume when finished with sensitive data.
* Artifacts are content-addressed (integrity is checked on read) but **not signed**. Signed manifests are Plan
  item C-120 and are not implemented.
* Compression can change safety behaviour; the `safety` eval suite is a probe set, not a guarantee.

## Reporting a vulnerability

Open a private security advisory on the GitHub repository (Security tab) or email the maintainers listed in the
repository profile. Please do not file public issues for vulnerabilities.
