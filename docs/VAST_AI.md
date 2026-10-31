# Running Forge on vast.ai (runbook)

This runbook takes you from "no instance" to "GPU tests reported and web UI open in my browser". It assumes a
Windows or Linux laptop with an SSH client. **No Forge GPU code has been validated on hardware yet** - the first
run on your instance *is* the validation; section 7 tells you what to send back.

vast.ai's console changes over time. Where this document names a UI label, treat it as a hint; the underlying
requirements (CUDA version, disk, open ports, SSH) are what matter.

## 1. Pick the hardware

| Goal | Suggested GPU | Min VRAM | Notes |
|------|---------------|----------|-------|
| Smoke test, kernel correctness, 1-3B models | RTX 3090/4090, A10, L4, A5000 | 24 GB | Cheapest way to validate Triton kernels on `sm_86`/`sm_89` |
| 8B LoRA fine-tune, W4A16/FP8 compression, 8B serving | RTX 4090 (24 GB), L40S/A6000 (48 GB), A100 40/80 GB | 24 GB (LoRA, 4-bit), 48-80 GB (BF16 full fine-tune) | Full BF16 fine-tune of 8B needs 80 GB or multiple GPUs |
| FP8 paths (`torch._scaled_mm`, FP8 training rung) | RTX 4090, L40S (`sm_89`), H100/H200 (`sm_90`) | 24 GB+ | FP8 tensor cores need `sm_89+` |
| Native NVFP4 / MXFP4 experiments | B200 (`sm_100`), RTX 5090 / PRO 6000 Blackwell (`sm_120`) | 32 GB+ | **Needs CUDA >= 12.8 image and a recent driver.** Native FP4 is only a hypothesis until `forge probe` confirms it |
| Everything on older cards | V100 (`sm_70`), T4 (`sm_75`) | 16 GB | fp16 only (no bf16); expect fallbacks |

Search filters worth setting in the vast.ai console:

* **Disk space: >= 200 GB** (500 GB if you will keep several 8B checkpoints and datasets). Disk size is fixed when
  the instance is created and cannot be grown later.
* **CUDA version >= 12.4** (12.8 for Blackwell). The host driver must support the CUDA version of your image.
* **Reliability >= 0.98**, **Direct port count >= 4** (so you can publish the web UI port without an SSH tunnel
  fallback), **Download speed >= 500 Mbps** (model downloads are the biggest wall-clock cost), PCIe/NVMe disk.
* Prefer **on-demand** for the first run; **interruptible (bid)** instances can be pre-empted mid-job (see 6).
* One GPU is enough for everything except multi-GPU tests (`tp_size > 1`, DDP/FSDP). Rent 2+ GPUs only for those.

## 2. Choose the image and launch mode

Any image with Python >= 3.10, a CUDA-enabled PyTorch >= 2.3 and `git` works. Good options:

* `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (Ada/Hopper/Ampere) - the `-devel` tag includes `nvcc`/headers
  and avoids Triton build surprises.
* A `pytorch/pytorch` or `vastai/pytorch` image built on **CUDA 12.8** for RTX 50xx / B200.
* The vast.ai "PyTorch" template (`vastai/pytorch`) is fine; it provides `/workspace`, Jupyter and SSH.

Settings when you create the instance:

* **Launch mode: SSH** (or "Jupyter + SSH"). Tick *Direct SSH* when offered - it is faster and more stable.
* **On-start script**: leave empty for the first run.
* **Docker options / port mapping**: add `-p 8765:8765` (Forge UI+API) and `-p 8800-8810:8800-8810` if you want to
  reach deployed model servers directly. Without this you will use SSH tunnelling (section 4), which is the safer default.
* **Environment variables** (optional): `HF_TOKEN=<your read token>` for gated models such as Llama. Do not put
  tokens anywhere that is committed to git.
* Disk: as above. `/workspace` is the persistent volume on vast.ai templates; Forge puts all state under
  `FORGE_HOME`, which the setup script defaults to `/workspace/forge-home` when `/workspace` exists.

## 3. First login and setup

From your machine (the console shows the exact host and port):

```bash
ssh -p <ssh_port> root@<ssh_host>            # Windows: same command in PowerShell or Git-Bash
```

On the instance:

```bash
cd /workspace
git clone https://github.com/<you>/forge-llm.git "Forge LLM"      # private repo: use a deploy key or a token (below)
cd "Forge LLM"
bash scripts/vast_setup.sh
```

Cloning a private repo: either `ssh -A` (agent forwarding) from your laptop and use the `git@github.com:` URL, or
create a **fine-grained read-only token** for this repo and use
`git clone https://<user>:<token>@github.com/<you>/<repo>.git`, then immediately
`git remote set-url origin https://github.com/<you>/<repo>.git` so the token is not stored in `.git/config`.
Alternatively `rsync`/`scp` the working tree from your laptop.

`vast_setup.sh` is idempotent. It:

1. verifies `nvidia-smi` works and lists the GPUs,
2. verifies Python >= 3.10 and that **torch sees CUDA** (it does not replace the image's torch unless you set
   `FORGE_INSTALL_TORCH=1`; Blackwell hosts may need `FORGE_TORCH_INDEX=https://download.pytorch.org/whl/cu128`),
3. sets `FORGE_HOME` (default `/workspace/forge-home`), warns if free space is below 100 GiB,
4. installs `forge[gpu,serve,parquet,dev]` in editable mode and Triton if missing,
5. writes `~/.forge_env` (sourced from `~/.bashrc`) and prints all relevant versions.

If it stops with "torch cannot see CUDA" the image/driver pair is wrong; pick a different image or host.

## 4. Starting the web app and reaching it

```bash
bash scripts/run_server.sh --tmux          # or --nohup; plain run is foreground
```

It prints the **API token** (also saved in `$FORGE_HOME/api_token`). Paste it into the UI under *Settings -> API
token*; API clients send `Authorization: Bearer <token>`.

### Option A (recommended): SSH tunnel

Run the server bound to loopback and forward the port from your laptop:

```bash
# on the instance
bash scripts/run_server.sh --host 127.0.0.1 --tmux
# on your laptop (keep this running)
ssh -N -L 8765:127.0.0.1:8765 -p <ssh_port> root@<ssh_host>
# then open http://localhost:8765
```

To also reach a model server started from the *Inference* page (default ports 8800+):
`ssh -N -L 8765:127.0.0.1:8765 -L 8800:127.0.0.1:8800 -p <ssh_port> root@<ssh_host>`.

### Option B: published port

If you mapped `-p 8765:8765`, the console shows the externally visible host:port in the instance's port list
(vast.ai assigns a random external port). Open `http://<public_ip>:<external_port>`. Traffic is **unencrypted
HTTP**; the token travels in clear text. Use this only for short sessions with disposable tokens, or put TLS in
front. See `SECURITY.md`.

### Persistence of the server

`--tmux` keeps the server alive after you disconnect (`tmux attach -t forge`, detach with `Ctrl-b d`).
`--nohup` logs to `$FORGE_HOME/server.log`. Stop with `bash scripts/run_server.sh --stop`.

## 5. Persistent storage and moving data

* Everything stateful lives in `FORGE_HOME`: `forge.db` (jobs, configs, metrics), `store/` (content-addressed
  artifacts and checkpoints), `models/`, `datasets/`, `runs/`, `autotune/` (kernel autotune cache),
  `hf_cache/` and `triton_cache/` (set by `vast_setup.sh`).
* **Stopping** an instance keeps its disk (you pay a small storage fee); **destroying** it deletes everything.
  Before destroying, copy out what you need:

```bash
# from your laptop
rsync -avP -e "ssh -p <ssh_port>" root@<ssh_host>:/workspace/forge-home/store/ ./forge-store-backup/
# or push finished models to the Hugging Face Hub / cloud storage from the instance
```

* Back up the **database and store together** (`docs/runbooks/restore-drill.md`); the store alone has no
  job history, the database alone has dangling references.
* Keep datasets and base models you can re-download out of backups; back up checkpoints, artifacts, `forge.db`
  and `autotune/` (autotune results are chip-specific and costly to regenerate).
* Use `Storage -> GC (dry run)` before deleting anything (`docs/runbooks/store-gc-drill.md`).
* vast.ai volumes can sometimes be re-attached to a new instance on the same host; do not rely on it - verify
  in the console before you plan around it.

## 6. Cost tips

* Do the cheap work on cheap GPUs: kernel correctness (`run_gpu_tests.sh`) needs minutes on any 24 GB card.
  Reserve H100/B200 time for the specific experiments that need `sm_90`/`sm_100`.
* Download models once to `FORGE_HOME/models` (or `hf_cache`) and stop - not destroy - the instance between
  sessions if you will return within days; compare storage fee vs re-download time.
* Set `FORGE_MAX_CONCURRENT_JOBS=1` (default) so jobs do not fight over one GPU.
* Use the planner (*Recipes -> fit/bound/ceiling panel*, `POST /planner/estimate`) before launching: it rejects
  recipes that cannot fit in VRAM. Use the `tiny-smoke` template to check the whole pipeline in minutes.
* Run long jobs with checkpoints enabled and `resume` on; interruptible instances then lose at most one
  checkpoint interval. A pre-empted instance does not restart the server for you: re-run
  `bash scripts/run_server.sh --tmux`; jobs that were `running` are marked `failed` on restart and can be
  resumed via *Jobs -> Clone*.
* Destroy instances you are not using. Check the vast.ai console balance, not Forge, for billing.

## 7. GPU smoke tests and what to send back

Run these in order on the instance, from the repo root:

```bash
bash scripts/gpu_smoke.sh          # ~2-10 min: torch/CUDA check, Triton sanity, forge doctor, forge probe --quick, quick kernel bench
bash scripts/run_gpu_tests.sh      # tests/ (CPU suite) + tests_gpu/ (correctness vs oracles) + quick bench
bash scripts/run_gpu_tests.sh -k w4a16 -x     # extra args go to pytest for the GPU suite
```

Both write to `./gpu_report/`:

| File | Content |
|------|---------|
| `report.md` | Human-readable summary: environment, per-step exit codes, per-file test counts, failing tests with first error line, skip reasons, failing-step output tails |
| `report.json` | Same, machine-readable |
| `env.json` | GPUs, driver, torch/triton/transformers versions |
| `steps.json`, `<step>.log` | Command, exit code, duration and full output of each step |
| `junit_gpu.xml`, `junit_cpu.xml` | Raw pytest results |
| `bench_summary.json` | Output of the quick kernel benchmark sweep, if it ran |

How to read it:

* `Overall: PASS` means *no recorded failures*. **Skipped tests are not passes** - check the skip reasons (a
  skipped FP8/FP4 test on an `sm_86` card is expected, a skipped W4A16 test is not).
* Benchmark numbers from the quick sweep are rough; the Plan KPIs (85% of bandwidth, 3.2x vs BF16) are *targets*,
  not results. Do not quote them before running the autotuner and a proper `bench` job.
* `forge probe --quick` creates a measured chip descriptor under `FORGE_HOME`; the *Hardware* page shows it.

**If something fails, send back:**

1. `gpu_report/report.md` (paste it) and the `.log` of any failing step,
2. `nvidia-smi` full output, `python -c "import torch,triton;print(torch.__version__,torch.version.cuda,triton.__version__)"`,
3. the vast.ai image name and the instance's CUDA/driver version,
4. for a failing kernel test: the test id, the shape/dtype in the assertion message, and
   `FORGE_KERNEL_DEBUG=1 FORGE_FORCE_REF=0 python -m pytest tests_gpu/<file>::<test> -x -s` output,
5. to rule out a kernel bug vs. an environment bug, also run with `FORGE_FORCE_REF=1` (forces the pure-torch
   reference path); if that passes and the Triton path fails, the kernel is the suspect.

Never paste `HF_TOKEN`, `FORGE_API_TOKEN` or `~/.forge_env` contents that contain secrets.

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `nvidia-smi` fails inside the container | Wrong instance type or driver mismatch; pick another host |
| `CUDA error: no kernel image is available for execution on the device` | torch wheel does not include your `sm_` (e.g. Blackwell needs a cu128 build) |
| Triton compile error mentioning `gcc`/`Python.h` | Install `build-essential python3-dev`, or use a `-devel` image |
| `ptxas` / unsupported `sm_` messages | Triton older than your GPU generation; `pip install -U triton` matching your torch, or force reference with `FORGE_FORCE_REF=1` |
| OOM at model load | Rent more VRAM; use W4A16/FP8 artifacts; lower `max_model_len` / `gpu_mem_util` |
| UI loads but API calls return 401 | Paste the token in *Settings* |
| Port not reachable | Check the container port mapping and instance's published port list; prefer the SSH tunnel |
| HF download 401/403 | Gated model: accept the licence on the Hub and set `HF_TOKEN` |
| `Disk quota exceeded` | `FORGE_HOME` is not on the big volume; `df -h`, move it, re-run `vast_setup.sh` |

## 9. Docker alternative

On a host with Docker and the NVIDIA Container Toolkit: `docker compose -f docker/docker-compose.yml up --build`
(see `docker/Dockerfile.cuda`; also unvalidated). On vast.ai you are already inside a container, so use the
scripts above instead.
