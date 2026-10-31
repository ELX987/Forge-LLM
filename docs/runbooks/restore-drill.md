# Runbook: backup and restore drill

Goal: show that `FORGE_HOME` can be rebuilt on a fresh machine and that artifacts are bit-identical.

## What to back up

| Path under `FORGE_HOME` | Needed? | Notes |
|-------------------------|---------|-------|
| `forge.db` (+ `-wal`, `-shm`) | yes | Jobs, configs, metrics, claims, deployments, lineage. Copy with the server stopped, or use the sqlite `.backup` command |
| `store/` | yes | Content-addressed chunks and manifests. Must be copied **consistently with** `forge.db` |
| `autotune/` | recommended | Chip-specific tuning results; slow to regenerate |
| `runs/` | optional | Logs and run directories |
| `models/`, `datasets/` | if not re-downloadable | Large |
| `hf_cache/`, `triton_cache/` | no | Re-created |

## Backup

```bash
bash scripts/run_server.sh --stop
sqlite3 "$FORGE_HOME/forge.db" ".backup $FORGE_HOME/forge.backup.db"
tar -C "$FORGE_HOME" -cf - forge.backup.db store autotune | zstd -T0 > "forge-backup-$(date +%F).tar.zst"
sha256sum forge-backup-*.tar.zst > forge-backup.sha256
```

(`sqlite3` and `zstd` binaries may need installing: `apt-get install -y sqlite3 zstd`. Plain `cp` of `forge.db`
with the server stopped is equally valid.)

Before stopping, record the hash of one model artifact and one checkpoint (UI *Storage*, or
`GET /api/v1/artifacts?kind=...`).

## Restore (fresh machine or empty FORGE_HOME)

```bash
export FORGE_HOME=/workspace/forge-home-restore
mkdir -p "$FORGE_HOME"
zstd -dc forge-backup-*.tar.zst | tar -C "$FORGE_HOME" -xf -
mv "$FORGE_HOME/forge.backup.db" "$FORGE_HOME/forge.db"
bash scripts/run_server.sh --tmux
```

## Verify

1. UI *Storage* shows the same logical and unique bytes as before.
2. The recorded artifact hashes exist (`GET /api/v1/artifacts/{hash}`) and verify.
3. Materialise the model artifact to a directory and compare hashes of the safetensors files with the originals.
4. Start a short job (`configs/train/tiny-smoke.yaml`) to confirm the restored database accepts writes and that
   migrations are idempotent.

## Pass criteria

All recorded hashes verify; job and config history are present; a new job runs. Note elapsed time and size.
