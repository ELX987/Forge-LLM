# forge.train

Training, fine-tuning and distillation (Plan 3.4 TR-1..TR-8, 7.4). Job entry points live in `run.py`
(`run_training`, `run_finetune`, `run_distill`); dataset ingestion is `data.ingest_dataset` (`dataset_ingest` job).

| file | what |
|---|---|
| `config.py` | `TrainConfig`, `FinetuneConfig`, `DistillConfig` (+ sections). Every field has a description / UI hints. |
| `data.py` | `IngestConfig`/`ingest_dataset` (jsonl/txt/parquet/HF/dir -> dedup -> tokenise -> uint16/32 shards + offsets + stats + `DatasetManifest`), `ShardedTokenDataset` (stateless deterministic order, exact resume), MinHash, contamination scan |
| `sft.py` | chat templates, assistant-only loss masks, packing, preference pairs, `SFTDataset` |
| `trainer.py` | the shared engine (`Trainer`, `Task`, `LMTask`): autocast/GradScaler, accumulation, clipping, DDP/FSDP, rollback on NaN/spike, eval/early-stop, checkpoints, MFU |
| `precision.py`, `fp4sim.py` | precision ladder R0/R1/R2, NVFP4 emulation (RTN/SR, 1x16 + 16x16 scales, RHT), exemptions, healing |
| `diagnostics.py` | TR-2: zero-gradient tile fraction, E4M3 scale flush/occupancy, SR bias, loss gap, low-norm-channel failure reproduction |
| `finetune.py`, `lora.py` | full / LoRA / QLoRA (NF4/int8 block-quantised base) / DPO-family; adapter save + merge |
| `distill.py` | licence gate (TR-8), top-k logit cache (Plan 8.5), offline / sequence-level / on-policy / reasoning-trace, diversity metrics |
| `dist.py`, `launch.py` | torchrun helpers (`launch_torchrun`, `python -m forge.train.launch`) |
| `checkpoint.py` | atomic local checkpoints (+ best / pre-anneal / final) mirrored into `forge.store` (delta-chained) |

Batches use **aligned labels** (`labels[t]` is the target of `logits[t]`, `-100` ignored). Metrics: `loss, lr, grad_norm,
tokens_per_s, tokens_seen, step_time_ms, gpu_mem_gb, mfu, update_weight_ratio` + task metrics (`ce, kd, teacher_agreement,
reward_acc, ...`) + `fp4/*` (R2) / `fp8/*` (R1) diagnostics + `restarts, skipped_steps`.

GPU status: everything here is unit-tested on CPU only. CUDA-specific behaviour (NCCL DDP/FSDP, torchao/TransformerEngine FP8,
fused AdamW, GradScaler, torch.compile, MFU against real GPUs) is covered by `tests_gpu/test_train_gpu.py` and has **not been
validated on hardware yet**.
