# Sharded ternary trainer (`parallel.strategy = zero_ternary`)

> Status: **phase P5, EXPERIMENTAL.** Verified on CPU with the gloo backend: sharded == unsharded within float tolerance, resume on a different
> number of ranks, shard-aware checkpoints, global clip, OOM policy. **The NCCL path (`reduce_scatter_tensor`, GPU buffers) has never been
> executed.** Gate G1 (`scripts/zero_ternary_parity.py`, `tests_gpu/test_zero_ternary.py`) decides the GPU cells; until then every GPU/perf cell is "no".

## What it does

DDP replicates the optimiser state on every GPU and FSDP shards parameters, neither of which fits ternary QAT, where the forward needs only 2-bit
codes. `zero_ternary` shards the *update*:

| Step | Where | Traffic |
|------|-------|---------|
| forward / backward | every rank, own micro-batches, full-size `weight.grad` | none |
| gradient exchange | `reduce_scatter` (NCCL) or `all_reduce` + slice (gloo): each rank keeps its rows | 1x dW |
| global-norm clip | one scalar all-reduce, one host sync | 4 bytes |
| optimiser update | each rank updates its row range and Adam/SGD state in place | none |
| code sync | re-quantise own rows, `all_gather` of packed 2-bit codes + fp32 group scales | ~1/8 of bf16 weights |

Scale groups run along K, so a row range quantises independently of the other rows: the sharded codes are bit-identical to the unsharded layer's.
Non-ternary parameters (embeddings, norms, head) are replicated and their gradients all-reduced.

## What is and is not sharded (v1)

* Sharded: optimiser state (`m`, `v`, momentum) and the update compute, by `1/world`; the weight-sync traffic is packed codes.
* **Not** sharded: the latent `weight` stays allocated at full size on every rank (it is the autograd leaf that receives `dW`; the rows a rank does
  not own are stale and unused because the forward decodes the gathered codes). Freeing it needs a grad-carrier-free autograd function: a follow-up.
  `describe()` / the run log print the optimiser-state saving; do not read it as a 1/world saving of total memory.
* During the quantisation warm-up (`precision.ternary.warmup_steps`, `lam < 1`) and around high-precision shadow/eval forwards the latent rows are
  all-gathered too, so those paths equal the unsharded trainer.

## Using it

```yaml
precision: {quant_aware: ternary, ternary: {group_size: -1}}   # -1 = one scale per row; 0 (per tensor) cannot be sharded
parallel: {strategy: zero_ternary, nproc_per_node: 4, zero_comm: auto}
optim: {name: adamw}          # adamw | lion | sgd (per-parameter state that can be row-sliced)
```

* `group_size: 0` is **rewritten to -1 with a warning** by the config validator, and the saved config shows the effective value.
* Needs `precision.quant_aware = ternary`; float16 + GradScaler and `adamw8bit` are rejected. A single process (`world_size = 1`) is allowed and equals
  the normal trainer.
* Jobs started with `nproc_per_node > 1` re-launch themselves under torchrun like ddp/fsdp. The checkpoint directory (`run_dir/checkpoints` by
  default) must be on storage shared by all ranks; a missing shard file is reported as an error at save time.

## Checkpoints

`step_NNNNNNNN/state.pt` holds everything replicated (non-ternary weights, replicated optimiser state, trainer/RNG state, the shard manifest). The
ternary latent rows and their optimiser state live in `shards/rank_XXXXX.safetensors` (no pickle) plus `shards/manifest.json` (row ranges per rank).
Loading reads exactly the rows a rank owns from whichever old shard files overlap them, so **a run resumes on a different number of GPUs** (tested
2 -> 3 -> 1). A checkpoint written by the plain trainer is adopted (latents load, optimiser state restarts, with a warning). Shards are not mirrored
into the content-addressed store (only `state.pt` is).

## OOM policy

`parallel.oom_halve_micro_batch` (on by default for `ternary_train`): on a CUDA out-of-memory error every micro-batch is split in two and the step is
retried; the gradient is unchanged because the loss is normalised by the step's total token count. **Single process only**: with `world_size > 1` it is
ignored with a warning (ranks cannot agree on a split in the middle of a collective, so an OOM on one rank fails the run).

## Collectives check

`python -m forge.train.collectives --backend nccl` (and the `collectives` probe check) runs broadcast, all_reduce, all_gather (uint8), and
`reduce_scatter_tensor` with known answers plus a timed all_reduce. Fewer than two GPUs is reported as **skipped**, never as passed. gloo records an
unsupported `reduce_scatter_tensor` as `None`, not as a failure; for NCCL it is required.

## Validation

* CPU/gloo (`tests/test_ternary_zero.py`): single-process equality with AdamW and SGD+momentum (clip, warm-up, eval, loss-gap shadow forwards, health
  metrics), 2-rank and 3-rank (uneven row split) equality with one process, 2 -> 3 -> 1 rank resume, plain-checkpoint adoption, OOM policy,
  the G1 harness itself on gloo, collectives check on gloo.
* Multi-rank AdamW is asserted as a trajectory (loss within 0.5%, mean weight difference < 2e-4), not bit equality: Adam normalises tiny gradients and so
  amplifies summation-order noise. SGD+momentum is asserted to 2e-5 (single resume hop) to 5e-5 (two hops).
* **G1 (not run):** `torchrun --nproc_per_node=2 scripts/zero_ternary_parity.py` and `pytest tests_gpu/test_zero_ternary.py` on a multi-GPU host.

## Known limits

* NCCL `reduce_scatter` path unexecuted; no throughput or memory numbers exist.
* Latent weights are not freed on non-owner rows (see above); bf16 stochastic-rounded optimiser state is not part of P5.
* No MoE-expert-parallel interaction, no LoRA/adapter-only training, no `torch.compile` validation, no `FSDP`/`ddp` mixing.
* `update_weight_ratio` is computed on rank 0's shard only.
