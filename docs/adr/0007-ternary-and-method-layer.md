# ADR 0007: Ternary (1.58-bit) support and the method catalog

- Status: proposed (approved defaults, 2026-11-01; see `plan-and-docs/Forge_Ternary_Merge_Plan.pdf`)
- Date: 2026-11-01
- Amends: ADR 0002 (adds an optional native CUDA tier)

## Context

The standalone project tern158 (ternary {-1, 0, +1} training, conversion and fine-tuning; ~2,570 lines of CUDA, ~4,330 lines
of Python) is being merged into Forge so Forge covers every training, quantization and fine-tuning method. tern158 has
single-GPU hardware results on sm_120 only (23/23 kernel selftests, 65/65 selfchecks, 346-411 INT8 TOPS, a 125M-parameter
run). Forge's GPU paths are likewise validated only on one sm_120 box.

## Decision

1. **Additive integration.** Ternary uses existing extension points: linear registry, job registry, recipe stages, kernel specs,
   gates, claims. tern158's monolithic model, orchestrator, UI server and run-directory protocol are not copied.
2. **Triton-first, native optional.** The ternary GEMM and quantiser are implemented as torch oracle + Triton (ADR 0002).
   tern158's CUDA kernels move to `forge/native/ternary` as an optional extension (`forge_ternary_C`, extra
   `forge-llm[ternary-cuda]`), selected through `Layout.executor='native'` only when its validation probe passes and it beats
   Triton on that device. Env mapping: `TERN158_FORCE_REF` -> `FORGE_FORCE_REF`, `allow_reference_fallback=false` ->
   `FORGE_KERNEL_STRICT`, `TERN158_ARCHS` -> `FORGE_TERNARY_ARCHS`.
3. **One format definition** in `forge/formats/ternary.py` (absmean/TWN/Lloyd quantisers, per-token int8 activation quant,
   STE), shared by PTQ, QAT, the oracle and tests.
4. **Two classes.** `fq_t158` (latent-weight Parameter, QAT) and `t158` (PackedLinear, buffers, inference). Reserved kind names;
   ternary is never modelled as `w2a16`.
5. **Storage.** Layout `ternary2_row`: 2 bits per weight, 16 codes per little-endian int32 along K, code j at bits [2j+1:2j],
   00=0, 01=+1, 11=-1, 10 reserved (decodes to 0). Byte-identical to `pack_bits(t & 3, 2)` (golden test required; K must be a
   multiple of 16). `.t158.safetensors` is import/export only; the Forge directory artifact is canonical.
6. **Method catalog** (`forge/methods`): metadata descriptors pointing at existing job types/stages/diagnostics. It does not
   refactor the Trainer. `PrecisionManager` gets a quant-mode plug-in interface (behaviour-preserving) before ternary is added.
7. **Sharded trainer** is a new `ParallelConfig.strategy = 'zero_ternary'` behind narrow Trainer hooks; latent shards are saved
   as safetensors, not pickle.
8. **Honesty.** Unvalidated paths sit behind explicit flags and EXPERIMENTAL labels; paper/report numbers are TARGET or
   untested claims; tern158 hardware numbers are recorded as "measured on RTX PRO 6000 Blackwell, sm_120" and not extrapolated.

## Consequences

* Ternary appears in New Job automatically; UI work is additive.
* Forge's chip table becomes canonical for the simulator and planner (`tern158/gpus.py` retires).
* `ChipInfo` gains `has_int8` / `int8_tc_fast` (false on sm_103).
* The Triton ternary kernel may fall short of the hand-written mma kernel; the native tier is the fallback.
* tern158 code is relicensed Apache-2.0 with a NOTICE entry (same owner).
