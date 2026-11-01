# forge/native/ternary: optional native CUDA tier (ADR 0007, tier 3)

CUDA sources imported from **tern158 v0.1** (same owner; relicensed Apache-2.0, see the repository `NOTICE`), module renamed `tern158_C` -> `forge_ternary_C`.
C symbols (`t158_*`) and the header directory (`kernel/include/tern158`) keep their names so diffs against the original stay readable.

| Item | State |
|---|---|
| Build | `setup.py` (torch CUDAExtension), `Makefile` (standalone libtern158.so + selftest), `CMakeLists.txt`; needs nvcc (12.8+ for sm_100/120, 12.9+ for sm_103/121) |
| Self-test | `kernel/tests/selftest.cu` (`make selftest`: validates every kernel against a host reference, prints TOPS per backend) |
| CPU-only check | `python tools/syntax_check.py` type-checks all `.cu` with g++ against stub headers (tested in `tests/test_native_ternary.py`); does NOT validate device semantics |
| Dispatch | NOT wired into `ops_gemm.ternary_linear` yet (Triton or the torch oracle run). Wiring requires a hardware run showing it passes its probe and beats Triton |
| Evidence | tern158's own run: sm_120, CUDA 13.4: 23/23 selftests, mma 346-411 TOPS, bf16/cuBLAS route 332-359 TOPS. Nothing reproduced through Forge |

Known quirks inherited from tern158 (to fix before relying on it; see the merge plan section 12): `gemm_fwd` routes M<=16 to the GEMV kernel regardless of the
requested backend, the bf16 backend allocates a workspace per call, the per-tensor quantiser uses a float scratch reinterpreted as double with double atomics.
