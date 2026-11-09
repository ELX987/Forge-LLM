# Ternary inference from a completed QAT run

Forge's `ternary_train` job exports a regular dense model directory containing the trained latent weights. Package that directory as a
T158 artifact with the `ternarize` + `pack` recipe, then deploy the registered artifact through the Inference page. The serving loader
rebuilds packed `t158` linear modules from the artifact manifest; it does not silently substitute the dense training export.

## 1. Find the QAT export

Use the successful job's `model_dir` output (or the model registered by that job on the Models page). It is usually under
`$FORGE_HOME/models/`. Use the exported model directory, **not** the checkpoint directory: the normal model loader accepts the exported
`config.json` + safetensors directory directly.

## 2. Create the packed inference artifact

From the Forge repository root on the Linux host, run this with your actual exported model path:

```bash
python -m forge.cli compress \
  --set model=/workspace/forge-home/models/YOUR_QAT_MODEL_DIRECTORY \
  --set recipe_path=configs/recipes/ternary-qat-inference.yaml \
  --set name=ternary-125m-inference \
  --set output_name=ternary-125m-t158 \
  --set dtype=bfloat16 \
  --set device=cuda \
  --set use_store=true \
  --set register_model=true \
  --gpus 0 --follow
```

For a CPU packaging smoke test, use `--set device=cpu --gpus cpu`; that checks serialization/loading only and says nothing about GPU kernels.
The job output should include a model directory, artifact digest and model id. Its artifact summary should identify `Ternary 1.58b` and
list `t158` layers. The packing report also records dense exclusions, so the final artifact is not exactly 1.58 bits for every model
parameter: embeddings, `lm_head`, and router/gate modules remain in their original dtype.

## 3. Start inference

Open **Inference**, choose the newly registered `ternary-125m-t158` model, keep backend `forge`, and start a deployment. Send a short
prompt and confirm the deployment status and generated tokens. Model identity/memory details should report a quantized `Ternary 1.58b`
artifact with packed bytes. If the picker does not show the model, use **Models → Import artifacts** and select the output directory.

## Validation status

CPU tests cover recipe packaging, the artifact manifest, packed-layer reload, and a short autoregressive generation. The Triton ternary GEMM
is implemented for supported NVIDIA GPUs, but has not yet been validated through live serving on your GPU. The dispatcher falls back to
the torch reference implementation if the kernel is unavailable or errors; that fallback is for correctness and may be slow. Do not treat
the current kernel as a demonstrated speedup until a GPU report records correctness and throughput.

After packaging the real model on the GPU host, run the small packed-load/generation check and then try the UI deployment:

```bash
python -m pytest tests_gpu/test_ternary_inference.py -q -s
```

The smoke test uses a tiny model and verifies artifact reload plus generation on CUDA; it does not establish production throughput or quality
for the 125M checkpoint.
