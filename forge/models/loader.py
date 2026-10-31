"""Checkpoint IO: HF safetensors <-> ForgeLM, streaming loader, sharded saver, tokenizer wrapper, ``backend='hf'`` fallback.

* Names: ForgeLM parameter names equal HF names without the ``model.`` prefix (``lm_head.weight`` unchanged); mixtral expert
  weights ``w1/w2/w3`` map to ``gate_proj/down_proj/up_proj`` (:func:`hf_to_forge_name` / :func:`forge_to_hf_name`).  Fused-expert
  tensors (``experts.gate_up_proj`` / ``experts.down_proj``, as in recent transformers) are split per expert on load.
  gemma4 checkpoints are multimodal (``model.language_model.*`` text tensors next to ``model.vision_tower.*`` / ``audio_tower.*``):
  the ``language_model.`` level is stripped, non-text towers are dropped, and the k/v tensors that ship for KV-shared layers (and
  the v_proj of ``attention_k_eq_v`` global layers) are ignored because the Forge model has no such modules.
  lfm2 / lfm2_moe use their own vocabulary of names (``operator_norm``, ``ffn_norm``, ``feed_forward.w1/w2/w3``, ``self_attn.out_proj``,
  ``q_layernorm`` / ``k_layernorm``, final ``embedding_norm``); they are renamed by ``_LFM2_TO_FORGE`` (and back on export).  The
  ``conv.{in_proj,conv,out_proj}`` tensors keep their names.
* Loading never holds more than one extra tensor in memory: the model is created on the ``meta`` device, materialised with
  ``to_empty`` on the target device and tensors are streamed shard by shard from ``safe_open``.
* ``config.json`` written by :func:`save_model` is HF-compatible and additionally carries a ``forge`` block with the exact
  :class:`ModelConfig` and the quantisation manifest (needed to rebuild packed linears before loading their buffers).
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError, NotFound
from forge.models.config import ModelConfig
from forge.models.linear import (
    apply_quantization_manifest,
    get_linear_cls,
    quantization_manifest,
    quantize_linears,
)
from forge.models.modeling import ForgeLM

log = logging.getLogger("forge.models.loader")

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype).lower().replace("torch.", "")]
    except KeyError:
        raise ConfigError(f"unknown dtype {dtype!r}; choose from {sorted(set(_DTYPES))}") from None


# ------------------------------------------------------------------------------------------------ name mapping
_MIXTRAL_EXPERT = re.compile(r"\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.")
_MIXTRAL_TO_FORGE = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_FORGE_TO_MIXTRAL = {v: k for k, v in _MIXTRAL_TO_FORGE.items()}
_FORGE_EXPERT = re.compile(r"\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.")
_FUSED = re.compile(r"^layers\.(\d+)\.(mlp|block_sparse_moe|feed_forward)\.experts\.(gate_up_proj|down_proj)$")
_KV_TENSOR = re.compile(r"^layers\.(\d+)\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\.")
_TOP_PREFIXES = ("layers.", "embed_tokens", "norm.", "per_layer_model_projection", "per_layer_projection_norm")
_LFM2_ARCHS = ("lfm2", "lfm2_moe")
# (HF pattern, Forge replacement); order matters (experts before the dense w1/w2/w3, `feed_forward.gate.` before nothing else)
_LFM2_TO_FORGE = [
    (re.compile(r"^embedding_norm\."), "norm."),
    (re.compile(r"\.operator_norm\."), ".input_layernorm."),
    (re.compile(r"\.ffn_norm\."), ".post_attention_layernorm."),
    (re.compile(r"\.self_attn\.out_proj\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.q_layernorm\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.k_layernorm\."), ".self_attn.k_norm."),
    (re.compile(r"\.feed_forward\.experts\.(\d+)\.w1\."), r".mlp.experts.\1.gate_proj."),
    (re.compile(r"\.feed_forward\.experts\.(\d+)\.w3\."), r".mlp.experts.\1.up_proj."),
    (re.compile(r"\.feed_forward\.experts\.(\d+)\.w2\."), r".mlp.experts.\1.down_proj."),
    (re.compile(r"\.feed_forward\.w1\."), ".mlp.gate_proj."),
    (re.compile(r"\.feed_forward\.w3\."), ".mlp.up_proj."),
    (re.compile(r"\.feed_forward\.w2\."), ".mlp.down_proj."),
    (re.compile(r"\.feed_forward\.gate\."), ".mlp.gate."),
    (re.compile(r"\.feed_forward\.expert_bias$"), ".mlp.expert_bias"),
]
_FORGE_TO_LFM2 = [
    (re.compile(r"^norm\."), "embedding_norm."),
    (re.compile(r"\.input_layernorm\."), ".operator_norm."),
    (re.compile(r"\.post_attention_layernorm\."), ".ffn_norm."),
    (re.compile(r"\.self_attn\.o_proj\."), ".self_attn.out_proj."),
    (re.compile(r"\.self_attn\.q_norm\."), ".self_attn.q_layernorm."),
    (re.compile(r"\.self_attn\.k_norm\."), ".self_attn.k_layernorm."),
    (re.compile(r"\.mlp\.experts\.(\d+)\.gate_proj\."), r".feed_forward.experts.\1.w1."),
    (re.compile(r"\.mlp\.experts\.(\d+)\.up_proj\."), r".feed_forward.experts.\1.w3."),
    (re.compile(r"\.mlp\.experts\.(\d+)\.down_proj\."), r".feed_forward.experts.\1.w2."),
    (re.compile(r"\.mlp\.gate_proj\."), ".feed_forward.w1."),
    (re.compile(r"\.mlp\.up_proj\."), ".feed_forward.w3."),
    (re.compile(r"\.mlp\.down_proj\."), ".feed_forward.w2."),
    (re.compile(r"\.mlp\.gate\."), ".feed_forward.gate."),
    (re.compile(r"\.mlp\.expert_bias$"), ".feed_forward.expert_bias"),
]


def hf_to_forge_name(name: str, cfg: ModelConfig | None = None) -> str | None:
    """Map an HF tensor name to the ForgeLM name; ``None`` = ignore (rotary buffers etc.)."""
    if "rotary_emb" in name:
        return None
    if name == "lm_head.weight":
        return name
    if name.startswith("model.language_model."):
        n = name[len("model.language_model.") :]
    else:
        n = name[len("model.") :] if name.startswith("model.") else name
    if cfg is not None and cfg.arch in _LFM2_ARCHS:  # `embedding_norm` is the FINAL norm; vision/projector towers of lfm2_vl are dropped
        for pat, rep in _LFM2_TO_FORGE:
            n = pat.sub(rep, n)
    if not n.startswith(_TOP_PREFIXES):
        return None
    if cfg is not None and cfg.arch == "gemma4":
        m = _KV_TENSOR.match(n)
        if m:
            li, kind = int(m.group(1)), m.group(2)
            if li >= cfg.num_layers or cfg.is_kv_shared_layer(li) or (kind == "v_proj" and not cfg.layer_has_v_proj(li)):
                return None
            if kind == "v_norm":  # scale-less in Forge (no parameter)
                return None
    if cfg is None or cfg.arch == "mixtral":
        n = n.replace(".block_sparse_moe.gate.", ".mlp.gate.")
        n = _MIXTRAL_EXPERT.sub(lambda m: f".mlp.experts.{m.group(1)}.{_MIXTRAL_TO_FORGE[m.group(2)]}.", n)
    return n


def forge_to_hf_name(name: str, cfg: ModelConfig | None = None) -> str:
    """Inverse of :func:`hf_to_forge_name`."""
    if name == "lm_head.weight" or name.startswith("lm_head."):
        return name
    n = name
    if cfg is not None and cfg.arch in _LFM2_ARCHS:
        for pat, rep in _FORGE_TO_LFM2:
            n = pat.sub(rep, n)
    if cfg is not None and cfg.arch == "mixtral":
        n = _FORGE_EXPERT.sub(lambda m: f".block_sparse_moe.experts.{m.group(1)}.{_FORGE_TO_MIXTRAL[m.group(2)]}.", n)
        n = n.replace(".mlp.gate.", ".block_sparse_moe.gate.")
    return "model." + n


def expand_hf_tensor(name: str, tensor: torch.Tensor, cfg: ModelConfig) -> list[tuple[str, torch.Tensor]]:
    """Map one checkpoint tensor to ``[(forge_name, tensor), ...]`` (splits fused expert tensors; empty list = skip)."""
    n = name[len("model.language_model.") :] if name.startswith("model.language_model.") else name
    n = n[len("model.") :] if n.startswith("model.") else n
    m = _FUSED.match(n)
    if m and cfg.moe is not None:
        layer, kind = int(m.group(1)), m.group(3)
        out = []
        for e in range(tensor.shape[0]):
            base = f"layers.{layer}.mlp.experts.{e}"
            if kind == "gate_up_proj":
                g, u = tensor[e].chunk(2, dim=0)
                out += [(f"{base}.gate_proj.weight", g), (f"{base}.up_proj.weight", u)]
            else:
                out.append((f"{base}.down_proj.weight", tensor[e]))
        return out
    fn = hf_to_forge_name(name, cfg)
    return [(fn, tensor)] if fn is not None else []


def forge_state_dict_to_hf(model_or_sd: nn.Module | dict[str, torch.Tensor], cfg: ModelConfig | None = None) -> dict[str, torch.Tensor]:
    """State dict with HF names (tied lm_head omitted)."""
    if isinstance(model_or_sd, nn.Module):
        cfg = cfg or getattr(model_or_sd, "config", None)
        sd = model_or_sd.state_dict()
    else:
        sd = model_or_sd
    out = {}
    for k, v in sd.items():
        if k == "lm_head.weight" and cfg is not None and cfg.tie_word_embeddings:
            continue
        out[forge_to_hf_name(k, cfg)] = v
    return out


# ------------------------------------------------------------------------------------------------ path / files
_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "tokenizer*"]


def resolve_model_path(path_or_repo: str | os.PathLike, revision: str | None = None, allow_patterns: list[str] | None = None) -> Path:
    """Local directory as-is; otherwise download (config, weights, tokenizer) from the HF hub into its cache."""
    p = Path(path_or_repo)
    if p.exists():
        return p
    from forge.core.modelref import get_hf_token, resolve_model_ref

    r = resolve_model_ref(path_or_repo)  # registered name/id, FORGE_HOME/models dir, or a validated org/name repo id
    if r.path is not None:
        return r.path
    path_or_repo = r.repo_id  # type: ignore[assignment]
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise NotFound(f"{path_or_repo!s} does not exist locally and huggingface_hub is not installed") from None
    return Path(snapshot_download(str(path_or_repo), revision=revision, allow_patterns=allow_patterns or _PATTERNS, token=get_hf_token()))


def checkpoint_files(path: Path) -> list[Path]:
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(set(wm.values()))
        for f in names:  # an index.json is untrusted input: shards must live next to it
            if f != Path(f).name or f in ("", ".", ".."):
                raise ConfigError(f"unsafe shard path {f!r} in {idx.name}: shard files must be plain file names inside the model directory")
        return [path / f for f in names]
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("adapter"))
    if files:
        return files
    bins = sorted(path.glob("pytorch_model*.bin"))
    if bins:
        return bins
    raise NotFound(f"no safetensors / pytorch_model*.bin weights found in {path}")


def iter_checkpoint(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(hf_name, cpu tensor)`` lazily, one tensor at a time for safetensors."""
    for f in checkpoint_files(path):
        if f.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    yield k, sf.get_tensor(k)
        else:
            sd = torch.load(str(f), map_location="cpu", weights_only=True)
            for k, v in sd.items():
                yield k, v
            del sd


def read_config(path: Path) -> tuple[ModelConfig, dict[str, Any]]:
    """``(ModelConfig, raw config.json dict)``; prefers the exact ``forge.model_config`` block when present."""
    cf = path / "config.json"
    if not cf.exists():
        raise NotFound(f"no config.json in {path}")
    raw = json.loads(cf.read_text(encoding="utf-8"))
    fb = raw.get("forge") or {}
    if fb.get("model_config"):
        return ModelConfig.model_validate(fb["model_config"]), raw
    return ModelConfig.from_hf_config(raw), raw


# ------------------------------------------------------------------------------------------------ materialise / load
def materialize(model: ForgeLM, device: str | torch.device = "cpu", dtype: str | torch.dtype | None = None, init: bool = True) -> ForgeLM:
    """Turn a meta-device model into real tensors on ``device`` (random init when ``init``); casts floating weights to ``dtype``."""
    dt = parse_dtype(dtype)
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        model.to_empty(device=device)
        if init:
            model.init_weights()
    else:
        model.to(device)
    model.tie_weights()
    if dt is not None:
        model.to(dt)
        model.tie_weights()
    return model


@dataclass
class LoadReport:
    loaded: int = 0
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    tied: bool = False


def _target_map(model: nn.Module) -> dict[str, torch.Tensor]:
    d: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        d[n] = p
    for n, b in model.named_buffers():
        d[n] = b
    return d


def load_hf_tensors(model: ForgeLM, tensors: Iterator[tuple[str, torch.Tensor]], *, strict: bool = True) -> LoadReport:
    """Copy ``(hf_name, tensor)`` pairs into a materialised model (casting to each target's dtype)."""
    cfg = model.config
    targets = _target_map(model)
    rep = LoadReport(tied=cfg.tie_word_embeddings)
    seen: set[str] = set()
    with torch.no_grad():
        for name, t in tensors:
            for fname, ft in expand_hf_tensor(name, t, cfg):
                if cfg.tie_word_embeddings and fname == "lm_head.weight":
                    continue
                tgt = targets.get(fname)
                if tgt is None:
                    rep.unexpected.append(name)
                    continue
                if tuple(tgt.shape) != tuple(ft.shape):
                    raise ConfigError(f"shape mismatch for {fname}: checkpoint {tuple(ft.shape)} vs model {tuple(tgt.shape)}")
                tgt.copy_(ft.to(tgt.device))
                seen.add(fname)
                rep.loaded += 1
    expected = {n for n in targets if not (cfg.tie_word_embeddings and n == "lm_head.weight")}
    rep.missing = sorted(expected - seen)
    if strict and rep.missing:
        raise ConfigError(f"checkpoint is missing {len(rep.missing)} tensors, e.g. {rep.missing[:3]}", details={"missing": rep.missing[:20]})
    if rep.unexpected:
        log.warning("ignored %d unexpected checkpoint tensors (e.g. %s)", len(rep.unexpected), rep.unexpected[:3])
    return rep


def _ensure_kind(kind: str) -> None:
    try:
        get_linear_cls(kind)
        return
    except ConfigError:
        pass
    for mod in ("forge.compress", "forge.compress.ptq", "forge.kernels", "forge.kernels.ops_gemm"):
        try:
            importlib.import_module(mod)
        except Exception:
            continue
    get_linear_cls(kind)  # raises ConfigError with the registered list


def load_model(
    path_or_repo: str | os.PathLike,
    *,
    dtype: str | torch.dtype = "bfloat16",
    device: str | torch.device | None = None,
    revision: str | None = None,
    quant: str | dict[str, Any] | None = None,
    backend: str = "forge",
    strict: bool = True,
    with_tokenizer: bool = True,
) -> tuple[Any, Any | None]:
    """Load a model (and tokenizer if present). Returns ``(model, tokenizer_or_None)``.

    ``device``: ``cuda`` | ``cpu`` | ``meta`` (default: cuda when available else cpu).  ``meta`` builds the structure only
    (no weights are read).  ``quant``: ``None``, a registered linear kind (``"fp8"``, ``"fq_int"`` ...) or
    ``{"kind": ..., **kwargs}`` applied after loading to all eligible linears.  ``backend='hf'`` returns a transformers
    ``AutoModelForCausalLM`` instead of :class:`ForgeLM`.
    """
    dt = parse_dtype(dtype) or torch.bfloat16
    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
    path = resolve_model_path(path_or_repo, revision)
    tok = load_tokenizer(path) if with_tokenizer else None
    if backend == "hf":
        return _load_hf_backend(path, dt, dev), tok
    if backend != "forge":
        raise ConfigError("backend must be 'forge' or 'hf'")
    cfg, raw = read_config(path)
    model = ForgeLM(cfg, dtype=dt, device="meta", init_weights=False)
    qm = (raw.get("forge") or {}).get("quantization")
    if qm:
        for ent in qm.values():
            _ensure_kind(ent["kind"])
        apply_quantization_manifest(model, qm, dt)
    if dev.type == "meta":
        return model, tok
    model.to_empty(device=dev)
    model.tie_weights()
    load_hf_tensors(model, iter_checkpoint(path), strict=strict)
    model.tie_weights()
    model.eval()
    if quant:
        spec = {"kind": quant} if isinstance(quant, str) else dict(quant)
        kind = spec.pop("kind")
        _ensure_kind(kind)
        quantize_linears(model, kind, **spec)
    return model, tok


def _load_hf_backend(path: Path, dt: torch.dtype, dev: torch.device):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ConfigError("backend='hf' requires the 'transformers' package") from None
    try:
        m = AutoModelForCausalLM.from_pretrained(str(path), dtype=dt)
    except TypeError:  # older transformers
        m = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dt)
    if dev.type != "meta":
        m = m.to(dev)
    m.eval()
    return m


# ------------------------------------------------------------------------------------------------ saving
def _castable(name: str, t: torch.Tensor) -> bool:
    return t.dtype in (torch.float32, torch.float16, torch.bfloat16) and (name.endswith(".weight") or name.endswith(".bias"))


def save_model(
    model: ForgeLM,
    directory: str | os.PathLike,
    *,
    as_dtype: str | torch.dtype | None = None,
    shard_gb: float = 4.0,
    tokenizer: Any | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Write an HF-compatible directory: sharded ``model-0000i-of-0000N.safetensors`` (+ index), ``config.json``, tokenizer.

    Floating ``.weight``/``.bias`` tensors are cast to ``as_dtype``; packed buffers (fp8/int/scales) are stored as they are.
    """
    from safetensors.torch import save_file

    from forge.models.lora import iter_lora

    if any(True for _ in iter_lora(model)):
        raise ConfigError("model has unmerged LoRA adapters: call merge_lora(model) or save_lora_adapter(model, dir)")
    cfg = model.config
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    dt = parse_dtype(as_dtype)
    sd = forge_state_dict_to_hf(model.state_dict(), cfg)

    def final(name: str, t: torch.Tensor) -> torch.Tensor:
        t = t.detach()
        if dt is not None and _castable(name, t):
            t = t.to(dt)
        return t.to("cpu").contiguous().clone()  # clone: shared/tied storage is rejected by safetensors

    limit = int(shard_gb * (1 << 30))
    groups: list[list[str]] = [[]]
    cur = 0
    for k, v in sd.items():
        sz = v.numel() * (dt.itemsize if (dt is not None and _castable(k, v)) else v.element_size())
        if groups[-1] and cur + sz > limit:
            groups.append([])
            cur = 0
        groups[-1].append(k)
        cur += sz
    # clean stale weight files
    for old in list(out.glob("model*.safetensors")) + list(out.glob("model.safetensors.index.json")):
        old.unlink()
    n = len(groups)
    weight_map: dict[str, str] = {}
    total = 0
    for i, keys in enumerate(groups):
        fname = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        shard = {k: final(k, sd[k]) for k in keys}
        save_file(shard, str(out / fname), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = fname
            total += t.numel() * t.element_size()
        del shard
    if n > 1:
        (out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2), encoding="utf-8"
        )
    hf = cfg.to_hf_config()
    store_dt = dt or model.dtype
    hf["torch_dtype"] = str(store_dt).replace("torch.", "")
    forge_block: dict[str, Any] = {"model_config": cfg.model_dump(mode="json")}
    qm = quantization_manifest(model)
    if qm:
        forge_block["quantization"] = qm
    if extra_config:
        forge_block["extra"] = extra_config
    hf["forge"] = forge_block
    (out / "config.json").write_text(json.dumps(hf, indent=2), encoding="utf-8")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(str(out))
    return out


# ------------------------------------------------------------------------------------------------ tokenizer
class SimpleTokenizer:
    """Thin wrapper over a ``tokenizers.Tokenizer`` (``tokenizer.json``) used when ``transformers`` is not installed."""

    def __init__(self, tk: Any, config: dict[str, Any] | None = None):
        self._tk = tk
        cfg = config or {}

        def tok_id(key: str) -> int | None:
            v = cfg.get(key)
            if isinstance(v, dict):
                v = v.get("content")
            return tk.token_to_id(v) if isinstance(v, str) else None

        self.eos_token_id = tok_id("eos_token")
        self.bos_token_id = tok_id("bos_token")
        self.pad_token_id = tok_id("pad_token")
        self.chat_template = cfg.get("chat_template")

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tk.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text: str | list[str], **_: Any) -> dict[str, Any]:
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        return {"input_ids": [self.encode(t) for t in text]}

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, *a: Any, **k: Any):
        raise ConfigError("chat templates need the 'transformers' package")


def load_tokenizer(path_or_repo: str | os.PathLike, revision: str | None = None) -> Any | None:
    """Load a tokenizer: ``transformers.AutoTokenizer`` when installed, else a minimal ``tokenizers`` wrapper, else ``None``."""
    try:
        p = resolve_model_path(path_or_repo, revision)
    except NotFound:
        return None
    if not (any((p / f).exists() for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json"))):
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(p))
    except ImportError:
        pass
    except Exception as e:  # corrupt / unsupported tokenizer
        log.warning("AutoTokenizer failed (%s); trying tokenizers", e)
    tj = p / "tokenizer.json"
    if tj.exists():
        try:
            from tokenizers import Tokenizer

            cfgp = p / "tokenizer_config.json"
            cfg = json.loads(cfgp.read_text(encoding="utf-8")) if cfgp.exists() else {}
            return SimpleTokenizer(Tokenizer.from_file(str(tj)), cfg)
        except ImportError:
            return None
    return None
