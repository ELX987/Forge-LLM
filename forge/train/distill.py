"""Distillation (Plan TR-3, TR-8, 7.4, 8.5): offline-logit (top-k cache), sequence-level, on-policy, reasoning-trace SFT.

* **Licence gate (TR-8)** -- :func:`check_teacher_licence` raises :class:`forge.core.errors.LicenceBlocked` before any teacher compute.
* **Top-k teacher cache (Plan 8.5)** -- :func:`build_topk_cache` / :class:`TopKLogitCache`: top-k indices + log-probs (fp16 or 8-bit),
  one residual-mass scalar per position, coverage statistics, cache key
  ``(teacher hash, tokenizer hash, data fingerprint, rows, seq len, k, quantisation, sampling)`` so one cache serves many students.
  Not implemented: varint-delta index packing (indices are stored as plain uint16/int32; use the store's lossless codec for shards).
* **Objectives** -- forward KL / reverse KL / generalised JSD / temperature (:mod:`forge.train.losses`), mixed with CE by ``alpha``.
* **On-policy** -- the student samples, the live teacher scores; naive KV-cache-free sampling (correct, slow -- use the serving
  engine for large-scale generation once wired).
* **Diversity monitoring** -- n-gram entropy, distinct-n, self-BLEU (pure functions, unit-tested) to watch for model collapse.
* **Lineage** -- student metadata records parent hash, pruning-plan hash, data fingerprint, teacher identity/hash/licence, tokens consumed.
"""
from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from forge.core.errors import ConfigError, LicenceBlocked
from forge.core.hashing import hash_bytes, hash_json
from forge.train import dist as D
from forge.train.common import build_model, check_vocab, coerce_cfg, load_tokenizer_for
from forge.train.config import (
    DistillConfig,
    DistillLossConfig,
    GenerationConfig,
    LogitCacheConfig,
    TeacherConfig,
)
from forge.train.data import _PermCache, step_positions
from forge.train.losses import ce_loss_sum, kd_loss_dense, kd_loss_topk, kd_token_divergence
from forge.train.sft import build_sft_datasets, load_prompts, load_records
from forge.train.tokenization import TokenizerAdapter
from forge.train.trainer import LMTask, StepLoss, Task, Trainer, forward_logits

log = logging.getLogger("forge.train.distill")

__all__ = [
    "OfflineKDTask", "OnPolicyTask", "TopKLogitCache", "build_topk_cache", "cache_key", "check_teacher_licence",
    "distinct_n", "diversity_report", "generate_tokens", "ngram_entropy", "run_distill", "self_bleu", "sentence_bleu",
]


# ---------------------------------------------------------------------------------------------------------------
# licence gate (TR-8)
# ---------------------------------------------------------------------------------------------------------------
_ALIASES = {
    "apache-2": "apache-2.0", "apache2": "apache-2.0", "apache-license-2.0": "apache-2.0", "apache-2.0-license": "apache-2.0",
    "mit-license": "mit", "bsd": "bsd-3-clause", "cc-by-4": "cc-by-4.0", "cc0": "cc0-1.0", "cc-by": "cc-by-4.0",
}


def normalise_licence(s: str | None) -> str:
    if not s:
        return ""
    x = re.sub(r"[\s_]+", "-", s.strip().lower())
    x = re.sub(r"-?licen[sc]e$", "", x)
    return _ALIASES.get(x, x)


def check_teacher_licence(t: TeacherConfig) -> dict[str, Any]:
    """Blocking policy gate: raises :class:`LicenceBlocked` unless distillation from this teacher is permitted.

    Allowed when ``distill_allowed`` is explicitly true (user attestation, recorded), the teacher is ``locally_owned``, or its
    licence is in ``permissive_licences``.  Explicit ``distill_allowed=false`` always blocks; unknown licences block.
    """
    lic = normalise_licence(t.licence)
    permissive = {normalise_licence(x) for x in t.permissive_licences}
    info: dict[str, Any] = {
        "teacher": t.name or t.path or "(unnamed)", "teacher_hash": t.hash, "licence": t.licence, "normalised_licence": lic,
        "locally_owned": t.locally_owned, "distill_allowed": t.distill_allowed,
    }
    if t.distill_allowed is False:
        raise LicenceBlocked(f"distillation from teacher {info['teacher']!r} is blocked: distill_allowed=false", details=info)
    if t.distill_allowed is True:
        info["basis"] = "attested distill_allowed=true"
    elif t.locally_owned:
        info["basis"] = "locally owned teacher"
    elif lic and lic in permissive:
        info["basis"] = f"permissive licence {lic}"
    else:
        raise LicenceBlocked(
            f"teacher {info['teacher']!r} (licence={t.licence!r}) is not locally owned or permissively licensed; review its terms and set "
            "teacher.distill_allowed=true (an attestation recorded in lineage) or teacher.locally_owned=true to proceed", details=info)
    info["checked_at"] = time.time()
    return info


# ---------------------------------------------------------------------------------------------------------------
# diversity metrics (pure functions over token-id / string sequences)
# ---------------------------------------------------------------------------------------------------------------
def _ngrams(seq: list[Any], n: int) -> list[tuple[Any, ...]]:
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(seqs: list[list[Any]], n: int = 2) -> float:
    """Unique n-grams / total n-grams over all sequences (1.0 = no repetition at all)."""
    total, uniq = 0, set()
    for s in seqs:
        g = _ngrams(s, n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / total if total else 0.0


def ngram_entropy(seqs: list[list[Any]], n: int = 2, normalise: bool = False) -> float:
    """Shannon entropy (bits) of the pooled n-gram distribution; ``normalise`` divides by log2(#unique n-grams)."""
    c: Counter = Counter()
    for s in seqs:
        c.update(_ngrams(s, n))
    tot = sum(c.values())
    if not tot:
        return 0.0
    h = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    if normalise:
        return h / math.log2(len(c)) if len(c) > 1 else 0.0
    return h


def sentence_bleu(hyp: list[Any], ref: list[Any], max_n: int = 4) -> float:
    """BLEU of ``hyp`` against a single reference, add-one smoothing for n>1."""
    if not hyp or not ref:
        return 0.0
    logp = 0.0
    for n in range(1, max_n + 1):
        hc, rc = Counter(_ngrams(hyp, n)), Counter(_ngrams(ref, n))
        tot = sum(hc.values())
        if tot == 0:
            p = 1.0 if n > 1 else 1e-9
        else:
            match = sum(min(c, rc[g]) for g, c in hc.items())
            p = (match / tot) if n == 1 else (match + 1) / (tot + 1)
            p = max(p, 1e-9)
        logp += math.log(p) / max_n
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return bp * math.exp(logp)


def self_bleu(seqs: list[list[Any]], n_pairs: int = 200, max_n: int = 4, seed: int = 0) -> float:
    """Pairwise self-BLEU estimate: mean BLEU of a random sequence against another random sequence (high = low diversity)."""
    if len(seqs) < 2:
        return 0.0
    rng = random.Random(seed)
    vals = []
    for _ in range(n_pairs):
        i, j = rng.sample(range(len(seqs)), 2)
        vals.append(sentence_bleu(seqs[i], seqs[j], max_n))
    return sum(vals) / len(vals)


def diversity_report(seqs: list[list[Any]], n: int = 2, self_bleu_pairs: int = 200, seed: int = 0) -> dict[str, float]:
    seqs = [s for s in seqs if len(s) > 0]
    return {
        f"distinct_{n}": distinct_n(seqs, n), "distinct_1": distinct_n(seqs, 1), f"ngram_entropy_{n}": ngram_entropy(seqs, n),
        f"ngram_entropy_{n}_norm": ngram_entropy(seqs, n, normalise=True), "self_bleu": self_bleu(seqs, self_bleu_pairs, seed=seed),
        "mean_len": (sum(len(s) for s in seqs) / len(seqs)) if seqs else 0.0, "n_samples": float(len(seqs)),
    }


# ---------------------------------------------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------------------------------------------
def _sample_next(logits: torch.Tensor, gen: GenerationConfig, g: torch.Generator) -> torch.Tensor:
    if gen.temperature <= 0:
        return logits.argmax(-1)
    lg = logits.float() / gen.temperature
    if gen.top_k and gen.top_k < lg.shape[-1]:
        kth = torch.topk(lg, gen.top_k, dim=-1).values[..., -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if gen.top_p < 1.0:
        sl, si = torch.sort(lg, dim=-1, descending=True)
        cum = torch.softmax(sl, dim=-1).cumsum(-1)
        drop = cum - torch.softmax(sl, dim=-1) > gen.top_p
        sl = sl.masked_fill(drop, float("-inf"))
        lg = torch.full_like(lg, float("-inf")).scatter(-1, si, sl)
    probs = torch.softmax(lg, dim=-1)
    return torch.multinomial(probs, 1, generator=g).squeeze(-1)


@torch.no_grad()
def generate_tokens(model: nn.Module, prompt: list[int], gen: GenerationConfig, eos_id: int | None, device: torch.device, seed: int = 0,
                    n: int | None = None, autocast: Any = None, max_context: int | None = None) -> list[list[int]]:
    """Sample ``n`` continuations of one prompt. Recomputes the full prefix at every step (no KV cache): correct, O(T^2)."""
    n = n or gen.num_samples
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    ids = torch.tensor([prompt] * n, device=device, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    out = torch.zeros(n, 0, dtype=torch.long, device=device)
    import contextlib

    with autocast if autocast is not None else contextlib.nullcontext():
        for _ in range(gen.max_new_tokens):
            if max_context and ids.shape[1] >= max_context:
                break
            logits, _ = forward_logits(model, ids)
            nxt = _sample_next(logits[:, -1], gen, g)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            out = torch.cat([out, nxt[:, None]], 1)
            if eos_id is not None:
                done = done | (nxt == eos_id)
                if bool(done.all()):
                    break
    res = []
    for row in out.tolist():
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id) + 1]
        res.append(row)
    return res


# ---------------------------------------------------------------------------------------------------------------
# top-k teacher cache
# ---------------------------------------------------------------------------------------------------------------
def row_crc(input_ids_row: torch.Tensor, length: int | None = None) -> int:
    a = input_ids_row if length is None else input_ids_row[:length]
    return zlib.crc32(a.detach().cpu().numpy().astype(np.int64).tobytes())


def cache_key(teacher_hash: str, tokenizer_hash: str, data_fingerprint: str, n_rows: int, seq_len: int, k: int, lp_quant: str,
              lp_floor: float, sampling: Any = None) -> str:
    """Plan 8.5 cache key: (teacher artifact hash, tokenizer hash, prompt/data hash, sampling params, k, quantisation scheme)."""
    return hash_json({"teacher": teacher_hash, "tokenizer": tokenizer_hash, "data": data_fingerprint, "rows": int(n_rows), "T": int(seq_len),
                      "k": int(k), "lp_quant": lp_quant, "lp_floor": lp_floor if lp_quant == "uint8" else None, "sampling": sampling,
                      "temperature": 1.0})


class TopKLogitCache:
    """Reader for a cache written by :func:`build_topk_cache` (memory-mapped shards)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        mp = self.path / "meta.json"
        if not mp.exists():
            raise ConfigError(f"no logit cache at {path}")
        self.meta: dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
        self.k, self.n, self.T, self.S = int(self.meta["k"]), int(self.meta["n"]), int(self.meta["T"]), int(self.meta["shard_sequences"])
        self._mm: dict[tuple[str, int], np.ndarray] = {}

    @property
    def complete(self) -> bool:
        return bool(self.meta.get("complete"))

    @property
    def stats(self) -> dict[str, Any]:
        return self.meta.get("stats", {})

    def close(self) -> None:
        """Release the memory maps (needed on Windows before the files can be modified/deleted)."""
        self._mm.clear()

    def _arr(self, kind: str, s: int) -> np.ndarray:
        key = (kind, s)
        a = self._mm.get(key)
        if a is None:
            a = np.load(self.path / f"{kind}_{s:05d}.npy", mmap_mode="r")
            self._mm[key] = a
        return a

    def get(self, positions: np.ndarray, T: int | None = None) -> dict[str, torch.Tensor]:
        """Rows for dataset ``positions``: ``idx [B,T,k] long, logp [B,T,k] float32, residual [B,T], crc [B]``."""
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size and (pos.min() < 0 or pos.max() >= self.n):
            raise ConfigError("cache does not cover the requested positions (restrict the dataset to the cache size)")
        Tt = min(T or self.T, self.T)
        B = len(pos)
        idx = np.zeros((B, Tt, self.k), dtype=np.int64)
        lp = np.zeros((B, Tt, self.k), dtype=np.float32)
        res = np.zeros((B, Tt), dtype=np.float32)
        crc = np.zeros(B, dtype=np.int64)
        floor = float(self.meta.get("lp_floor", -20.0))
        for b, p in enumerate(pos):
            s, r = int(p // self.S), int(p % self.S)
            idx[b] = self._arr("idx", s)[r, :Tt].astype(np.int64)
            raw = self._arr("lp", s)[r, :Tt]
            lp[b] = raw.astype(np.float32) if self.meta["lp_quant"] == "fp16" else raw.astype(np.float32) / 255.0 * (-floor) + floor
            res[b] = self._arr("res", s)[r, :Tt]
            crc[b] = int(self._arr("crc", s)[r])
        return {"idx": torch.from_numpy(idx), "logp": torch.from_numpy(lp), "residual": torch.from_numpy(res), "crc": torch.from_numpy(crc)}


def _coverage_stats(cov: np.ndarray, min_warn: float) -> dict[str, Any]:
    if cov.size == 0:
        return {"positions": 0}
    return {
        "positions": int(cov.size), "mean_coverage": float(cov.mean()), "p1": float(np.percentile(cov, 1)), "p5": float(np.percentile(cov, 5)),
        "p50": float(np.percentile(cov, 50)), "frac_below_0.9": float((cov < 0.9).mean()), "frac_below_0.95": float((cov < 0.95).mean()),
        "frac_below_0.99": float((cov < 0.99).mean()), "k_too_small": bool(cov.mean() < min_warn),
    }


@torch.no_grad()
def build_topk_cache(path: str | Path, teacher: nn.Module, source: Any, n: int, *, k: int, lp_quant: str = "fp16", lp_floor: float = -20.0,
                     shard_sequences: int = 256, batch_size: int = 4, device: torch.device | str = "cpu", T: int | None = None,
                     meta_extra: dict[str, Any] | None = None, ctx: Any = None, autocast: Any = None, min_coverage_warn: float = 0.9,
                     should_stop: Any = None, rebuild: bool = False) -> TopKLogitCache:
    """Run the teacher over positions ``0..n-1`` of ``source.get_batch`` and store top-k log-probs + residual mass.

    Resumable per shard: finished shards are skipped only when the build parameters (k, n, T, shard size, quantisation, ``meta_extra``
    such as the cache key / teacher hash) equal those recorded in ``build.json``; any difference, or ``rebuild=True``, discards the
    stale shards first so shards from different configurations can never be mixed.
    """
    import contextlib

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    teacher.eval()
    S = int(shard_sequences)
    probe = source.get_batch(np.arange(0, min(n, batch_size)))
    V = forward_logits(teacher, probe["input_ids"].to(device))[0].shape[-1]
    idx_dtype = np.uint16 if V <= 65536 else np.int32
    Tfix = int(T or probe["input_ids"].shape[1])
    n_shards = -(-n // S)
    sig = {"k": int(k), "n": int(n), "T": Tfix, "S": S, "V": int(V), "lp_quant": lp_quant, "lp_floor": float(lp_floor),
           "extra": json.loads(json.dumps(meta_extra or {}, default=str))}
    sig_path = path / "build.json"
    old_sig = None
    if sig_path.exists():
        try:
            old_sig = json.loads(sig_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old_sig = None
    if rebuild or old_sig != sig:
        if rebuild or any(path.glob("done_*")) or (path / "meta.json").exists():
            log.info("logit cache %s: %s -- discarding existing shards", path, "rebuild requested" if rebuild else "build parameters changed")
        for f in list(path.glob("done_*")) + list(path.glob("*_[0-9][0-9][0-9][0-9][0-9].npy")) + [path / "meta.json"]:
            f.unlink(missing_ok=True)
        sig_path.write_text(json.dumps(sig, indent=1), encoding="utf-8")
    t0 = time.time()
    for s in range(n_shards):
        done_marker = path / f"done_{s:05d}"
        if done_marker.exists():
            continue
        lo, hi = s * S, min(n, (s + 1) * S)
        m = hi - lo
        idx_a = np.zeros((m, Tfix, k), dtype=idx_dtype)
        lp_a = np.zeros((m, Tfix, k), dtype=np.float16 if lp_quant == "fp16" else np.uint8)
        res_a = np.zeros((m, Tfix), dtype=np.float32)
        crc_a = np.zeros(m, dtype=np.int64)
        cov_list: list[np.ndarray] = []
        for b0 in range(lo, hi, batch_size):
            pos = np.arange(b0, min(hi, b0 + batch_size))
            batch = source.get_batch(pos)
            ids = batch["input_ids"].to(device)
            with autocast if autocast is not None else contextlib.nullcontext():
                logits, _ = forward_logits(teacher, ids)
            lp = torch.log_softmax(logits.float(), dim=-1)
            v, i = lp.topk(k, dim=-1)
            mass = v.exp().sum(-1)
            resid = (1.0 - mass).clamp(min=0.0)
            Tb = min(ids.shape[1], Tfix)
            r0 = b0 - lo
            rr = slice(r0, r0 + len(pos))
            idx_a[rr, :Tb] = i[:, :Tb].cpu().numpy().astype(idx_dtype)
            vv = v[:, :Tb].cpu()
            if lp_quant == "fp16":
                lp_a[rr, :Tb] = vv.numpy().astype(np.float16)
            else:
                lp_a[rr, :Tb] = np.round((vv.clamp(min=lp_floor, max=0.0) - lp_floor) / (-lp_floor) * 255.0).numpy().astype(np.uint8)
            res_a[rr, :Tb] = resid[:, :Tb].cpu().numpy()
            lengths = batch.get("lengths")
            for j in range(len(pos)):
                crc_a[r0 + j] = row_crc(batch["input_ids"][j], int(lengths[j]) if lengths is not None else None)
            lab = batch["labels"][:, :Tb]
            msk = (lab != -100).cpu().numpy()
            cov_list.append(mass[:, :Tb].cpu().numpy()[msk])
        for name, arr in (("idx", idx_a), ("lp", lp_a), ("res", res_a), ("crc", crc_a)):
            np.save(path / f"{name}_{s:05d}.npy", arr)
        np.save(path / f"cov_{s:05d}.npy", np.concatenate(cov_list) if cov_list else np.zeros(0, dtype=np.float32))
        done_marker.write_text("1")
        if ctx is not None:
            ctx.progress((s + 1) / n_shards, f"teacher cache shard {s + 1}/{n_shards}")
            ctx.log(f"logit cache: shard {s + 1}/{n_shards} done ({time.time() - t0:.1f}s)")
            if should_stop is not None and should_stop() and s + 1 < n_shards:
                raise ConfigError("cancelled while building the logit cache (partial cache kept; re-run to resume)")
    cov = np.concatenate([np.load(path / f"cov_{s:05d}.npy") for s in range(n_shards)]) if n_shards else np.zeros(0)
    stats = _coverage_stats(cov, min_coverage_warn)
    meta = {"k": k, "n": n, "T": Tfix, "shard_sequences": S, "vocab_size": int(V), "idx_dtype": np.dtype(idx_dtype).name, "lp_quant": lp_quant,
            "lp_floor": lp_floor, "complete": True, "stats": stats, "created": time.time(), **(meta_extra or {})}
    (path / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if stats.get("k_too_small"):
        msg = f"top-{k} cache covers only {stats['mean_coverage']:.3f} of the teacher mass on average (p5={stats['p5']:.3f}): increase k"
        (ctx.log(msg, "warning") if ctx is not None else log.warning(msg))
    return TopKLogitCache(path)


# ---------------------------------------------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------------------------------------------
class OfflineKDTask(LMTask):
    """Student loss = alpha * KD(teacher || student) + (1 - alpha) * CE, teacher from a top-k cache or a live forward."""

    name = "kd"

    def __init__(self, ds: Any, cfg: DistillConfig, val_ds: Any | None, seq_len: int, cache: TopKLogitCache | None, teacher: nn.Module | None,
                 num_workers: int = 0, prefetch_steps: int = 2) -> None:
        super().__init__(ds, cfg, val_ds, seq_len, num_workers, prefetch_steps)
        self.dcfg = cfg
        self.cache, self.teacher = cache, teacher
        if cache is None and teacher is None and cfg.loss.kind != "ce_only":
            raise ConfigError("offline distillation needs a logit cache or a live teacher")

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:
        mb, acc = self.cfg.micro_batch_size, self.cfg.grad_accum_steps
        per = mb * acc
        pos = self.ds.positions_at(step, per * world)[rank * per:(rank + 1) * per]
        return [self.ds.get_batch(pos[i:i + mb]) for i in range(0, per, mb)]

    def tokens_per_step(self, global_batch: int) -> float | None:
        if self.seq_len_hint:
            return float(global_batch * self.seq_len_hint)
        return float(self.ds.tokens_per_epoch) / len(self.ds) * global_batch

    def eval_batches(self) -> list[dict[str, Any]]:
        b = super().eval_batches()
        for x in b:
            x["eval"] = True
        return b

    def loss(self, model: nn.Module, batch: dict[str, Any], trainer: Trainer) -> StepLoss:
        L: DistillLossConfig = self.dcfg.loss
        c = self.dcfg
        ids, labels = batch["input_ids"], batch["labels"]
        logits, _ = forward_logits(model, ids)
        mask = labels != -100
        ce_total, n, parts = ce_loss_sum(logits, labels, label_smoothing=c.label_smoothing, z_loss=c.z_loss)
        m: dict[str, Any] = {"ce": parts["ce_sum"]}
        if batch.get("eval") or L.kind == "ce_only":
            return StepLoss(ce_total, m)
        agree = None
        if self.cache is not None and batch.get("positions") is not None:
            T = ids.shape[1]
            cd = self.cache.get(batch["positions"].cpu().numpy(), T)
            self._verify(cd["crc"], batch)
            dev = logits.device
            idx, lp, res = cd["idx"].to(dev), cd["logp"].to(dev), cd["residual"].to(dev)
            if idx.shape[1] < T:
                pad = T - idx.shape[1]
                idx = torch.nn.functional.pad(idx, (0, 0, 0, pad))
                lp = torch.nn.functional.pad(lp, (0, 0, 0, pad))
                res = torch.nn.functional.pad(res, (0, pad))
            kd_sum, _ = kd_loss_topk(logits, idx, lp, res, mask, L.kind, L.temperature, L.jsd_beta, L.tail_mode)
            if L.token_agreement_metric:
                agree = ((logits.argmax(-1) == idx[..., 0]) & mask).sum()
        else:
            assert self.teacher is not None
            with torch.no_grad():
                tl, _ = forward_logits(self.teacher, ids)
            if tl.shape[-1] != logits.shape[-1]:
                raise ConfigError("teacher and student vocabularies differ; logit distillation needs identical vocabularies")
            kd_sum, _ = kd_loss_dense(logits, tl, mask, L.kind, L.temperature, L.jsd_beta)
            if L.token_agreement_metric:
                agree = ((logits.argmax(-1) == tl.argmax(-1)) & mask).sum()
        total = L.alpha * kd_sum + (1.0 - L.alpha) * ce_total
        m["kd"] = kd_sum.detach()
        if agree is not None:
            m["teacher_agreement"] = agree.float()
        return StepLoss(total, m)

    @staticmethod
    def _verify(crc: torch.Tensor, batch: dict[str, Any]) -> None:
        ids = batch["input_ids"]
        lengths = batch.get("lengths")
        for j in range(ids.shape[0]):
            got = row_crc(ids[j], int(lengths[j]) if lengths is not None else None)
            if got != int(crc[j]):
                raise ConfigError("logit cache does not match the data (token mismatch): the dataset, seed, seq_len or tokenizer changed since the cache was built")


class OnPolicyTask(Task):
    """On-policy distillation: sample from the student, score with the live teacher, minimise per-token divergence on the samples."""

    name = "on_policy"

    def __init__(self, prompts: list[list[int]], cfg: DistillConfig, teacher: nn.Module, tok: TokenizerAdapter) -> None:
        self.prompts, self.cfg, self.teacher, self.tok = prompts, cfg, teacher, tok
        self.op = cfg.on_policy
        self.perm = _PermCache(len(prompts), cfg.seed + self.op.generation.seed, 7, True)
        self.num_workers, self.prefetch_steps = 0, 2
        self.seq_len_hint = max(len(p) for p in prompts) + self.op.generation.max_new_tokens
        self._samples: list[list[int]] = []
        self._extra: dict[str, float] = {}
        self._last_text: str | None = None

    def get_batches(self, step: int, rank: int, world: int) -> list[dict[str, Any]]:
        mb, acc = self.op.prompts_per_step, self.cfg.grad_accum_steps
        per = mb * acc
        pos = step_positions(self.perm, step, per * world)[rank * per:(rank + 1) * per]
        return [{"prompt_idx": torch.as_tensor(pos[i:i + mb]), "step": int(step), "mb": i // mb} for i in range(0, per, mb)]

    def count(self, batch: dict[str, Any]) -> float:
        return float(len(batch["prompt_idx"]) * self.op.generation.num_samples)

    def tokens_in_batch(self, batch: dict[str, Any]) -> int:
        return int(self.count(batch) * self.seq_len_hint)

    def steps_per_epoch(self, global_batch: int) -> float | None:
        return None

    def tokens_per_step(self, global_batch: int) -> float | None:
        return None

    def loss(self, model: nn.Module, batch: dict[str, Any], trainer: Trainer) -> StepLoss:
        L = self.cfg.loss
        gen = self.op.generation
        dev = trainer.device
        raw = trainer.raw
        seqs: list[tuple[list[int], int]] = []  # (full ids, prompt_len)
        raw.eval()
        try:
            for j, pi in enumerate(batch["prompt_idx"].tolist()):
                p = self.prompts[pi]
                seed = gen.seed * 1_000_003 + batch["step"] * 4099 + batch["mb"] * 131 + j
                for samp in generate_tokens(raw, p, gen, self.tok.eos_id, dev, seed=seed, autocast=trainer.precision.autocast()):
                    if samp:
                        seqs.append((p + samp, len(p)))
        finally:
            raw.train()
        if not seqs:
            raise ConfigError("on-policy sampling produced no tokens")
        self._remember(seqs)
        Tm = max(len(s) for s, _ in seqs) - 1
        pad = self.tok.pad_id or 0
        x = torch.full((len(seqs), Tm), pad, dtype=torch.long)
        y = torch.full((len(seqs), Tm), -100, dtype=torch.long)
        for r, (s, pl) in enumerate(seqs):
            x[r, : len(s) - 1] = torch.tensor(s[:-1])
            y[r, pl - 1: len(s) - 1] = torch.tensor(s[pl:])
        x, y = x.to(dev), y.to(dev)
        logits, _ = forward_logits(model, x)
        with torch.no_grad():
            tl, _ = forward_logits(self.teacher, x)
        if tl.shape[-1] != logits.shape[-1]:
            raise ConfigError("teacher and student vocabularies differ")
        kind = "reverse_kl" if L.kind == "ce_only" else L.kind
        d = kd_token_divergence(logits, tl, kind, L.temperature, L.jsd_beta)
        mask = (y != -100).float()
        per_seq = (d * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        loss_sum = per_seq.sum()
        with torch.no_grad():
            agree = (((logits.argmax(-1) == tl.argmax(-1)).float() * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).sum()
            glen = mask.sum(-1).sum()
        return StepLoss(loss_sum, {"kd": per_seq.detach().sum(), "teacher_agreement": agree, "gen_len": glen})

    def _remember(self, seqs: list[tuple[list[int], int]]) -> None:
        for s, pl in seqs:
            self._samples.append(s[pl:])
        cap = max(2, self.cfg.diversity.max_samples)
        self._samples = self._samples[-cap:]
        self._last_text = self.tok.decode(seqs[0][0], skip_special_tokens=False)

    def extra_metrics(self) -> dict[str, float]:
        d = self.cfg.diversity
        if not d.enabled or len(self._samples) < 2:
            return {}
        rep = diversity_report(self._samples, d.ngram, d.self_bleu_pairs)
        out = {f"diversity/{k}": v for k, v in rep.items()}
        if rep.get(f"distinct_{d.ngram}", 1.0) < d.min_distinct_n:
            log.warning("generation diversity low: distinct-%d=%.3f < %.3f (possible collapse)", d.ngram, rep[f"distinct_{d.ngram}"], d.min_distinct_n)
        return out


# ---------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------
def _teacher_hash(t: TeacherConfig) -> str:
    if t.hash:
        return t.hash
    p = Path(t.path) if t.path else None
    if p is not None and p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix in (".safetensors", ".json", ".bin"))
        return hash_json({"dir": p.name, "files": [(f.name, f.stat().st_size) for f in files]})
    return hash_json({"path": t.path, "name": t.name, "source": t.source, "arch": t.arch_overrides})


def _load_teacher(t: TeacherConfig, device: torch.device, tok: TokenizerAdapter, seed: int, ctx: Any) -> nn.Module:
    from forge.train.config import ModelSourceConfig

    src = ModelSourceConfig(source="tiny" if t.source == "tiny" else "pretrained", path=t.path, backend=t.backend, dtype=t.dtype, arch_overrides=t.arch_overrides)
    model, _ = build_model(src, device=device, seed=seed + 1000 if t.source == "tiny" else seed, tok=tok, ctx=ctx, frozen_base=True, role="teacher")
    model.eval().requires_grad_(False)
    return model


def _check_tokenizers(t: TeacherConfig, student_tok: TokenizerAdapter, ctx: Any) -> TokenizerAdapter:
    """Teacher tokenizer (defaults to the student's). Logit-level modes require identical vocabularies."""
    if not t.tokenizer:
        return student_tok
    from forge.train.tokenization import load_tokenizer_any

    tt = load_tokenizer_any(t.tokenizer)
    if tt.hash != student_tok.hash:
        raise ConfigError("teacher and student tokenizers differ: logit/on-policy distillation needs the same vocabulary "
                          "(use mode=sequence_level or reasoning_sft across tokenizers)")
    return student_tok


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _prepare_trace_records(cfg: DistillConfig, ctx: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    recs = load_records(cfg.sft_data.train_files)
    out, dropped = [], 0
    for r in recs:
        trace = r.get("trace", r.get("reasoning", r.get("reasoning_content", "")))
        answer = r.get("answer", r.get("response", r.get("completion", "")))
        prompt = r.get("prompt", r.get("question", r.get("instruction")))
        if prompt is None or (not trace and not answer):
            dropped += 1
            continue
        if cfg.verified_only and ("verified" in r or "verifier_outcome" in r):
            ok = r.get("verified", r.get("verifier_outcome"))
            if ok in (False, 0, "fail", "failed", "incorrect"):
                dropped += 1
                continue
        content = cfg.trace_template.format(trace=trace, answer=answer)
        out.append({"messages": [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": content}],
                    "teacher": cfg.teacher.name or cfg.teacher.path})
    return out, {"read": len(recs), "kept": len(out), "dropped": dropped}


def run_distill(cfg: DistillConfig | dict, ctx: Any) -> dict[str, Any]:
    """Distillation job entry (job type ``distill``)."""
    from forge.train.finetune import SFTTask
    from forge.train.run import _finish, _maybe_launch, build_token_datasets

    cfg = coerce_cfg(cfg, DistillConfig)
    gate = check_teacher_licence(cfg.teacher)  # TR-8: blocking, before any compute or launch
    ctx.log(f"licence gate passed: {gate['basis']} (teacher={gate['teacher']})")
    launched = _maybe_launch("distill", cfg, ctx)
    if launched is not None:
        return launched
    if cfg.mode == "on_policy" and cfg.parallel.strategy == "fsdp":
        raise ConfigError("on_policy distillation does not support FSDP (student sampling needs the unsharded module); use ddp or none")
    dinfo = D.init_distributed(cfg.parallel)
    cache: TopKLogitCache | None = None
    try:
        tok = load_tokenizer_for(cfg.model, ctx)
        teacher_hash = _teacher_hash(cfg.teacher)
        lineage_extra: dict[str, Any] = {"distill": {"mode": cfg.mode, "teacher": {**gate, "hash": teacher_hash, "path": cfg.teacher.path},
                                                      "loss": cfg.loss.model_dump(mode="json"), "pruning_plan_hash": cfg.pruning_plan_hash}}
        teacher: nn.Module | None = None
        train_ds: Any
        val_ds: Any = None
        seq_len = 0
        if cfg.mode == "offline_logit":
            _check_tokenizers(cfg.teacher, tok, ctx)
            if cfg.data_kind == "tokens":
                train_ds, val_ds = build_token_datasets(cfg, cfg.token_data, cfg.seed)
                seq_len = cfg.token_data.seq_len
            else:
                train_ds, val_ds = build_sft_datasets(cfg.sft_data, tok, cfg.seed + cfg.sft_data.data_seed)
                seq_len = 0
            use_cache = cfg.cache.use_cache and cfg.loss.kind != "ce_only"
            if use_cache:
                n = len(train_ds) if cfg.cache.max_sequences is None else min(len(train_ds), cfg.cache.max_sequences)
                train_ds.restrict(n)
                cache = _get_or_build_cache(cfg, ctx, dinfo, tok, train_ds, n, teacher_hash, seq_len, lineage_extra)
            elif cfg.loss.kind != "ce_only":
                teacher = _load_teacher(cfg.teacher, dinfo.device, tok, cfg.seed, ctx)
            task: Task = OfflineKDTask(train_ds, cfg, val_ds, seq_len, cache, teacher,
                                       num_workers=cfg.token_data.num_workers if cfg.data_kind == "tokens" else 0)
        elif cfg.mode == "on_policy":
            _check_tokenizers(cfg.teacher, tok, ctx)
            prompts = load_prompts(cfg.on_policy.prompts, tok, cfg.sft_data.system_prompt)
            teacher = _load_teacher(cfg.teacher, dinfo.device, tok, cfg.seed, ctx)
            task = OnPolicyTask(prompts, cfg, teacher, tok)
            train_ds = _FP(f"prompts:{hash_json(prompts[:64])}")
        elif cfg.mode == "sequence_level":
            files = _generate_sequence_level_data(cfg, ctx, dinfo, tok, lineage_extra)
            sft = cfg.sft_data.model_copy(update={"train_files": files, "val_files": []})
            train_ds, val_ds = build_sft_datasets(sft, tok, cfg.seed + cfg.sft_data.data_seed)
            task = SFTTask(train_ds, cfg, val_ds, sft.max_seq_len)
        elif cfg.mode == "reasoning_sft":
            recs, st = _prepare_trace_records(cfg, ctx)
            lineage_extra["distill"]["trace_filter"] = st
            ctx.log(f"reasoning traces: {st}")
            f = _write_jsonl(Path(ctx.run_dir) / "reasoning_traces.jsonl", recs)
            sft = cfg.sft_data.model_copy(update={"train_files": [str(f)], "val_files": []})
            train_ds, val_ds = build_sft_datasets(sft, tok, cfg.seed + cfg.sft_data.data_seed)
            task = SFTTask(train_ds, cfg, val_ds, sft.max_seq_len)
        else:  # pragma: no cover
            raise ConfigError(f"unknown mode {cfg.mode}")

        student, _ = build_model(cfg.model, device=dinfo.device, seed=cfg.seed, tok=tok, ctx=ctx)
        check_vocab(student, tok)
        trainer = Trainer(student, cfg, task, ctx, dinfo=dinfo)
        result = trainer.fit()
        lineage_extra["distill"]["token_budget_consumed"] = result["tokens_seen"]
        if isinstance(task, OnPolicyTask) and task._samples:
            lineage_extra["distill"]["diversity"] = diversity_report(task._samples, cfg.diversity.ngram)
        result["distill"] = lineage_extra["distill"]
        parents = [cfg.model.parent_hash, cfg.teacher.hash]
        _finish(cfg, ctx, trainer, tok, result, getattr(train_ds, "fingerprint", None), parents, "model", lineage_extra)
        return result
    finally:
        if cache is not None:
            cache.close()
        D.shutdown_distributed()


class _FP:
    def __init__(self, fp: str) -> None:
        self.fingerprint = fp


def _get_or_build_cache(cfg: DistillConfig, ctx: Any, dinfo: D.DistInfo, tok: TokenizerAdapter, ds: Any, n: int, teacher_hash: str, seq_len: int,
                        lineage_extra: dict[str, Any]) -> TopKLogitCache:
    from forge.core.paths import cache_dir

    cc: LogitCacheConfig = cfg.cache
    T = seq_len or max(len(it[0]) - 1 for it in ds.items)
    ids = getattr(ds, "ids", None)  # token datasets: positions map to sequence ids (depends on holdout / max_sequences)
    data_fp = hash_json([ds.fingerprint, hash_bytes(np.ascontiguousarray(ids[:n]).tobytes()) if ids is not None else ""])
    key = cache_key(teacher_hash, tok.hash, data_fp, n, T, cc.k, cc.lp_quant, cc.lp_floor)
    path = Path(cc.dir) if cc.dir else cache_dir() / "distill" / key.split(":")[-1][:24]
    lineage_extra["distill"]["cache_key"] = key
    lineage_extra["distill"]["cache_dir"] = str(path)
    if (path / "meta.json").exists() and not cc.rebuild:
        c = TopKLogitCache(path)
        if c.complete and c.n >= n and c.k == cc.k:
            ctx.log(f"reusing logit cache {path} (key {key[:24]}...) coverage={c.stats.get('mean_coverage')}")
            lineage_extra["distill"]["cache_stats"] = c.stats
            return c
    if dinfo.is_main:
        ctx.log(f"building top-{cc.k} teacher logit cache for {n} sequences at {path}")
        teacher = _load_teacher(cfg.teacher, dinfo.device, tok, cfg.seed, ctx)
        try:
            from forge.train.precision import resolve_compute_dtype

            adt = resolve_compute_dtype(cfg.precision.compute_dtype, dinfo.device.type)
            ac = torch.autocast(device_type=dinfo.device.type, dtype=adt) if adt is not None else None
            build_topk_cache(path, teacher, ds, n, k=cc.k, lp_quant=cc.lp_quant, lp_floor=cc.lp_floor, shard_sequences=cc.shard_sequences,
                             batch_size=cc.teacher_batch_size, device=dinfo.device, T=T, ctx=ctx, autocast=ac,
                             min_coverage_warn=cc.min_coverage_warn, should_stop=ctx.should_stop, rebuild=cc.rebuild,
                             meta_extra={"key": key, "teacher_hash": teacher_hash, "tokenizer_hash": tok.hash, "data_fingerprint": data_fp})
        finally:
            del teacher
            if dinfo.device.type == "cuda":
                torch.cuda.empty_cache()
    if dinfo.initialized:
        import torch.distributed as dist

        dist.barrier()
    c = TopKLogitCache(path)
    lineage_extra["distill"]["cache_stats"] = c.stats
    store = getattr(ctx, "store", None)
    if store is not None and dinfo.is_main:
        try:
            lineage_extra["distill"]["cache_meta_hash"] = store.put_file(path / "meta.json", kind="logit_cache_meta")
        except Exception as e:  # noqa: BLE001
            ctx.log(f"could not store cache metadata: {e!r}", "warning")
    return c


def _generate_sequence_level_data(cfg: DistillConfig, ctx: Any, dinfo: D.DistInfo, student_tok: TokenizerAdapter, lineage_extra: dict[str, Any]) -> list[str]:
    """Teacher generates completions (sequence-level KD); returns the jsonl files for the student SFT stage."""
    out_path = Path(ctx.run_dir) / "sequence_level_data.jsonl"
    if dinfo.is_main:
        ttok = student_tok
        if cfg.teacher.tokenizer:
            from forge.train.tokenization import load_tokenizer_any

            ttok = load_tokenizer_any(cfg.teacher.tokenizer)
        prompts = load_prompts(cfg.on_policy.prompts, ttok, cfg.sft_data.system_prompt)
        teacher = _load_teacher(cfg.teacher, dinfo.device, ttok, cfg.seed, ctx)
        rows: list[dict[str, Any]] = []
        gen = cfg.generation
        from forge.train.precision import resolve_compute_dtype

        adt = resolve_compute_dtype(cfg.precision.compute_dtype, dinfo.device.type)
        t0 = time.time()
        for i, p in enumerate(prompts):
            if ctx.should_stop():
                break
            ac = torch.autocast(device_type=dinfo.device.type, dtype=adt) if adt is not None else None
            outs = generate_tokens(teacher, p, gen, ttok.eos_id, dinfo.device, seed=gen.seed * 7919 + i, autocast=ac)
            ptxt = ttok.decode(p, skip_special_tokens=True)
            for o in outs:
                rows.append({"messages": [{"role": "user", "content": ptxt}, {"role": "assistant", "content": ttok.decode(o, skip_special_tokens=True)}],
                             "teacher": cfg.teacher.name or cfg.teacher.path, "teacher_tokens": o})
            ctx.progress(0.5 * (i + 1) / len(prompts), f"teacher generation {i + 1}/{len(prompts)}")
        del teacher
        div = diversity_report([r["teacher_tokens"] for r in rows], cfg.diversity.ngram, cfg.diversity.self_bleu_pairs)
        lineage_extra["distill"]["teacher_output_diversity"] = div
        ctx.log(f"teacher generated {len(rows)} samples in {time.time() - t0:.1f}s; diversity={div}")
        for r in rows:
            r.pop("teacher_tokens", None)
        n_synth = len(rows)
        mix = cfg.mix_non_synthetic
        if mix > 0 and cfg.sft_data.train_files and n_synth:
            real = load_records(cfg.sft_data.train_files)
            want = int(round(mix / max(1e-9, 1.0 - mix) * n_synth)) if mix < 1 else len(real)
            rng = random.Random(cfg.seed)
            rng.shuffle(real)
            rows.extend(real[:want])
            lineage_extra["distill"]["mix_non_synthetic"] = {"synthetic": n_synth, "real": min(want, len(real))}
        _write_jsonl(out_path, rows)
    if dinfo.initialized:
        import torch.distributed as dist

        dist.barrier()
    return [str(out_path)]
