"""Supervised fine-tuning data: chat templates, assistant-only loss masks, packing, preference pairs.

Every conversation is rendered into *segments* ``(text-or-token-id, trainable)``; segments are tokenised separately and
concatenated, which gives exact per-token loss masks without offset mapping.  Batches use aligned labels
(``labels[t]`` is the target of ``logits[t]``; ``-100`` = ignored).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.errors import ConfigError, NotFound
from forge.core.hashing import hash_file, hash_json
from forge.train.config import PromptSourceConfig, SFTDataConfig
from forge.train.data import _PermCache, step_positions
from forge.train.tokenization import TokenizerAdapter

log = logging.getLogger("forge.train.sft")

__all__ = [
    "CHAT_TEMPLATES", "IGNORE_INDEX", "SFTDataset", "build_sft_datasets", "collate_examples", "encode_conversation",
    "load_records", "normalise_record", "render_segments", "load_prompts",
]

IGNORE_INDEX = -100
Segment = tuple["str | int", bool]  # (text or token id, trainable)
CHAT_TEMPLATES = ("tokenizer", "chatml", "llama3", "mistral", "alpaca", "plain")
_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "model": "assistant",
             "system": "system", "tool": "tool", "function": "tool"}


# ---------------------------------------------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------------------------------------------
def _norm_messages(messages: list[dict], system_prompt: str | None) -> list[dict]:
    msgs = [{"role": _ROLE_MAP.get(str(m.get("role", m.get("from", "user"))).lower(), "user"), "content": str(m.get("content", m.get("value", "")))} for m in messages]
    if system_prompt and not any(m["role"] == "system" for m in msgs):
        msgs.insert(0, {"role": "system", "content": system_prompt})
    return msgs


def render_segments(template: str, messages: list[dict], tok: TokenizerAdapter, *, system_prompt: str | None = None,
                    add_generation_prompt: bool = False, train_all_assistant: bool = True) -> list[Segment]:
    """Render ``messages`` into ``(piece, trainable)`` segments for a named template."""
    msgs = _norm_messages(messages, system_prompt)
    last_asst = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    segs: list[Segment] = []

    def train_flag(i: int) -> bool:
        return msgs[i]["role"] == "assistant" and (train_all_assistant or i == last_asst)

    eos = tok.eos_id
    if template == "chatml":
        for i, m in enumerate(msgs):
            segs.append((f"<|im_start|>{m['role']}\n", False))
            segs.append((m["content"] + "<|im_end|>", train_flag(i)))
            segs.append(("\n", False))
        if add_generation_prompt:
            segs.append(("<|im_start|>assistant\n", False))
    elif template == "llama3":
        segs.append(("<|begin_of_text|>", False))
        for i, m in enumerate(msgs):
            segs.append((f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n", False))
            segs.append((m["content"] + "<|eot_id|>", train_flag(i)))
        if add_generation_prompt:
            segs.append(("<|start_header_id|>assistant<|end_header_id|>\n\n", False))
    elif template == "mistral":
        segs.append(("<s>", False))
        sys_txt = ""
        body = msgs
        if body and body[0]["role"] == "system":
            sys_txt, body = body[0]["content"] + "\n\n", body[1:]
        first_user = True
        offset = len(msgs) - len(body)
        for j, m in enumerate(body):
            if m["role"] == "assistant":
                segs.append((" " + m["content"], train_flag(j + offset)))
                segs.append((eos if eos is not None else "</s>", train_flag(j + offset)))
            else:
                pre = sys_txt if first_user else ""
                first_user = False
                segs.append((f"[INST] {pre}{m['content']} [/INST]", False))
    elif template in ("alpaca", "plain"):
        sys_txt = ""
        body = msgs
        if body and body[0]["role"] == "system":
            sys_txt, body = body[0]["content"].strip() + "\n\n", body[1:]
        offset = len(msgs) - len(body)
        for j, m in enumerate(body):
            if m["role"] == "assistant":
                segs.append((m["content"], train_flag(j + offset)))
                segs.append((eos if eos is not None else "\n", train_flag(j + offset)))
            elif template == "alpaca":
                segs.append(((sys_txt if j == 0 else "") + f"### Instruction:\n{m['content']}\n\n### Response:\n", False))
            else:
                segs.append(((sys_txt if j == 0 else "") + f"{m['role']}: {m['content']}\n", False))
        if add_generation_prompt and template == "plain":
            segs.append(("assistant: ", False))
    else:
        raise ConfigError(f"unknown chat template {template!r}", details={"known": list(CHAT_TEMPLATES)})
    return segs


def _segments_to_ids(segs: list[Segment], tok: TokenizerAdapter) -> tuple[list[int], list[bool]]:
    ids: list[int] = []
    mask: list[bool] = []
    for piece, flag in segs:
        sub = [int(piece)] if isinstance(piece, int) else tok.encode(piece)
        ids.extend(sub)
        mask.extend([flag] * len(sub))
    return ids, mask


def _tokenizer_template_ids(messages: list[dict], tok: TokenizerAdapter, system_prompt: str | None, train_all: bool) -> tuple[list[int], list[bool]]:
    msgs = _norm_messages(messages, system_prompt)
    full = tok.apply_chat_template(msgs, add_generation_prompt=False)
    mask = [False] * len(full)
    last_asst = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=-1)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or not (train_all or i == last_asst):
            continue
        pre = tok.apply_chat_template(msgs[:i], add_generation_prompt=True) if i > 0 else []
        upto = tok.apply_chat_template(msgs[: i + 1], add_generation_prompt=False)
        if upto[: len(pre)] == pre and upto == full[: len(upto)]:
            for t in range(len(pre), len(upto)):
                mask[t] = True
        else:  # not prefix-consistent: fall back to training on everything of this turn's span (best effort)
            log.warning("tokenizer chat template is not prefix-consistent; training on the whole sequence")
            return full, [True] * len(full)
    return full, mask


def encode_conversation(messages: list[dict], tok: TokenizerAdapter, template: str = "chatml", *, system_prompt: str | None = None,
                        train_on_prompt: bool = False, train_all_assistant: bool = True) -> tuple[list[int], list[bool]]:
    """Token ids and per-token trainable mask (True = contributes to the loss) for a conversation."""
    if template == "tokenizer":
        if not tok.has_chat_template():
            raise ConfigError("template=tokenizer but the tokenizer has no chat template")
        ids, mask = _tokenizer_template_ids(messages, tok, system_prompt, train_all_assistant)
    else:
        segs = render_segments(template, messages, tok, system_prompt=system_prompt, train_all_assistant=train_all_assistant)
        ids, mask = _segments_to_ids(segs, tok)
    if train_on_prompt:
        mask = [True] * len(ids)
    return ids, mask


# ---------------------------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------------------------
def load_records(files: list[str]) -> list[dict]:
    recs: list[dict] = []
    for f in files:
        p = Path(f)
        if not p.exists():
            raise NotFound(f"SFT file not found: {f}")
        name = p.name.lower()
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq

            recs.extend(pq.read_table(str(p)).to_pylist())
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        s = text.lstrip()
        if s.startswith("["):
            recs.extend(json.loads(s))
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("skipping malformed jsonl line in %s", f)
    return recs


def _as_messages(v: Any, role: str) -> list[dict]:
    if isinstance(v, str):
        return [{"role": role, "content": v}]
    if isinstance(v, list):
        return [{"role": _ROLE_MAP.get(str(m.get("role", m.get("from", role))).lower(), "user"), "content": m.get("content", m.get("value", ""))} for m in v]
    raise ConfigError(f"cannot interpret {type(v).__name__} as messages")


def normalise_record(rec: dict, fmt: str = "auto", field_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Return ``{'kind': 'chat', 'messages': [...]}`` | ``{'kind': 'text', 'text': str}`` | ``{'kind': 'pref', 'prompt': [...], 'chosen': str|[...], 'rejected': ...}``."""
    fm = field_map or {}
    r = {k: rec.get(fm.get(k, k)) if fm.get(k, k) in rec else rec.get(k) for k in
         ("messages", "conversations", "instruction", "input", "output", "prompt", "completion", "chosen", "rejected", "text", "system", "response")}
    if fmt == "auto":
        if r["chosen"] is not None and r["rejected"] is not None:
            fmt = "preference"
        elif r["messages"] is not None:
            fmt = "messages"
        elif r["conversations"] is not None:
            fmt = "sharegpt"
        elif r["instruction"] is not None:
            fmt = "alpaca"
        elif r["prompt"] is not None and (r["completion"] is not None or r["response"] is not None):
            fmt = "prompt_completion"
        elif r["text"] is not None:
            fmt = "text"
        else:
            raise ConfigError("cannot detect record format", details={"keys": list(rec.keys())})
    sys_msg = [{"role": "system", "content": r["system"]}] if r.get("system") else []
    if fmt == "messages":
        return {"kind": "chat", "messages": sys_msg + _as_messages(r["messages"], "user")}
    if fmt == "sharegpt":
        return {"kind": "chat", "messages": sys_msg + _as_messages(r["conversations"], "user")}
    if fmt == "alpaca":
        user = str(r["instruction"]) + (f"\n\n{r['input']}" if r.get("input") else "")
        out = r["output"] if r["output"] is not None else r["response"]
        return {"kind": "chat", "messages": sys_msg + [{"role": "user", "content": user}, {"role": "assistant", "content": str(out)}]}
    if fmt == "prompt_completion":
        comp = r["completion"] if r["completion"] is not None else r["response"]
        return {"kind": "chat", "messages": sys_msg + _as_messages(r["prompt"], "user") + [{"role": "assistant", "content": str(comp)}]}
    if fmt == "preference":
        return {"kind": "pref", "prompt": sys_msg + _as_messages(r["prompt"], "user") if r["prompt"] is not None else sys_msg,
                "chosen": _as_messages(r["chosen"], "assistant"), "rejected": _as_messages(r["rejected"], "assistant")}
    if fmt == "text":
        return {"kind": "text", "text": str(r["text"])}
    raise ConfigError(f"unknown record format {fmt!r}")


# ---------------------------------------------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------------------------------------------
def collate_examples(items: list[tuple[list[int], list[bool]]], pad_id: int, max_len: int | None = None) -> dict[str, Any]:
    """Right-pad ``(ids, mask)`` items into aligned ``input_ids``/``labels`` tensors."""
    import torch

    seqs_in, seqs_lab = [], []
    for ids, mask in items:
        inp = ids[:-1]
        lab = [ids[t + 1] if mask[t + 1] else IGNORE_INDEX for t in range(len(ids) - 1)]
        seqs_in.append(inp)
        seqs_lab.append(lab)
    T = max(1, max(len(s) for s in seqs_in))
    if max_len:
        T = min(T, max_len)
    B = len(items)
    x = torch.full((B, T), int(pad_id), dtype=torch.long)
    y = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (a, b) in enumerate(zip(seqs_in, seqs_lab, strict=False)):
        n = min(len(a), T)
        if n:
            x[i, :n] = torch.tensor(a[:n])
            y[i, :n] = torch.tensor(b[:n])
        lengths[i] = n
    return {"input_ids": x, "labels": y, "lengths": lengths}


class SFTDataset:
    """Tokenised SFT examples with deterministic, stateless batch order (same contract as ``ShardedTokenDataset``)."""

    def __init__(self, records: list[dict], tok: TokenizerAdapter, cfg: SFTDataConfig, *, seed: int = 0, shuffle: bool | None = None,
                 fingerprint_extra: Any = None) -> None:
        self.tok, self.cfg = tok, cfg
        self.seed = int(seed) + int(cfg.data_seed)
        self.pad_id = tok.pad_id if tok.pad_id is not None else 0
        self.kind = "chat"
        self.items: list[Any] = []
        self.stats = {"records": len(records), "dropped_long": 0, "dropped_unsupervised": 0, "truncated": 0, "tokens": 0, "supervised_tokens": 0}
        n_pref = 0
        for rec in records:
            norm = normalise_record(rec, cfg.format, cfg.field_map)
            if norm["kind"] == "pref":
                n_pref += 1
                self._add_pref(norm)
            elif norm["kind"] == "text":
                ids = tok.encode(norm["text"]) + ([tok.eos_id] if tok.eos_id is not None else [])
                self._add_ids(ids, [True] * len(ids))
            else:
                ids, mask = encode_conversation(norm["messages"], tok, cfg.template, system_prompt=cfg.system_prompt,
                                                train_on_prompt=cfg.train_on_prompt, train_all_assistant=cfg.train_on_all_assistant_turns)
                self._add_ids(ids, mask)
        if n_pref:
            if n_pref != len(records):
                raise ConfigError("mixing preference and non-preference records is not supported")
            self.kind = "preference"
        elif cfg.packing:
            self._pack()
        if not self.items:
            raise ConfigError("SFT dataset is empty after filtering", details=self.stats)
        self._perms = _PermCache(len(self.items), self.seed, 0, cfg.shuffle if shuffle is None else shuffle)
        self.fingerprint = hash_json({"n": len(self.items), "stats": self.stats, "extra": fingerprint_extra, "tok": tok.hash,
                                      "template": cfg.template, "max": cfg.max_seq_len,
                                      "sample": hashlib.blake2b(repr(self.items[:8]).encode(), digest_size=8).hexdigest()})

    # -- building -----------------------------------------------------------------------------------
    def _fit(self, ids: list[int], mask: list[bool]) -> tuple[list[int], list[bool]] | None:
        L = self.cfg.max_seq_len + 1
        if len(ids) > L:
            if self.cfg.truncation == "drop":
                self.stats["dropped_long"] += 1
                return None
            self.stats["truncated"] += 1
            ids, mask = (ids[:L], mask[:L]) if self.cfg.truncation == "right" else (ids[-L:], mask[-L:])
        return ids, mask

    def _add_ids(self, ids: list[int], mask: list[bool]) -> None:
        fitted = self._fit(ids, mask)
        if fitted is None:
            return
        ids, mask = fitted
        sup = sum(mask[1:])
        if sup < max(1, self.cfg.min_supervised_tokens):
            self.stats["dropped_unsupervised"] += 1
            return
        self.items.append((ids, mask))
        self.stats["tokens"] += len(ids)
        self.stats["supervised_tokens"] += sup

    def _add_pref(self, norm: dict) -> None:
        cfg, tok = self.cfg, self.tok
        pair = []
        for key in ("chosen", "rejected"):
            ids, mask = encode_conversation(norm["prompt"] + norm[key], tok, cfg.template, system_prompt=cfg.system_prompt,
                                            train_on_prompt=False, train_all_assistant=False)
            fitted = self._fit(ids, mask)
            if fitted is None or sum(fitted[1][1:]) < 1:
                self.stats["dropped_long"] += 1
                return
            pair.append(fitted)
        self.items.append(tuple(pair))
        self.stats["tokens"] += len(pair[0][0]) + len(pair[1][0])
        self.stats["supervised_tokens"] += sum(pair[0][1][1:]) + sum(pair[1][1][1:])

    def _pack(self) -> None:
        """Greedy sequential packing of examples into <= max_seq_len+1 token units (static, epoch-independent)."""
        L = self.cfg.max_seq_len + 1
        packed: list[tuple[list[int], list[bool]]] = []
        cur_ids: list[int] = []
        cur_mask: list[bool] = []
        for ids, mask in self.items:
            if cur_ids and len(cur_ids) + len(ids) > L:
                packed.append((cur_ids, cur_mask))
                cur_ids, cur_mask = [], []
            # first token of a non-first example has no predecessor in its own context: it is still predicted from the
            # previous example's tail, so mask it out to avoid learning cross-document transitions
            m = list(mask)
            if cur_ids:
                m[0] = False
            cur_ids.extend(ids)
            cur_mask.extend(m)
        if cur_ids:
            packed.append((cur_ids, cur_mask))
        self.stats["packed_sequences"] = len(packed)
        self.items = packed

    # -- access ---------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def get_batch(self, positions: np.ndarray) -> dict[str, Any]:
        sel = [self.items[int(p)] for p in positions]
        if self.kind == "preference":
            chosen = collate_examples([s[0] for s in sel], self.pad_id)
            rejected = collate_examples([s[1] for s in sel], self.pad_id)
            import torch

            T = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

            def padto(t: Any, val: int) -> Any:
                return torch.nn.functional.pad(t, (0, T - t.shape[1]), value=val)

            return {
                "input_ids": torch.cat([padto(chosen["input_ids"], self.pad_id), padto(rejected["input_ids"], self.pad_id)]),
                "labels": torch.cat([padto(chosen["labels"], IGNORE_INDEX), padto(rejected["labels"], IGNORE_INDEX)]),
                "n_pairs": len(sel),
            }
        out = collate_examples(sel, self.pad_id)
        import torch

        out["positions"] = torch.as_tensor(np.asarray(positions, dtype=np.int64))
        return out

    def batch_at(self, step: int, global_batch_size: int, rank: int = 0, world: int = 1) -> dict[str, Any]:
        if global_batch_size % world:
            raise ConfigError("global batch size must be divisible by world size")
        pos = step_positions(self._perms, step, global_batch_size)
        per = global_batch_size // world
        return self.get_batch(pos[rank * per:(rank + 1) * per])

    def positions_at(self, step: int, global_batch_size: int) -> np.ndarray:
        return step_positions(self._perms, step, global_batch_size)

    def restrict(self, n: int) -> None:
        """Keep only the first ``n`` examples (cache-bound distillation)."""
        self.items = self.items[: int(n)]
        self._perms = _PermCache(len(self.items), self.seed, 0, self._perms.shuffle)

    def eval_batches(self, micro_batch: int, n_batches: int) -> list[dict[str, Any]]:
        total = min(len(self), micro_batch * n_batches)
        return [self.get_batch(np.arange(i, min(i + micro_batch, total))) for i in range(0, total, micro_batch)]

    @property
    def tokens_per_epoch(self) -> int:
        return int(self.stats["tokens"])

    def describe(self) -> dict[str, Any]:
        return {"examples": len(self), "kind": self.kind, **self.stats, "fingerprint": self.fingerprint}


def build_sft_datasets(cfg: SFTDataConfig, tok: TokenizerAdapter, seed: int = 0) -> tuple[SFTDataset, SFTDataset | None]:
    """Train / validation datasets (val from ``val_files`` or a deterministic ``val_fraction`` hold-out)."""
    if not cfg.train_files:
        raise ConfigError("sft data: train_files is empty")
    fp = [hash_file(f) for f in cfg.train_files if Path(f).exists()]
    recs = load_records(cfg.train_files)
    if cfg.val_files:
        val_recs = load_records(cfg.val_files)
    elif cfg.val_fraction > 0 and len(recs) >= 4:
        cut = int(cfg.val_fraction * 10000)
        key = hash_json(fp)
        is_val = [int.from_bytes(hashlib.blake2b(f"{key}:{i}".encode(), digest_size=4).digest(), "little") % 10000 < cut for i in range(len(recs))]
        if not any(is_val):
            is_val[0] = True  # at least one example when a fraction was requested
        val_recs = [r for r, v in zip(recs, is_val, strict=False) if v]
        recs = [r for r, v in zip(recs, is_val, strict=False) if not v]
    else:
        val_recs = []
    train = SFTDataset(recs, tok, cfg, seed=seed, fingerprint_extra=fp)
    val = SFTDataset(val_recs, tok, cfg, seed=seed, shuffle=False, fingerprint_extra=("val", fp)) if val_recs else None
    return train, val


def load_prompts(cfg: PromptSourceConfig, tok: TokenizerAdapter, system_prompt: str | None = None) -> list[list[int]]:
    """Prompt token-id lists (templated as a user turn when ``use_chat_template``)."""
    prompts: list[list[int]] = []
    for f in cfg.files:
        p = Path(f)
        if not p.exists():
            raise NotFound(f"prompt file not found: {f}")
        if p.suffix == ".txt":
            items: list[Any] = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            items = [r.get(cfg.field, r.get("messages")) if isinstance(r, dict) else r for r in load_records([f])]
        for it in items:
            if it is None:
                continue
            msgs = it if isinstance(it, list) else [{"role": "user", "content": str(it)}]
            if cfg.use_chat_template:
                if cfg.template == "tokenizer":
                    ids = tok.apply_chat_template(_norm_messages(msgs, system_prompt), add_generation_prompt=True)
                else:
                    ids, _ = _segments_to_ids(render_segments(cfg.template, msgs, tok, system_prompt=system_prompt, add_generation_prompt=True), tok)
            else:
                ids = tok.encode(msgs[-1]["content"])
            if len(ids) > cfg.max_prompt_tokens:
                ids = ids[-cfg.max_prompt_tokens:]
            prompts.append(ids)
            if cfg.max_prompts and len(prompts) >= cfg.max_prompts:
                return prompts
    if not prompts:
        raise ConfigError("no prompts loaded")
    return prompts
