"""Tokenizer adapter used by the data pipeline.

A tiny uniform interface over (a) the built-in dependency-free :class:`ByteTokenizer` (CPU tests, smoke runs, byte-level
models), (b) tokenizers returned by ``forge.models.loader.load_tokenizer`` and (c) Hugging Face / ``tokenizers`` objects.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from forge.core.errors import ConfigError
from forge.core.hashing import hash_json

log = logging.getLogger("forge.train.tokenization")

__all__ = ["BYTE_SPECIALS", "ByteTokenizer", "TokenizerAdapter", "load_tokenizer_any", "wrap_tokenizer"]

BYTE_SPECIALS: list[str] = [
    "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>", "<|begin_of_text|>", "<|eot_id|>",
    "<|start_header_id|>", "<|end_header_id|>", "[INST]", "[/INST]", "<think>", "</think>", "<|system|>",
    "<|user|>", "<|assistant|>",
]


class ByteTokenizer:
    """Byte-level tokenizer: ids 0..255 are raw bytes, 256.. are special tokens (see ``BYTE_SPECIALS``)."""

    def __init__(self) -> None:
        self.specials = list(BYTE_SPECIALS)
        self.special_to_id = {s: 256 + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = 256 + len(self.specials)
        self.bos_token_id = self.special_to_id["<s>"]
        self.eos_token_id = self.special_to_id["</s>"]
        self.pad_token_id = self.special_to_id["<pad>"]
        self._split = re.compile("(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")")
        self.chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        if add_special_tokens:
            out.append(self.bos_token_id)
        for part in self._split.split(text):
            if not part:
                continue
            sid = self.special_to_id.get(part)
            if sid is not None:
                out.append(sid)
            else:
                out.extend(part.encode("utf-8"))
        return out

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        buf = bytearray()
        parts: list[str] = []
        for i in ids:
            i = int(i)
            if i < 256:
                buf.append(i)
            else:
                if buf:
                    parts.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                if not skip_special_tokens:
                    parts.append(self.id_to_special.get(i, ""))
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def get_vocab(self) -> dict[str, int]:
        v = {f"<0x{b:02X}>": b for b in range(256)}
        v.update(self.special_to_id)
        return v


class TokenizerAdapter:
    """Uniform wrapper: ``encode(text) -> list[int]`` (no automatic BOS/EOS), ids, ``hash``."""

    def __init__(self, tok: Any, name: str = "") -> None:
        self.tok = tok
        self.name = name or type(tok).__name__
        self.is_byte = isinstance(tok, ByteTokenizer)

    # -- ids --------------------------------------------------------------------------------
    def _id(self, attr: str) -> int | None:
        v = getattr(self.tok, attr, None)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @property
    def vocab_size(self) -> int:
        vs = getattr(self.tok, "vocab_size", None)
        if vs is None and hasattr(self.tok, "get_vocab_size"):
            vs = self.tok.get_vocab_size()
        if vs is None and hasattr(self.tok, "__len__"):
            vs = len(self.tok)
        if hasattr(self.tok, "__len__") and not self.is_byte:
            try:
                vs = max(int(vs or 0), len(self.tok))
            except Exception:
                pass
        return int(vs or 0)

    @property
    def eos_id(self) -> int | None:
        return self._id("eos_token_id")

    @property
    def bos_id(self) -> int | None:
        return self._id("bos_token_id")

    @property
    def pad_id(self) -> int | None:
        p = self._id("pad_token_id")
        return p if p is not None else self.eos_id

    # -- api ------------------------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        try:
            out = self.tok.encode(text, add_special_tokens=False)
        except TypeError:
            out = self.tok.encode(text)
        if hasattr(out, "ids"):
            out = out.ids
        return [int(i) for i in out]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        try:
            return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
        except TypeError:
            return self.tok.decode(list(ids))

    def has_chat_template(self) -> bool:
        return bool(getattr(self.tok, "chat_template", None)) and hasattr(self.tok, "apply_chat_template")

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
        out = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        elif isinstance(out, dict):
            out = out["input_ids"]
        return [int(i) for i in out]

    def special_id(self, token: str) -> int | None:
        """Id of a special token string if it maps to a single id."""
        if self.is_byte:
            return self.tok.special_to_id.get(token)
        try:
            ids = self.encode(token)
        except Exception:
            return None
        return ids[0] if len(ids) == 1 else None

    @property
    def hash(self) -> str:
        """Stable identity of the vocabulary (used in dataset manifests and cache keys)."""
        try:
            vocab = self.tok.get_vocab()
            return hash_json({"vocab": sorted(vocab.items())})
        except Exception:
            return hash_json({"name": self.name, "vocab_size": self.vocab_size})


def wrap_tokenizer(tok: Any) -> TokenizerAdapter:
    return tok if isinstance(tok, TokenizerAdapter) else TokenizerAdapter(tok)


def load_tokenizer_any(spec: str | None) -> TokenizerAdapter:
    """Load a tokenizer from ``spec``: 'byte', a local dir/file, or a Hugging Face repo id."""
    if spec is None or str(spec).strip().lower() in ("byte", "bytes", "builtin:byte"):
        return TokenizerAdapter(ByteTokenizer(), "byte")
    spec = str(spec)
    errors: list[str] = []
    try:  # 1) Forge loader
        from forge.models.loader import load_tokenizer  # type: ignore

        tok = load_tokenizer(spec)
        if tok is not None:
            return TokenizerAdapter(tok, spec)
    except Exception as e:  # noqa: BLE001
        errors.append(f"forge.models.loader: {e!r}")
    try:  # 2) transformers
        from transformers import AutoTokenizer  # type: ignore

        return TokenizerAdapter(AutoTokenizer.from_pretrained(spec), spec)
    except Exception as e:  # noqa: BLE001
        errors.append(f"transformers: {e!r}")
    try:  # 3) bare tokenizers json
        from tokenizers import Tokenizer  # type: ignore

        p = Path(spec)
        f = p / "tokenizer.json" if p.is_dir() else p
        if f.exists():
            return TokenizerAdapter(Tokenizer.from_file(str(f)), spec)
    except Exception as e:  # noqa: BLE001
        errors.append(f"tokenizers: {e!r}")
    raise ConfigError(f"could not load tokenizer {spec!r}", details={"attempts": errors})
