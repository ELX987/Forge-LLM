"""Forge LLM: unified training, compression, fine-tuning and serving system.

Importing ``forge`` is cheap: heavy dependencies (torch, triton, transformers, pynvml)
are only imported lazily inside the functions that need them.
"""
from __future__ import annotations

from forge.__version__ import __version__

__all__ = ["__version__"]
