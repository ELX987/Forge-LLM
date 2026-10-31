"""Job configuration for the hardware probe (job type ``probe``).  Also re-exported from :mod:`forge.hw.probe`.

The UI renders the probe form from this model's JSON schema, so every field has a description.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.core.jsonschema import ui

BenchName = Literal["bandwidth", "gemm", "onchip", "launch", "dequant", "p2p", "extra_checks"]


class ProbeConfig(BaseModel):
    """Hardware capability probe + micro-benchmark suite (Plan HW-2/HW-3): builds a measured chip descriptor of
    the selected GPU (or a CPU tier-C descriptor on hosts without an NVIDIA GPU) and stores it."""

    model_config = ConfigDict(extra="forbid")

    device: int = Field(
        default=0, ge=0, description="CUDA device index to probe (index among the GPUs visible to the job).",
        json_schema_extra=ui(group="Target"),
    )
    quick: bool = Field(
        default=False,
        description="Quick mode: smaller problem sizes and fewer iterations (about 10 s). Numbers are noisier "
                    "and the multi-GPU test is skipped.",
        json_schema_extra=ui(group="Target"),
    )
    microbench: bool = Field(
        default=True,
        description="Run the micro-benchmarks. If off, only the datasheet descriptor completed with driver-reported "
                    "properties is produced (no measured fields; fp8/fp4 stay unconfirmed).",
        json_schema_extra=ui(group="Benchmarks"),
    )
    benchmarks: list[BenchName] = Field(
        default_factory=lambda: ["bandwidth", "gemm", "onchip", "launch", "dequant", "p2p", "extra_checks"],
        description="Which benchmark groups to run: memory bandwidth, tensor-core GEMM per dtype, on-chip SRAM "
                    "re-read proxy, kernel launch latency, dequantisation ALU rate, peer-to-peer copy, and "
                    "checks registered by other modules.",
        json_schema_extra=ui(group="Benchmarks"),
    )
    dtypes: list[str] = Field(
        default_factory=list,
        description="GEMM dtypes to measure (fp32, tf32, fp16, bf16, fp8, int8, fp4). Empty = every dtype the "
                    "device is expected to support.",
        json_schema_extra=ui(group="Benchmarks", advanced=True),
    )
    record_throttling: bool = Field(
        default=True,
        description="Sample clocks, temperature, power and throttle reasons (NVML / nvidia-smi) while benchmarking "
                    "and store a summary; flags runs distorted by throttling.",
        json_schema_extra=ui(group="Benchmarks", advanced=True),
    )
    save: bool = Field(
        default=True, description="Store the descriptor and raw report in the Forge database / home directory.",
        json_schema_extra=ui(group="Output"),
    )
    label: str = Field(
        default="", max_length=64,
        description="Optional free-text label stored in the report (e.g. 'vast.ai instance 1234, locked clocks').",
        json_schema_extra=ui(group="Output", advanced=True),
    )
