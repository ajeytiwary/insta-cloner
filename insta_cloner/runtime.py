from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RuntimeStatus:
    python: str
    cuda: bool
    gpu: str | None
    vram_gb: float | None
    torch: str | None
    diffusers: str | None
    accelerate: bool
    bitsandbytes: bool
    huggingface_hub: bool


def runtime_status() -> dict[str, Any]:
    torch_version = None
    diffusers_version = None
    cuda = False
    gpu = None
    vram = None
    try:
        import torch
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            vram = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception:
        pass
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except Exception:
        pass
    return asdict(
        RuntimeStatus(
            python=platform.python_version(),
            cuda=cuda,
            gpu=gpu,
            vram_gb=vram,
            torch=torch_version,
            diffusers=diffusers_version,
            accelerate=importlib.util.find_spec("accelerate") is not None,
            bitsandbytes=importlib.util.find_spec("bitsandbytes") is not None,
            huggingface_hub=importlib.util.find_spec("huggingface_hub") is not None,
        )
    )


def model_readiness(spec: dict[str, Any]) -> dict[str, Any]:
    status = runtime_status()
    warnings: list[str] = []
    if spec.get("local") and not status["cuda"]:
        warnings.append("CUDA GPU not detected.")
    if not status["diffusers"] and spec.get("kind") == "diffusers":
        warnings.append("Diffusers is not installed; install the generate/full extra.")
    if spec.get("low_vram") and not status["accelerate"]:
        warnings.append("Accelerate is required for CPU offload.")
    if spec.get("quantization") == "bitsandbytes_8bit" and not status["bitsandbytes"]:
        warnings.append("bitsandbytes is required for the selected 8-bit mode.")
    if status["vram_gb"] and status["vram_gb"] <= 12.5 and not spec.get("recommended_12gb", False):
        warnings.append(
            "This preset is not marked as comfortable on 12 GB VRAM; use low-VRAM "
            "offload/quantization and expect system-RAM use."
        )
    return {"ready": not warnings, "warnings": warnings, "runtime": status}


def apply_memory_optimizations(pipe: Any, low_vram: bool = True) -> Any:
    if not low_vram:
        if hasattr(pipe, "to"):
            pipe.to("cuda")
        return pipe
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe


def hf_cache_dir() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.expanduser("~/.cache/huggingface"),
    )
