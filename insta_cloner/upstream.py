from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

@dataclass
class UpstreamStatus:
    name: str
    import_name: str
    installed: bool
    version: str | None = None


def _version(dist: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def upstream_status() -> list[UpstreamStatus]:
    checks=[
        ("Instaloader","instaloader","instaloader"),
        ("CLIP Interrogator","clip_interrogator","clip-interrogator"),
        ("ControlNet Aux","controlnet_aux","controlnet-aux"),
        ("Diffusers / IP-Adapter","diffusers","diffusers"),
    ]
    out=[]
    for label,module,dist in checks:
        try:
            __import__(module); ok=True
        except Exception:
            ok=False
        out.append(UpstreamStatus(label,module,ok,_version(dist) if ok else None))
    return out


def make_ip_adapter_pipeline(model_id: str, ip_adapter_repo: str,
                             weight_name: str="ip-adapter_sdxl.bin",
                             subfolder: str="sdxl_models",
                             scale: float=0.45):
    import torch
    from diffusers import AutoPipelineForText2Image
    pipe=AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.load_ip_adapter(ip_adapter_repo,subfolder=subfolder,weight_name=weight_name)
    pipe.set_ip_adapter_scale(scale)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe,"enable_vae_slicing"): pipe.enable_vae_slicing()
    if hasattr(pipe,"enable_vae_tiling"): pipe.enable_vae_tiling()
    return pipe


def make_controlnet_pipeline(base_model: str, controlnet_model: str):
    import torch
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    controlnet=ControlNetModel.from_pretrained(controlnet_model,torch_dtype=torch.float16)
    pipe=StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model,controlnet=controlnet,torch_dtype=torch.float16,use_safetensors=True
    )
    pipe.enable_model_cpu_offload()
    return pipe


def generate_with_ip_adapter(prompt: str, reference: Image.Image, pipe, width: int=768,
                             height: int=1024, steps: int=25, seed: int=42):
    import torch
    return pipe(
        prompt=prompt,
        ip_adapter_image=reference,
        width=width,height=height,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]


def generate_with_controlnet(prompt: str, control_image: Image.Image, pipe,
                             width: int=768,height: int=1024,steps: int=25,seed: int=42):
    import torch
    return pipe(
        prompt=prompt,image=control_image,width=width,height=height,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
