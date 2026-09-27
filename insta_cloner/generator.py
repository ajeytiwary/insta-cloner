from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def _reference_images(media_dir: Path, limit: int = 4) -> list[Image.Image]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in sorted(media_dir.rglob("*")) if p.suffix.lower() in suffixes][:limit]
    return [Image.open(p).convert("RGB") for p in paths]


def _dry_image(path: Path, concept: dict[str, Any], width: int, height: int) -> None:
    img = Image.new("RGB", (width, height), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    text = "DRY RUN\\n" + str(concept.get("title", "concept")) + "\\n\\n" + str(concept.get("prompt", ""))[:220]
    draw.multiline_text((40, 40), text, fill=(235, 235, 235), spacing=8)
    img.save(path)


def generate_images(concepts: list[dict[str, Any]], output_dir: Path, media_dir: Path,
                    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                    ip_adapter_repo: str | None = None,
                    ip_adapter_weight: str = "ip-adapter_sdxl.bin",
                    ip_scale: float = 0.45, steps: int = 25,
                    width: int = 768, height: int = 1024,
                    seed: int = 42, dry_run: bool = False) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        made = []
        for i, concept in enumerate(concepts):
            path = output_dir / f"{i + 1:03d}.png"
            _dry_image(path, concept, width, height)
            made.append(path)
        return made

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError('Install generation dependencies: pip install -e ".[generate]"') from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Use --dry-run-generation or run on a CUDA machine.")

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    refs = _reference_images(media_dir)
    use_ip = bool(ip_adapter_repo and refs)
    if use_ip:
        pipe.load_ip_adapter(ip_adapter_repo, subfolder="sdxl_models", weight_name=ip_adapter_weight)
        pipe.set_ip_adapter_scale(ip_scale)

    made = []
    for i, concept in enumerate(concepts):
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        kwargs: dict[str, Any] = {
            "prompt": concept.get("prompt", ""),
            "negative_prompt": concept.get("negative_prompt", "watermark, logo, text"),
            "num_inference_steps": steps, "width": width, "height": height, "generator": generator,
        }
        if use_ip:
            kwargs["ip_adapter_image"] = refs[i % len(refs)]
        image = pipe(**kwargs).images[0]
        path = output_dir / f"{i + 1:03d}.png"
        image.save(path)
        made.append(path)
    return made
