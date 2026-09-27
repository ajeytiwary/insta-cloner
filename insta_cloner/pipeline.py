from __future__ import annotations

from pathlib import Path
from typing import Any

from .analyzer import analyze_images, save_style_profile
from .brief import create_generation_brief
from .downloader import download_sample
from .frames import extract_keyframes
from .generator import generate_images
from .llm import generate_concepts
from .pose import extract_poses
from .utils import profile_name


def run_pipeline(profile: str, output: Path = Path("output"), max_posts: int = 12,
                 login: str | None = None, use_clip: bool = False, use_pose: bool = False,
                 video_frames: int = 4, concept_count: int = 12,
                 llm_base_url: str = "http://127.0.0.1:8080/v1",
                 llm_model: str = "local-model", llm_api_key: str = "local",
                 generate: bool = False,
                 image_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 ip_adapter_repo: str | None = None,
                 dry_run_llm: bool = False, dry_run_generation: bool = False) -> dict[str, Any]:
    username = profile_name(profile)
    out = output / username
    out.mkdir(parents=True, exist_ok=True)

    manifest = download_sample(username, out, max_posts=max_posts, login=login)
    frames = extract_keyframes(out / "media", out / "frames", per_video=video_frames)
    records = analyze_images([out / "media", out / "frames"], out, use_clip=use_clip)
    style = save_style_profile(records, out)
    create_generation_brief(username, style, out / "generation_brief.md")

    poses = []
    if use_pose:
        poses = extract_poses([out / "media", out / "frames"], out / "poses")

    concepts = generate_concepts(
        style, out / "concepts.json", count=concept_count,
        base_url=llm_base_url, model=llm_model, api_key=llm_api_key, dry_run=dry_run_llm,
    )

    images = []
    if generate:
        images = generate_images(
            concepts, out / "generated", out / "media",
            model_id=image_model, ip_adapter_repo=ip_adapter_repo, dry_run=dry_run_generation,
        )

    return {"username": username, "output": str(out), "posts": len(manifest),
            "frames": len(frames), "analyzed": len(records), "poses": len(poses),
            "concepts": len(concepts), "generated": len(images)}
