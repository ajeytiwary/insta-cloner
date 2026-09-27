from __future__ import annotations

import colorsys
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageStat

from .utils import write_json

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _brightness(img: Image.Image) -> float:
    gray = img.convert("L")
    return float(ImageStat.Stat(gray).mean[0] / 255.0)


def _contrast(img: Image.Image) -> float:
    gray = img.convert("L")
    return float(ImageStat.Stat(gray).stddev[0] / 255.0)


def _saturation(img: Image.Image) -> float:
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


def _dominant_colors(img: Image.Image, n: int = 5) -> list[str]:
    thumb = img.convert("RGB")
    thumb.thumbnail((192, 192))
    pal = thumb.quantize(colors=n, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts = Counter(pal.getdata())
    return [
        "#{:02x}{:02x}{:02x}".format(*rgb)
        for rgb, _ in counts.most_common(n)
    ]


def _orientation(img: Image.Image) -> str:
    w, h = img.size
    ratio = w / max(h, 1)
    if ratio > 1.15:
        return "landscape"
    if ratio < 0.87:
        return "portrait"
    return "square-ish"


def _clip_interrogator():
    from clip_interrogator import Config, Interrogator

    config = Config(
        clip_model_name="ViT-L-14/openai",
        quiet=True,
    )
    return Interrogator(config)


def analyze_images(
    roots: list[Path],
    output_dir: Path,
    use_clip: bool = False,
) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    paths = sorted(set(paths))
    ci = _clip_interrogator() if use_clip else None

    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        record: dict[str, Any] = {
            "path": str(path),
            "width": img.width,
            "height": img.height,
            "orientation": _orientation(img),
            "brightness": round(_brightness(img), 4),
            "contrast": round(_contrast(img), 4),
            "saturation": round(_saturation(img), 4),
            "dominant_colors": _dominant_colors(img),
        }
        if ci is not None:
            record["interrogation"] = ci.interrogate(img)
        records.append(record)

    write_json(output_dir / "per_image_analysis.json", records)
    return records


def aggregate_style(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "sample_size": 0,
            "warning": "No analyzable media was found.",
        }

    orientations = Counter(r["orientation"] for r in records)
    colors = Counter(c for r in records for c in r["dominant_colors"][:3])

    brightness = statistics.fmean(r["brightness"] for r in records)
    contrast = statistics.fmean(r["contrast"] for r in records)
    saturation = statistics.fmean(r["saturation"] for r in records)

    descriptors = [r.get("interrogation") for r in records if r.get("interrogation")]

    return {
        "sample_size": len(records),
        "composition": {
            "orientation_distribution": dict(orientations),
            "dominant_orientation": orientations.most_common(1)[0][0],
        },
        "look": {
            "mean_brightness": round(brightness, 4),
            "mean_contrast": round(contrast, 4),
            "mean_saturation": round(saturation, 4),
            "brightness_character": "dark" if brightness < 0.38 else "bright" if brightness > 0.68 else "mid-key",
            "contrast_character": "soft" if contrast < 0.18 else "punchy" if contrast > 0.28 else "moderate",
            "saturation_character": "muted" if saturation < 0.25 else "vivid" if saturation > 0.5 else "balanced",
        },
        "palette": [color for color, _ in colors.most_common(12)],
        "clip_descriptions": descriptors,
        "usage": {
            "recommended": "Use these recurring characteristics as a visual system, while changing exact subject identity, scene, wardrobe, props and composition.",
            "avoid": "Do not present generated work as originating from the analyzed account or reproduce its creator's identity.",
        },
    }


def save_style_profile(records: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    profile = aggregate_style(records)
    write_json(output_dir / "style_profile.json", profile)
    return profile
