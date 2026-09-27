from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .utils import write_json

AESTHETIC_SCHEMA = """Analyze ONLY visual/aesthetic properties. Return JSON with:
medium, genre, art_style, photographic_style, composition, framing, camera_angle,
lens_character, depth_of_field, lighting, color_palette, color_grading, texture,
subject_styling, pose, environment, visual_motifs, typography, post_processing,
animation_style, camera_motion, subject_motion, transition_style, prompt_terms.
Be precise and descriptive. Do not identify the person or infer sensitive attributes."""


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def analyze_with_vlm(paths: list[Path], output_path: Path,
                     base_url: str = "http://127.0.0.1:8000/v1",
                     model: str = "Qwen/Qwen3-VL-4B-Instruct",
                     api_key: str = "local", limit: int = 12) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install the LLM extra: pip install -e ".[llm]"') from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    results = []
    for path in paths[:limit]:
        response = client.chat.completions.create(
            model=model, temperature=0.1,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": AESTHETIC_SCHEMA},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ]}],
        )
        raw = response.choices[0].message.content or ""
        try:
            start, end = raw.find("{"), raw.rfind("}")
            parsed = json.loads(raw[start:end + 1])
        except Exception:
            parsed = {"raw_analysis": raw}
        results.append({"path": str(path), "analysis": parsed})

    write_json(output_path, results)
    return results


def merge_vlm_style(style: dict[str, Any], analyses: list[dict[str, Any]]) -> dict[str, Any]:
    style = dict(style)
    style["vlm_aesthetic_analysis"] = [x["analysis"] for x in analyses]
    return style
