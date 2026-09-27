from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .utils import write_json

SYSTEM_PROMPT = """You are an art director. Convert an extracted visual-style profile into ORIGINAL social-media concepts.
Do not reproduce a real person's identity, exact image composition, logos, watermarks, or captions.
Return valid JSON only: an array of objects with keys title, prompt, negative_prompt, camera, lighting, pose, wardrobe_props, palette.
Make each concept materially different in subject action, location, wardrobe/props and exact composition while retaining broad visual-language characteristics."""


def _extract_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("~~~"):
        text = re.sub(r"^~~~(?:json)?\\s*", "", text)
        text = re.sub(r"\\s*~~~$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Local model did not return a JSON array.")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array.")
    return data


def mock_concepts(style: dict[str, Any], count: int) -> list[dict[str, Any]]:
    look = style.get("look", {})
    palette = style.get("palette", [])[:6]
    return [{
        "title": f"Original concept {i + 1}",
        "prompt": f"original editorial social photograph, {look.get('brightness_character', 'mid-key')} lighting, {look.get('contrast_character', 'moderate')} contrast, {look.get('saturation_character', 'balanced')} saturation, new subject and location, concept {i + 1}",
        "negative_prompt": "watermark, logo, text, duplicate composition, recognizable public figure",
        "camera": style.get("composition", {}).get("dominant_orientation", "portrait"),
        "lighting": look.get("brightness_character", "mid-key"),
        "pose": "natural original pose",
        "wardrobe_props": "new wardrobe and props",
        "palette": palette,
    } for i in range(count)]


def generate_concepts(style: dict[str, Any], output_path: Path, count: int = 12,
                      base_url: str = "http://127.0.0.1:8080/v1", model: str = "local-model",
                      api_key: str = "local", dry_run: bool = False) -> list[dict[str, Any]]:
    if dry_run:
        concepts = mock_concepts(style, count)
        write_json(output_path, concepts)
        return concepts
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('Install the LLM extra: pip install -e ".[llm]"') from exc

    payload = {"requested_count": count, "style_profile": style,
               "instruction": "Create the requested number of original concepts. Return JSON only."}
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model, temperature=0.8,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    concepts = _extract_json(response.choices[0].message.content or "")
    if not concepts:
        raise ValueError("Local model returned no concepts.")
    write_json(output_path, concepts[:count])
    return concepts[:count]
