from __future__ import annotations

from pathlib import Path
from typing import Any


def create_generation_brief(username: str, style: dict[str, Any], output_path: Path) -> str:
    look = style.get("look", {})
    comp = style.get("composition", {})
    palette = style.get("palette", [])
    descriptions = style.get("clip_descriptions", [])

    examples = "\n".join(f"- {d}" for d in descriptions[:8]) or "- No CLIP descriptions generated."

    text = f"""# Original-content generation brief inspired by @{username}

## Visual system

- Dominant framing: {comp.get('dominant_orientation', 'unknown')}
- Brightness: {look.get('brightness_character', 'unknown')}
- Contrast: {look.get('contrast_character', 'unknown')}
- Saturation: {look.get('saturation_character', 'unknown')}
- Recurring palette: {', '.join(palette) if palette else 'unknown'}

## Reference descriptions

{examples}

## Local-LLM instruction

Create original Instagram content concepts that use the visual system above as inspiration.

Preserve high-level properties such as:
- lighting character;
- color relationships;
- broad camera/framing tendencies;
- rhythm and visual density;
- broad pose categories when pose references are provided.

Change:
- exact person/identity;
- exact pose coordinates;
- exact location;
- exact wardrobe;
- props;
- scene arrangement;
- text/captions;
- narrative premise.

For each concept return:
1. concept title;
2. image/video prompt;
3. camera/framing;
4. lighting;
5. subject action or pose;
6. wardrobe/props;
7. palette;
8. negative prompt;
9. optional IP-Adapter reference recommendation;
10. optional ControlNet pose-map recommendation.

The resulting content should read as part of your own coherent visual brand, not as a repost or impersonation of @{username}.
"""
    output_path.write_text(text, encoding="utf-8")
    return text
