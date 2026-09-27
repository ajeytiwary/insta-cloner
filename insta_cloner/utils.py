from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def profile_name(value: str) -> str:
    value = value.strip().rstrip("/")
    m = re.search(r"instagram\.com/([^/?#]+)", value, flags=re.I)
    if m:
        return m.group(1).lstrip("@")
    return value.lstrip("@")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
