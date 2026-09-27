from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from .models import MODELS, ANALYZERS

CONFIG_PATH=Path.home()/".config"/"insta-cloner"/"models.json"

def load_registry(path: Path=CONFIG_PATH) -> dict[str,Any]:
    data={"image_models":deepcopy(MODELS),"analyzers":deepcopy(ANALYZERS)}
    if path.exists():
        custom=json.loads(path.read_text(encoding="utf-8"))
        data["image_models"].update(custom.get("image_models",{}))
        data["analyzers"].update(custom.get("analyzers",{}))
    return data

def save_registry(data: dict[str,Any], path: Path=CONFIG_PATH) -> Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    return path

def resolve_image_model(name: str, custom_model_id: str|None=None,
                        custom_kind: str="diffusers") -> dict[str,Any]:
    if custom_model_id:
        return {"kind":custom_kind,"model_id":custom_model_id,"local":custom_kind=="diffusers"}
    registry=load_registry()["image_models"]
    if name not in registry: raise ValueError(f"Unknown image model: {name}")
    return registry[name]
