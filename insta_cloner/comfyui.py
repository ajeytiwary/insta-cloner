from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import requests


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def system_stats(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/system_stats", timeout=15)
        response.raise_for_status()
        return response.json()

    def queue(self, workflow: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30
        )
        response.raise_for_status()
        payload = response.json()
        if "prompt_id" not in payload:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {payload}")
        return payload["prompt_id"]

    def wait(self, prompt_id: str) -> dict[str, Any]:
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            response = requests.get(
                f"{self.base_url}/history/{prompt_id}", timeout=30
            )
            response.raise_for_status()
            payload = response.json()
            if prompt_id in payload:
                return payload[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"ComfyUI generation timed out: {prompt_id}")

    def download_outputs(self, history: dict[str, Any], output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        made: list[Path] = []
        for node in history.get("outputs", {}).values():
            for image in node.get("images", []):
                params = {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
                response = requests.get(
                    f"{self.base_url}/view", params=params, timeout=120
                )
                response.raise_for_status()
                suffix = Path(image["filename"]).suffix or ".png"
                target = output_dir / f"{len(made)+1:03d}{suffix}"
                target.write_bytes(response.content)
                made.append(target)
        if not made:
            raise RuntimeError("ComfyUI history contained no image outputs.")
        return made


def load_api_workflow(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ComfyUI API workflow must be a JSON object.")
    return payload


def configure_workflow(
    workflow: dict[str, Any],
    prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    result = deepcopy(workflow)
    values = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
    }
    for field, value in values.items():
        target = mapping.get(field)
        if not target:
            continue
        node_id = str(target["node"])
        input_name = target["input"]
        if node_id not in result:
            raise KeyError(f"ComfyUI workflow node {node_id!r} not found for {field}.")
        result[node_id].setdefault("inputs", {})[input_name] = value
    return result


def generate_comfy(
    prompts: list[str],
    output_dir: Path,
    workflow_path: Path,
    mapping: dict[str, Any],
    base_url: str,
    width: int,
    height: int,
    seed: int,
    steps: int,
) -> list[Path]:
    client = ComfyUIClient(base_url)
    client.system_stats()
    template = load_api_workflow(workflow_path)
    made: list[Path] = []
    for index, prompt in enumerate(prompts):
        workflow = configure_workflow(
            template, prompt, width, height, seed + index, steps, mapping
        )
        prompt_id = client.queue(workflow)
        history = client.wait(prompt_id)
        generated = client.download_outputs(
            history, output_dir / f"_comfy_{index+1:03d}"
        )
        for source in generated:
            target = output_dir / f"{len(made)+1:03d}{source.suffix}"
            source.replace(target)
            made.append(target)
    return made
