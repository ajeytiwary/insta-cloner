from __future__ import annotations

MODELS = {
    "sdxl": {
        "kind": "diffusers", "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local": True, "recommended_12gb": True,
    },
    "qwen-image-2.1": {
        "kind": "comfyui", "model_id": "Comfy-Org/Qwen-Image-2.1",
        "local": True, "recommended_12gb": False,
        "note": "Use the INT8 ComfyUI workflow on constrained GPUs; full BF16 is too large for 12 GB VRAM.",
    },
    "krea2-turbo": {
        "kind": "diffusers", "model_id": "krea/Krea-2-Turbo",
        "local": True, "recommended_12gb": False,
        "steps": 8, "guidance_scale": 0.0,
    },
    "krea2-api-medium": {
        "kind": "krea_api", "endpoint": "https://api.krea.ai/generate/image/krea/krea-2/medium",
        "local": False,
    },
    "krea2-api-large": {
        "kind": "krea_api", "endpoint": "https://api.krea.ai/generate/image/krea/krea-2/large",
        "local": False,
    },
    "ideogram4-nf4": {
        "kind": "diffusers", "model_id": "ideogram-ai/ideogram-4-nf4",
        "local": True, "recommended_12gb": False,
        "note": "Gated, non-commercial open weights; accept the Hugging Face terms first.",
    },
    "ideogram4-api": {
        "kind": "ideogram_api", "endpoint": "https://api.ideogram.ai/v1/ideogram-v4/generate",
        "local": False,
    },
}

ANALYZERS = {
    "qwen3-vl-4b": {
        "kind": "openai_vision",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "recommended_12gb": True,
        "note": "Serve quantized with an OpenAI-compatible multimodal server such as vLLM.",
    },
    "openai-compatible-vlm": {
        "kind": "openai_vision", "model_id": "local-vlm", "recommended_12gb": True,
    },
}


def get_model(name: str) -> dict:
    if name not in MODELS:
        raise ValueError(f"Unknown image model {name!r}. Available: {', '.join(MODELS)}")
    return MODELS[name]


def get_analyzer(name: str) -> dict:
    if name not in ANALYZERS:
        raise ValueError(f"Unknown analyzer {name!r}. Available: {', '.join(ANALYZERS)}")
    return ANALYZERS[name]
