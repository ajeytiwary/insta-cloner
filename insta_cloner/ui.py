from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from .config import load_registry, save_registry
from .pipeline import run_pipeline
from .upstream import upstream_status


def _choices():
    registry = load_registry()
    return list(registry["image_models"]), list(registry["analyzers"])


def _upstream_table():
    return [
        [item.name, "yes" if item.installed else "no", item.version or ""]
        for item in upstream_status()
    ]


def run(
    profile,
    max_posts,
    login,
    clip,
    pose,
    use_vlm,
    vlm_url,
    vlm_model,
    llm_url,
    llm_model,
    llm_key,
    concepts,
    generate,
    image_model,
    output,
    dry_llm,
    dry_image,
    krea_key,
    ideogram_key,
    ip_adapter,
    ip_scale,
    ip_repo,
    ip_weight,
    controlnet,
    control_scale,
    control_model,
):
    if krea_key:
        os.environ["KREA_API_KEY"] = krea_key
    if ideogram_key:
        os.environ["IDEOGRAM_API_KEY"] = ideogram_key

    result = run_pipeline(
        profile=profile,
        output=Path(output or "output"),
        max_posts=int(max_posts),
        login=login or None,
        use_clip=clip,
        use_pose=pose,
        concept_count=int(concepts),
        vlm=use_vlm,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        llm_base_url=llm_url,
        llm_model=llm_model,
        llm_api_key=llm_key or "local",
        generate=generate,
        image_model=image_model,
        dry_run_llm=dry_llm,
        dry_run_generation=dry_image,
        ip_adapter=ip_adapter,
        ip_scale=float(ip_scale),
        ip_repo=ip_repo,
        ip_weight=ip_weight,
        controlnet=controlnet,
        control_scale=float(control_scale),
        control_model=control_model,
    )

    out = Path(result["output"])
    style_path = out / "style_profile.json"
    concepts_path = out / "concepts.json"
    generated_dir = out / "generated"

    style = style_path.read_text(encoding="utf-8") if style_path.exists() else "{}"
    concepts_json = (
        concepts_path.read_text(encoding="utf-8") if concepts_path.exists() else "[]"
    )
    gallery = (
        [str(path) for path in sorted(generated_dir.glob("*.png"))]
        if generated_dir.exists()
        else []
    )
    return json.dumps(result, indent=2), style, concepts_json, gallery


def save_models(raw):
    data = json.loads(raw)
    path = save_registry(data)
    return f"Saved {path}"


def app():
    images, analyzers = _choices()
    loaded = load_registry()
    registry_json = json.dumps(loaded, indent=2)
    analyzer_models = [loaded["analyzers"][name]["model_id"] for name in analyzers]

    with gr.Blocks(title="Insta Cloner") as demo:
        gr.Markdown(
            "# Insta Cloner\n"
            "Built on upstream Instaloader, CLIP Interrogator, ControlNet tooling "
            "and Diffusers IP-Adapter support."
        )

        with gr.Tab("Create"):
            profile = gr.Textbox(label="Instagram profile URL / username")
            with gr.Row():
                max_posts = gr.Slider(1, 50, 12, step=1, label="Posts to sample")
                concepts = gr.Slider(1, 50, 12, step=1, label="Concepts")
                output = gr.Textbox(value="output", label="Output directory")

            login = gr.Textbox(label="Instaloader session username (optional)")

            with gr.Accordion("Aesthetic analysis", open=True):
                with gr.Row():
                    clip = gr.Checkbox(label="CLIP Interrogator (upstream)")
                    pose = gr.Checkbox(label="ControlNet/OpenPose maps (upstream)")
                    use_vlm = gr.Checkbox(
                        value=True, label="Local VLM aesthetic analysis"
                    )
                vlm_url = gr.Textbox(
                    value="http://127.0.0.1:8000/v1",
                    label="VLM OpenAI-compatible URL",
                )
                vlm_model = gr.Dropdown(
                    choices=analyzer_models,
                    value="Qwen/Qwen3-VL-4B-Instruct",
                    allow_custom_value=True,
                    label="VLM model ID",
                )

            with gr.Accordion("Concept LLM", open=True):
                llm_url = gr.Textbox(
                    value="http://127.0.0.1:8080/v1", label="LLM URL"
                )
                llm_model = gr.Textbox(value="local-model", label="LLM model name")
                llm_key = gr.Textbox(
                    value="local", type="password", label="LLM API key"
                )

            with gr.Accordion("Image generation", open=True):
                generate = gr.Checkbox(label="Generate images")
                image_model = gr.Dropdown(
                    choices=images,
                    value="sdxl",
                    allow_custom_value=True,
                    label="Image model preset",
                )

                with gr.Accordion("Style transfer - IP-Adapter", open=True):
                    ip_adapter = gr.Checkbox(label="Enable IP-Adapter")
                    ip_scale = gr.Slider(
                        0, 1, 0.45, step=0.05, label="Style/reference strength"
                    )
                    ip_repo = gr.Textbox(
                        value="h94/IP-Adapter", label="IP-Adapter repository"
                    )
                    ip_weight = gr.Textbox(
                        value="ip-adapter_sdxl.bin", label="IP-Adapter weight"
                    )

                with gr.Accordion("Pose / structure - ControlNet", open=True):
                    controlnet = gr.Checkbox(label="Enable OpenPose ControlNet")
                    control_scale = gr.Slider(
                        0,
                        1.5,
                        0.8,
                        step=0.05,
                        label="ControlNet conditioning strength",
                    )
                    control_model = gr.Textbox(
                        value="thibaud/controlnet-openpose-sdxl-1.0",
                        label="ControlNet model",
                    )

                with gr.Row():
                    krea_key = gr.Textbox(
                        type="password", label="Krea API key (optional)"
                    )
                    ideogram_key = gr.Textbox(
                        type="password", label="Ideogram API key (optional)"
                    )

            with gr.Accordion("Testing"):
                dry_llm = gr.Checkbox(label="Dry-run concept LLM")
                dry_image = gr.Checkbox(label="Dry-run image generation")

            go = gr.Button("Analyze & Generate", variant="primary")
            status = gr.Code(label="Run result", language="json")
            with gr.Tab("Style profile"):
                style = gr.Code(language="json")
            with gr.Tab("Concepts"):
                concept_out = gr.Code(language="json")
            gallery = gr.Gallery(label="Generated images", columns=4)

            inputs = [
                profile,
                max_posts,
                login,
                clip,
                pose,
                use_vlm,
                vlm_url,
                vlm_model,
                llm_url,
                llm_model,
                llm_key,
                concepts,
                generate,
                image_model,
                output,
                dry_llm,
                dry_image,
                krea_key,
                ideogram_key,
                ip_adapter,
                ip_scale,
                ip_repo,
                ip_weight,
                controlnet,
                control_scale,
                control_model,
            ]
            go.click(run, inputs, [status, style, concept_out, gallery])

        with gr.Tab("Model configuration"):
            gr.Markdown(
                "Edit presets or add your own providers/model IDs. "
                "Secrets are not stored here."
            )
            editor = gr.Code(
                value=registry_json,
                language="json",
                label="~/.config/insta-cloner/models.json",
            )
            save = gr.Button("Save model configuration")
            saved = gr.Textbox(label="Status")
            save.click(save_models, [editor], [saved])

        with gr.Tab("Upstream components"):
            gr.Markdown(
                "External projects/libraries used directly by the orchestration layer."
            )
            table = gr.Dataframe(
                headers=["Component", "Installed", "Version"],
                value=_upstream_table(),
                interactive=False,
            )
            refresh = gr.Button("Refresh")
            refresh.click(_upstream_table, outputs=[table])

    return demo


def launch(host="127.0.0.1", port=7860, share=False):
    app().launch(server_name=host, server_port=port, share=share)
