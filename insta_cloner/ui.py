from __future__ import annotations
import json
import os
from pathlib import Path
import gradio as gr
from .config import load_registry, save_registry
from .pipeline import run_pipeline

def _choices():
    r=load_registry()
    return list(r["image_models"]),list(r["analyzers"])

def run(profile,max_posts,login,clip,pose,use_vlm,vlm_url,vlm_model,
        llm_url,llm_model,llm_key,concepts,generate,image_model,output,
        dry_llm,dry_image,krea_key,ideogram_key):
    if krea_key: os.environ["KREA_API_KEY"]=krea_key
    if ideogram_key: os.environ["IDEOGRAM_API_KEY"]=ideogram_key
    result=run_pipeline(profile=profile,output=Path(output or "output"),max_posts=int(max_posts),
        login=login or None,use_clip=clip,use_pose=pose,concept_count=int(concepts),
        vlm=use_vlm,vlm_url=vlm_url,vlm_model=vlm_model,
        llm_base_url=llm_url,llm_model=llm_model,llm_api_key=llm_key or "local",
        generate=generate,image_model=image_model,dry_run_llm=dry_llm,dry_run_generation=dry_image)
    out=Path(result["output"])
    style=(out/"style_profile.json").read_text(encoding="utf-8") if (out/"style_profile.json").exists() else "{}"
    concepts_json=(out/"concepts.json").read_text(encoding="utf-8") if (out/"concepts.json").exists() else "[]"
    gallery=[str(p) for p in sorted((out/"generated").glob("*.png"))] if (out/"generated").exists() else []
    return json.dumps(result,indent=2),style,concepts_json,gallery

def save_models(raw):
    data=json.loads(raw); path=save_registry(data)
    return f"Saved {path}"

def app():
    images,analyzers=_choices(); registry=json.dumps(load_registry(),indent=2)
    with gr.Blocks(title="Insta Cloner") as demo:
        gr.Markdown("# Insta Cloner\nAnalyze a profile's visual language and generate original inspired concepts/content.")
        with gr.Tab("Create"):
            profile=gr.Textbox(label="Instagram profile URL / username")
            with gr.Row():
                max_posts=gr.Slider(1,50,12,step=1,label="Posts to sample")
                concepts=gr.Slider(1,50,12,step=1,label="Concepts")
                output=gr.Textbox(value="output",label="Output directory")
            login=gr.Textbox(label="Instaloader session username (optional)")
            with gr.Accordion("Aesthetic analysis",open=True):
                with gr.Row():
                    clip=gr.Checkbox(label="CLIP Interrogator")
                    pose=gr.Checkbox(label="Pose maps")
                    use_vlm=gr.Checkbox(value=True,label="Local VLM aesthetic analysis")
                vlm_url=gr.Textbox(value="http://127.0.0.1:8000/v1",label="VLM OpenAI-compatible URL")
                vlm_model=gr.Dropdown(choices=[load_registry()["analyzers"][x]["model_id"] for x in analyzers],
                                      value="Qwen/Qwen3-VL-4B-Instruct",allow_custom_value=True,label="VLM model ID")
            with gr.Accordion("Concept LLM",open=True):
                llm_url=gr.Textbox(value="http://127.0.0.1:8080/v1",label="LLM URL")
                llm_model=gr.Textbox(value="local-model",label="LLM model name")
                llm_key=gr.Textbox(value="local",type="password",label="LLM API key")
            with gr.Accordion("Image generation",open=True):
                generate=gr.Checkbox(label="Generate images")
                image_model=gr.Dropdown(choices=images,value="sdxl",allow_custom_value=True,label="Image model preset")
                with gr.Row():
                    krea_key=gr.Textbox(type="password",label="Krea API key (optional)")
                    ideogram_key=gr.Textbox(type="password",label="Ideogram API key (optional)")
            with gr.Accordion("Testing"):
                dry_llm=gr.Checkbox(label="Dry-run concept LLM")
                dry_image=gr.Checkbox(label="Dry-run image generation")
            go=gr.Button("Analyze & Generate",variant="primary")
            status=gr.Code(label="Run result",language="json")
            with gr.Tab("Style profile"): style=gr.Code(language="json")
            with gr.Tab("Concepts"): concept_out=gr.Code(language="json")
            gallery=gr.Gallery(label="Generated images",columns=4)
            go.click(run,[profile,max_posts,login,clip,pose,use_vlm,vlm_url,vlm_model,llm_url,llm_model,llm_key,
                          concepts,generate,image_model,output,dry_llm,dry_image,krea_key,ideogram_key],
                     [status,style,concept_out,gallery])
        with gr.Tab("Model configuration"):
            gr.Markdown("Edit presets or add your own providers/model IDs. Secrets are not stored here.")
            editor=gr.Code(value=registry,language="json",label="~/.config/insta-cloner/models.json")
            save=gr.Button("Save model configuration")
            saved=gr.Textbox(label="Status")
            save.click(save_models,[editor],[saved])
    return demo

def launch(host="127.0.0.1",port=7860,share=False):
    app().launch(server_name=host,server_port=port,share=share)
