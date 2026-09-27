from pathlib import Path
from PIL import Image
from insta_cloner.analyzer import analyze_images, aggregate_style
from insta_cloner.generator import generate_images
from insta_cloner.llm import _extract_json, generate_concepts
from insta_cloner.models import get_model, get_analyzer
from insta_cloner.utils import profile_name
from insta_cloner.vlm import merge_vlm_style

def test_profile_name():
    assert profile_name("https://www.instagram.com/example/")=="example"
    assert profile_name("@example")=="example"

def test_analyzer(tmp_path: Path):
    media=tmp_path/"media"; media.mkdir()
    Image.new("RGB",(300,500),(120,80,40)).save(media/"a.jpg")
    records=analyze_images([media],tmp_path,use_clip=False)
    profile=aggregate_style(records)
    assert len(records)==1 and profile["composition"]["dominant_orientation"]=="portrait"

def test_llm_json_and_dry_run(tmp_path: Path):
    assert _extract_json('[{"title":"x"}]')[0]["title"]=="x"
    concepts=generate_concepts({"look":{},"composition":{},"palette":[]},tmp_path/"concepts.json",count=3,dry_run=True)
    assert len(concepts)==3

def test_generation_dry_run(tmp_path: Path):
    media=tmp_path/"media"; media.mkdir()
    made=generate_images([{"title":"x","prompt":"original portrait"}],tmp_path/"generated",media,
                         model_name="qwen-image-2.1",dry_run=True,width=128,height=128)
    assert len(made)==1 and made[0].exists()

def test_registry():
    qwen=get_model("qwen-image-2.1")
    assert qwen["kind"]=="diffusers"
    assert qwen["pipeline"]=="qwen_image_2_1"
    assert qwen["native"] is True
    assert qwen["workflow_path"].endswith("qwen_image_2_1_t2i_api.json")
    assert get_model("krea2-turbo")["model_id"]=="krea/Krea-2-Turbo"
    assert get_model("ideogram4-nf4")["kind"]=="diffusers"
    assert get_analyzer("qwen3-vl-4b")["recommended_12gb"] is True

def test_vlm_merge():
    result=merge_vlm_style({"palette":[]},[{"analysis":{"art_style":"editorial"}}])
    assert result["vlm_aesthetic_analysis"][0]["art_style"]=="editorial"


def test_custom_registry(tmp_path: Path):
    from insta_cloner.config import load_registry, save_registry, resolve_image_model
    path=tmp_path/"models.json"
    save_registry({"image_models":{"custom":{"kind":"diffusers","model_id":"org/model"}},"analyzers":{}},path)
    data=load_registry(path)
    assert data["image_models"]["custom"]["model_id"]=="org/model"
    assert resolve_image_model("ignored","org/other")["model_id"]=="org/other"

def test_ui_import():
    from insta_cloner.ui import app
    assert callable(app)


def test_upstream_status_shape():
    from insta_cloner.upstream import upstream_status
    rows=upstream_status()
    names={x.name for x in rows}
    assert "Instaloader" in names
    assert "CLIP Interrogator" in names
    assert "ControlNet Aux" in names
    assert "Diffusers / IP-Adapter" in names


def test_conditioning_dry_run_does_not_load_models(tmp_path: Path):
    media=tmp_path/"media"; media.mkdir()
    Image.new("RGB",(64,64),(1,2,3)).save(media/"ref.jpg")
    made=generate_images([{"title":"x","prompt":"new scene"}],tmp_path/"out",media,model_name="sdxl",
        ip_adapter=True,controlnet=True,dry_run=True,width=64,height=64)
    assert made[0].exists()


def test_comfy_workflow_mapping():
    from insta_cloner.comfyui import configure_workflow
    workflow={
        "1":{"class_type":"Text","inputs":{"text":"old"}},
        "2":{"class_type":"Latent","inputs":{"width":1,"height":1}},
        "3":{"class_type":"Sampler","inputs":{"seed":0,"steps":1}},
    }
    mapping={
        "prompt":{"node":"1","input":"text"},
        "width":{"node":"2","input":"width"},
        "height":{"node":"2","input":"height"},
        "seed":{"node":"3","input":"seed"},
        "steps":{"node":"3","input":"steps"},
    }
    out=configure_workflow(workflow,"hello",768,1024,42,25,mapping)
    assert out["1"]["inputs"]["text"]=="hello"
    assert out["2"]["inputs"]["width"]==768
    assert out["2"]["inputs"]["height"]==1024
    assert out["3"]["inputs"]["seed"]==42
    assert out["3"]["inputs"]["steps"]==25
    assert workflow["1"]["inputs"]["text"]=="old"


def test_runtime_status_shape():
    from insta_cloner.runtime import runtime_status
    status=runtime_status()
    assert "cuda" in status
    assert "vram_gb" in status
    assert "diffusers" in status


def test_model_readiness_without_gpu_is_explanatory():
    from insta_cloner.runtime import model_readiness
    result=model_readiness({"kind":"diffusers","local":True,"recommended_12gb":False})
    assert "runtime" in result
    assert isinstance(result["warnings"],list)
