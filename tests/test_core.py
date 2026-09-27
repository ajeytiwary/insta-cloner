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
    assert get_model("qwen-image-2.1")["kind"]=="comfyui"
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
