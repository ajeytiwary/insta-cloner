from pathlib import Path
from PIL import Image

from insta_cloner.analyzer import analyze_images, aggregate_style
from insta_cloner.generator import generate_images
from insta_cloner.llm import _extract_json, generate_concepts
from insta_cloner.utils import profile_name


def test_profile_name():
    assert profile_name("https://www.instagram.com/example/") == "example"
    assert profile_name("@example") == "example"


def test_analyzer(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    Image.new("RGB", (300, 500), (120, 80, 40)).save(media / "a.jpg")
    records = analyze_images([media], tmp_path, use_clip=False)
    assert len(records) == 1
    profile = aggregate_style(records)
    assert profile["sample_size"] == 1
    assert profile["composition"]["dominant_orientation"] == "portrait"


def test_llm_json_and_dry_run(tmp_path: Path):
    assert _extract_json('[{"title":"x"}]')[0]["title"] == "x"
    style = {"look": {}, "composition": {}, "palette": ["#111111"]}
    concepts = generate_concepts(style, tmp_path / "concepts.json", count=3, dry_run=True)
    assert len(concepts) == 3
    assert (tmp_path / "concepts.json").exists()


def test_generation_dry_run(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    concepts = [{"title": "x", "prompt": "original portrait"}]
    made = generate_images(concepts, tmp_path / "generated", media, dry_run=True, width=128, height=128)
    assert len(made) == 1
    assert made[0].exists()
