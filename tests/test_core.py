from pathlib import Path

from PIL import Image

from insta_cloner.analyzer import analyze_images, aggregate_style
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
