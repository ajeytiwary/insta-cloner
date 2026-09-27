from __future__ import annotations

from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def extract_poses(roots: list[Path], output_dir: Path) -> list[Path]:
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError as exc:
        raise RuntimeError(
            'Pose extraction requires the optional dependency: pip install -e ".[pose]"'
        ) from exc

    detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    output_dir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    for index, path in enumerate(sorted(set(paths))):
        try:
            img = Image.open(path).convert("RGB")
            pose = detector(img)
        except Exception:
            continue
        out = output_dir / f"{index:04d}_{path.stem}_pose.png"
        pose.save(out)
        made.append(out)

    return made
