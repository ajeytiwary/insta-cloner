from __future__ import annotations

from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def extract_keyframes(media_dir: Path, frames_dir: Path, per_video: int = 4) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for video in sorted(p for p in media_dir.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES):
        cap = cv2.VideoCapture(str(video))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            cap.release()
            continue

        positions = [
            int((frame_count - 1) * i / max(per_video - 1, 1))
            for i in range(per_video)
        ]
        for i, pos in enumerate(positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if not ok:
                continue
            out = frames_dir / f"{video.stem}_frame_{i:02d}.jpg"
            cv2.imwrite(str(out), frame)
            created.append(out)
        cap.release()

    return created
