from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import instaloader

from .utils import write_json


def _loader(download_dir: Path, login: str | None = None) -> instaloader.Instaloader:
    loader = instaloader.Instaloader(
        dirname_pattern=str(download_dir),
        filename_pattern="{date_utc:%Y-%m-%d_%H-%M-%S}_{shortcode}",
        download_pictures=True,
        download_videos=True,
        download_video_thumbnails=True,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
        post_metadata_txt_pattern="",
        max_connection_attempts=3,
    )
    if login:
        loader.load_session_from_file(login)
    return loader


def download_sample(
    username: str,
    output_dir: Path,
    max_posts: int = 12,
    login: str | None = None,
) -> list[dict[str, Any]]:
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    loader = _loader(media_dir, login=login)
    profile = instaloader.Profile.from_username(loader.context, username)

    manifest: list[dict[str, Any]] = []
    for index, post in enumerate(profile.get_posts()):
        if index >= max_posts:
            break

        loader.download_post(post, target=".")
        manifest.append(
            {
                "shortcode": post.shortcode,
                "date_utc": post.date_utc.isoformat(),
                "typename": post.typename,
                "is_video": bool(post.is_video),
                "caption": post.caption or "",
                "likes": getattr(post, "likes", None),
                "comments": getattr(post, "comments", None),
                "url": f"https://www.instagram.com/p/{post.shortcode}/",
            }
        )

    write_json(output_dir / "sample_manifest.json", manifest)
    return manifest
