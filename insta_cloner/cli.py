from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .analyzer import analyze_images, save_style_profile
from .brief import create_generation_brief
from .downloader import download_sample
from .frames import extract_keyframes
from .pose import extract_poses
from .utils import profile_name

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def analyze(
    profile: str = typer.Argument(..., help="Instagram profile URL or username"),
    max_posts: int = typer.Option(12, min=1, max=100),
    output: Path = typer.Option(Path("output")),
    login: str | None = typer.Option(None, help="Instaloader login/session username"),
    clip: bool = typer.Option(False, help="Use CLIP Interrogator"),
    pose: bool = typer.Option(False, help="Extract OpenPose-compatible pose maps"),
    video_frames: int = typer.Option(4, min=1, max=20),
) -> None:
    username = profile_name(profile)
    out = output / username
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Profile:[/bold] @{username}")
    console.print("[bold]1/4[/bold] Downloading bounded sample...")
    manifest = download_sample(username, out, max_posts=max_posts, login=login)
    console.print(f"Downloaded metadata for {len(manifest)} posts.")

    console.print("[bold]2/4[/bold] Extracting video keyframes...")
    frames = extract_keyframes(out / "media", out / "frames", per_video=video_frames)
    console.print(f"Created {len(frames)} keyframes.")

    console.print("[bold]3/4[/bold] Analyzing visual language...")
    records = analyze_images([out / "media", out / "frames"], out, use_clip=clip)
    style = save_style_profile(records, out)
    console.print(f"Analyzed {len(records)} images/frames.")

    if pose:
        console.print("[bold]Pose[/bold] Extracting pose maps...")
        poses = extract_poses([out / "media", out / "frames"], out / "poses")
        console.print(f"Created {len(poses)} pose maps.")

    console.print("[bold]4/4[/bold] Writing local-model brief...")
    create_generation_brief(username, style, out / "generation_brief.md")

    console.print(f"[green]Done.[/green] Results: {out}")


if __name__ == "__main__":
    app()
