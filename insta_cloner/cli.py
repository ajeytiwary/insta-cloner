from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console

from .pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def clone(
    profile: str = typer.Argument(..., help="Instagram profile URL or username"),
    max_posts: int = typer.Option(12, min=1, max=100),
    output: Path = typer.Option(Path("output")),
    login: str | None = typer.Option(None),
    clip: bool = typer.Option(False, help="Enable CLIP Interrogator"),
    pose: bool = typer.Option(False, help="Extract pose maps"),
    concepts: int = typer.Option(12, min=1, max=100),
    llm_url: str = typer.Option("http://127.0.0.1:8080/v1"),
    llm_model: str = typer.Option("local-model"),
    llm_api_key: str = typer.Option("local"),
    generate: bool = typer.Option(False, help="Generate images after concepts"),
    image_model: str = typer.Option("stabilityai/stable-diffusion-xl-base-1.0"),
    ip_adapter_repo: str | None = typer.Option(None, help="e.g. h94/IP-Adapter"),
    dry_run_llm: bool = typer.Option(False),
    dry_run_generation: bool = typer.Option(False),
) -> None:
    result = run_pipeline(
        profile=profile, output=output, max_posts=max_posts, login=login,
        use_clip=clip, use_pose=pose, concept_count=concepts,
        llm_base_url=llm_url, llm_model=llm_model, llm_api_key=llm_api_key,
        generate=generate, image_model=image_model, ip_adapter_repo=ip_adapter_repo,
        dry_run_llm=dry_run_llm, dry_run_generation=dry_run_generation,
    )
    console.print("[green]Complete[/green]")
    for key, value in result.items():
        console.print(f"{key}: {value}")


@app.command()
def analyze(profile: str = typer.Argument(...), max_posts: int = 12,
            output: Path = Path("output"), login: str | None = None,
            clip: bool = False, pose: bool = False) -> None:
    result = run_pipeline(
        profile=profile, output=output, max_posts=max_posts, login=login,
        use_clip=clip, use_pose=pose, concept_count=1, dry_run_llm=True,
    )
    console.print(result)


if __name__ == "__main__":
    app()
