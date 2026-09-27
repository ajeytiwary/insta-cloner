from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import load_registry
from .pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("ui")
def ui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the local Gradio UI."""
    from .ui import launch
    launch(host, port, share)


@app.command("models")
def models_cmd() -> None:
    """Show bundled image-generator and analyzer presets."""
    console.print("[bold]Image models[/bold]")
    registry = load_registry()
    for name, spec in registry["image_models"].items():
        console.print(f"{name}: {spec['kind']} | {spec.get('model_id', spec.get('endpoint'))}")
    console.print("[bold]Aesthetic analyzers[/bold]")
    for name, spec in registry["analyzers"].items():
        console.print(f"{name}: {spec['model_id']}")


@app.command("gpu-smoke")
def gpu_smoke(
    model: str = typer.Option("qwen-image-2.1", help="Image model preset"),
    output: Path = typer.Option(Path("/data/smoke"), help="Smoke-test output directory"),
    width: int = typer.Option(512, min=256),
    height: int = typer.Option(512, min=256),
    steps: int = typer.Option(4, min=1),
) -> None:
    """Load a native image model on CUDA and generate one small validation image."""
    from .generator import generate_images
    from .runtime import runtime_status
    status = runtime_status()
    console.print(status)
    if not status["cuda"]:
        raise typer.Exit(code=2)
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    made = generate_images(
        [{"title": "gpu-smoke", "prompt": "A small green leaf on a clean white background, studio photograph"}],
        output / "generated",
        media,
        model_name=model,
        width=width,
        height=height,
        seed=42,
        steps=steps,
        dry_run=False,
        backend="native",
        low_vram=True,
    )
    if not made or not made[0].exists() or made[0].stat().st_size == 0:
        raise RuntimeError("GPU smoke test did not produce a valid image.")
    console.print(f"[green]GPU smoke test passed[/green]: {made[0]}")


@app.command()
def clone(
    profile: str = typer.Argument(..., help="Instagram profile URL or username"),
    max_posts: int = typer.Option(12, min=1, max=100),
    output: Path = Path("output"),
    login: str | None = None,
    clip: bool = False,
    pose: bool = False,
    concepts: int = typer.Option(12, min=1, max=100),
    vlm: bool = typer.Option(False, help="Use a local/OpenAI-compatible vision LLM"),
    vlm_url: str = "http://127.0.0.1:8000/v1",
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    llm_url: str = "http://127.0.0.1:8080/v1",
    llm_model: str = "local-model",
    llm_api_key: str = "local",
    generate: bool = False,
    image_model: str = typer.Option("sdxl", help="Run 'insta-cloner models' for choices"),
    ip_adapter: bool = typer.Option(False, help="Condition generation on sampled reference images"),
    ip_scale: float = typer.Option(0.45, min=0.0, max=1.0),
    ip_repo: str = "h94/IP-Adapter",
    ip_weight: str = "ip-adapter_sdxl.bin",
    controlnet: bool = typer.Option(False, help="Use extracted OpenPose maps during generation"),
    control_scale: float = typer.Option(0.8, min=0.0, max=1.5),
    control_model: str = "thibaud/controlnet-openpose-sdxl-1.0",
    dry_run_llm: bool = False,
    dry_run_generation: bool = False,
) -> None:
    result = run_pipeline(
        profile=profile,
        output=output,
        max_posts=max_posts,
        login=login,
        use_clip=clip,
        use_pose=pose,
        concept_count=concepts,
        vlm=vlm,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        llm_base_url=llm_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        generate=generate,
        image_model=image_model,
        ip_adapter=ip_adapter,
        ip_scale=ip_scale,
        ip_repo=ip_repo,
        ip_weight=ip_weight,
        controlnet=controlnet,
        control_scale=control_scale,
        control_model=control_model,
        dry_run_llm=dry_run_llm,
        dry_run_generation=dry_run_generation,
    )
    console.print("[green]Complete[/green]")
    for key, value in result.items():
        console.print(f"{key}: {value}")


@app.command()
def analyze(
    profile: str = typer.Argument(...),
    max_posts: int = 12,
    output: Path = Path("output"),
    login: str | None = None,
    clip: bool = False,
    pose: bool = False,
    vlm: bool = False,
    vlm_url: str = "http://127.0.0.1:8000/v1",
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
) -> None:
    console.print(
        run_pipeline(
            profile=profile,
            output=output,
            max_posts=max_posts,
            login=login,
            use_clip=clip,
            use_pose=pose,
            vlm=vlm,
            vlm_url=vlm_url,
            vlm_model=vlm_model,
            concept_count=1,
            dry_run_llm=True,
        )
    )


if __name__ == "__main__":
    app()
