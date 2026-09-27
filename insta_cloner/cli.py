from __future__ import annotations
from pathlib import Path
import typer
from rich.console import Console
from .models import MODELS, ANALYZERS
from .pipeline import run_pipeline

app=typer.Typer(no_args_is_help=True); console=Console()

@app.command("models")
def models_cmd() -> None:
    console.print("[bold]Image models[/bold]")
    for n,s in MODELS.items(): console.print(f"{n}: {s['kind']} | {s.get('model_id',s.get('endpoint'))}")
    console.print("[bold]Aesthetic analyzers[/bold]")
    for n,s in ANALYZERS.items(): console.print(f"{n}: {s['model_id']}")

@app.command()
def clone(profile: str=typer.Argument(...), max_posts: int=12, output: Path=Path("output"),
          login: str|None=None, clip: bool=False, pose: bool=False, concepts: int=12,
          vlm: bool=typer.Option(False,help="Use local vision LLM for aesthetic analysis"),
          vlm_url: str="http://127.0.0.1:8000/v1",
          vlm_model: str="Qwen/Qwen3-VL-4B-Instruct",
          llm_url: str="http://127.0.0.1:8080/v1", llm_model: str="local-model",
          llm_api_key: str="local", generate: bool=False,
          image_model: str=typer.Option("sdxl",help="Run 'insta-cloner models' for choices"),
          dry_run_llm: bool=False, dry_run_generation: bool=False) -> None:
    result=run_pipeline(profile=profile,output=output,max_posts=max_posts,login=login,use_clip=clip,
        use_pose=pose,concept_count=concepts,vlm=vlm,vlm_url=vlm_url,vlm_model=vlm_model,
        llm_base_url=llm_url,llm_model=llm_model,llm_api_key=llm_api_key,generate=generate,
        image_model=image_model,dry_run_llm=dry_run_llm,dry_run_generation=dry_run_generation)
    console.print("[green]Complete[/green]")
    for k,v in result.items(): console.print(f"{k}: {v}")

@app.command()
def analyze(profile: str=typer.Argument(...), max_posts: int=12, output: Path=Path("output"),
            login: str|None=None, clip: bool=False, pose: bool=False, vlm: bool=False,
            vlm_url: str="http://127.0.0.1:8000/v1",
            vlm_model: str="Qwen/Qwen3-VL-4B-Instruct") -> None:
    console.print(run_pipeline(profile=profile,output=output,max_posts=max_posts,login=login,
        use_clip=clip,use_pose=pose,vlm=vlm,vlm_url=vlm_url,vlm_model=vlm_model,
        concept_count=1,dry_run_llm=True))

if __name__=="__main__": app()
