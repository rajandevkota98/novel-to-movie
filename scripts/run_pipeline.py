"""Command-line interface for the Novel-to-Movie AI Pipeline.

Follows strict functional programming conventions and is run via 'uv run'.
"""

from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.table import Table

from pipeline.assemble import build_concat_list_file, concatenate_clips, mix_scene_audio
from pipeline.bibles import export_character_sheet_json, load_character_bibles_from_yaml, load_voice_bibles_from_yaml
from pipeline.breakdown import breakdown_story_and_characters
from pipeline.generate import process_single_shot
from pipeline.models import CharacterProfile
from pipeline.state_manager import (
    get_character_dict,
    get_shot,
    init_database,
    list_characters,
    list_shots,
    save_characters_batch,
    save_shots_batch,
)

app = typer.Typer(help="Novel-to-Movie AI Pipeline CLI")
console = Console()


def _load_config(config_path: str = "config.yaml") -> dict:
    """Reads configuration YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def init_db(config: str = "config.yaml"):
    """Initializes the SQLite shot database."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]
    init_database(db_path)
    console.print(f"[green]✓ Database initialized at {db_path}[/green]")


@app.command()
def breakdown(
    chapter_file: str = typer.Argument(..., help="Path to raw chapter text file"),
    chapter_num: int = typer.Option(1, help="Chapter number"),
    config: str = "config.yaml",
):
    """Dynamically discovers characters and parses shots from ANY novel text."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]
    init_database(db_path)

    text_path = Path(chapter_file)
    if not text_path.exists():
        console.print(f"[red]Error: file {chapter_file} not found.[/red]")
        raise typer.Exit(1)

    with open(text_path, "r", encoding="utf-8") as f:
        chapter_text = f.read()

    console.print(f"[cyan]Analyzing story with model {cfg['models']['breakdown']}...[/cyan]")
    result = breakdown_story_and_characters(
        story_text=chapter_text,
        chapter_num=chapter_num,
        model=cfg["models"]["breakdown"],
    )

    save_characters_batch(db_path, result.characters)
    save_shots_batch(db_path, result.shots)
    export_character_sheet_json(result.characters, "output/character_sheet.json")

    # Display Character Sheet
    char_table = Table(title="🎭 Discovered Character Sheet")
    char_table.add_column("ID", style="cyan")
    char_table.add_column("Name", style="green")
    char_table.add_column("Gender/Age", style="yellow")
    char_table.add_column("Voice Timbre", style="magenta")
    char_table.add_column("Appearance Spec", style="white")

    for c in result.characters:
        char_table.add_row(
            c.char_id,
            c.name,
            f"{c.gender}, {c.age}",
            c.voice_timbre,
            c.appearance_description[:60] + "..." if len(c.appearance_description) > 60 else c.appearance_description,
        )
    console.print(char_table)

    console.print(f"[green]✓ Generated {len(result.characters)} character profiles and {len(result.shots)} shots.[/green]")


@app.command()
def render_shot(
    shot_id: str = typer.Argument(..., help="Shot ID to render, e.g. ch01_sc01_sh001"),
    config: str = "config.yaml",
    mock: bool = typer.Option(True, help="Run with local mock rendering without remote GPU"),
):
    """Executes the rendering and QA loop for a specific shot."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]
    shot = get_shot(db_path, shot_id)

    if not shot:
        console.print(f"[red]Shot {shot_id} not found in database.[/red]")
        raise typer.Exit(1)

    char_dict = get_character_dict(db_path)

    console.print(f"[cyan]Processing shot {shot_id} (Status: {shot.status})...[/cyan]")
    result = process_single_shot(
        shot=shot,
        db_path=db_path,
        visual_profiles=char_dict,
        voice_profiles=char_dict,
        output_dir=cfg["project"]["output_dir"],
        retry_cap=cfg["quality_gate"]["retry_cap"],
        mock_rendering=mock,
    )
    console.print(f"[green]✓ Shot {shot_id} finished with status: {result.status}[/green]")


@app.command()
def render_all(
    config: str = "config.yaml",
    mock: bool = typer.Option(True, help="Run with local mock rendering"),
):
    """Iterates through all queued shots and executes the rendering pipeline."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]
    shots = list_shots(db_path, status="queued")

    if not shots:
        console.print("[yellow]No queued shots found.[/yellow]")
        return

    char_dict = get_character_dict(db_path)

    console.print(f"[cyan]Rendering {len(shots)} shots with dynamic character sheets...[/cyan]")
    for shot in shots:
        process_single_shot(
            shot=shot,
            db_path=db_path,
            visual_profiles=char_dict,
            voice_profiles=char_dict,
            output_dir=cfg["project"]["output_dir"],
            retry_cap=cfg["quality_gate"]["retry_cap"],
            mock_rendering=mock,
        )
    console.print("[green]✓ Completed rendering queue pass.[/green]")


@app.command()
def assemble(
    config: str = "config.yaml",
    output_filename: str = "final_movie.mp4",
):
    """Concatenates all approved shots into the final movie using FFmpeg."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]
    shots = list_shots(db_path, status="passed")

    if not shots:
        console.print("[red]No passed shots available to assemble.[/red]")
        raise typer.Exit(1)

    video_files = [s.synced_path for s in shots if s.synced_path and Path(s.synced_path).exists()]
    if not video_files:
        console.print("[red]No valid synced video files found for passed shots.[/red]")
        raise typer.Exit(1)

    out_dir = Path(cfg["project"]["output_dir"]) / "final"
    concat_txt = out_dir / "concat_list.txt"
    final_video = out_dir / output_filename

    console.print(f"[cyan]Assembling {len(video_files)} clips...[/cyan]")
    build_concat_list_file(video_files, str(concat_txt))
    concatenate_clips(str(concat_txt), str(final_video))
    console.print(f"[green]✓ Final video successfully assembled at: {final_video.resolve()}[/green]")


@app.command()
def status(config: str = "config.yaml"):
    """Displays current shot queue status and progression."""
    cfg = _load_config(config)
    db_path = cfg["storage"]["database_path"]

    # Display Discovered Characters
    chars = list_characters(db_path)
    if chars:
        char_table = Table(title="🎭 Active Character Sheet")
        char_table.add_column("ID", style="cyan")
        char_table.add_column("Name", style="green")
        char_table.add_column("Gender/Age", style="yellow")
        char_table.add_column("Voice Timbre", style="magenta")
        for c in chars:
            char_table.add_row(c.char_id, c.name, f"{c.gender}, {c.age}", c.voice_timbre)
        console.print(char_table)

    shots = list_shots(db_path)
    if not shots:
        console.print("[yellow]No shots found in queue. Run breakdown first.[/yellow]")
        return

    table = Table(title="🎞️ Production Shot Queue")
    table.add_column("Shot ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Speaker", style="yellow")
    table.add_column("Lip-Sync", style="blue")
    table.add_column("Status", style="green")
    table.add_column("Attempts", style="white")

    for s in shots:
        table.add_row(
            s.shot_id,
            s.shot_type,
            s.speaker or "-",
            "Yes" if s.lip_sync_required else "No",
            s.status,
            str(s.attempts),
        )

    console.print(table)

@app.command()
def deploy_modal():
    """Deploys ComfyUI and F5-TTS apps to Modal."""
    import subprocess
    console.print("[cyan]Deploying ComfyUI server to Modal...[/cyan]")
    subprocess.run(["modal", "deploy", "modal_app/comfy_server.py"], check=True)
    console.print("[cyan]Deploying F5-TTS server to Modal...[/cyan]")
    subprocess.run(["modal", "deploy", "modal_app/f5_tts_server.py"], check=True)
    console.print("[green]✓ Modal apps deployed successfully![/green]")


@app.command()
def comfy_ui():
    """Launches an interactive ComfyUI session on Modal with public URL."""
    import subprocess
    console.print("[cyan]Starting interactive ComfyUI session on Modal...[/cyan]")
    subprocess.run(["modal", "run", "modal_app/comfy_server.py::ui"])


@app.command()
def download_models():
    """Runs the model downloader on Modal to populate persistent Volume."""
    import subprocess
    console.print("[cyan]Starting model weights downloader on Modal Volume...[/cyan]")
    subprocess.run(["modal", "run", "modal_app/download_models.py::download_checkpoints"])


if __name__ == "__main__":
    app()
