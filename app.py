"""FastAPI Web Studio application for the Novel-to-Movie AI Pipeline.

Provides interactive endpoints and a dark-mode web dashboard for:
- Viewing character & voice bibles
- Triggering script breakdown
- Monitoring and dispatching shot rendering
- Previewing clips and assembling final film
"""

import os
from pathlib import Path
from typing import Optional
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pipeline.assemble import build_concat_list_file, concatenate_clips
from pipeline.bibles import export_character_sheet_json, load_character_bibles_from_yaml
from pipeline.breakdown import breakdown_story_and_characters
from pipeline.generate import process_single_shot
from pipeline.models import CharacterProfile, DialogueLine, ShotJob
from pipeline.state_manager import (
    get_character_dict,
    get_shot,
    init_database,
    list_characters,
    list_shots,
    save_characters_batch,
    save_shots_batch,
)

app = FastAPI(title="Novel-to-Movie AI Studio")

# Static and template configuration
TEMPLATES_DIR = Path("web/templates")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount output directory to serve media files
output_path = Path("output")
output_path.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_path)), name="output")


def _get_config() -> dict:
    """Reads configuration YAML."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BreakdownRequest(BaseModel):
    chapter_text: str
    chapter_num: int = 1


@app.on_event("startup")
def on_startup():
    """Initializes database on app boot."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]
    init_database(db_path)

    # Seed initial characters if database is empty
    if not list_characters(db_path):
        legacy = load_character_bibles_from_yaml("assets/characters.yaml")
        initial_chars = [
            CharacterProfile(
                char_id=cid,
                name=c.name,
                gender=c.gender,
                age=c.age,
                appearance_description=c.appearance_description,
                personality_tone="dramatic",
                voice_timbre="British period drama voice",
            )
            for cid, c in legacy.items()
        ]
        save_characters_batch(db_path, initial_chars)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the central studio web UI."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]

    # Load state dynamically from SQLite
    shots = list_shots(db_path)
    characters = list_characters(db_path)

    sample_text = ""
    sample_file = Path("data/sample_chapter.txt")
    if sample_file.exists():
        sample_text = sample_file.read_text(encoding="utf-8")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "shots": shots,
            "characters": characters,
            "sample_text": sample_text,
        },
    )


@app.post("/api/breakdown")
async def api_breakdown(req: BreakdownRequest):
    """Dynamically extracts Character Sheet and Shot List from ANY story text."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]

    try:
        result = breakdown_story_and_characters(
            story_text=req.chapter_text,
            chapter_num=req.chapter_num,
            model=cfg["models"]["breakdown"],
        )
        save_characters_batch(db_path, result.characters)
        save_shots_batch(db_path, result.shots)
        export_character_sheet_json(result.characters, "output/character_sheet.json")
        return {
            "status": "success",
            "characters_count": len(result.characters),
            "shots_count": len(result.shots),
        }
    except Exception as e:
        # Fallback dynamic discovery for testing when no API key is provided
        # Generates character sheets based on the text submitted
        words = [w.strip('",.?!') for w in req.chapter_text.split() if w.istitle() and len(w) > 3][:4]
        primary_name = words[0] if words else "Protagonist"
        second_name = words[1] if len(words) > 1 else "Companion"

        mock_chars = (
            CharacterProfile(
                char_id=primary_name.lower(),
                name=primary_name,
                gender="female" if "she" in req.chapter_text.lower() else "male",
                age="20s-30s",
                appearance_description=f"Distinctive appearance for {primary_name}, detailed cinematic costume and features.",
                personality_tone="determined",
                voice_timbre="clear expressive dramatic voice",
            ),
            CharacterProfile(
                char_id=second_name.lower(),
                name=second_name,
                gender="male" if "he" in req.chapter_text.lower() else "female",
                age="30s-40s",
                appearance_description=f"Distinctive appearance for {second_name}, contrasting lighting and presence.",
                personality_tone="intense",
                voice_timbre="deep resonant vocal timbre",
            ),
        )

        mock_shots = (
            ShotJob(
                shot_id=f"ch{req.chapter_num:02d}_sc01_sh001",
                sequence_order=1,
                chapter_num=req.chapter_num,
                scene_num=1,
                shot_type="wide",
                characters=(mock_chars[0].char_id, mock_chars[1].char_id),
                speaker=None,
                dialogue=(),
                lip_sync_required=False,
                visual_prompt=f"Establishing shot of the story environment. {mock_chars[0].name} and {mock_chars[1].name} in composition.",
                mood="atmospheric",
                target_duration_sec=3.5,
                status="queued",
            ),
            ShotJob(
                shot_id=f"ch{req.chapter_num:02d}_sc01_sh002",
                sequence_order=2,
                chapter_num=req.chapter_num,
                scene_num=1,
                shot_type="close_up",
                characters=(mock_chars[0].char_id,),
                speaker=mock_chars[0].char_id,
                dialogue=(DialogueLine(character=mock_chars[0].char_id, text=f"Dialogue spoken by {mock_chars[0].name}."),),
                lip_sync_required=True,
                visual_prompt=f"Close-up portrait of {mock_chars[0].name}, focused lighting.",
                mood="tense",
                target_duration_sec=3.0,
                status="queued",
            ),
        )

        save_characters_batch(db_path, mock_chars)
        save_shots_batch(db_path, mock_shots)
        export_character_sheet_json(mock_chars, "output/character_sheet.json")
        return {
            "status": "success (mock discovery)",
            "characters_count": len(mock_chars),
            "shots_count": len(mock_shots),
            "warning": str(e),
        }


@app.post("/api/render/{shot_id}")
async def api_render_shot(shot_id: str, mock: bool = True):
    """Renders a single shot."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]
    shot = get_shot(db_path, shot_id)

    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    chars = load_character_bibles_from_yaml("assets/characters.yaml")
    voices = load_voice_bibles_from_yaml("assets/characters.yaml")

    result = process_single_shot(
        shot=shot,
        db_path=db_path,
        visual_profiles=chars,
        voice_profiles=voices,
        output_dir=cfg["project"]["output_dir"],
        retry_cap=cfg["quality_gate"]["retry_cap"],
        mock_rendering=mock,
    )
    return {"status": result.status, "shot_id": result.shot_id, "video": result.synced_path}


@app.post("/api/render-all")
async def api_render_all(mock: bool = True):
    """Batch renders all queued shots."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]
    shots = list_shots(db_path, status="queued")

    if not shots:
        return {"status": "success", "message": "No queued shots to render"}

    chars = load_character_bibles_from_yaml("assets/characters.yaml")
    voices = load_voice_bibles_from_yaml("assets/characters.yaml")

    for s in shots:
        process_single_shot(
            shot=s,
            db_path=db_path,
            visual_profiles=chars,
            voice_profiles=voices,
            output_dir=cfg["project"]["output_dir"],
            retry_cap=cfg["quality_gate"]["retry_cap"],
            mock_rendering=mock,
        )
    return {"status": "success", "message": f"Processed {len(shots)} shots"}


@app.post("/api/assemble")
async def api_assemble():
    """Assembles all passed shots into final movie."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]
    passed_shots = list_shots(db_path, status="passed")

    if not passed_shots:
        raise HTTPException(status_code=400, detail="No approved shots to assemble")

    video_files = [s.synced_path for s in passed_shots if s.synced_path and Path(s.synced_path).exists()]
    if not video_files:
        raise HTTPException(status_code=400, detail="No valid video files found")

    out_dir = Path(cfg["project"]["output_dir"]) / "final"
    concat_txt = out_dir / "concat_list.txt"
    final_video = out_dir / "test_scene_assembled.mp4"

    build_concat_list_file(video_files, str(concat_txt))
    concatenate_clips(str(concat_txt), str(final_video))

    return {"status": "success", "output": str(final_video)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
