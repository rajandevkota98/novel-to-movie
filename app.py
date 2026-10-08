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
from pipeline.bibles import load_character_bibles_from_yaml, load_voice_bibles_from_yaml
from pipeline.breakdown import breakdown_chapter_to_shots
from pipeline.generate import process_single_shot
from pipeline.models import ShotJob
from pipeline.state_manager import get_shot, init_database, list_shots, save_shots_batch

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
    init_database(cfg["storage"]["database_path"])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Renders the central studio web UI."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]

    # Load state
    shots = list_shots(db_path)
    characters = load_character_bibles_from_yaml("assets/characters.yaml")

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
    """Executes script breakdown on provided chapter text."""
    cfg = _get_config()
    db_path = cfg["storage"]["database_path"]

    try:
        shots = breakdown_chapter_to_shots(
            chapter_text=req.chapter_text,
            chapter_num=req.chapter_num,
            model=cfg["models"]["breakdown"],
        )
        save_shots_batch(db_path, shots)
        return {"status": "success", "count": len(shots)}
    except Exception as e:
        # Fallback for testing when no OpenRouter API key is provided
        from pipeline.models import DialogueLine
        mock_shots = (
            ShotJob(
                shot_id=f"ch{req.chapter_num:02d}_sc01_sh001",
                sequence_order=1,
                chapter_num=req.chapter_num,
                scene_num=1,
                shot_type="wide",
                characters=("mrs_bennet", "mr_bennet"),
                speaker=None,
                dialogue=(),
                lip_sync_required=False,
                visual_prompt="Regency parlour room, sunlight streaming through windows.",
                mood="domestic",
                target_duration_sec=3.0,
                status="queued",
            ),
            ShotJob(
                shot_id=f"ch{req.chapter_num:02d}_sc01_sh002",
                sequence_order=2,
                chapter_num=req.chapter_num,
                scene_num=1,
                shot_type="close_up",
                characters=("mrs_bennet",),
                speaker="mrs_bennet",
                dialogue=(DialogueLine(character="mrs_bennet", text="My dear Mr. Bennet!"),),
                lip_sync_required=True,
                visual_prompt="Close-up of Mrs. Bennet, expressive eager eyes.",
                mood="excitable",
                target_duration_sec=3.5,
                status="queued",
            ),
        )
        save_shots_batch(db_path, mock_shots)
        return {"status": "success (mock)", "count": len(mock_shots), "warning": str(e)}


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
