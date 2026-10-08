"""Functional visual character sheet generation and anchoring engine.

Generates canonical turnaround character portrait sheets using Flux.1 (or mock generator)
and updates the project's character profiles in SQLite so every subsequent shot keyframe
can load that exact image into PuLID/IP-Adapter for zero facial fluctuation.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from pipeline.comfy_client import inject_workflow_params, load_workflow_template, save_media_bytes
from pipeline.models import CharacterProfile
from pipeline.state_manager import list_characters, save_character


def _create_mock_character_sheet(char: CharacterProfile, out_path: str) -> str:
    """Generates a structured, high-resolution visual character card placeholder using PIL."""
    width, height = 1024, 1024
    # Deterministic palette based on char_id
    color_seed = int(hashlib.md5(char.char_id.encode("utf-8")).hexdigest()[:6], 16)
    r = (color_seed >> 16) & 255
    g = (color_seed >> 8) & 255
    b = color_seed & 255

    # Dark background with character-toned hue
    bg_color = (max(15, r // 8), max(18, g // 8), max(25, b // 8))
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Outer decorative frame
    draw.rectangle([30, 30, width - 30, height - 30], outline=(100, 116, 139), width=3)
    draw.rectangle([45, 45, width - 45, height - 45], outline=(r, g, b), width=2)

    # Portrait silhouette placeholder
    avatar_center = (width // 2, 380)
    avatar_radius = 180
    draw.ellipse(
        [
            avatar_center[0] - avatar_radius,
            avatar_center[1] - avatar_radius,
            avatar_center[0] + avatar_radius,
            avatar_center[1] + avatar_radius,
        ],
        fill=(max(30, r // 4), max(35, g // 4), max(45, b // 4)),
        outline=(r, g, b),
        width=4,
    )

    # Character head/body stylized silhouette
    draw.ellipse(
        [avatar_center[0] - 65, avatar_center[1] - 110, avatar_center[0] + 65, avatar_center[1] + 20],
        fill=(r, g, b),
    )
    draw.arc(
        [avatar_center[0] - 120, avatar_center[1] - 10, avatar_center[0] + 120, avatar_center[1] + 160],
        start=180,
        end=360,
        fill=(r, g, b),
        width=6,
    )

    # Metadata & Typography
    draw.text((width // 2, 600), char.name.upper(), fill=(248, 250, 252), anchor="mm")
    tagline = f"{char.gender.capitalize()} • {char.age} • Voice: {char.voice_timbre}"
    draw.text((width // 2, 640), tagline, fill=(148, 163, 184), anchor="mm")

    # Appearance spec preview
    desc = char.appearance_description
    if len(desc) > 160:
        desc = desc[:157] + "..."
    # Word wrap description into lines
    words = desc.split()
    lines = []
    curr = []
    for w in words:
        curr.append(w)
        if len(" ".join(curr)) > 55:
            lines.append(" ".join(curr))
            curr = []
    if curr:
        lines.append(" ".join(curr))

    y_offset = 700
    for line in lines[:4]:
        draw.text((width // 2, y_offset), line, fill=(203, 213, 225), anchor="mm")
        y_offset += 32

    draw.text(
        (width // 2, 940),
        "CANONICAL FACIAL ANCHOR • NOVEL-TO-MOVIE PIPELINE",
        fill=(100, 116, 139),
        anchor="mm",
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return str(Path(out_path).resolve())


def generate_character_sheet_image(
    char: CharacterProfile,
    output_dir: str = "output",
    mock: bool = False,
) -> str:
    """Renders or mocks a canonical character sheet portrait used as PuLID face anchor."""
    out_dir = Path(output_dir) / "characters"
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / f"{char.char_id}_sheet.png"

    if mock or not os.environ.get("MODAL_TOKEN_ID"):
        return _create_mock_character_sheet(char, str(sheet_path))

    # Remote Modal ComfyUI Flux.1 text2img generation
    import modal

    worker = modal.Cls.from_name("novel-to-movie-comfy", "KeyframeComfyWorker")()
    template = load_workflow_template("modal_app/workflows/flux_text2img_api.json")

    flux_prompt = (
        f"Cinematic studio character portrait, 3/4 angle turnaround reference sheet, "
        f"neutral clean background, sharp facial details, 8k photorealistic. "
        f"Subject: {char.name}, {char.gender}, {char.age}. "
        f"Visual details: {char.appearance_description}."
    )

    seed = (int(hashlib.md5(char.char_id.encode("utf-8")).hexdigest()[:8], 16) % 1_000_000) + 42
    wf = inject_workflow_params(template, {"PROMPT": flux_prompt, "seed": seed})

    img_bytes = worker.run_workflow.remote(wf)
    return save_media_bytes(img_bytes, str(sheet_path))


def ensure_character_sheets(
    db_path: str,
    output_dir: str = "output",
    mock: bool = False,
) -> Dict[str, CharacterProfile]:
    """Ensures every registered character in SQLite has an anchored visual character sheet."""
    characters = list_characters(db_path)
    updated_dict: Dict[str, CharacterProfile] = {}

    for char in characters:
        existing_ref = char.reference_image_paths[0] if char.reference_image_paths else None
        if existing_ref and Path(existing_ref).is_file():
            updated_dict[char.char_id] = char
            continue

        sheet_path = generate_character_sheet_image(char, output_dir=output_dir, mock=mock)
        updated_char = char.model_copy(update={"reference_image_paths": (sheet_path,)})
        save_character(db_path, updated_char)
        updated_dict[char.char_id] = updated_char

    return updated_dict
