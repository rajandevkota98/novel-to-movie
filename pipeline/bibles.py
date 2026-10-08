"""Functional character and voice profile management.

Supports both:
- Dynamic Character Sheets discovered by LLM from any novel/text.
- Optional legacy YAML bibles for fixed projects.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import yaml

from pipeline.models import (
    CharacterProfile,
    CharacterVisualProfile,
    CharacterVoiceProfile,
)


def export_character_sheet_json(
    characters: Sequence[CharacterProfile], output_path: str
) -> str:
    """Exports dynamically discovered character profiles to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in characters]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(out.resolve())


def load_character_sheet_json(json_path: str) -> Dict[str, CharacterProfile]:
    """Reads a dynamic character sheet JSON file into a mapping of char_id -> CharacterProfile."""
    path = Path(json_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = {}
    for item in data:
        profile = CharacterProfile(
            char_id=item["char_id"],
            name=item["name"],
            gender=item.get("gender", "unknown"),
            age=str(item.get("age", "adult")),
            appearance_description=item.get("appearance_description", ""),
            personality_tone=item.get("personality_tone", "dramatic"),
            voice_timbre=item.get("voice_timbre", "natural speaking voice"),
            reference_image_paths=tuple(item.get("reference_image_paths", [])),
            reference_audio_path=item.get("reference_audio_path"),
        )
        profiles[profile.char_id] = profile
    return profiles


def load_character_bibles_from_yaml(yaml_path: str) -> Dict[str, CharacterVisualProfile]:
    """Reads character visual definitions from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    characters_raw = data.get("characters", {})
    profiles = {}
    for char_id, info in characters_raw.items():
        refs = tuple(info.get("reference_image_paths", []))
        profiles[char_id] = CharacterVisualProfile(
            name=info.get("name", char_id.title()),
            gender=info.get("gender", "unknown"),
            age=str(info.get("age", "adult")),
            appearance_description=info.get("appearance_description", ""),
            reference_image_paths=refs,
        )
    return profiles


def load_voice_bibles_from_yaml(yaml_path: str) -> Dict[str, CharacterVoiceProfile]:
    """Reads character voice definitions from a YAML file."""
    path = Path(yaml_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    characters_raw = data.get("characters", {})
    profiles = {}
    for char_id, info in characters_raw.items():
        audio_path = info.get("reference_audio", "")
        profiles[char_id] = CharacterVoiceProfile(
            name=info.get("name", char_id.title()),
            reference_audio_path=audio_path,
            reference_transcript=info.get("reference_transcript", ""),
        )
    return profiles
