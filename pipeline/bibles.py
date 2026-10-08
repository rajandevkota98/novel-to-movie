"""Functional character and voice bible management.

Provides immutable character profiles and reference asset lookups.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

from pipeline.models import CharacterVisualProfile, CharacterVoiceProfile


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
