"""Pure immutable data models for the Novel-to-Movie AI Pipeline.

Adheres strictly to functional programming principles:
- All models are immutable (frozen=True).
- No mutating methods or stateful classes.
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, ConfigDict, Field


class DialogueLine(BaseModel):
    """Immutable representation of a single line of spoken dialogue."""
    model_config = ConfigDict(frozen=True)

    character: str
    text: str


class ShotJob(BaseModel):
    """Immutable representation of a single shot unit in the production queue."""
    model_config = ConfigDict(frozen=True)

    shot_id: str                      # Format: "ch01_sc02_sh04"
    sequence_order: int
    chapter_num: int
    scene_num: int
    shot_type: str                    # "close_up", "medium", "wide", "over_the_shoulder"
    characters: Tuple[str, ...]       # Tuple for immutability
    speaker: Optional[str] = None
    dialogue: Tuple[DialogueLine, ...] = Field(default_factory=tuple)
    lip_sync_required: bool = False
    visual_prompt: str
    mood: str
    target_duration_sec: float = 5.0
    status: str = "queued"            # queued, processing, passed, needs_review, failed
    attempts: int = 0
    keyframe_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    synced_path: Optional[str] = None
    judge_feedback: Optional[str] = None
    seed: int = 42


class SceneOutline(BaseModel):
    """Immutable representation of a dramatic scene extracted from a chapter."""
    model_config = ConfigDict(frozen=True)

    chapter_num: int
    scene_num: int
    location: str
    time_of_day: str
    mood: str
    characters: Tuple[str, ...]
    summary: str
    shots: Tuple[ShotJob, ...] = Field(default_factory=tuple)


class CharacterVisualProfile(BaseModel):
    """Visual identity bible entry for a character."""
    model_config = ConfigDict(frozen=True)

    name: str
    gender: str
    age: str
    appearance_description: str
    reference_image_paths: Tuple[str, ...] = Field(default_factory=tuple)


class CharacterVoiceProfile(BaseModel):
    """Voice identity bible entry for a character."""
    model_config = ConfigDict(frozen=True)

    name: str
    reference_audio_path: str         # 10s sample for F5-TTS zero-shot cloning
    reference_transcript: str = ""    # Optional transcript of the reference sample


class JudgeVerdict(BaseModel):
    """Multimodal evaluation score from the automated quality gate."""
    model_config = ConfigDict(frozen=True)

    passed: bool
    character_consistency: int       # 1 to 5 scale
    prompt_adherence: int            # 1 to 5 scale
    motion_quality: int              # 1 to 5 scale
    reason: str
    seed_nudge: Optional[int] = None
    prompt_nudge: Optional[str] = None
