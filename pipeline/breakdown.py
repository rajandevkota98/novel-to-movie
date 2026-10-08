"""Functional script breakdown and shot extraction module.

Calls frontier LLMs (GPT-5 Luna / OpenRouter) to parse literature chapters
into structured dramatic scenes and atomic shot jobs.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple
from dotenv import load_dotenv
import httpx

load_dotenv()

from pipeline.models import BreakdownResult, CharacterProfile, DialogueLine, SceneOutline, ShotJob


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _call_llm(
    prompt: str,
    system_prompt: str,
    model: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Pure I/O function to call OpenRouter / OpenAI API with JSON mode."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/novel-to-movie",
        "X-Title": "Novel to Movie AI Pipeline",
    }

    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(OPENROUTER_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        return json.loads(content)


def breakdown_story_and_characters(
    story_text: str,
    chapter_num: int = 1,
    model: str = "openai/gpt-5-luna",
    api_key: Optional[str] = None,
) -> BreakdownResult:
    """Discovers characters dynamically from ANY story/novel text and generates mapped shots."""
    system_prompt = (
        "You are an elite Hollywood director and screenwriter adapting literature into cinema. "
        "Read the provided text and do two things:\n"
        "1. DYNAMIC CHARACTER DISCOVERY: Extract all characters present in the text to create a Character Sheet. "
        "For each character specify: id (lowercase slug e.g. 'alice', 'dracula', 'victor'), name, gender, age, "
        "appearance (rich visual details for Flux diffusion: clothing, face, hair, lighting), "
        "personality_tone, voice_timbre (vocal traits for audio synthesis).\n"
        "2. CINEMATIC SHOT BREAKDOWN: Break the story chronologically into film shots mapped to those character IDs. "
        "For each shot specify: shot_type (close_up, medium, wide, over_the_shoulder), characters (list of ids), "
        "speaker (character id or null), dialogue (list of {character: id, text: str}), "
        "lip_sync_required (true for speaking close-ups/mediums), visual_prompt (for Flux), mood, duration_sec (3.0-6.0).\n"
        "Respond ONLY with a JSON object with keys: 'characters' (array of character objects) and 'shots' (array of shot objects)."
    )

    prompt = f"Story / Chapter {chapter_num} text:\n\n{story_text}"
    response_data = _call_llm(prompt, system_prompt, model, api_key)

    raw_chars = response_data.get("characters", [])
    character_profiles = []
    for c in raw_chars:
        cid = str(c.get("id") or c.get("name", "char")).lower().replace(" ", "_")
        profile = CharacterProfile(
            char_id=cid,
            name=c.get("name", cid.title()),
            gender=c.get("gender", "unknown"),
            age=str(c.get("age", "adult")),
            appearance_description=c.get("appearance", c.get("appearance_description", "")),
            personality_tone=c.get("personality_tone", "dramatic"),
            voice_timbre=c.get("voice_timbre", "natural speaking voice"),
        )
        character_profiles.append(profile)

    raw_shots = response_data.get("shots", [])
    shot_jobs = []
    for idx, s in enumerate(raw_shots, start=1):
        dialogue_tuples = tuple(
            DialogueLine(character=d.get("character", ""), text=d.get("text", ""))
            for d in s.get("dialogue", [])
        )
        characters_tuple = tuple(s.get("characters", []))

        shot_id = f"ch{chapter_num:02d}_sc01_sh{idx:03d}"
        shot = ShotJob(
            shot_id=shot_id,
            sequence_order=idx,
            chapter_num=chapter_num,
            scene_num=1,
            shot_type=s.get("shot_type", "medium"),
            characters=characters_tuple,
            speaker=s.get("speaker"),
            dialogue=dialogue_tuples,
            lip_sync_required=bool(s.get("lip_sync_required", False)),
            visual_prompt=s.get("visual_prompt", ""),
            mood=s.get("mood", "neutral"),
            target_duration_sec=float(s.get("duration_sec", 5.0)),
            status="queued",
            attempts=0,
            seed=42 + idx,
        )
        shot_jobs.append(shot)

    return BreakdownResult(
        characters=tuple(character_profiles),
        shots=tuple(shot_jobs),
    )


def breakdown_chapter_to_shots(
    chapter_text: str,
    chapter_num: int = 1,
    model: str = "openai/gpt-5-luna",
    api_key: Optional[str] = None,
) -> Tuple[ShotJob, ...]:
    """Backward-compatible helper returning only shots."""
    result = breakdown_story_and_characters(chapter_text, chapter_num, model, api_key)
    return result.shots
