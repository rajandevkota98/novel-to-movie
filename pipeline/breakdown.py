"""Functional script breakdown and shot extraction module.

Calls frontier LLMs (GPT-5 Luna / OpenRouter) to parse literature chapters
into structured dramatic scenes and atomic shot jobs.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple
import httpx

from pipeline.models import DialogueLine, SceneOutline, ShotJob


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
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


def breakdown_chapter_to_shots(
    chapter_text: str,
    chapter_num: int,
    model: str = "openai/gpt-5-luna",
    api_key: Optional[str] = None,
) -> Tuple[ShotJob, ...]:
    """Transforms a raw novel chapter text into an immutable sequence of ShotJobs."""
    system_prompt = (
        "You are an elite cinematic director and screenwriter. "
        "Your task is to break down the provided novel text into a chronological sequence of film shots. "
        "For each shot, specify: shot_type (close_up, medium, wide, over_the_shoulder), "
        "characters present, speaker (if any), dialogue lines, whether lip_sync is required, "
        "a visual prompt optimized for diffusion (Flux), mood, and estimated duration in seconds (usually 3.0 to 6.0s). "
        "Respond ONLY with a JSON object containing a 'shots' array."
    )

    prompt = f"Chapter {chapter_num} text:\n\n{chapter_text}"
    response_data = _call_llm(prompt, system_prompt, model, api_key)
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

    return tuple(shot_jobs)
