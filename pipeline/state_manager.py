"""Functional SQLite state management for idempotent shot tracking.

Follows strict functional programming:
- Pure functions only, no stateful class instances or connections stored on 'self'.
- Connections are opened, executed with context managers, and closed per transaction.
- Returns immutable ShotJob tuples.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from pipeline.models import CharacterProfile, DialogueLine, ShotJob


def init_database(db_path: str) -> None:
    """Creates the SQLite database and tables if they do not exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript(
            """
            CREATE TABLE IF NOT EXISTS characters (
                char_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                gender TEXT NOT NULL,
                age TEXT NOT NULL,
                appearance_description TEXT NOT NULL,
                personality_tone TEXT NOT NULL,
                voice_timbre TEXT NOT NULL,
                reference_images_json TEXT NOT NULL DEFAULT '[]',
                reference_audio_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS shots (
                shot_id TEXT PRIMARY KEY,
                sequence_order INTEGER NOT NULL,
                chapter_num INTEGER NOT NULL,
                scene_num INTEGER NOT NULL,
                shot_type TEXT NOT NULL,
                characters TEXT NOT NULL,
                speaker TEXT,
                dialogue_json TEXT NOT NULL,
                lip_sync_required INTEGER NOT NULL,
                visual_prompt TEXT NOT NULL,
                mood TEXT NOT NULL,
                target_duration_sec REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                attempts INTEGER NOT NULL DEFAULT 0,
                keyframe_path TEXT,
                video_path TEXT,
                audio_path TEXT,
                synced_path TEXT,
                judge_feedback TEXT,
                seed INTEGER NOT NULL DEFAULT 42,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _row_to_shot_job(row: sqlite3.Row) -> ShotJob:
    """Converts a database row into an immutable ShotJob."""
    dialogue_raw = json.loads(row["dialogue_json"])
    dialogue_items = tuple(
        DialogueLine(character=d["character"], text=d["text"])
        for d in dialogue_raw
    )
    characters_list = tuple(json.loads(row["characters"]))

    return ShotJob(
        shot_id=row["shot_id"],
        sequence_order=row["sequence_order"],
        chapter_num=row["chapter_num"],
        scene_num=row["scene_num"],
        shot_type=row["shot_type"],
        characters=characters_list,
        speaker=row["speaker"],
        dialogue=dialogue_items,
        lip_sync_required=bool(row["lip_sync_required"]),
        visual_prompt=row["visual_prompt"],
        mood=row["mood"],
        target_duration_sec=row["target_duration_sec"],
        status=row["status"],
        attempts=row["attempts"],
        keyframe_path=row["keyframe_path"],
        video_path=row["video_path"],
        audio_path=row["audio_path"],
        synced_path=row["synced_path"],
        judge_feedback=row["judge_feedback"],
        seed=row["seed"],
    )


def save_shot(db_path: str, shot: ShotJob) -> None:
    """Inserts or replaces a single shot in the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        dialogue_json = json.dumps(
            [{"character": d.character, "text": d.text} for d in shot.dialogue]
        )
        characters_json = json.dumps(list(shot.characters))

        cursor.execute(
            """
            INSERT INTO shots (
                shot_id, sequence_order, chapter_num, scene_num, shot_type,
                characters, speaker, dialogue_json, lip_sync_required,
                visual_prompt, mood, target_duration_sec, status, attempts,
                keyframe_path, video_path, audio_path, synced_path,
                judge_feedback, seed, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, CURRENT_TIMESTAMP
            )
            ON CONFLICT(shot_id) DO UPDATE SET
                sequence_order=excluded.sequence_order,
                chapter_num=excluded.chapter_num,
                scene_num=excluded.scene_num,
                shot_type=excluded.shot_type,
                characters=excluded.characters,
                speaker=excluded.speaker,
                dialogue_json=excluded.dialogue_json,
                lip_sync_required=excluded.lip_sync_required,
                visual_prompt=excluded.visual_prompt,
                mood=excluded.mood,
                target_duration_sec=excluded.target_duration_sec,
                status=excluded.status,
                attempts=excluded.attempts,
                keyframe_path=excluded.keyframe_path,
                video_path=excluded.video_path,
                audio_path=excluded.audio_path,
                synced_path=excluded.synced_path,
                judge_feedback=excluded.judge_feedback,
                seed=excluded.seed,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                shot.shot_id,
                shot.sequence_order,
                shot.chapter_num,
                shot.scene_num,
                shot.shot_type,
                characters_json,
                shot.speaker,
                dialogue_json,
                int(shot.lip_sync_required),
                shot.visual_prompt,
                shot.mood,
                shot.target_duration_sec,
                shot.status,
                shot.attempts,
                shot.keyframe_path,
                shot.video_path,
                shot.audio_path,
                shot.synced_path,
                shot.judge_feedback,
                shot.seed,
            ),
        )
        conn.commit()


def save_shots_batch(db_path: str, shots: Sequence[ShotJob]) -> None:
    """Inserts or replaces multiple shots atomically."""
    for shot in shots:
        save_shot(db_path, shot)


def get_shot(db_path: str, shot_id: str) -> Optional[ShotJob]:
    """Retrieves a single shot by ID, returning None if not found."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM shots WHERE shot_id = ?", (shot_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_shot_job(row)


def list_shots(
    db_path: str, status: Optional[str] = None
) -> Tuple[ShotJob, ...]:
    """Lists shots ordered by sequence_order, optionally filtered by status."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM shots WHERE status = ? ORDER BY sequence_order ASC",
                (status,),
            )
        else:
            cursor.execute("SELECT * FROM shots ORDER BY sequence_order ASC")
        rows = cursor.fetchall()
        return tuple(_row_to_shot_job(r) for r in rows)


def update_shot_state(
    db_path: str, shot_id: str, new_status: str, **kwargs
) -> Optional[ShotJob]:
    """Updates status and specific optional fields on a shot, returning the updated ShotJob."""
    existing = get_shot(db_path, shot_id)
    if not existing:
        return None

    # Functional dictionary replacement
    shot_dict = existing.model_dump()
    shot_dict["status"] = new_status
    for k, v in kwargs.items():
        if k in shot_dict:
            shot_dict[k] = v

    updated_shot = ShotJob(**shot_dict)
    save_shot(db_path, updated_shot)
    return updated_shot


def _row_to_character_profile(row: sqlite3.Row) -> CharacterProfile:
    """Converts a database row into an immutable CharacterProfile."""
    ref_images = tuple(json.loads(row["reference_images_json"]))
    return CharacterProfile(
        char_id=row["char_id"],
        name=row["name"],
        gender=row["gender"],
        age=row["age"],
        appearance_description=row["appearance_description"],
        personality_tone=row["personality_tone"],
        voice_timbre=row["voice_timbre"],
        reference_image_paths=ref_images,
        reference_audio_path=row["reference_audio_path"],
    )


def save_character(db_path: str, char: CharacterProfile) -> None:
    """Inserts or updates a dynamic character sheet profile in SQLite."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        ref_images_json = json.dumps(list(char.reference_image_paths))
        cursor.execute(
            """
            INSERT INTO characters (
                char_id, name, gender, age, appearance_description,
                personality_tone, voice_timbre, reference_images_json,
                reference_audio_path, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(char_id) DO UPDATE SET
                name=excluded.name,
                gender=excluded.gender,
                age=excluded.age,
                appearance_description=excluded.appearance_description,
                personality_tone=excluded.personality_tone,
                voice_timbre=excluded.voice_timbre,
                reference_images_json=excluded.reference_images_json,
                reference_audio_path=excluded.reference_audio_path,
                updated_at=CURRENT_TIMESTAMP
            """,
            (
                char.char_id,
                char.name,
                char.gender,
                char.age,
                char.appearance_description,
                char.personality_tone,
                char.voice_timbre,
                ref_images_json,
                char.reference_audio_path,
            ),
        )
        conn.commit()


def save_characters_batch(db_path: str, chars: Sequence[CharacterProfile]) -> None:
    """Saves multiple discovered characters atomically."""
    for c in chars:
        save_character(db_path, c)


def get_character(db_path: str, char_id: str) -> Optional[CharacterProfile]:
    """Retrieves a character by ID from SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM characters WHERE char_id = ?", (char_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_character_profile(row)


def list_characters(db_path: str) -> Tuple[CharacterProfile, ...]:
    """Lists all dynamically discovered characters for the project."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM characters ORDER BY name ASC")
        rows = cursor.fetchall()
        return tuple(_row_to_character_profile(r) for r in rows)


def get_character_dict(db_path: str) -> Dict[str, CharacterProfile]:
    """Returns a dictionary mapping char_id -> CharacterProfile."""
    return {c.char_id: c for c in list_characters(db_path)}
