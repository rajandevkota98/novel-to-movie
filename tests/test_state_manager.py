"""Tests for the functional SQLite state manager."""

import tempfile
from pathlib import Path
from pipeline.models import DialogueLine, ShotJob
from pipeline.state_manager import (
    init_database,
    save_shot,
    save_shots_batch,
    get_shot,
    list_shots,
    update_shot_state,
)


def test_sqlite_state_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_shots.db")
        init_database(db_path)

        shot1 = ShotJob(
            shot_id="ch01_sc01_sh01",
            sequence_order=1,
            chapter_num=1,
            scene_num=1,
            shot_type="wide",
            characters=("elizabeth", "darcy"),
            speaker="darcy",
            dialogue=(DialogueLine(character="darcy", text="Good evening."),),
            lip_sync_required=True,
            visual_prompt="Grand ballroom in Netherfield, opulent chandeliers.",
            mood="formal",
            target_duration_sec=4.5,
            status="queued",
        )

        save_shot(db_path, shot1)

        retrieved = get_shot(db_path, "ch01_sc01_sh01")
        assert retrieved is not None
        assert retrieved.shot_id == "ch01_sc01_sh01"
        assert retrieved.speaker == "darcy"
        assert len(retrieved.dialogue) == 1
        assert retrieved.dialogue[0].text == "Good evening."

        # Update state
        updated = update_shot_state(
            db_path,
            "ch01_sc01_sh01",
            "keyframe_ready",
            keyframe_path="/tmp/shot1.png",
        )
        assert updated is not None
        assert updated.status == "keyframe_ready"
        assert updated.keyframe_path == "/tmp/shot1.png"

        # List filter
        all_shots = list_shots(db_path)
        assert len(all_shots) == 1


def test_character_profile_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_chars.db")
        init_database(db_path)

        from pipeline.models import CharacterProfile
        from pipeline.state_manager import (
            get_character,
            get_character_dict,
            list_characters,
            save_character,
        )

        char1 = CharacterProfile(
            char_id="dracula",
            name="Count Dracula",
            gender="male",
            age="ancient",
            appearance_description="Pale aristocratic vampire in black cape.",
            personality_tone="menacing",
            voice_timbre="deep aristocratic Romanian accent",
        )
        save_character(db_path, char1)

        retrieved = get_character(db_path, "dracula")
        assert retrieved is not None
        assert retrieved.name == "Count Dracula"
        assert retrieved.voice_timbre == "deep aristocratic Romanian accent"

        all_chars = list_characters(db_path)
        assert len(all_chars) == 1
        assert all_chars[0].char_id == "dracula"

        char_dict = get_character_dict(db_path)
        assert "dracula" in char_dict
