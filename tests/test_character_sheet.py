"""Tests for visual character sheet generation and PuLID face anchor workflows."""

from pathlib import Path
import tempfile
from PIL import Image

from pipeline.character_sheet import ensure_character_sheets, generate_character_sheet_image
from pipeline.generate import process_single_shot
from pipeline.models import CharacterProfile, DialogueLine, ShotJob
from pipeline.state_manager import get_character, init_database, save_character, save_shot


def test_generate_character_sheet_image():
    with tempfile.TemporaryDirectory() as tmp_dir:
        char = CharacterProfile(
            char_id="elizabeth",
            name="Elizabeth Bennet",
            gender="female",
            age="early 20s",
            appearance_description="Dark hair, luminous expressive dark eyes, Regency period walking dress.",
            personality_tone="spirited",
            voice_timbre="articulate Regency British voice",
        )
        sheet_path = generate_character_sheet_image(char, output_dir=tmp_dir, mock=True)
        assert Path(sheet_path).is_file()
        assert sheet_path.endswith("elizabeth_sheet.png")

        # Verify it's a valid 1024x1024 PNG image
        with Image.open(sheet_path) as img:
            assert img.size == (1024, 1024)
            assert img.format == "PNG"


def test_ensure_character_sheets():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test.db")
        init_database(db_path)

        char1 = CharacterProfile(
            char_id="darcy",
            name="Mr. Darcy",
            gender="male",
            age="late 20s",
            appearance_description="Tall, noble mien, dark brooding features, dark riding coat.",
            personality_tone="reserved",
            voice_timbre="deep aristocratic baritone",
        )
        save_character(db_path, char1)

        # Before ensure: reference_image_paths is empty
        fetched_before = get_character(db_path, "darcy")
        assert len(fetched_before.reference_image_paths) == 0

        # Run ensure_character_sheets
        updated_dict = ensure_character_sheets(db_path, output_dir=tmp_dir, mock=True)
        assert "darcy" in updated_dict
        assert len(updated_dict["darcy"].reference_image_paths) == 1
        assert Path(updated_dict["darcy"].reference_image_paths[0]).is_file()

        # Database state is persisted
        fetched_after = get_character(db_path, "darcy")
        assert len(fetched_after.reference_image_paths) == 1
        assert fetched_after.reference_image_paths[0] == updated_dict["darcy"].reference_image_paths[0]


def test_keyframe_generation_with_character_sheet_anchor():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test.db")
        init_database(db_path)

        char = CharacterProfile(
            char_id="darcy",
            name="Mr. Darcy",
            gender="male",
            age="late 20s",
            appearance_description="Tall, noble mien, dark brooding features.",
            personality_tone="reserved",
            voice_timbre="deep aristocratic baritone",
        )
        save_character(db_path, char)
        ensure_character_sheets(db_path, output_dir=tmp_dir, mock=True)

        shot = ShotJob(
            shot_id="ch01_sc01_sh002",
            sequence_order=1,
            chapter_num=1,
            scene_num=1,
            shot_type="close_up",
            characters=("darcy",),
            speaker="darcy",
            dialogue=(DialogueLine(character="darcy", text="She is tolerable; but not handsome enough to tempt me."),),
            lip_sync_required=True,
            visual_prompt="Close up portrait of Darcy with proud expression.",
            mood="haughty",
            target_duration_sec=3.0,
            status="queued",
        )
        save_shot(db_path, shot)

        result = process_single_shot(
            shot=shot,
            db_path=db_path,
            output_dir=tmp_dir,
            mock_rendering=True,
        )

        assert result.keyframe_path is not None
        assert Path(result.keyframe_path).is_file()
        assert Path(result.synced_path).is_file()
