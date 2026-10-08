"""End-to-end and unit tests for pipeline functional components."""

import tempfile
from pathlib import Path
from pipeline.assemble import build_concat_list_file
from pipeline.audio import get_audio_duration_seconds
from pipeline.bibles import load_character_bibles_from_yaml, load_voice_bibles_from_yaml
from pipeline.generate import process_single_shot
from pipeline.models import DialogueLine, ShotJob
from pipeline.state_manager import init_database, save_shot


def test_bibles_loading():
    chars = load_character_bibles_from_yaml("assets/characters.yaml")
    assert "elizabeth" in chars
    assert chars["elizabeth"].gender == "female"

    voices = load_voice_bibles_from_yaml("assets/characters.yaml")
    assert "mr_bennet" in voices
    assert voices["mr_bennet"].name == "Mr. Bennet"


def test_concat_file_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        list_file = Path(tmp_dir) / "list.txt"
        videos = ["/tmp/vid1.mp4", "/tmp/vid2.mp4"]
        build_concat_list_file(videos, str(list_file))
        assert list_file.exists()
        content = list_file.read_text()
        assert "file '/tmp/vid1.mp4'" in content
        assert "file '/tmp/vid2.mp4'" in content


def test_process_single_shot_mock():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test.db")
        init_database(db_path)

        shot = ShotJob(
            shot_id="ch01_sc01_sh001",
            sequence_order=1,
            chapter_num=1,
            scene_num=1,
            shot_type="close_up",
            characters=("elizabeth",),
            speaker="elizabeth",
            dialogue=(DialogueLine(character="elizabeth", text="Testing spoken line."),),
            lip_sync_required=True,
            visual_prompt="Close up portrait of Elizabeth.",
            mood="reflective",
            target_duration_sec=3.0,
            status="queued",
        )
        save_shot(db_path, shot)

        chars = load_character_bibles_from_yaml("assets/characters.yaml")
        voices = load_voice_bibles_from_yaml("assets/characters.yaml")

        result = process_single_shot(
            shot=shot,
            db_path=db_path,
            visual_profiles=chars,
            voice_profiles=voices,
            output_dir=tmp_dir,
            mock_rendering=True,
        )

        assert result.status == "passed"
        assert result.audio_path is not None
        assert Path(result.audio_path).exists()
        assert result.keyframe_path is not None
        assert Path(result.keyframe_path).exists()
        assert result.video_path is not None
        assert Path(result.video_path).exists()
        assert result.synced_path is not None
        assert Path(result.synced_path).exists()


def test_dynamic_character_sheet_export_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        from pipeline.bibles import export_character_sheet_json, load_character_sheet_json
        from pipeline.models import CharacterProfile

        profiles = [
            CharacterProfile(
                char_id="victor",
                name="Victor Frankenstein",
                gender="male",
                age="20s",
                appearance_description="Pale gaunt Swiss student in velvet coat with disheveled brown hair.",
                personality_tone="obsessive",
                voice_timbre="intense nervous Swiss-accented voice",
            ),
            CharacterProfile(
                char_id="creature",
                name="The Creature",
                gender="male",
                age="unknown",
                appearance_description="Eight-foot-tall figure with watery yellow eyes, translucent yellowish skin.",
                personality_tone="melancholic, vengeful",
                voice_timbre="deep guttural resonant tone",
            ),
        ]

        json_path = Path(tmp_dir) / "character_sheet.json"
        export_character_sheet_json(profiles, str(json_path))
        assert json_path.exists()

        loaded = load_character_sheet_json(str(json_path))
        assert "victor" in loaded
        assert "creature" in loaded
        assert loaded["victor"].name == "Victor Frankenstein"
        assert loaded["creature"].personality_tone == "melancholic, vengeful"
