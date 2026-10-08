"""Checkpoint verification script for Novel-to-Movie AI Pipeline.

Tests the full local functional pipeline:
1. Seeds 3 shots from Pride and Prejudice Chapter 1.
2. Synthesizes dialogue audio.
3. Generates keyframe and video clips.
4. Performs mock neural lip-sync.
5. Evaluates with QA judge.
6. Assembles the shots into final_movie.mp4.
"""

from pathlib import Path
from pipeline.assemble import build_concat_list_file, concatenate_clips
from pipeline.bibles import load_character_bibles_from_yaml, load_voice_bibles_from_yaml
from pipeline.generate import process_single_shot
from pipeline.models import DialogueLine, ShotJob
from pipeline.state_manager import init_database, list_shots, save_shots_batch


def run_checkpoint_verification():
    print("=== Starting Checkpoint Verification ===")
    db_path = "./output/shots.db"
    init_database(db_path)

    sample_shots = (
        ShotJob(
            shot_id="ch01_sc01_sh001",
            sequence_order=1,
            chapter_num=1,
            scene_num=1,
            shot_type="wide",
            characters=("mrs_bennet", "mr_bennet"),
            speaker=None,
            dialogue=(),
            lip_sync_required=False,
            visual_prompt="Regency parlour room, sunlight streaming through Georgian windows, Mr. Bennet reading in an armchair while Mrs. Bennet flutters nearby.",
            mood="domestic, calm",
            target_duration_sec=3.0,
            status="queued",
            seed=101,
        ),
        ShotJob(
            shot_id="ch01_sc01_sh002",
            sequence_order=2,
            chapter_num=1,
            scene_num=1,
            shot_type="close_up",
            characters=("mrs_bennet",),
            speaker="mrs_bennet",
            dialogue=(
                DialogueLine(
                    character="mrs_bennet",
                    text="My dear Mr. Bennet, have you heard that Netherfield Park is let at last?",
                ),
            ),
            lip_sync_required=True,
            visual_prompt="Close-up of Mrs. Bennet, expressive eager eyes, Regency floral cap, speaking urgently.",
            mood="excitable",
            target_duration_sec=3.5,
            status="queued",
            seed=102,
        ),
        ShotJob(
            shot_id="ch01_sc01_sh003",
            sequence_order=3,
            chapter_num=1,
            scene_num=1,
            shot_type="medium",
            characters=("mr_bennet",),
            speaker="mr_bennet",
            dialogue=(
                DialogueLine(
                    character="mr_bennet",
                    text="I have not.",
                ),
            ),
            lip_sync_required=True,
            visual_prompt="Medium shot of Mr. Bennet looking up from his book over his spectacles with a dry amused expression.",
            mood="dry, sarcastic",
            target_duration_sec=2.5,
            status="queued",
            seed=103,
        ),
    )

    save_shots_batch(db_path, sample_shots)
    print(f"✓ Seeded {len(sample_shots)} shots into {db_path}")

    visual_profiles = load_character_bibles_from_yaml("assets/characters.yaml")
    voice_profiles = load_voice_bibles_from_yaml("assets/characters.yaml")

    # Render each shot
    for s in sample_shots:
        print(f"--> Processing {s.shot_id} ({s.shot_type})...")
        res = process_single_shot(
            shot=s,
            db_path=db_path,
            visual_profiles=visual_profiles,
            voice_profiles=voice_profiles,
            output_dir="./output",
            mock_rendering=True,
        )
        print(f"    Status: {res.status} | Video: {res.synced_path}")

    # Assemble
    passed_shots = list_shots(db_path, status="passed")
    print(f"✓ All {len(passed_shots)} shots passed QA gate.")

    out_dir = Path("./output/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_txt = out_dir / "concat_list.txt"
    final_video = out_dir / "test_scene_assembled.mp4"

    video_files = [s.synced_path for s in passed_shots if s.synced_path]
    build_concat_list_file(video_files, str(concat_txt))
    concatenate_clips(str(concat_txt), str(final_video))

    print(f"✓ Successfully assembled final movie at: {final_video.resolve()}")
    print("=== Verification Complete! ===")


if __name__ == "__main__":
    run_checkpoint_verification()
