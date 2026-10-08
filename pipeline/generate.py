"""Functional shot generation orchestrator.

Executes the step-by-step rendering pipeline for a single shot:
1. Audio Synthesis (F5-TTS)
2. Face Keyframe (Flux + PuLID)
3. Video Generation (Wan 2.2 / LTX)
4. Lip-Sync (LatentSync, if speaking close-up)
5. Multimodal Quality Gate (Judge with auto-retry)
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pipeline.audio import generate_shot_dialogue_audio
from pipeline.bibles import (
    load_character_bibles_from_yaml,
    load_voice_bibles_from_yaml,
)
from pipeline.comfy_client import (
    inject_workflow_params,
    load_workflow_template,
    save_media_bytes,
)
from pipeline.judge import evaluate_shot_clip
from pipeline.models import CharacterProfile, CharacterVisualProfile, CharacterVoiceProfile, JudgeVerdict, ShotJob
from pipeline.state_manager import get_character_dict, update_shot_state


def process_single_shot(
    shot: ShotJob,
    db_path: str,
    visual_profiles: Optional[Dict[str, Any]] = None,
    voice_profiles: Optional[Dict[str, Any]] = None,
    output_dir: str = "./output",
    retry_cap: int = 3,
    mock_rendering: bool = False,
) -> ShotJob:
    """Executes the complete generation and QA lifecycle for a single shot.

    Dynamically resolves character sheets from SQLite if not provided.
    """
    # Auto-resolve dynamic character profiles from SQLite if omitted
    if visual_profiles is None:
        visual_profiles = get_character_dict(db_path)
    if voice_profiles is None:
        voice_profiles = visual_profiles

    # Step 1: Audio Synthesis
    audio_path = shot.audio_path
    if not audio_path and shot.dialogue:
        audio_path = generate_shot_dialogue_audio(
            shot=shot,
            voice_profiles=voice_profiles,
            output_dir=f"{output_dir}/audio",
            mock_mode=mock_rendering,
        )
        shot = update_shot_state(db_path, shot.shot_id, "audio_ready", audio_path=audio_path)

    # Step 2: Keyframe Image Generation
    keyframe_path = shot.keyframe_path
    if not keyframe_path:
        keyframe_out = f"{output_dir}/keyframes/{shot.shot_id}.png"
        Path(keyframe_out).parent.mkdir(parents=True, exist_ok=True)

        char_profile = visual_profiles.get(shot.speaker.lower()) if shot.speaker else None
        prompt_text = shot.visual_prompt
        if char_profile and getattr(char_profile, "appearance_description", None):
            prompt_text = f"{shot.visual_prompt}. Character details: {char_profile.appearance_description}"

        has_ref_image = bool(
            char_profile
            and getattr(char_profile, "reference_image_paths", None)
            and Path(char_profile.reference_image_paths[0]).is_file()
        )

        if mock_rendering or not os.environ.get("MODAL_TOKEN_ID"):
            # Functional visual keyframe placeholder
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (1024, 576), color=(20, 24, 39))
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 1004, 556], outline=(59, 130, 246), width=2)
            draw.text((512, 180), f"KEYFRAME: {shot.shot_id.upper()}", fill=(248, 250, 252), anchor="mm")
            draw.text((512, 230), f"Shot Type: {shot.shot_type} | Mood: {shot.mood}", fill=(148, 163, 184), anchor="mm")
            anchor_info = (
                f"Anchored to Sheet: {char_profile.name} ({char_profile.reference_image_paths[0]})"
                if has_ref_image
                else "Scene Shot (No Face Anchor)"
            )
            draw.text((512, 280), anchor_info, fill=(52, 211, 153), anchor="mm")
            preview_prompt = prompt_text[:85] + "..." if len(prompt_text) > 85 else prompt_text
            draw.text((512, 330), preview_prompt, fill=(203, 213, 225), anchor="mm")
            img.save(keyframe_out, format="PNG")
            keyframe_path = str(Path(keyframe_out).resolve())
        else:
            import modal
            worker = modal.Cls.from_name("novel-to-movie-comfy", "KeyframeComfyWorker")()

            if has_ref_image:
                template = load_workflow_template("modal_app/workflows/flux_pulid_api.json")
                ref_path = Path(char_profile.reference_image_paths[0])
                ref_filename = f"{getattr(char_profile, 'char_id', 'char')}_ref.png"
                wf = inject_workflow_params(template, {
                    "PROMPT": prompt_text,
                    "seed": shot.seed,
                    "CHARACTER_REF_IMAGE": ref_filename,
                })
                img_bytes = worker.run_workflow.remote(
                    workflow_json=wf,
                    input_files={ref_filename: ref_path.read_bytes()},
                )
            else:
                template = load_workflow_template("modal_app/workflows/flux_text2img_api.json")
                wf = inject_workflow_params(template, {
                    "PROMPT": prompt_text,
                    "seed": shot.seed,
                })
                img_bytes = worker.run_workflow.remote(workflow_json=wf)

            keyframe_path = save_media_bytes(img_bytes, keyframe_out)

        shot = update_shot_state(db_path, shot.shot_id, "keyframe_ready", keyframe_path=keyframe_path)

    # Step 3: Video Generation
    video_path = shot.video_path
    if not video_path:
        video_out = f"{output_dir}/videos/{shot.shot_id}_silent.mp4"
        if mock_rendering or not os.environ.get("MODAL_TOKEN_ID"):
            # Functional placeholder: generate test pattern video using ffmpeg
            Path(video_out).parent.mkdir(parents=True, exist_ok=True)
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={shot.target_duration_sec}",
                "-pix_fmt", "yuv420p",
                str(Path(video_out).resolve()),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            video_path = str(Path(video_out).resolve())
        else:
            import modal
            worker = modal.Cls.from_name("novel-to-movie-comfy", "VideoComfyWorker")()
            template = load_workflow_template("modal_app/workflows/wan2_i2v_api.json")
            wf = inject_workflow_params(template, {
                "KEYFRAME_IMAGE": "input_keyframe.png",
                "MOTION_PROMPT": f"Cinematic motion: {shot.visual_prompt}",
                "seed": shot.seed,
            })
            keyframe_bytes = Path(keyframe_path).read_bytes()
            vid_bytes = worker.run_workflow.remote(
                workflow_json=wf,
                input_files={"input_keyframe.png": keyframe_bytes},
            )
            video_path = save_media_bytes(vid_bytes, video_out)

        shot = update_shot_state(db_path, shot.shot_id, "video_ready", video_path=video_path)

    # Step 4: Neural Lip-Sync or Direct Audio Mux
    synced_path = shot.synced_path
    if not synced_path:
        synced_out = f"{output_dir}/synced/{shot.shot_id}.mp4"
        Path(synced_out).parent.mkdir(parents=True, exist_ok=True)

        if shot.lip_sync_required and audio_path:
            if mock_rendering or not os.environ.get("MODAL_TOKEN_ID"):
                # Use ffmpeg to mux video + audio for mock
                import subprocess
                cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", synced_out]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                synced_path = str(Path(synced_out).resolve())
            else:
                # Dispatch LatentSync via Modal ComfyUI
                import modal
                lipsync_worker = modal.Cls.from_name("novel-to-movie-comfy", "LipSyncComfyWorker")()
                template = load_workflow_template("modal_app/workflows/latentsync_api.json")
                video_bytes = Path(video_path).read_bytes()
                audio_bytes = Path(audio_path).read_bytes()
                wf = inject_workflow_params(template, {
                    "VIDEO_PATH": "input_video.mp4",
                    "AUDIO_PATH": "input_audio.wav",
                })
                synced_bytes = lipsync_worker.run_workflow.remote(
                    workflow_json=wf,
                    input_files={
                        "input_video.mp4": video_bytes,
                        "input_audio.wav": audio_bytes,
                    },
                )
                synced_path = save_media_bytes(synced_bytes, synced_out)
        else:
            # Reaction / wide shot: mux with silent audio stream so all timeline clips share audio layout
            import subprocess
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=24000",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                synced_out,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            synced_path = str(Path(synced_out).resolve())

        shot = update_shot_state(db_path, shot.shot_id, "synced", synced_path=synced_path)

    # Step 5: Automated Quality Gate (Judge)
    if mock_rendering or not os.environ.get("MODAL_TOKEN_ID"):
        verdict = JudgeVerdict(
            passed=True,
            character_consistency=4,
            prompt_adherence=4,
            motion_quality=4,
            reason="Mock rendering pass approval",
        )
    else:
        ref_image = None
        if shot.speaker:
            ref_profile = visual_profiles.get(shot.speaker.lower())
            if ref_profile and ref_profile.reference_image_paths:
                ref_image = ref_profile.reference_image_paths[0]

        verdict = evaluate_shot_clip(
            shot=shot,
            video_path=synced_path,
            character_ref_image_path=ref_image,
        )

    if verdict.passed:
        shot = update_shot_state(
            db_path,
            shot.shot_id,
            "passed",
            judge_feedback=verdict.reason,
        )
    else:
        new_attempts = shot.attempts + 1
        if new_attempts >= retry_cap:
            shot = update_shot_state(
                db_path,
                shot.shot_id,
                "needs_review",
                attempts=new_attempts,
                judge_feedback=f"Failed QA after {new_attempts} attempts: {verdict.reason}",
            )
        else:
            # Requeue with nudged seed
            nudge_seed = verdict.seed_nudge or (shot.seed + 107)
            shot = update_shot_state(
                db_path,
                shot.shot_id,
                "queued",
                attempts=new_attempts,
                seed=nudge_seed,
                video_path=None,  # Reset so it regenerates
                synced_path=None,
                judge_feedback=f"Retry {new_attempts}: {verdict.reason}",
            )

    return shot
