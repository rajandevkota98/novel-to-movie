# PROJECT MEMORY: Novel-to-Movie AI Pipeline

> **Single Source of Truth** for architectural principles, code conventions, model choices, data schemas, and deployment infrastructure.

---

## 1. System Vision & Core Principles

The **Novel-to-Movie AI Pipeline** autonomously transforms public-domain literature into hour-long cinematic drama videos with consistent cast identities, persistent character voices, and coherent scene progression.

### Core Architecture Philosophy
* **Frontier Reasoning for Creative Direction**: Use frontier LLMs (**GPT-5 Luna** / OpenRouter) for scene chunking, dramatic pacing, shot list generation, and multimodal quality gates.
* **Open-Source for Media Synthesis**: Use self-hosted open-source models (**Flux + PuLID**, **Wan 2.2 / LTX**, **F5-TTS**, **LatentSync**) running in **ComfyUI on Modal** for rendering.
* **Zero Visual Drift via Decoupled Keyframes**: Never generate raw text-to-video directly. Always lock character identity onto a still keyframe (Flux + PuLID) before animating with Image-to-Video (Wan 2.2).
* **Idempotency & Resumability**: Every task is keyed by `shot_id`. Crashes or spot preemptions can be resumed without re-rendering completed shots.

---

## 2. Hard Engineering Constraints

### A. Python: Functional Paradigm Only
* **No Stateful OOP Classes**: Strictly avoid classes with mutable state (`self.state`, `self.items.append()`, stateful manager objects).
* **Pure Functions First**: Write functions that take explicit inputs and return new outputs without side effects.
* **Immutable Data Contracts**: All data transfer objects must be immutable Pydantic models (`model_config = ConfigDict(frozen=True)`) or Python `dataclass(frozen=True)`.
* **Isolated Side Effects**: Confine disk, network, and database I/O to explicitly named I/O functions (e.g., `save_shot_to_db()`, `fetch_workflow_result()`).
* **Composition Over Inheritance**: Compose pipeline steps using simple functional pipelines (`pipe(data, step1, step2, step3)`).

### B. Package & Runtime Management: `uv`
* **Exclusively `uv`**: Manage all dependencies, virtual environments, and script executions with `uv`:
  ```bash
  uv add <package>
  uv run python -m pipeline.<module>
  ```
* **No manual `pip install` or `requirements.txt` desync**: The canonical configuration is `pyproject.toml` and `uv.lock`.

### C. Compute & Server Infrastructure: Modal + Headless ComfyUI
* **Modal App Infrastructure**:
  * Headless ComfyUI server executed inside a dedicated Modal container.
  * Shared model checkpoints cached on a persistent **Modal Volume** (`comfy-models-volume`).
  * Tiered GPU allocation:
    * `A10G` / `L4`: Flux + PuLID Keyframes, F5-TTS, LatentSync Lip-Sync.
    * `H100` / `A100`: Wan 2.2 14B Video Generation.
* **ComfyUI API Integration**: ComfyUI workflows are stored as parameterized JSON templates (`modal_app/workflows/*.json`). Functions inject prompt tokens, seeds, and image latents, then submit via ComfyUI's `/prompt` HTTP endpoint.

---

## 3. Model Stack & Selection

| Role | Model | Engine / Platform | Details |
| :--- | :--- | :--- | :--- |
| **Script Breakdown** | **GPT-5 Luna** | OpenRouter / OpenAI API | Deep scene breakdown, shot planning, and dialogue attribution. |
| **Character Visual Keyframes** | **Flux.1-Dev + PuLID-Flux** | ComfyUI on Modal (A10G) | Zero-shot face identity locking without per-character LoRAs. |
| **Video Generation (Quality)** | **Wan 2.2 (14B)** | ComfyUI on Modal (H100) | Apache 2.0 I2V model with high motion coherence. |
| **Video Generation (Fast Mode)** | **LTX-Video** | ComfyUI on Modal (A100) | 2–3x faster iteration speed for draft cuts. |
| **Voice Synthesis (TTS)** | **F5-TTS** | Modal Worker (L4) | Open-source zero-shot voice cloning from 10s audio reference samples. |
| **Neural Lip-Sync** | **LatentSync** | ComfyUI on Modal (A10G) | Latent inpainting for speaking close-ups. |
| **Quality Gate (Judge)** | **GPT-5 Multimodal / Gemini 2.5 Pro** | OpenRouter | Automated QA scoring on sampled clip frames (consistency, adherence, motion). |
| **Assembly & Mixing** | **FFmpeg** | Local / Modal CPU | Lossless concatenation, audio ducking, final 1080p mux. |

---

## 4. Repository Layout

```
vidoe-generation/
├── PROJECT_MEMORY.md               # This document (Architecture & conventions)
├── pyproject.toml                  # UV project configuration and dependencies
├── uv.lock                         # UV lockfile
├── config.yaml                     # Central pipeline configuration
├── assets/
│   ├── characters/                 # Canonical reference portraits (PNG)
│   ├── voices/                     # 10s reference audio files for voice cloning
│   └── ambient/                    # Background ambient music and Foley soundscapes
├── modal_app/
│   ├── __init__.py
│   ├── common.py                   # Modal images, persistent volumes, shared constants
│   ├── comfy_server.py             # Headless ComfyUI runner deployed as Modal App
│   ├── f5_tts_server.py            # F5-TTS zero-shot voice synthesis Modal worker
│   └── workflows/
│       ├── flux_text2img_api.json   # ComfyUI API template for canonical character sheet synthesis
│       ├── flux_pulid_api.json      # ComfyUI API template for Flux + PuLID identity locking
│       ├── wan2_i2v_api.json        # ComfyUI API template for Wan 2.2 I2V
│       ├── ltx_i2v_api.json         # ComfyUI API template for LTX-Video
│       └── latentsync_api.json      # ComfyUI API template for LatentSync
├── pipeline/
│   ├── __init__.py
│   ├── models.py                   # Pure immutable data models (Pydantic frozen=True)
│   ├── state_manager.py            # Functional SQLite state tracking & idempotency
│   ├── breakdown.py                # Functional novel chunking & dynamic cast extractor
│   ├── character_sheet.py          # Functional visual character turnaround sheet synthesis
│   ├── bibles.py                   # Character & voice profile JSON export/import helpers
│   ├── comfy_client.py             # Functional HTTP client for Modal ComfyUI endpoints
│   ├── audio.py                    # Functional TTS dispatcher and audio duration analyzer
│   ├── judge.py                    # Functional multimodal QA gate & retry logic
│   └── assemble.py                 # Functional FFmpeg timeline composer & audio muxer
└── scripts/
    ├── run_pipeline.py             # Main CLI orchestrator
    └── test_checkpoint.py          # Functional checkpoint verification runner
```

---

## 5. Core Data Contracts (`pipeline/models.py`)

All data structures are immutable:

```python
from typing import List, Optional, Tuple
from pydantic import BaseModel, ConfigDict

class ShotDialogue(BaseModel):
    model_config = ConfigDict(frozen=True)
    character: str
    text: str

class ShotJob(BaseModel):
    model_config = ConfigDict(frozen=True)
    shot_id: str                      # Format: "ch01_sc02_sh04"
    sequence_order: int
    chapter_num: int
    scene_num: int
    shot_type: str                    # "close_up", "medium", "wide", "over_the_shoulder"
    characters: Tuple[str, ...]
    speaker: Optional[str]
    dialogue: Tuple[ShotDialogue, ...]
    lip_sync_required: bool
    visual_prompt: str
    mood: str
    target_duration_sec: float
    status: str = "queued"            # queued, processing, passed, needs_review, failed
    attempts: int = 0
    keyframe_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    synced_path: Optional[str] = None

class JudgeVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)
    passed: bool
    character_consistency: int       # 1-5
    prompt_adherence: int            # 1-5
    motion_quality: int              # 1-5
    feedback: str
    retry_prompt_nudge: Optional[str] = None
```

---

## 6. Pipeline Execution Lifecycle

```
[Novel Text]
     │
     ▼
1. breakdown_novel()          ──> SQLite 'shots' table initialized with status='queued'
     │
     ▼
2. generate_character_bibles() ──> Face reference PNGs + Voice reference WAVs registered
     │
     ▼
3. For each shot_id:
     ├─ generate_dialogue_audio()     ──> assets/audio/{shot_id}.wav
     ├─ generate_flux_keyframe()      ──> assets/keyframes/{shot_id}.png
     ├─ generate_wan_video()          ──> assets/video/{shot_id}_silent.mp4
     ├─ apply_lipsync_if_needed()     ──> assets/video/{shot_id}_synced.mp4
     └─ evaluate_clip_with_judge()    ──> If pass: status='passed'
                                          If fail & attempts < 3: status='retry' (nudge seed)
                                          If fail & attempts >= 3: status='needs_review'
     │
     ▼
4. assemble_final_cut()        ──> FFmpeg stitches approved clips, mixes ambient sound, exports 1080p MP4
```

---

## 7. Operational Guidelines for Coding Agents

1. **Keep Functions Pure and Modular**: Functions must have single responsibilities. Do not write a 300-line monolithic function.
2. **Strictly Functional**: Do not introduce classes with mutable properties. If an operation needs shared state, pass state explicitly as immutable arguments and return updated copies.
3. **Always Run Commands via `uv run`**: Never execute bare `python ...` or invoke global environments.
4. **Log State Transitions**: Every transition of `shot_id` status must be recorded in SQLite.
