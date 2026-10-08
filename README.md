# Novel-to-Movie AI Pipeline

Autonomous end-to-end pipeline that transforms public-domain literature into an hour-long cinematic drama.

## Core Stack
- **Language & Paradigm**: Python 3.11+ strictly following **Functional Programming** (pure functions, immutable Pydantic models).
- **Package Manager**: Managed with [`uv`](https://docs.astral.sh/uv/).
- **Creative Reasoning**: Frontier LLMs (**GPT-5 Luna** via OpenRouter) for scene breakdown, shot extraction, and multimodal QA.
- **Rendering & Synthesis**: Open-source models running in headless **ComfyUI on Modal**:
  - **Flux.1-Dev + PuLID-Flux**: Identity-locked facial keyframes (A10G).
  - **Wan 2.2 14B / LTX-Video**: High-coherence Image-to-Video generation (H100/A100).
  - **F5-TTS**: Zero-shot voice cloning from 10s audio reference samples (L4).
  - **LatentSync**: Diffusion-based neural lip-sync for speaking close-ups (A10G).
  - **FFmpeg**: Lossless timeline concatenation and audio mixing.

---

## Quickstart with `uv`

### 1. Environment Setup
```bash
# Sync dependencies
uv sync

# Run tests
uv run pytest
```

### 2. Initialize Database & Inspect Status
```bash
# Initialize SQLite shot queue
uv run python scripts/run_pipeline.py init-db

# Check queue status
uv run python scripts/run_pipeline.py status
```

### 3. Parse Novel Chapter
```bash
export OPENROUTER_API_KEY="your-key"
uv run python scripts/run_pipeline.py breakdown path/to/chapter1.txt --chapter-num 1
```

For full architectural contracts and operational rules, see [PROJECT_MEMORY.md](PROJECT_MEMORY.md).
