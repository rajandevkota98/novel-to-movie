"""Common Modal configuration, container images, and persistent volumes.

Pure functional specifications for remote GPU execution.
"""

import modal

APP_NAME = "novel-to-movie-comfy"
MODELS_VOLUME_NAME = "comfy-models-volume"

# Modal persistent volume to cache multi-gigabyte models (Wan 2.2, Flux.1, LatentSync)
models_volume = modal.Volume.from_name(MODELS_VOLUME_NAME, create_if_missing=True)

# CUDA-enabled base image for ComfyUI
comfy_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "wget",
        "curl",
    )
    .pip_install(
        "torch>=2.4.0",
        "torchvision",
        "torchaudio",
        "accelerate",
        "transformers",
        "safetensors",
        "sentencepiece",
        "aiohttp",
        "websocket-client",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "pydantic>=2.10.0",
        "onnxruntime",
        "insightface",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        # Clone headless ComfyUI into /root/ComfyUI
        "git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI",
        "pip install -r /root/ComfyUI/requirements.txt",
        # Clone essential custom nodes
        "git clone https://github.com/ltdrdata/ComfyUI-Manager.git /root/ComfyUI/custom_nodes/ComfyUI-Manager",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /root/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper || true",
        "git clone https://github.com/Balus-T/ComfyUI-PuLID-Flux.git /root/ComfyUI/custom_nodes/ComfyUI-PuLID-Flux || true",
        "git clone https://github.com/chaojie/ComfyUI-LatentSync.git /root/ComfyUI/custom_nodes/ComfyUI-LatentSync || true",
    )
)
