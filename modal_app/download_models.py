"""Modal script to download open-source model weights into the persistent Volume.

Populates 'comfy-models-volume' with:
- Flux.1-Dev & VAE
- PuLID-Flux weights
- Wan 2.2 14B / Wan 2.1 I2V 14B checkpoint
- LatentSync checkpoints
"""

import modal
from modal_app.common import APP_NAME, models_volume

app = modal.App(f"{APP_NAME}-downloader")

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "curl", "git")
    .pip_install("huggingface_hub")
)


@app.function(
    image=download_image,
    volumes={"/models": models_volume},
    timeout=3600,
)
def download_checkpoints():
    import os
    from huggingface_hub import hf_hub_download

    print("Beginning model checkpoint downloads into Modal volume...")
    os.makedirs("/models/checkpoints", exist_ok=True)
    os.makedirs("/models/pulid", exist_ok=True)
    os.makedirs("/models/vae", exist_ok=True)
    os.makedirs("/models/clip", exist_ok=True)
    os.makedirs("/models/diffusion_models", exist_ok=True)

    print("1. Downloading PuLID-Flux weights...")
    hf_hub_download(
        repo_id="guozinan/PuLID",
        filename="pulid_flux_v0.9.1.safetensors",
        local_dir="/models/pulid",
    )

    print("2. Downloading Wan 2.1 / Wan 2.2 I2V 14B weights...")
    hf_hub_download(
        repo_id="Wan-AI/Wan2.1-I2V-14B-480P",
        filename="diffusion_pytorch_model.safetensors",
        local_dir="/models/diffusion_models/wan2.1_i2v_14B",
    )

    print("3. Downloading LatentSync weights...")
    hf_hub_download(
        repo_id="ByteDance/LatentSync",
        filename="latentsync_unet.pt",
        local_dir="/models/checkpoints/latentsync",
    )

    models_volume.commit()
    print("All checkpoints successfully downloaded and committed to volume!")
