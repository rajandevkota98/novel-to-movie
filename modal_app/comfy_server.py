"""Headless ComfyUI Server on Modal.

Provides GPU-accelerated endpoints for:
- Flux + PuLID Keyframe generation (A10G)
- Wan 2.2 / LTX Image-to-Video generation (H100)
- LatentSync Neural Lip-Sync (A10G)
- Interactive ComfyUI Web UI (for workflow inspection)
"""

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import modal
from modal_app.common import APP_NAME, comfy_image, models_volume

app = modal.App(APP_NAME)


def _wait_for_comfy(port: int = 8188, timeout_sec: int = 60) -> None:
    """Blocks until local ComfyUI instance responds on HTTP port."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/system_stats"
    while time.time() - start < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"ComfyUI failed to start within {timeout_sec}s")


def _queue_workflow_and_get_output(prompt_workflow: Dict[str, Any], port: int = 8188) -> bytes:
    """Submits workflow API JSON to ComfyUI, polls for generation, and fetches output bytes."""
    prompt_url = f"http://127.0.0.1:{port}/prompt"
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(prompt_url, data=data, headers={"Content-Type": "application/json"})

    with urllib.request.urlopen(req) as resp:
        res_data = json.loads(resp.read().decode("utf-8"))
        prompt_id = res_data["prompt_id"]

    # Poll history endpoint until finished
    history_url = f"http://127.0.0.1:{port}/history/{prompt_id}"
    while True:
        with urllib.request.urlopen(history_url) as resp:
            history = json.loads(resp.read().decode("utf-8"))
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                for node_id, node_output in outputs.items():
                    # Handle image output
                    if "images" in node_output and node_output["images"]:
                        img_info = node_output["images"][0]
                        view_url = (
                            f"http://127.0.0.1:{port}/view?"
                            f"filename={urllib.parse.quote(img_info['filename'])}"
                            f"&subfolder={urllib.parse.quote(img_info.get('subfolder', ''))}"
                            f"&type={urllib.parse.quote(img_info.get('type', 'output'))}"
                        )
                        with urllib.request.urlopen(view_url) as view_resp:
                            return view_resp.read()
                    # Handle video/gifs output (Wan 2.2 / VHS_VideoCombine)
                    if "gifs" in node_output and node_output["gifs"]:
                        video_info = node_output["gifs"][0]
                        view_url = (
                            f"http://127.0.0.1:{port}/view?"
                            f"filename={urllib.parse.quote(video_info['filename'])}"
                            f"&subfolder={urllib.parse.quote(video_info.get('subfolder', ''))}"
                            f"&type={urllib.parse.quote(video_info.get('type', 'output'))}"
                        )
                        with urllib.request.urlopen(view_url) as view_resp:
                            return view_resp.read()
        time.sleep(1)


@app.cls(
    gpu="A10G",
    image=comfy_image,
    volumes={"/root/ComfyUI/models": models_volume},
    container_idle_timeout=120,
    timeout=600,
)
class KeyframeComfyWorker:
    """Worker handling Flux.1 + PuLID keyframe rendering on A10G."""

    @modal.enter()
    def start_comfy(self):
        cmd = ["python", "/root/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
        self.process = subprocess.Popen(cmd)
        _wait_for_comfy(port=8188)

    @modal.method()
    def run_workflow(self, workflow_json: Dict[str, Any], input_files: Optional[Dict[str, bytes]] = None) -> bytes:
        if input_files:
            input_dir = Path("/root/ComfyUI/input")
            input_dir.mkdir(parents=True, exist_ok=True)
            for filename, file_bytes in input_files.items():
                (input_dir / filename).write_bytes(file_bytes)
        return _queue_workflow_and_get_output(workflow_json, port=8188)


@app.cls(
    gpu="H100",
    image=comfy_image,
    volumes={"/root/ComfyUI/models": models_volume},
    container_idle_timeout=180,
    timeout=1200,
)
class VideoComfyWorker:
    """Worker handling Wan 2.2 / LTX video rendering on H100."""

    @modal.enter()
    def start_comfy(self):
        cmd = ["python", "/root/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
        self.process = subprocess.Popen(cmd)
        _wait_for_comfy(port=8188)

    @modal.method()
    def run_workflow(self, workflow_json: Dict[str, Any], input_files: Optional[Dict[str, bytes]] = None) -> bytes:
        if input_files:
            input_dir = Path("/root/ComfyUI/input")
            input_dir.mkdir(parents=True, exist_ok=True)
            for filename, file_bytes in input_files.items():
                (input_dir / filename).write_bytes(file_bytes)
        return _queue_workflow_and_get_output(workflow_json, port=8188)


@app.cls(
    gpu="A10G",
    image=comfy_image,
    volumes={"/root/ComfyUI/models": models_volume},
    container_idle_timeout=120,
    timeout=600,
)
class LipSyncComfyWorker:
    """Worker handling LatentSync neural lip-sync on A10G."""

    @modal.enter()
    def start_comfy(self):
        cmd = ["python", "/root/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"]
        self.process = subprocess.Popen(cmd)
        _wait_for_comfy(port=8188)

    @modal.method()
    def run_workflow(self, workflow_json: Dict[str, Any], input_files: Optional[Dict[str, bytes]] = None) -> bytes:
        if input_files:
            input_dir = Path("/root/ComfyUI/input")
            input_dir.mkdir(parents=True, exist_ok=True)
            for filename, file_bytes in input_files.items():
                (input_dir / filename).write_bytes(file_bytes)
        return _queue_workflow_and_get_output(workflow_json, port=8188)


@app.function(
    gpu="A10G",
    image=comfy_image,
    volumes={"/root/ComfyUI/models": models_volume},
    timeout=3600,
)
@modal.web_server(port=8188, startup_timeout=120)
def ui():
    """Interactive ComfyUI web UI session on Modal for visual testing and node setup."""
    subprocess.Popen(["python", "/root/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"])
