"""F5-TTS zero-shot voice cloning server deployed on Modal.

Runs on an L4 GPU to synthesize dialogue matching character reference audio.
"""

import io
import modal

APP_NAME = "novel-to-movie-tts"
tts_volume = modal.Volume.from_name("tts-models-volume", create_if_missing=True)

f5_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4.0",
        "torchaudio",
        "soundfile",
        "f5-tts",
        "cached_path",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App(APP_NAME)


@app.cls(
    gpu="L4",
    image=f5_image,
    volumes={"/root/.cache": tts_volume},
    container_idle_timeout=120,
    timeout=300,
)
class F5TTSEngine:
    """Zero-shot voice cloning inference engine."""

    @modal.enter()
    def load_model(self):
        from f5_tts.api import F5TTS
        self.tts = F5TTS(model_type="F5-TTS_Base")

    @modal.method()
    def synthesize(
        self,
        text: str,
        ref_audio_bytes: bytes,
        ref_text: str = "",
    ) -> bytes:
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp:
            ref_tmp.write(ref_audio_bytes)
            ref_tmp.flush()
            ref_audio_path = ref_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
            out_path = out_tmp.name

        # Run F5-TTS inference
        wav, sr, _ = self.tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=text,
            file_wave_dir=None,
        )
        sf.write(out_path, wav, sr)

        with open(out_path, "rb") as f:
            generated_bytes = f.read()

        return generated_bytes
