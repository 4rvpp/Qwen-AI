"""Local Qwen dictation: stop recording -> recognize -> append editable text."""
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache" / "audio"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
import torch
from audio_utils import append_text, prepare_audio

MODEL_ID = os.getenv("QWEN_MODEL", "Qwen/Qwen3-ASR-0.6B")
DEVICE = os.getenv("QWEN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
_model = None
_lock = Lock()


def get_model():
    global _model
    if _model is None:
        from qwen_asr import Qwen3ASRModel
        dtype = torch.float32
        if DEVICE.startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        _model = Qwen3ASRModel.from_pretrained(
            MODEL_ID, device_map=DEVICE, dtype=dtype,
            max_inference_batch_size=1, max_new_tokens=512,
        )
    return _model


def recognize(audio, language, existing):
    start = perf_counter()
    try:
        samples, sample_rate = prepare_audio(audio)
        # A basic silence guard, not a speech detector. Noise can still pass.
        if float(np.sqrt(np.mean(samples ** 2))) < 0.001:
            return existing, "Very quiet recording. Speak closer to the microphone."
        with _lock, torch.inference_mode():
            result = get_model().transcribe(
                audio=(samples, sample_rate),
                language=None if language == "Auto" else language,
            )[0]
        text = result.text.strip()
        if not text:
            return existing, "No speech recognized. Try another recording."
        return append_text(existing, text), (
            f"Added text · language: {result.language} · "
            f"{perf_counter() - start:.1f}s · {DEVICE}"
        )
    except ValueError as error:
        return existing, str(error)
    except Exception as error:
        # Existing user text survives a failed recognition request.
        return existing, f"Recognition failed ({type(error).__name__}): {error}"


def dictate(audio, language, existing):
    yield gr.update(interactive=False), "Recognizing speech... First use also downloads and loads the model.", gr.update(interactive=False)
    text, message = recognize(audio, language, existing)
    yield gr.update(value=text, interactive=True), message, gr.update(interactive=True)


with gr.Blocks(title="Qwen Voice Typing", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# Qwen Voice Typing\nRecord a sentence, then press **Stop**. Your words appear below automatically.")
    gr.Markdown(
        f"Model: `{MODEL_ID}` · Device: `{DEVICE}`. "
        "The first transcription downloads and loads the model. "
        "Keep each clip under 60 seconds; wait for recognition before editing or recording again."
    )
    language = gr.Dropdown(["Auto", "English", "Malay", "Indonesian", "Chinese", "Arabic"], value="Auto", label="Spoken language")
    audio = gr.Audio(sources=["microphone"], type="numpy", format="wav", label="Microphone", buttons=[])
    transcript = gr.Textbox(label="Your text", lines=12, interactive=True, buttons=["copy"])
    status = gr.Textbox(label="Status", value="Ready. Allow microphone access when your browser asks.", interactive=False)
    gr.Markdown("Audio is processed on this computer. Temporary recordings may remain in `.cache/audio` for about an hour while the app runs. Text is not saved automatically; copy it before closing.")
    audio.stop_recording(dictate, inputs=[audio, language, transcript], outputs=[transcript, status, audio], concurrency_limit=1, concurrency_id="asr", trigger_mode="once")


if __name__ == "__main__":
    demo.queue(max_size=4).launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)
