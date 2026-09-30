"""One shared local Qwen model, initialized lazily."""
import os
from pathlib import Path
from threading import Lock
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
MODEL_ID = os.getenv('QWEN_MODEL', 'Qwen/Qwen3-ASR-0.6B')
DEVICE = os.getenv('QWEN_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
_model = None
_lock = Lock()

def get_model():
    global _model
    with _lock:
        if _model is None:
            from qwen_asr import Qwen3ASRModel
            dtype = torch.float32
            if DEVICE.startswith('cuda'):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            _model = Qwen3ASRModel.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=dtype,
                max_inference_batch_size=1, max_new_tokens=256)
    return _model

def transcribe(samples, rate, language):
    model = get_model()
    with _lock, torch.inference_mode():
        return model.transcribe(audio=(samples, rate), language=None if language == 'Auto' else language)[0].text.strip()
