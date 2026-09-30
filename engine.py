"""One shared local Qwen model, initialized lazily."""
import os
import sys
import types
from importlib.machinery import ModuleSpec
from math import gcd
from pathlib import Path
from threading import Lock
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
MODEL_ID = os.getenv('QWEN_MODEL', 'Qwen/Qwen3-ASR-0.6B')
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = os.getenv('QWEN_DEVICE', 'cuda:0' if CUDA_AVAILABLE else 'cpu')
# Qwen ASR's nested configurations can otherwise select PyTorch SDPA on CUDA.
# Use the portable eager implementation by default: PyTorch 2.7/cu128 has had
# SDPA failures on some Blackwell laptop GPUs. Set QWEN_ATTN_IMPLEMENTATION=sdpa
# only after confirming it is stable on this machine.
ATTN_IMPLEMENTATION = os.getenv('QWEN_ATTN_IMPLEMENTATION', 'eager')
_model = None
_lock = Lock()

def install_librosa_compat():
    """Provide Qwen's two librosa calls without importing numba/llvmlite.

    On this Linux build, loading llvmlite through librosa raises SIGBUS before
    Qwen sees the microphone samples. Qwen ASR uses only ``load`` and
    ``resample`` from librosa, both of which have safe equivalents here.
    """
    if 'librosa' in sys.modules:
        return
    from scipy.signal import resample_poly
    import soundfile as sf

    compat = types.ModuleType('librosa')
    # Transformers checks package availability using find_spec().
    compat.__spec__ = ModuleSpec('librosa', loader=None)

    def load(path, sr=None, mono=True):
        samples, rate = sf.read(path, dtype='float32', always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if mono and samples.ndim == 2:
            samples = np.mean(samples, axis=-1, dtype=np.float32)
        if sr is not None and rate != sr:
            samples = resample(samples, orig_sr=rate, target_sr=sr)
            rate = sr
        return samples, rate

    def resample(samples, *, orig_sr, target_sr, **_):
        if orig_sr == target_sr:
            return np.asarray(samples, dtype=np.float32)
        divisor = gcd(int(orig_sr), int(target_sr))
        return resample_poly(samples, int(target_sr) // divisor, int(orig_sr) // divisor).astype(np.float32)

    compat.load = load
    compat.resample = resample
    sys.modules['librosa'] = compat

def validate_cuda():
    """Fail before model loading if the installed wheel cannot run this GPU."""
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            'QWEN_DEVICE requests CUDA, but this PyTorch installation cannot access an NVIDIA GPU. '
            'Run install-gpu.sh (Linux) or install-gpu.ps1 (Windows), or set QWEN_DEVICE=cpu.'
        )
    capability = torch.cuda.get_device_capability(0)
    architecture = f'sm_{capability[0]}{capability[1]}'
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f'The installed PyTorch wheel does not support {architecture} ({torch.cuda.get_device_name(0)}). '
            'Install a CUDA 12.8+ PyTorch build with install-gpu.sh or install-gpu.ps1.'
        )

def configure_attention(model):
    """Apply the selected backend to every nested Qwen ASR configuration.

    Qwen ASR contains several nested models. Passing ``attn_implementation``
    only to ``from_pretrained`` does not reliably reach all of them.
    """
    for module in model.modules():
        config = getattr(module, 'config', None)
        if config is not None:
            config._attn_implementation = ATTN_IMPLEMENTATION
            config._attn_implementation_internal = ATTN_IMPLEMENTATION

def configure_generation(model):
    """Remove Qwen's sampling temperature from greedy transcription.

    The shipped generation config sets ``do_sample=false`` and a non-default
    temperature. Newer Transformers correctly ignores that combination but
    logs a warning for every process.
    """
    for module in model.modules():
        generation_config = getattr(module, 'generation_config', None)
        if generation_config is not None and not generation_config.do_sample:
            generation_config.temperature = None

def get_model():
    global _model
    with _lock:
        if _model is None:
            if DEVICE.startswith('cuda'):
                validate_cuda()
            # Qwen's bundled generation_config has a sampling temperature
            # together with do_sample=False. Transformers warns about it while
            # loading; it is irrelevant to greedy ASR output.
            from transformers.utils import logging as transformers_logging
            transformers_logging.set_verbosity_error()
            install_librosa_compat()
            from qwen_asr import Qwen3ASRModel
            dtype = torch.float32
            if DEVICE.startswith('cuda'):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            _model = Qwen3ASRModel.from_pretrained(
                MODEL_ID,
                device_map=DEVICE,
                dtype=dtype,
                attn_implementation=ATTN_IMPLEMENTATION,
                max_inference_batch_size=1,
                max_new_tokens=256,
            )
            configure_attention(_model.model)
            configure_generation(_model.model)
    return _model

def transcribe(samples, rate, language):
    model = get_model()
    with _lock, torch.inference_mode():
        return model.transcribe(audio=(samples, rate), language=None if language == 'Auto' else language)[0].text.strip()
