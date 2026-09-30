"""Audio validation separate from model loading for lightweight testing."""
import numpy as np


def prepare_audio(audio, max_seconds=60):
    if audio is None:
        raise ValueError("Record some speech first.")
    sample_rate, samples = audio
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("Invalid audio sample rate.")
    samples = np.asarray(samples)
    if samples.ndim not in (1, 2) or samples.size == 0:
        raise ValueError("The recording is empty or has an unsupported shape.")
    if len(samples) / sample_rate > max_seconds:
        raise ValueError(f"Please record at most {max_seconds} seconds per clip.")
    if np.issubdtype(samples.dtype, np.signedinteger):
        scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / scale
    elif np.issubdtype(samples.dtype, np.unsignedinteger):
        midpoint = (np.iinfo(samples.dtype).max + 1) / 2
        samples = (samples.astype(np.float32) - midpoint) / midpoint
    else:
        samples = samples.astype(np.float32)
    if not np.isfinite(samples).all():
        raise ValueError("The recording contains invalid samples.")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return np.clip(samples, -1, 1), int(sample_rate)


def append_text(existing, recognized):
    recognized = recognized.strip()
    if not recognized:
        return existing or ""
    return f"{existing.rstrip()}\n{recognized}" if existing and existing.strip() else recognized
