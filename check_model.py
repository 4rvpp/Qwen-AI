"""Download/load Qwen and transcribe the public example from its official README."""
import time
from engine import DEVICE, MODEL_ID, get_model

if __name__ == "__main__":
    print(f"Loading {MODEL_ID} on {DEVICE}", flush=True)
    start = time.perf_counter()
    model = get_model()
    print(f"Model loaded in {time.perf_counter() - start:.1f}s", flush=True)
    results = model.transcribe(
        audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        language="English",
    )
    if not results or not results[0].text.strip():
        raise RuntimeError("The sample did not produce a transcript.")
    print("Language:", results[0].language, flush=True)
    print("Transcript:", results[0].text, flush=True)
