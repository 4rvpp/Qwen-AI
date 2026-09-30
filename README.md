# Qwen Live Dictation

Click **Start listening** once. After the model is ready, speak continuously.
Text previews appear while recording remains active. Pause briefly to finalize
a phrase; press **Stop** when you want to finish the session.

## Start

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:7861 in Chrome or Edge and allow microphone access.
The previous record/stop app can remain on port 7860 while you copy its text.
Stop its Python process when no longer needed to free memory. The new process
loads its own model once and reuses it across listening sessions.

For a fresh installation, run `install.ps1` then `run.ps1`. Manual installation
instructions are in RECORD_MODE.md. Use the current requirements.txt. The ZIP
excludes Python, the virtual environment, and the model cache.

## What live means here

This Windows version uses Qwen's Transformers backend. It re-recognizes the
active phrase after at least two more seconds of audio arrive. The preview
replaces the previous version of that phrase instead of duplicating words.
This is not Qwen's native vLLM streaming API.

Two seconds is a scheduling interval, not a response-time guarantee. Actual
delay includes inference and any backlog. CPU processing may be slower than
speech. This removes the need to press Stop after every sentence, but does
not guarantee immediate word-by-word output.

## Flow

1. **Prepare:** Start requests microphone access, connects a local WebSocket,
   and loads Qwen. Audio is sent only after the server reports Ready.
2. **Capture:** An AudioWorklet sends mono float32 frames approximately every
   250 ms, preserving the AudioContext's real sample rate.
3. **Buffer:** Python receives audio independently of model inference. An RMS
   energy threshold estimates speech activity. A small pre-roll protects initial
   sounds. This is a heuristic, not a trained voice activity detector.
4. **Preview:** After two additional seconds, the worker recognizes a snapshot
   of the current phrase. Capture continues during inference. Outdated pending
   previews are not individually queued.
5. **Finalize:** About 0.8 seconds of quiet finalizes the phrase. Results carry
   phrase IDs, allowing a final result to replace its preview exactly once.
   Final phrases take priority over new previews.
6. **Bound memory:** Phrases split at 12 seconds even without a pause. Words at
   that boundary can be affected. Pending audio is capped at 45 seconds, plus
   the one snapshot currently in inference. If the CPU falls behind, capture
   stops with a message and accepted audio is processed. The overflow frame
   itself is rejected.
7. **Stop:** The browser flushes its last short frame and releases the microphone.
   Python finalizes the tail and processes accepted pending audio.
8. **Edit/save:** The editor unlocks. Copy or Save .txt before closing. Starting
   again appends to existing text.

Processing time measures one inference call. Waiting audio is queued audio
duration, not an estimate of the time needed to process it.

## Data and failure behavior

- Live audio and transcripts stay in memory; there is no server-side recording
  or transcript database. Save .txt is a user-triggered download.
- Models remain cached under `.cache/huggingface`.
- The server listens on localhost and rejects other-origin WebSocket requests.
- One active recording session per server is supported.
- A disconnect or inference error preserves displayed text; pending audio is
  not recoverable and an unfinished preview may be inaccurate.
- Editing is disabled during listening and finalization to protect manual edits.
- Cancelling preparation releases the microphone, but cannot immediately
  interrupt an already running Python model load or inference thread.
- Quiet speech can be missed by the energy threshold; noise can delay phrase
  finalization. Test with your own microphone. A trained VAD is a future upgrade.

## Files

| File | Role |
| --- | --- |
| app.py | Live mode entry point |
| live_app.py | FastAPI server, WebSocket capture, inference worker |
| live.html | Controls, microphone lifecycle, replaceable previews, export |
| capture.js | AudioWorklet delivering microphone frames |
| stream_buffer.py | Phrase boundaries, partial scheduling, bounded backlog |
| engine.py | Shared Qwen initialization and inference |
| record_app.py | Previous Gradio record/stop implementation |
| check_model.py | Public sample recognition check |
| test_streaming.py | Buffer and protocol tests using a fake recognizer |
| VERIFICATION.md | Verification results and limitations |

Run tests with `.\.venv\Scripts\python.exe -m unittest -v`.
Fake-recognizer tests verify buffering and protocol behavior, not ASR accuracy.

## Future lower-latency options

For native Qwen streaming, connect browser capture to its supported vLLM
streaming state API on a compatible runtime and hardware setup. A hosted
streaming service is another architecture and would send audio to its provider;
this project does not use one.

References: [Qwen ASR](https://github.com/QwenLM/Qwen3-ASR),
[PyTorch installation](https://pytorch.org/get-started/locally/).
