# Qwen Live Dictation

This project types speech into an editable browser text box using a local
Qwen3-ASR model. Click **Start listening** once, speak, and pause briefly between
phrases. The current phrase is updated while recording remains active. Click
**Stop** when you want to finish; the last phrase is processed before editing
is enabled. No API key is needed.

## Set up on this computer

The project is already installed at
`C:\Users\arifi\Documents\Codex\2026-09-30\help\outputs\qwen-voice-typing`.
Open **PowerShell**, then run:

```powershell
cd "C:\Users\arifi\Documents\Codex\2026-09-30\help\outputs\qwen-voice-typing"
.\run.ps1
```

If PowerShell prevents scripts from running, use the equivalent Python command:

```powershell
.\.venv\Scripts\python.exe app.py
```

Leave the PowerShell window open while using the app. Open
http://127.0.0.1:7861 in Chrome or Edge. Click **Start listening**, allow that
page to use the microphone when the browser asks, wait for **Listening**, and
then speak. Click **Stop** to finish. Use **Copy text** or **Save .txt** before
closing or refreshing the page. Press **Ctrl+C** in PowerShell to stop the server.

The old record/stop app can remain open on port 7860 while you copy its text.
The live app uses a separate Python process and loads another copy of the model,
so close the old app when finished to free memory. Text from the old tab does
not automatically move to the new tab; copy and paste it if you want to keep it.

## Set up from the ZIP on another Windows computer

1. Extract `qwen-voice-typing.zip` to a folder you can write to. Open PowerShell
   in that extracted folder. The ZIP contains source files, not Python packages
   or model weights.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
   from its official instructions if `uv --version` does not work. You need an
   internet connection for the initial setup and model download.
3. In the extracted project folder, run:

   ```powershell
   .\install.ps1
   .\run.ps1
   ```

   `install.ps1` installs Python 3.12, creates a private `.venv` environment,
   and installs `requirements.txt`. It can download several hundred megabytes
   of dependencies. `run.ps1` starts the live app on port 7861.

4. Open http://127.0.0.1:7861 and allow microphone access. The first use also
   downloads the Qwen3-ASR-0.6B model, whose weight file is about 1.88 GB, then
   loads it into memory. Later listening sessions in the same running server
   reuse that in-memory model. A restart reloads it from the local cache.

If PowerShell blocks `.ps1` scripts, run these commands directly in the project
folder without changing the system execution policy:

```powershell
$env:UV_CACHE_DIR = "$PWD\.cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.cache\python"
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If port 7861 is already in use, close the previous live server with **Ctrl+C**.
If you intentionally need another port, set `$env:QWEN_PORT = '7862'` before
starting Python and open the matching localhost URL.

## Use an NVIDIA GPU

The app automatically uses `cuda:0` when PyTorch can see an NVIDIA CUDA GPU;
otherwise it uses CPU. The normal installer can install a CPU-only PyTorch
build, so run the matching setup once after installing the application.

### Windows

```powershell
.\install-gpu.ps1
```

### Linux

```bash
chmod +x install-gpu.sh
./install-gpu.sh
```

Both scripts install the current CUDA 12.8 PyTorch wheel, required for NVIDIA Blackwell
GPUs such as the RTX 5070 Laptop (`sm_120`), and verify that the operating
system can access the GPU. Restart the app afterward; its first line must say
`Live dictation (cuda:0)`. If the script reports CUDA unavailable, install or
update the NVIDIA driver and rerun it. The full CUDA Toolkit is not required.
To force CPU mode, set `QWEN_DEVICE=cpu` before starting the app.

If PyTorch warns that it supports only up to `sm_90`, the old wheel is still
active. Stop every running app process, rerun the matching GPU installer, then
verify the exact environment used by the app:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

The output must show CUDA `12.8` (or newer) and include `sm_120` before you
start `app.py`.

Run `.venv/bin/python check_gpu.py` before the app. It verifies a BF16 CUDA
matrix multiply independently of Qwen. If it passes but Qwen later crashes,
the failure is within Qwen/Transformers rather than the NVIDIA driver or the
PyTorch CUDA installation.

On Blackwell laptop GPUs the app defaults to Qwen's portable `eager` attention
backend, avoiding the CUDA SDPA path that can crash during the first generated
token. Startup reports `attention=eager`; leave that setting in place unless
you have separately verified that `QWEN_ATTN_IMPLEMENTATION=sdpa` is stable.

## Check that setup worked

From the project folder, check the installed packages and run the tests:

```powershell
uv pip check --python .venv\Scripts\python.exe
.\.venv\Scripts\python.exe -m unittest -v
```

The package check should report that all installed packages are compatible.
The tests should report **9 tests, OK**. They check audio buffering and the
local WebSocket behavior; they do not measure recognition speed or microphone
quality. To check Qwen with its public sample:

```powershell
.\.venv\Scripts\python.exe -u check_model.py
```

That command can take a long time on CPU. For a first microphone test, say one
short sentence, wait for text, then click Stop. A browser permission error means
the browser or Windows microphone access must be enabled. A Qwen download error
usually means the model files could not be fetched; confirm internet access and
retry `check_model.py`.

## What live means here

This Windows version uses Qwen's Transformers backend. It re-recognizes the
active phrase after at least two more seconds of audio arrive. The preview
replaces the previous version of that phrase instead of duplicating words.
This is not Qwen's native vLLM streaming API.

Two seconds is a scheduling interval, not a response-time guarantee. Actual
delay includes inference and any backlog. On the CPU used for this project's
test, the first previews took about **31–37 seconds**; a longer phrase took
about **250 seconds**. This removes the need to press Stop after every sentence,
but this computer does not deliver immediate word-by-word output. You would
need a faster inference setup to reduce that delay substantially.

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
