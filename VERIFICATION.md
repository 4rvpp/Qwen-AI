# Verification: live dictation update

2026-09-30, Windows / Python 3.12.13 / CPU.

- The user confirmed the original record/stop transcription completed.
- Model weights are now present in the local cache.
- Added WebSockets dependency and launched live mode on localhost:7861.
- Nine tests passed: audio conversion, invalid audio, text preservation, live
  preview/final phrase IDs, final short-frame flush, pause detection, bounded
  backlog, capture during delayed inference, error reporting, and origin checks
  are covered across those tests.
- Verified the page opens and renders in the browser.
- Real Qwen public-sample test produced a nonempty preview before Stop, then
  revised that same phrase with more audio. These were actual model results.
- First measured preview inference: 30.8 seconds. Second: 37.0 seconds.
  This CPU configuration does NOT achieve immediate or real-time typing.
- The real WebSocket sample test completed successfully: previews were returned
  before Stop, followed by two nonempty final phrases and a completion message.
  Final inference times were 249.6 seconds for the longer phrase and 16.3 seconds
  for the remaining tail. Capture/protocol worked; CPU throughput is inadequate
  for immediate dictation. The test session ended and released the server.
- End-to-end browser microphone timing has not been measured with the user's
  microphone. Protocol tests inject audio directly and do not verify microphone
  permissions or hardware capture.

The server is intended for one local user. Chunked inference uses the existing
Transformers model; it is not native Qwen vLLM streaming. Native streaming or
faster hardware would require a separate setup and performance verification.

The previous record/stop source is retained as record_app.py. Its old guide is
RECORD_MODE.md. Current behavior and instructions are in README.md.
