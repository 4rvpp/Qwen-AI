"""Continuous capture; inference runs separately so incoming audio keeps flowing."""
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from stream_buffer import StreamBuffer

ROOT = Path(__file__).resolve().parent

def create_app(load_model=None, recognize=None):
    if load_model is None or recognize is None:
        from engine import get_model, transcribe
        load_model, recognize = get_model, transcribe
    app = FastAPI()
    gate = asyncio.Lock()

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'live.html')

    @app.get('/capture.js')
    async def worklet():
        return FileResponse(ROOT / 'capture.js', media_type='text/javascript')

    @app.websocket('/listen')
    async def listen(ws: WebSocket):
        origin = urlparse(ws.headers.get('origin', ''))
        if origin.netloc != ws.headers.get('host') or origin.scheme not in ('http', 'https'):
            await ws.close(code=1008)
            return
        await ws.accept()
        if gate.locked():
            await ws.send_json({'type':'error','message':'Another recording is active. Finish it first.'})
            await ws.close()
            return
        async with gate:
            worker = None
            try:
                config = await asyncio.wait_for(ws.receive_json(), 15)
                rate = int(config['rate'])
                language = config.get('language', 'Auto')
                if language not in ['Auto','English','Malay','Indonesian','Chinese','Arabic']:
                    raise ValueError('Unsupported language selection')
                buffer = StreamBuffer(rate)
                await ws.send_json({'type':'status','message':'Loading Qwen into memory. Capture starts when ready.'})
                await asyncio.to_thread(load_model)
                await ws.send_json({'type':'ready'})
                changed = asyncio.Event()

                async def process():
                    try:
                        while True:
                            job = buffer.next_job()
                            if job is None:
                                if buffer.stopped:
                                    await ws.send_json({'type':'done'})
                                    await ws.close()
                                    return
                                changed.clear()
                                await changed.wait()
                                continue
                            phrase, samples, final = job
                            started = perf_counter()
                            await ws.send_json({'type':'status','message':'Recognizing while listening...' if not buffer.stopped else 'Finishing remaining speech...'})
                            text = await asyncio.to_thread(recognize, samples, rate, language)
                            await ws.send_json({'type':'text','id':phrase,'text':text,'final':final,
                                'seconds':round(perf_counter()-started,1),'backlog':round(buffer.backlog,1)})
                    except Exception as error:
                        try:
                            await ws.send_json({'type':'error','message':f'Recognition failed: {error}. Earlier text is preserved.'})
                            await ws.close()
                        except Exception:
                            pass

                worker = asyncio.create_task(process())
                while True:
                    packet = await ws.receive()
                    if packet['type'] == 'websocket.disconnect':
                        break
                    if packet.get('bytes') is not None:
                        raw = packet['bytes']
                        if len(raw) % 4 or len(raw) > rate*4:
                            raise ValueError('Invalid audio packet')
                        try:
                            buffer.add(np.frombuffer(raw, dtype='<f4'))
                        except OverflowError as error:
                            buffer.stop()
                            await ws.send_json({'type':'stop_capture','message':str(error)})
                        changed.set()
                    elif packet.get('text') is not None and json.loads(packet['text']).get('type') == 'stop':
                        buffer.stop()
                        changed.set()
                        await worker
                        break
            except WebSocketDisconnect:
                pass
            except Exception as error:
                try:
                    await ws.send_json({'type':'error','message':str(error)})
                    await ws.close()
                except Exception:
                    pass
            finally:
                if worker and not worker.done():
                    worker.cancel()
                    try:
                        await worker
                    except asyncio.CancelledError:
                        pass
    return app

def main():
    import uvicorn
    from engine import ATTN_IMPLEMENTATION, DEVICE
    port = int(os.getenv('QWEN_PORT', '7861'))
    print(f'Live dictation ({DEVICE}; attention={ATTN_IMPLEMENTATION}): http://127.0.0.1:{port}')
    uvicorn.run(create_app(), host='127.0.0.1', port=port, ws_max_size=400000)

if __name__ == '__main__':
    main()
