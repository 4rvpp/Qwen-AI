import unittest
import time
import numpy as np
from fastapi.testclient import TestClient
from live_app import create_app
from stream_buffer import StreamBuffer

RATE=16000
VOICE=np.full(4000,.1,dtype=np.float32)
QUIET=np.zeros(4000,dtype=np.float32)

class BufferTests(unittest.TestCase):
    def test_partial_replaces_same_phrase_and_stop_flushes_tail(self):
        buf=StreamBuffer(RATE)
        for _ in range(8): buf.add(VOICE)
        first=buf.next_job()
        self.assertEqual((first[0],first[2]),(0,False))
        self.assertIsNone(buf.next_job())
        buf.add(VOICE)
        buf.stop()
        final=buf.next_job()
        self.assertEqual((final[0],final[2],len(final[1])),(0,True,36000))
        self.assertIsNone(buf.next_job())

    def test_pause_finalizes_once_and_silence_is_not_transcribed(self):
        buf=StreamBuffer(RATE)
        for _ in range(20): buf.add(QUIET)
        self.assertIsNone(buf.next_job())
        buf.add(VOICE)
        for _ in range(4): buf.add(QUIET)
        self.assertTrue(buf.next_job()[2])
        self.assertIsNone(buf.next_job())
        buf.stop()
        self.assertIsNone(buf.next_job())

    def test_backlog_is_bounded(self):
        buf=StreamBuffer(RATE,max_pending=1)
        for _ in range(4): buf.add(VOICE)
        with self.assertRaises(OverflowError): buf.add(VOICE)
        buf.stop()
        self.assertEqual(len(buf.next_job()[1]),RATE)

class ProtocolTests(unittest.TestCase):
    def test_capture_continues_during_inference_and_stop_drains(self):
        calls=[]
        def recognize(samples,rate,language):
            time.sleep(.05)
            calls.append(len(samples))
            return f'words {len(samples)}'
        with TestClient(create_app(lambda:None,recognize)) as client:
            self.assertEqual(client.get('/').status_code,200)
            with client.websocket_connect('/listen',headers={'origin':'http://testserver'}) as ws:
                ws.send_json({'rate':RATE,'language':'English'})
                self.assertEqual(ws.receive_json()['type'],'status')
                self.assertEqual(ws.receive_json()['type'],'ready')
                for _ in range(8): ws.send_bytes(VOICE.tobytes())
                self.assertEqual(ws.receive_json()['type'],'status')
                ws.send_bytes(VOICE.tobytes())
                ws.send_json({'type':'stop'})
                messages=[]
                while True:
                    message=ws.receive_json();messages.append(message)
                    if message['type']=='done':break
                texts=[m for m in messages if m['type']=='text']
                self.assertEqual([(m['id'],m['final']) for m in texts],[(0,False),(0,True)])
                self.assertEqual(calls,[32000,36000])

    def test_error_is_reported(self):
        def broken(*args):raise RuntimeError('test failure')
        with TestClient(create_app(lambda:None,broken)) as client:
            with client.websocket_connect('/listen',headers={'origin':'http://testserver'}) as ws:
                ws.send_json({'rate':RATE});ws.receive_json();ws.receive_json()
                ws.send_bytes(VOICE.tobytes());ws.send_json({'type':'stop'})
                self.assertEqual(ws.receive_json()['type'],'status')
                self.assertIn('test failure',ws.receive_json()['message'])

    def test_cross_origin_rejected(self):
        from starlette.websockets import WebSocketDisconnect
        with TestClient(create_app(lambda:None,lambda *args:'')) as client:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect('/listen',headers={'origin':'https://example.com'}):pass

if __name__=='__main__':unittest.main()
