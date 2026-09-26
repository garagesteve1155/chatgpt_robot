#!/usr/bin/env python3
"""Shared direct Internet link for Echo's desktop brain and Raspberry Pi body."""
from __future__ import annotations
import asyncio, hashlib, hmac, json, queue, ssl, struct, threading, time, uuid
from pathlib import Path
from typing import Any, Optional
import numpy as np

PACKET_AUDIO=1
PACKET_VIDEO=2
PACKET_HEADER=struct.Struct("!BQ")

def load_json(path,default=None):
    path=Path(path)
    if not path.exists():return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

def normalize_fingerprint(value):
    return "".join(ch for ch in str(value).lower() if ch in "0123456789abcdef")

def certificate_sha256_from_der(cert_der):
    return hashlib.sha256(cert_der).hexdigest()

def auth_digest(shared_secret,nonce,device_id):
    msg=f"{nonce}|{device_id}".encode()
    return hmac.new(shared_secret.encode(),msg,hashlib.sha256).hexdigest()

def pack_binary(kind,payload):
    return PACKET_HEADER.pack(int(kind),int(time.monotonic()*1000))+payload

def unpack_binary(raw):
    if len(raw)<PACKET_HEADER.size:raise ValueError("short Echo binary packet")
    kind,stamp=PACKET_HEADER.unpack(raw[:PACKET_HEADER.size])
    return kind,stamp,raw[PACKET_HEADER.size:]

class DesktopBodyServer:
    def __init__(self,host,port,cert_file,key_file,shared_secret,expected_device_id="echo"):
        self.host=host; self.port=int(port); self.cert_file=str(cert_file); self.key_file=str(key_file)
        self.shared_secret=str(shared_secret); self.expected_device_id=str(expected_device_id)
        self.connected=threading.Event(); self.stop_event=threading.Event()
        self.loop=None; self.thread=None; self.websocket=None; self._send_lock=None
        self._rpc_lock=threading.Lock(); self._pending={}
        self._frame_lock=threading.Lock(); self._latest_frame_jpeg=None; self._frame_counter=0
        self.audio_queue=queue.Queue(maxsize=2500)
        self.status_lock=threading.Lock(); self.status={}

    def start(self):
        if self.thread and self.thread.is_alive():return
        self.thread=threading.Thread(target=self._thread_main,name="EchoBodyServer",daemon=True)
        self.thread.start()

    def _thread_main(self):
        self.loop=asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._run_server())

    async def _run_server(self):
        import websockets
        ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert_file,self.key_file)
        self._send_lock=asyncio.Lock()
        async with websockets.serve(
            self._handler,self.host,self.port,ssl=ctx,max_size=None,
            ping_interval=20,ping_timeout=20,compression=None
        ):
            print(f"[body-link] listening on wss://{self.host}:{self.port}",flush=True)
            while not self.stop_event.is_set():await asyncio.sleep(.5)

    async def _handler(self,websocket,*_):
        if self.websocket is not None:
            try:await self.websocket.close(code=4001,reason="new Echo body connection replaced old connection")
            except Exception:pass
        nonce=uuid.uuid4().hex+uuid.uuid4().hex
        await websocket.send(json.dumps({"type":"challenge","nonce":nonce}))
        raw=await asyncio.wait_for(websocket.recv(),timeout=15)
        if not isinstance(raw,str):
            await websocket.close(code=4003,reason="authentication required"); return
        try:auth=json.loads(raw)
        except Exception:
            await websocket.close(code=4003,reason="invalid authentication"); return
        device_id=str(auth.get("device_id") or "")
        supplied=str(auth.get("digest") or "")
        expected=auth_digest(self.shared_secret,nonce,device_id)
        if device_id!=self.expected_device_id or not hmac.compare_digest(supplied,expected):
            await websocket.close(code=4003,reason="authentication failed"); return
        self.websocket=websocket; self.connected.set()
        await websocket.send(json.dumps({"type":"auth_ok","server_time":time.time()}))
        print(f"[body-link] authenticated body {device_id}",flush=True)
        try:
            async for raw in websocket:
                if isinstance(raw,bytes):self._handle_binary(raw)
                else:await self._handle_text(raw)
        except Exception as exc:
            print(f"[body-link] connection ended: {type(exc).__name__}: {exc}",flush=True)
        finally:
            if self.websocket is websocket:self.websocket=None
            self.connected.clear(); self._fail_all_pending("body disconnected"); self._drain_audio()
            print("[body-link] waiting for Echo body to reconnect...",flush=True)

    def _handle_binary(self,raw):
        kind,_,payload=unpack_binary(raw)
        if kind==PACKET_VIDEO:
            with self._frame_lock:
                self._latest_frame_jpeg=payload; self._frame_counter+=1
            return
        if kind==PACKET_AUDIO:
            try:self.audio_queue.put_nowait(payload)
            except queue.Full:
                for _ in range(100):
                    try:self.audio_queue.get_nowait()
                    except queue.Empty:break
                try:self.audio_queue.put_nowait(payload)
                except queue.Full:pass

    async def _handle_text(self,raw):
        try:msg=json.loads(raw)
        except Exception:return
        if msg.get("type")=="rpc_result":
            req_id=str(msg.get("id") or "")
            with self._rpc_lock:item=self._pending.get(req_id)
            if item:
                event,holder=item; holder.update(msg); event.set()
        elif msg.get("type")=="status":
            with self.status_lock:self.status=dict(msg)

    async def _send_json_async(self,value):
        ws=self.websocket
        if ws is None:raise ConnectionError("Echo body is not connected")
        async with self._send_lock:
            await ws.send(json.dumps(value,separators=(",",":"),ensure_ascii=False))

    def send_json(self,value,timeout=10):
        self.wait_for_connection(None)
        fut=asyncio.run_coroutine_threadsafe(self._send_json_async(value),self.loop)
        fut.result(timeout=timeout)

    def rpc(self,op,timeout=20,**kwargs):
        self.wait_for_connection(None)
        req_id=uuid.uuid4().hex; event=threading.Event(); holder={}
        with self._rpc_lock:self._pending[req_id]=(event,holder)
        try:
            self.send_json({"type":"rpc","id":req_id,"op":op,"args":kwargs},timeout=min(timeout,10))
            if not event.wait(timeout):raise TimeoutError(f"Echo body RPC timed out: {op}")
            if not holder.get("ok",False):raise RuntimeError(str(holder.get("error") or f"Echo body RPC failed: {op}"))
            return holder.get("result")
        finally:
            with self._rpc_lock:self._pending.pop(req_id,None)

    def _fail_all_pending(self,reason):
        with self._rpc_lock:pending=list(self._pending.values())
        for event,holder in pending:
            holder.update({"ok":False,"error":reason}); event.set()

    def wait_for_connection(self,timeout=None):
        if timeout is None:
            while not self.connected.wait(1):
                print("[body-link] waiting for Echo body...",flush=True)
            return
        if not self.connected.wait(timeout):raise TimeoutError("Echo body did not connect in time")

    def latest_frame(self,timeout=10):
        import cv2
        deadline=time.time()+timeout
        while True:
            with self._frame_lock:data=self._latest_frame_jpeg
            if data is not None:
                frame=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if frame is None:raise RuntimeError("Echo body sent an undecodable camera frame")
                return frame
            if time.time()>=deadline:raise TimeoutError("No camera frame received from Echo body")
            time.sleep(.01)

    def speak_text(self,text):
        self.rpc("speaker_say",timeout=max(30,len(str(text))/5),text=str(text))
    def set_speaker_volume(self):
        self.rpc("speaker_set_volume",timeout=10)
    def set_mic_enabled(self,enabled):
        self.send_json({"type":"mic_enabled","enabled":bool(enabled)},timeout=5)
        if not enabled:self._drain_audio()
    def _drain_audio(self):
        while True:
            try:self.audio_queue.get_nowait()
            except queue.Empty:break

class RemotePicamera2:
    def __init__(self,body):self.body=body
    def create_still_configuration(self):return {"controls":{}}
    def configure(self,config):
        try:self.body.rpc("camera_config",timeout=10,config=dict(config or {}))
        except Exception:pass
    def start(self):return None
    def capture_array(self):return self.body.latest_frame(timeout=15)

class RemoteAudioStream:
    def __init__(self,body,frames_per_buffer=320):
        self.body=body; self.frames_per_buffer=int(frames_per_buffer); self._buffer=bytearray()
        self.body.set_mic_enabled(True)
    def read(self,frames,exception_on_overflow=False):
        needed=int(frames)*2
        while len(self._buffer)<needed:
            try:chunk=self.body.audio_queue.get(timeout=5)
            except queue.Empty:
                if not self.body.connected.is_set():self.body.wait_for_connection(None)
                continue
            self._buffer.extend(chunk)
        out=bytes(self._buffer[:needed]); del self._buffer[:needed]; return out
    def stop_stream(self):
        try:self.body.set_mic_enabled(False)
        except Exception:pass
    def start_stream(self):
        try:self.body.set_mic_enabled(True)
        except Exception:pass
    def close(self):self.stop_stream()

class RemotePyAudio:
    def __init__(self,body):self.body=body
    def get_sample_size(self,fmt):return 2
    def open(self,*args,**kwargs):return RemoteAudioStream(self.body,kwargs.get("frames_per_buffer",320))
    def terminate(self):pass

class RemoteSMBus:
    def __init__(self,body,bus_number):self.body=body; self.bus_number=int(bus_number)
    def read_i2c_block_data(self,addr,register,length):
        return list(self.body.rpc("i2c_read_block",bus=self.bus_number,addr=int(addr),register=int(register),length=int(length)))
    def write_i2c_block_data(self,addr,register,data):
        return self.body.rpc("i2c_write_block",bus=self.bus_number,addr=int(addr),register=int(register),data=[int(x) for x in data])
    def close(self):pass

class RemoteBluetoothSocket:
    def __init__(self,body,proto=None):self.body=body
    def connect(self,endpoint):
        address,port=endpoint; self.body.rpc("arduino_connect",address=address,port=int(port or 1))
    def send(self,data):
        raw=data if isinstance(data,bytes) else str(data).encode()
        return int(self.body.rpc("arduino_send",data_hex=raw.hex()) or len(raw))
    def recv(self,size):
        return str(self.body.rpc("arduino_recv",size=int(size)) or "").encode()
    def close(self):
        try:self.body.rpc("arduino_close")
        except Exception:pass

def build_remote_modules(body):
    class BluetoothModule:
        RFCOMM=1
        BluetoothError=RuntimeError
        @staticmethod
        def discover_devices(lookup_names=False):
            return body.rpc("bluetooth_discover",lookup_names=bool(lookup_names),timeout=45) or []
        @staticmethod
        def BluetoothSocket(proto):return RemoteBluetoothSocket(body,proto)
    class SMBusModule:
        @staticmethod
        def SMBus(bus_number):return RemoteSMBus(body,bus_number)
    class PyAudioModule:
        paInt16=8
        @staticmethod
        def PyAudio():return RemotePyAudio(body)
    return BluetoothModule,SMBusModule,PyAudioModule
