#!/usr/bin/env python3
"""Echo Raspberry Pi body endpoint: camera/mic/speaker/Arduino/I2C only."""
from __future__ import annotations
import argparse, asyncio, contextlib, json, os, queue, ssl, subprocess, tempfile, threading, time, traceback
from pathlib import Path
import cv2, pyaudio, smbus
from picamera2 import Picamera2
try: import bluetooth
except ImportError: bluetooth=None
from echo_link import PACKET_AUDIO,PACKET_VIDEO,auth_digest,certificate_sha256_from_der,load_json,normalize_fingerprint,pack_binary

BASE_DIR=Path(__file__).resolve().parent
DEFAULT_CONFIG=BASE_DIR/"echo_pi_config.json"
CHUNK=320
RATE=16000
CHANNELS=1
FORMAT=pyaudio.paInt16
MOTION_WATCHDOG_SECONDS=2.0

class EchoBody:
    def __init__(self,config):
        self.config=config
        self.host=str(config.get("brain_host") or "").strip()
        self.port=int(config.get("brain_port",8765))
        self.device_id=str(config.get("device_id","echo"))
        self.shared_secret=str(config.get("shared_secret") or "")
        self.server_cert_sha256=normalize_fingerprint(config.get("server_cert_sha256") or "")
        self.video_fps=max(1,min(float(config.get("video_fps",6)),20))
        self.jpeg_quality=max(35,min(int(config.get("jpeg_quality",75)),95))
        if not self.host: raise SystemExit("echo_pi_config.json is missing brain_host")
        if not self.shared_secret: raise SystemExit("echo_pi_config.json is missing shared_secret")
        if len(self.server_cert_sha256)!=64: raise SystemExit("echo_pi_config.json needs desktop certificate SHA-256 fingerprint")
        self.ws=None; self.send_lock=None; self.connected=threading.Event(); self.mic_enabled=threading.Event()
        self.mic_enabled.set(); self.stop_event=threading.Event()
        self.audio_queue=queue.Queue(maxsize=500)
        self.camera=None; self.camera_lock=threading.Lock()
        self.bt_lock=threading.RLock(); self.bt_sock=None; self.arduino_address=None; self.bt_port=1
        self.i2c_lock=threading.RLock(); self.i2c_buses={}
        self.motion_lock=threading.Lock(); self.motion_deadline=0.0
        self.audio_card_number=self._get_audio_card_number()

    def initialize_hardware(self):
        self.camera=Picamera2()
        cfg=self.camera.create_still_configuration()
        cfg["controls"]={"AeEnable":False,"AnalogueGain":8.0,"ExposureTime":50000,"AwbEnable":True}
        self.camera.configure(cfg); self.camera.start()
        threading.Thread(target=self._audio_worker,name="EchoMic",daemon=True).start()
        print("[pi] hardware initialized",flush=True)

    def _get_audio_card_number(self):
        try:
            out=subprocess.run(["aplay","-l"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True).stdout
            import re
            m=re.search(r"card (\d+): Device",out) or re.search(r"card (\d+): wm8960sound",out)
            return m.group(1) if m else None
        except Exception:return None

    def _set_max_volume(self):
        if self.audio_card_number is not None:
            subprocess.run(["amixer","-c",str(self.audio_card_number),"sset","Speaker","100%"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)

    def _speak_text(self,text):
        text=str(text or "").strip()
        if not text:return
        was=self.mic_enabled.is_set(); self.mic_enabled.clear()
        try:
            fd,path=tempfile.mkstemp(prefix="echo_tts_",suffix=".wav"); os.close(fd)
            try:
                subprocess.check_call(["espeak","-v","en-us","-s","180","-p","130","-a","200","-w",path,text])
                self._set_max_volume()
                cmd=["aplay",path] if self.audio_card_number is None else ["aplay","-D",f"plughw:{self.audio_card_number}",path]
                subprocess.check_call(cmd)
            finally:
                with contextlib.suppress(OSError):os.remove(path)
        finally:
            if was:self.mic_enabled.set()

    def _audio_worker(self):
        p=pyaudio.PyAudio(); stream=None; running=False
        try:
            stream=p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK); running=True
            while not self.stop_event.is_set():
                if not self.mic_enabled.is_set():
                    if running:
                        with contextlib.suppress(Exception):stream.stop_stream()
                        running=False
                    time.sleep(.02); continue
                if not running:
                    with contextlib.suppress(Exception):stream.start_stream()
                    running=True
                try:frame=stream.read(CHUNK,exception_on_overflow=False)
                except Exception:time.sleep(.05); continue
                try:self.audio_queue.put_nowait(frame)
                except queue.Full:
                    with contextlib.suppress(queue.Empty):self.audio_queue.get_nowait()
                    with contextlib.suppress(queue.Full):self.audio_queue.put_nowait(frame)
        finally:
            if stream:
                with contextlib.suppress(Exception):stream.stop_stream()
                with contextlib.suppress(Exception):stream.close()
            p.terminate()

    async def send_json(self,value):
        if self.ws is None:raise ConnectionError("desktop disconnected")
        async with self.send_lock:await self.ws.send(json.dumps(value,separators=(",",":"),ensure_ascii=False))
    async def send_binary(self,kind,payload):
        if self.ws is not None:
            async with self.send_lock:await self.ws.send(pack_binary(kind,payload))

    def _pin_certificate(self,ws):
        ssl_object=ws.transport.get_extra_info("ssl_object")
        if ssl_object is None:raise RuntimeError("connection is not TLS")
        actual=normalize_fingerprint(certificate_sha256_from_der(ssl_object.getpeercert(binary_form=True)))
        if actual!=self.server_cert_sha256:
            raise RuntimeError(f"desktop TLS fingerprint mismatch: expected {self.server_cert_sha256}, got {actual}")

    async def _authenticate(self,ws):
        raw=await asyncio.wait_for(ws.recv(),timeout=15)
        challenge=json.loads(raw) if isinstance(raw,str) else {}
        if challenge.get("type")!="challenge":raise RuntimeError("bad desktop authentication challenge")
        nonce=str(challenge.get("nonce") or "")
        await ws.send(json.dumps({"type":"auth","device_id":self.device_id,"digest":auth_digest(self.shared_secret,nonce,self.device_id)}))
        raw=await asyncio.wait_for(ws.recv(),timeout=15)
        if not isinstance(raw,str) or json.loads(raw).get("type")!="auth_ok":raise RuntimeError("desktop rejected Echo authentication")

    def _capture_frame(self):
        with self.camera_lock:return self.camera.capture_array()

    async def _camera_task(self):
        period=1.0/self.video_fps
        while self.ws is not None:
            start=time.monotonic()
            try:
                frame=await asyncio.to_thread(self._capture_frame)
                ok,enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),self.jpeg_quality])
                if ok:await self.send_binary(PACKET_VIDEO,enc.tobytes())
            except asyncio.CancelledError:raise
            except Exception as exc:print("[pi] camera error:",exc,flush=True)
            await asyncio.sleep(max(0,period-(time.monotonic()-start)))

    async def _audio_task(self):
        while self.ws is not None:
            try:frame=await asyncio.to_thread(self.audio_queue.get,True,1)
            except queue.Empty:continue
            if self.mic_enabled.is_set():await self.send_binary(PACKET_AUDIO,frame)

    async def _status_task(self):
        while self.ws is not None:
            await self.send_json({"type":"status","time":time.time(),"device_id":self.device_id,
                                  "mic_enabled":self.mic_enabled.is_set(),"arduino_connected":self.bt_sock is not None})
            await asyncio.sleep(5)

    async def _watchdog_task(self):
        while self.ws is not None:
            stop=False
            with self.motion_lock:
                if self.motion_deadline and time.monotonic()>=self.motion_deadline:
                    self.motion_deadline=0; stop=True
            if stop:await asyncio.to_thread(self._emergency_stop)
            await asyncio.sleep(.05)

    async def _receiver_task(self):
        async for raw in self.ws:
            if not isinstance(raw,str):continue
            try:msg=json.loads(raw)
            except Exception:continue
            if msg.get("type")=="mic_enabled":
                self.mic_enabled.set() if msg.get("enabled") else self.mic_enabled.clear()
            elif msg.get("type")=="rpc":
                await self._handle_rpc(msg)

    async def _handle_rpc(self,msg):
        rid=str(msg.get("id") or ""); op=str(msg.get("op") or ""); args=dict(msg.get("args") or {})
        try:
            result=await asyncio.to_thread(self._rpc_blocking,op,args)
            out={"type":"rpc_result","id":rid,"ok":True,"result":result}
        except Exception as exc:
            out={"type":"rpc_result","id":rid,"ok":False,"error":f"{type(exc).__name__}: {exc}"}
        await self.send_json(out)

    def _rpc_blocking(self,op,args):
        if op=="ping":return "pong"
        if op=="camera_config":
            controls=dict((args.get("config") or {}).get("controls") or {})
            if controls:
                with self.camera_lock:
                    with contextlib.suppress(Exception):self.camera.set_controls(controls)
            return True
        if op=="speaker_set_volume":self._set_max_volume(); return True
        if op=="speaker_say":self._speak_text(args.get("text")); return True
        if op=="bluetooth_discover":
            if bluetooth is None:raise RuntimeError("PyBluez is not installed")
            return bluetooth.discover_devices(lookup_names=bool(args.get("lookup_names",False)))
        if op=="arduino_connect":return self._arduino_connect(args.get("address"),int(args.get("port",1)))
        if op=="arduino_send":return self._arduino_send(bytes.fromhex(str(args.get("data_hex") or "")))
        if op=="arduino_recv":return self._arduino_recv(int(args.get("size",1024)))
        if op=="arduino_close":return self._arduino_close()
        if op=="i2c_read_block":
            b=self._i2c_bus(int(args["bus"]))
            with self.i2c_lock:return list(b.read_i2c_block_data(int(args["addr"]),int(args["register"]),int(args["length"])))
        if op=="i2c_write_block":
            b=self._i2c_bus(int(args["bus"]))
            with self.i2c_lock:b.write_i2c_block_data(int(args["addr"]),int(args["register"]),[int(x) for x in args["data"]])
            return True
        raise ValueError(f"unknown RPC: {op}")

    def _arduino_connect(self,address,port):
        if bluetooth is None:raise RuntimeError("PyBluez is not installed")
        with self.bt_lock:
            if self.bt_sock is not None:return True
            if not address:
                for addr,name in bluetooth.discover_devices(lookup_names=True):
                    if name=="HC-05":address=addr; break
            if not address:raise RuntimeError("HC-05 not found")
            sock=bluetooth.BluetoothSocket(bluetooth.RFCOMM); sock.connect((address,int(port or 1)))
            self.bt_sock=sock; self.arduino_address=address; self.bt_port=int(port or 1)
            print(f"[pi] connected HC-05 {address}",flush=True); return True

    def _arduino_send(self,raw):
        if not raw:return 0
        with self.bt_lock:
            if self.bt_sock is None:self._arduino_connect(self.arduino_address,self.bt_port)
            sent=self.bt_sock.send(raw)
            for byte in raw:
                ch=chr(byte)
                if ch in {"w","s","a","d"}:
                    with self.motion_lock:self.motion_deadline=time.monotonic()+MOTION_WATCHDOG_SECONDS
                elif ch=="x":
                    with self.motion_lock:self.motion_deadline=0
            return int(sent or len(raw))

    def _arduino_recv(self,size):
        with self.bt_lock:
            if self.bt_sock is None:raise RuntimeError("Arduino not connected")
            return self.bt_sock.recv(size).decode(errors="replace").strip()

    def _arduino_close(self):
        with self.bt_lock:
            if self.bt_sock is not None:
                with contextlib.suppress(Exception):self.bt_sock.close()
            self.bt_sock=None
        return True

    def _i2c_bus(self,n):
        with self.i2c_lock:
            if n not in self.i2c_buses:self.i2c_buses[n]=smbus.SMBus(n)
            return self.i2c_buses[n]

    def _emergency_stop(self):
        with self.bt_lock:
            if self.bt_sock is not None:
                with contextlib.suppress(Exception):self.bt_sock.send(b"x")
        with self.motion_lock:self.motion_deadline=0

    async def session(self):
        import websockets
        ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
        host=f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        uri=f"wss://{host}:{self.port}"
        print("[pi] direct connect",uri,flush=True)
        async with websockets.connect(uri,ssl=ctx,max_size=None,ping_interval=20,ping_timeout=20,open_timeout=20,compression=None) as ws:
            self._pin_certificate(ws); await self._authenticate(ws)
            self.ws=ws; self.send_lock=asyncio.Lock(); self.connected.set()
            tasks=[
                asyncio.create_task(self._camera_task()),
                asyncio.create_task(self._audio_task()),
                asyncio.create_task(self._status_task()),
                asyncio.create_task(self._watchdog_task()),
                asyncio.create_task(self._receiver_task()),
            ]
            try:
                done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    exc=task.exception()
                    if exc:raise exc
            finally:
                for task in tasks:task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError,Exception):await task
                self._emergency_stop(); self.connected.clear(); self.ws=None; self.send_lock=None

    async def run(self):
        self.initialize_hardware(); delay=1.0
        while not self.stop_event.is_set():
            try:await self.session(); delay=1.0
            except asyncio.CancelledError:raise
            except Exception as exc:
                self._emergency_stop()
                print(f"[pi] desktop unavailable: {type(exc).__name__}: {exc}",flush=True)
                await asyncio.sleep(delay); delay=min(delay*1.7,15)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(DEFAULT_CONFIG))
    args=parser.parse_args()
    body=EchoBody(load_json(args.config))
    try:asyncio.run(body.run())
    except KeyboardInterrupt:
        body.stop_event.set(); body._emergency_stop()
    except Exception:
        body._emergency_stop(); traceback.print_exc(); raise

if __name__=="__main__":main()
