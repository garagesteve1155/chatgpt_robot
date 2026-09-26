#!/usr/bin/env python3
"""Setup utility for Echo's split desktop-brain / Pi-body architecture."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, secrets, subprocess, sys, urllib.request
from datetime import datetime,timedelta,timezone
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DEFAULT_QWEN_DIR=r"C:\Users\garag\OneDrive\Desktop\MME\qwen3_14b"
DEFAULT_PORT=8765
RUNTIME_DIRS=["History","History_dataset","History_dataset_mental","People","Pictures","Videos","Convo","Current_people"]

def run(cmd,check=True):
    print("+"," ".join(map(str,cmd)),flush=True); return subprocess.run(cmd,check=check)
def pip_install(*pkgs,break_system=False):
    cmd=[sys.executable,"-m","pip","install","--upgrade",*pkgs]
    if break_system:cmd.append("--break-system-packages")
    run(cmd)
def ensure_runtime_dirs():
    for name in RUNTIME_DIRS:(BASE_DIR/name).mkdir(parents=True,exist_ok=True)
def download(url,path):
    if path.exists() and path.stat().st_size>1024:return
    print("Downloading",path.name)
    req=urllib.request.Request(url,headers={"User-Agent":"EchoRobotSetup/2.0"})
    with urllib.request.urlopen(req,timeout=120) as r:path.write_bytes(r.read())

def generate_tls_and_configs(model_dir,port):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    cert_path=BASE_DIR/"echo_cert.pem"; key_path=BASE_DIR/"echo_key.pem"
    if not cert_path.exists() or not key_path.exists():
        key=rsa.generate_private_key(public_exponent=65537,key_size=3072)
        name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,"Echo Desktop Brain")])
        now=datetime.now(timezone.utc)
        cert=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=5))
              .not_valid_after(now+timedelta(days=3650))
              .add_extension(x509.BasicConstraints(ca=True,path_length=None),critical=True)
              .sign(key,hashes.SHA256()))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.TraditionalOpenSSL,serialization.NoEncryption()))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    cert=x509.load_pem_x509_certificate(cert_path.read_bytes())
    fp=hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    dpath=BASE_DIR/"echo_desktop_config.json"
    old=json.loads(dpath.read_text()) if dpath.exists() else {}
    secret=str(old.get("shared_secret") or secrets.token_urlsafe(48))
    old.update({
        "listen_host":"0.0.0.0","listen_port":int(port),"tls_cert":str(cert_path),"tls_key":str(key_path),
        "shared_secret":secret,"device_id":"echo","qwen_model_dir":model_dir,"whisper_model":"small.en",
    })
    dpath.write_text(json.dumps(old,indent=2),encoding="utf-8")
    pconfig={
        "brain_host":"PUT_YOUR_HOME_PUBLIC_IP_OR_DNS_NAME_HERE","brain_port":int(port),"device_id":"echo",
        "shared_secret":secret,"server_cert_sha256":fp,"video_fps":6,"jpeg_quality":75,
    }
    (BASE_DIR/"echo_pi_config.generated.json").write_text(json.dumps(pconfig,indent=2),encoding="utf-8")
    print("\nGenerated echo_desktop_config.json and echo_pi_config.generated.json")
    print("Certificate SHA-256:",fp)
    print("Copy the generated Pi config to the Pi as echo_pi_config.json and set brain_host.")
    print(f"Forward TCP {port} from your home router to the desktop if using IPv4 NAT.")

def desktop_install(model_dir,port,open_firewall):
    ensure_runtime_dirs()
    pip_install("transformers>=4.52.4","accelerate","bitsandbytes","safetensors","websockets>=12",
                "numpy","scipy","opencv-python","SpeechRecognition","webrtcvad-wheels","cryptography","faster-whisper")
    download("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",BASE_DIR/"yolov4-tiny.cfg")
    download("https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",BASE_DIR/"yolov4-tiny.weights")
    download("https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",BASE_DIR/"haarcascade_frontalface_default.xml")
    download("https://huggingface.co/onnxmodelzoo/mobilenetv2-7/resolve/main/mobilenetv2-7.onnx?download=true",BASE_DIR/"mobilenetv2.onnx")
    generate_tls_and_configs(model_dir,port)
    if open_firewall and platform.system().lower()=="windows":
        run(["netsh","advfirewall","firewall","add","rule","name=Echo Desktop Brain","dir=in","action=allow",
             "protocol=TCP",f"localport={port}"],check=False)

def pi_install():
    ensure_runtime_dirs()
    run(["sudo","apt","update"])
    run(["sudo","apt","install","-y","python3-pip","python3-picamera2","python3-opencv","python3-pyaudio",
         "python3-smbus","bluez","bluetooth","libbluetooth-dev","portaudio19-dev","alsa-utils","espeak"])
    pip_install("websockets>=12","pybluez",break_system=True)
    print("Copy echo_pi_config.generated.json from the desktop to this folder as echo_pi_config.json.")
    print("Then run: python3 main.py")

def test(role):
    if role=="desktop":
        names=["echo_cert.pem","echo_key.pem","yolov4-tiny.cfg","yolov4-tiny.weights",
               "haarcascade_frontalface_default.xml","mobilenetv2.onnx","echo_desktop_config.json"]
    else:names=["echo_pi_config.json"]
    for n in names:print(n,"OK" if (BASE_DIR/n).exists() else "MISSING")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--role",choices=["desktop","pi"],required=True)
    p.add_argument("--mode",choices=["install","test"],default="install")
    p.add_argument("--model-dir",default=os.environ.get("ECHO_QWEN_DIR",DEFAULT_QWEN_DIR))
    p.add_argument("--port",type=int,default=DEFAULT_PORT)
    p.add_argument("--open-firewall",action="store_true")
    a=p.parse_args()
    if a.mode=="test":test(a.role)
    elif a.role=="desktop":desktop_install(a.model_dir,a.port,a.open_firewall)
    else:pi_install()
if __name__=="__main__":main()
