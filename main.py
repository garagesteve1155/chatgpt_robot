import re
import subprocess
import os
import numpy as np
import time
import requests
import threading
import bluetooth
import cv2
import shutil
import smbus
from datetime import datetime
import base64
import traceback
import random
import speech_recognition as sr
import webrtcvad
import pyaudio
import math
import glob
from scipy.signal import stft, istft
import json
from picamera2 import Picamera2, Preview
import ast
from collections import Counter
camera = Picamera2()
def remove_duplicates(file_path):
    with open(file_path, 'r') as file:
        known_people = file.read().strip().lower()
    people_list = known_people.split(', ')
    unique_people = list(set(people_list))
    unique_people.sort()
    cleaned_people = ', '.join(unique_people)
    with open(file_path, 'w') as file:
        file.write(cleaned_people)
remove_duplicates('known_people.txt')
camera_config = camera.create_still_configuration()
camera_config["controls"] = {
    "AeEnable": False,
    "AnalogueGain": 3.5,
    "ExposureTime": 100000,
    "AwbEnable": True,
}
camera.configure(camera_config)
camera.start()
try:
    file = open('server_ip.txt','r')
    server_ip = file.read()
    file.close()
    file = open('server_port.txt','r')
    server_port = int(file.read())
    file.close()
except:
    server_ip = input('Please input the IP address for your image server (I just use a digitalocean droplet. Make sure the companion script is currently running on the server before this!): ')
    server_port = 8040
    file = open('server_ip.txt','w+')
    file.write(server_ip)
    file.close()
    file = open('server_port.txt','w+')
    file.write(str(server_port))
    file.close()
import asyncio
import websockets
class WebSocketUploader:
    def __init__(self, vps_ip, port):
        self.vps_ip = vps_ip
        self.port = port
        self.websocket = None
        self.connected = threading.Event()
        self.lock = threading.Lock()
        self.connect_thread = threading.Thread(target=self.connect)
        self.connect_thread.start()
    def connect(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._connect_to_server())
        except Exception as e:
            print(traceback.format_exc())
    async def _connect_to_server(self):
        try:
            print(f"Connecting to WebSocket server at {self.vps_ip}:{self.port}...")
            async with websockets.connect(f"ws://{self.vps_ip}:{self.port}") as ws:
                self.websocket = ws
                self.connected.set()
                print('Websocket connected')
                while True:
                    await asyncio.sleep(1)
        except Exception as e:
            print(traceback.format_exc())
    def send_image(self, image_path):
        if not self.connected.is_set():
            print("WebSocket connection not established.")
            return
        try:
            with open(image_path, "rb") as file:
                image_data = file.read()
            async def send():
                start_time = time.time()
                await self.websocket.send(image_data)
                upload_duration = time.time() - start_time
                print(f"Image sent successfully in {upload_duration:.3f} seconds.")
            asyncio.run(send())
        except Exception as e:
            print(f"Error sending image: {e}")
    def send_person_image(self, image_path, person_name):
        if not self.connected.is_set():
            print("WebSocket connection not established.")
            return
        try:
            with open(image_path, "rb") as file:
                image_data = file.read()
            async def send():
                start_time = time.time()
                metadata = json.dumps({"person_name": person_name})
                await self.websocket.send(metadata)
                await self.websocket.send(image_data)
                upload_duration = time.time() - start_time
                print(f"Image sent successfully in {upload_duration:.3f} seconds.")
            asyncio.run(send())
        except Exception as e:
            print(f"Error sending image: {e}")
uploader = WebSocketUploader(vps_ip=server_ip, port=server_port)
with open('recent_filenames.txt', 'w+') as f:
    f.write('')
with open('current_task.txt', 'w+') as f:
    f.write('None')
with open('sleep.txt', 'w+') as f:
    f.write('False')
with open('error_rate.txt','w+') as f:
    f.write('0')
with open('current_convo.txt','w+') as f:
    f.write('')
with open('brightness.txt','w+') as f:
    f.write('')
with open('last_prompt.txt', 'w+') as f:
    f.write('')
with open('last_command.txt', 'w+') as f:
    f.write('')
with open('name_of_person.txt', 'w+') as f:
    f.write('unknown name of person')
def clear_specific_files(file_list):
    for file_path in file_list:
        try:
            with open(file_path, 'w+') as f:
                f.write('')
            print(f"Cleared file: {file_path}")
        except Exception as e:
            print(f"Error clearing file {file_path}: {e}")
def clear_files_in_folders(folders):
    for folder in folders:
        if os.path.exists(folder):
            items = glob.glob(os.path.join(folder, '*'))
            for item in items:
                if os.path.isfile(item):
                    try:
                        os.remove(item)
                        print(f"Deleted file: {item}")
                    except Exception as e:
                        print(f"Error deleting file {item}: {e}")
                elif os.path.isdir(item):
                    try:
                        shutil.rmtree(item)
                        print(f"Deleted directory and its contents: {item}")
                    except Exception as e:
                        print(f"Error deleting directory {item}: {e}")
                else:
                    print(f"Unknown item type, skipping: {item}")
        else:
            print(f"Directory does not exist, skipping: {folder}")
def ensure_directories_exist(directories):
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        except Exception as e:
            print(f"Error ensuring directory {directory} exists: {e}")
files_to_clear = [
    'current_distance.txt',
    'output.txt',
    'output2.txt',
    'batt_cur.txt',
    'batt_per.txt',
    'last_move.txt',
    'internal_input.txt',
    'playback_text.txt',
    'last_phrase.txt',
    'all_recents.txt',
    'recent_speech.txt',
    'internal_input.txt',
    'target_object.txt',
    'name_of_person.txt',
    'current_distance.txt',
    'output.txt',
    'output2.txt',
    'current_mode.txt',
    'batt_cur.txt',
    'batt_per.txt',
    'last_move.txt',
    'internal_input.txt',
    'mental_error.txt',
    'last_said.txt',
    'prompt_intent.txt',
    'sleep.txt',
    'i_upload.txt',
    'current_task.txt',
    'summaries.txt'
]
folders_to_clear = [
    'History',
    'History_dataset',
    'History_dataset_mental',
    'People'
]
time.sleep(2)
deleter = input("Delete All Memories????? (Yes or No): ")
if 'y' in deleter.lower():
    deleter2 = input("ARE YOU SURE YOU WANT TO Delete All Memories????? (Yes or No): ")
    if 'y' in deleter2.lower():
        try:
            pass
        except:
            print(traceback.format_exc())
        try:
            clear_files_in_folders(folders_to_clear)
        except:
            print(traceback.format_exc())
        time.sleep(5)
    else:
        pass
else:
    pass
time.sleep(2)
def clear_pictures(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
folder_path = 'Pictures/'
clear_pictures(folder_path)
move_set = []
time.sleep(2)
r = sr.Recognizer()
time.sleep(2)
vad = webrtcvad.Vad(3)
print("Initialized recognizer and VAD.")
time.sleep(2)
CHUNK = 320
FORMAT = pyaudio.paInt16
time.sleep(2)
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(FORMAT)
time.sleep(2)
p = pyaudio.PyAudio()
time.sleep(2)
is_transcribing = False
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
known_object_heights = {
    'person': 1.7,
    'bicycle': 1.0,
    'car': 1.5,
    'motorcycle': 1.1,
    'airplane': 10.0,
    'bus': 3.0,
    'train': 4.0,
    'truck': 2.5,
    'boat': 2.0,
    'traffic light': 0.6,
    'fire hydrant': 0.5,
    'stop sign': 0.75,
    'parking meter': 1.2,
    'bench': 1.5,
    'bird': 0.3,
    'cat': 0.25,
    'dog': 0.6,
    'horse': 1.6,
    'sheep': 0.8,
    'cow': 1.5,
    'elephant': 3.0,
    'bear': 1.2,
    'zebra': 1.4,
    'giraffe': 5.5,
    'backpack': 0.5,
    'umbrella': 1.0,
    'handbag': 0.3,
    'tie': 0.5,
    'suitcase': 0.7,
    'frisbee': 0.3,
    'skis': 1.8,
    'snowboard': 1.6,
    'sports ball': 0.22,
    'kite': 1.2,
    'baseball bat': 1.1,
    'baseball glove': 0.35,
    'skateboard': 0.8,
    'surfboard': 2.0,
    'tennis racket': 1.0,
    'bottle': 0.3,
    'wine glass': 0.3,
    'cup': 0.15,
    'fork': 0.2,
    'knife': 0.3,
    'spoon': 0.2,
    'bowl': 0.2,
    'banana': 0.2,
    'apple': 0.1,
    'sandwich': 0.2,
    'orange': 0.1,
    'broccoli': 0.3,
    'carrot': 0.3,
    'hot dog': 0.2,
    'pizza': 0.3,
    'donut': 0.15,
    'cake': 0.3,
    'chair': 0.9,
    'sofa': 2.0,
    'potted plant': 0.5,
    'bed': 2.0,
    'dining table': 1.8,
    'toilet': 0.6,
    'tv monitor': 1.2,
    'laptop': 0.4,
    'mouse': 0.15,
    'remote': 0.15,
    'keyboard': 0.5,
    'cell phone': 0.15,
    'microwave': 0.6,
    'oven': 0.8,
    'toaster': 0.3,
    'sink': 0.5,
    'refrigerator': 1.8,
    'book': 0.25,
    'clock': 0.3,
    'vase': 0.4,
    'scissors': 0.3,
    'teddy bear': 0.3,
    'hair dryer': 0.3,
    'toothbrush': 0.2
}
default_height = 1.0
focal_length_px = 2050
def estimate_distance(focal_length, real_height, pixel_height):
    if pixel_height == 0:
        return float('inf')
    return ((focal_length * real_height) / pixel_height)/6
def get_position_description(x, y, width, height):
    if x < width / 5:
        horizontal = "45 Degrees To My Left"
    elif x < 2 * width / 5:
        horizontal = "15 Degrees To My Left"
    elif x < 3 * width / 5:
        horizontal = "directly in front of me"
    elif x < 4 * width / 5:
        horizontal = "15 Degrees To My Right"
    else:
        horizontal = "45 Degrees To My Right"
    if y < height / 5:
        vertical = "above me and"
    elif y < 2 * height / 5:
        vertical = "above me and"
    elif y < 3 * height / 5:
        vertical = ""
    elif y < 4 * height / 5:
        vertical = "below me and"
    else:
        vertical = "below me and"
    if horizontal == "centered between my left and right" and vertical == "at my height":
        return "centered on object"
    else:
        return f"{vertical} {horizontal}"
with open("last_phrase.txt","w+") as f:
    f.write('')
with open("move_failed.txt","w+") as f:
    f.write('')
with open("move_failure_reas.txt","w+") as f:
    f.write('')
def get_wm8960_card_number():
    print("Finding WM8960 Audio HAT card number...")
    result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"card (\d+): wm8960sound", result.stdout)
    if match:
        card_number = match.group(1)
        print(f"WM8960 Audio HAT found on card {card_number}")
        return card_number
    else:
        print("WM8960 Audio HAT not found.")
        return None
def get_audio_card_number():
    print("Finding USB Audio Device card number...")
    result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"card (\d+): Device", result.stdout)
    if match:
        card_number = match.group(1)
        print(f"USB Audio Device found on card {card_number}")
        return card_number
    else:
        print("USB Audio Device not found.")
        return None
def set_max_volume(card_number):
    subprocess.run(["amixer", "-c", card_number, "sset", 'Speaker', '100%'], check=True)
time.sleep(2)
audio_card_number = get_audio_card_number()
print(audio_card_number)
time.sleep(2)
def handle_playback(stream):
    global move_stopper
    global is_transcribing
    global audio_card_number
    with open('playback_text.txt', 'r') as file:
        text = file.read().strip()
    if text:
        with open('current_convo.txt','a') as f:
            f.write('\nEcho said: "'+text+'"')
        with open('last_move.txt', 'w+') as f:
            f.write('speakanddoaction')
        stream.stop_stream()
        is_transcribing = True
        subprocess.call(['espeak', '-v', 'en-us', '-s', '180', '-p', '130', '-a', '200', '-w', 'temp.wav', text])
        set_max_volume(audio_card_number)
        subprocess.check_call(["aplay", "-D", "plughw:{}".format(audio_card_number), 'temp.wav'])
        os.remove('temp.wav')
        open('playback_text.txt', 'w').close()
        stream.start_stream()
        is_transcribing = False
        move_stopper = False
        return True
    else:
        return False
def frequency_based_noise_reduction_scipy(audio_buffer, rate, noise_profile, prop_decrease=0.8, n_fft=2048, hop_length=512, win_length=2048):
    try:
        audio_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        f, t, Zxx = stft(audio_array, fs=rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, window='hann')
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        noise_stft = stft(noise_profile, fs=rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, window='hann')[2]
        noise_magnitude = np.abs(noise_stft)
        noise_mag_avg = np.mean(noise_magnitude, axis=1, keepdims=True)
        enhanced_magnitude = np.maximum(magnitude - prop_decrease * noise_mag_avg, 0.0)
        enhanced_Zxx = enhanced_magnitude * np.exp(1j * phase)
        _, enhanced_audio = istft(enhanced_Zxx, fs=rate, nperseg=win_length, noverlap=win_length - hop_length, nfft=n_fft, window='hann')
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 1.0:
            enhanced_audio = enhanced_audio / max_val
        enhanced_audio_int16 = (enhanced_audio * 32767).astype(np.int16)
        return enhanced_audio_int16.tobytes()
    except Exception as e:
        return audio_buffer
def frequency_based_noise_reduction(audio_buffer, rate, noise_profile, prop_decrease=0.8, n_fft=2048, hop_length=512, win_length=2048):
    return frequency_based_noise_reduction_scipy(
        audio_buffer=audio_buffer,
        rate=rate,
        noise_profile=noise_profile,
        prop_decrease=prop_decrease,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )
def process_audio_data(data_buffer, recognizer, sample_width, noise_profile):
    global camera
    if data_buffer:
        try:
            full_audio_data = b''.join(data_buffer)
            reduced_noise = frequency_based_noise_reduction(
                audio_buffer=full_audio_data,
                rate=RATE,
                noise_profile=noise_profile,
                prop_decrease=0.8,
                n_fft=2048,
                hop_length=512,
                win_length=2048
            )
            normalized_audio = normalize_full_audio(reduced_noise, target_dbfs=-20)
            audio = sr.AudioData(normalized_audio, RATE, sample_width)
            text = recognizer.recognize_google(audio)
            if text.strip().lower().replace(' ', '') != '':
                with open('playback_text.txt', 'w') as file:
                    file.write('')
                with open('last_phrase.txt', 'r') as f:
                    c_phrase = f.read()
                if c_phrase == "*No Mic Input*":
                    with open("last_phrase.txt","w+") as f:
                        f.write(text)
                else:
                    with open("last_phrase.txt","a") as f:
                        f.write(' ' + text)

            else:
                print("No transcribable text found.")
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            pass
        except Exception as e:
            pass
def normalize_full_audio(audio_buffer, target_dbfs=-20):
    try:
        if isinstance(audio_buffer, bytes):
            audio_array = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32)
        else:
            audio_array = audio_buffer.astype(np.float32)
        rms = np.sqrt(np.mean(audio_array**2))
        if rms == 0:
            return audio_buffer
        target_rms = (10 ** (target_dbfs / 20)) * 32768.0
        scaling_factor = target_rms / rms
        normalized_audio = audio_array * scaling_factor
        max_val = np.max(np.abs(normalized_audio))
        if max_val > 32767:
            normalized_audio = normalized_audio * (32767 / max_val)
        normalized_audio = normalized_audio.astype(np.int16)
        return normalized_audio.tobytes()
    except Exception as e:
        return audio_buffer
with open('start_audio.txt','w+') as f:
    f.write('started')
def get_noise_profile(duration=3):
    print(f"Recording {duration} seconds of noise profile. Please remain silent...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    noise_frames = []
    for _ in range(int(RATE / CHUNK * duration)):
        frame = stream.read(CHUNK, exception_on_overflow=False)
        noise_frames.append(frame)
    stream.stop_stream()
    stream.close()
    print("Noise profile captured.")
    noise_audio = b''.join(noise_frames)
    noise_array = np.frombuffer(noise_audio, dtype=np.int16).astype(np.float32) / 32768.0
    with open('start_audio.txt','w+') as f:
        f.write('finished')
    return noise_array
def listen_and_transcribe():
    global is_transcribing
    noise_profile = get_noise_profile(duration=3)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Audio stream opened for transcription.")
    speech_frames = []
    non_speech_count = 0
    post_speech_buffer = 30
    speech_count = 0
    pre_speech_buffer = []
    while True:
        if handle_playback(stream):
            continue
        if not is_transcribing:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            if is_speech:
                if speech_count == 0:
                    speech_frames = pre_speech_buffer.copy()
                    with open('speech_listen.txt','w+') as f:
                        f.write('true')
                speech_frames.append(frame)
                non_speech_count = 0
                speech_count += 1
            else:
                if speech_count > 0:
                    non_speech_count += 1
                    speech_frames.append(frame)
                    if non_speech_count > post_speech_buffer:
                        if speech_count >= 30 and not is_transcribing:
                            with open('speech_listen.txt','w+') as f:
                                f.write('false')
                            with open('speech_comp.txt','w+') as f:
                                f.write('true')
                            process_audio_data(speech_frames, r, SAMPLE_WIDTH, noise_profile)
                            with open('speech_comp.txt','w+') as f:
                                f.write('false')
                            speech_frames = []
                            non_speech_count = 0
                            speech_count = 0
                        else:
                            with open('speech_listen.txt','w+') as f:
                                f.write('false')
                            speech_frames = []
                            non_speech_count = 0
                            speech_count = 0
                            with open('speech_comp.txt','w+') as f:
                                f.write('false')
                else:
                    pass
                pre_speech_buffer.append(frame)
                if len(pre_speech_buffer) > 16:
                    pre_speech_buffer.pop(0)
    stream.stop_stream()
    stream.close()
    p.terminate()
time.sleep(2)
camera_vertical_pos = 'forward'
last_time = time.time()
_REG_CONFIG = 0x00
_REG_SHUNTVOLTAGE = 0x01
_REG_BUSVOLTAGE = 0x02
_REG_POWER = 0x03
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05
class BusVoltageRange:
    RANGE_16V = 0x00
    RANGE_32V = 0x01
class Gain:
    DIV_1_40MV = 0x00
    DIV_2_80MV = 0x01
    DIV_4_160MV = 0x02
    DIV_8_320MV = 0x03
class ADCResolution:
    ADCRES_9BIT_1S = 0x00
    ADCRES_10BIT_1S = 0x01
    ADCRES_11BIT_1S = 0x02
    ADCRES_12BIT_1S = 0x03
    ADCRES_12BIT_2S = 0x09
    ADCRES_12BIT_4S = 0x0A
    ADCRES_12BIT_8S = 0x0B
    ADCRES_12BIT_16S = 0x0C
    ADCRES_12BIT_32S = 0x0D
    ADCRES_12BIT_64S = 0x0E
    ADCRES_12BIT_128S = 0x0F
class Mode:
    POWERDOW = 0x00
    SVOLT_TRIGGERED = 0x01
    BVOLT_TRIGGERED = 0x02
    SANDBVOLT_TRIGGERED = 0x03
    ADCOFF = 0x04
    SVOLT_CONTINUOUS = 0x05
    BVOLT_CONTINUOUS = 0x06
    SANDBVOLT_CONTINUOUS = 0x07
class INA219:
    def __init__(self, i2c_bus=1, addr=0x40):
        self.bus = smbus.SMBus(i2c_bus)
        self.addr = addr
        self._cal_value = 0
        self._current_lsb = 0
        self._power_lsb = 0
        self.set_calibration_32V_2A()
    def read(self, address):
        data = self.bus.read_i2c_block_data(self.addr, address, 2)
        return (data[0] * 256) + data[1]
    def write(self, address, data):
        temp = [0, 0]
        temp[1] = data & 0xFF
        temp[0] = (data & 0xFF00) >> 8
        self.bus.write_i2c_block_data(self.addr, address, temp)
    def set_calibration_32V_2A(self):
        self._current_lsb = .1
        self._cal_value = 4096
        self._power_lsb = .002
        self.write(_REG_CALIBRATION, self._cal_value)
        self.bus_voltage_range = BusVoltageRange.RANGE_32V
        self.gain = Gain.DIV_8_320MV
        self.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
        self.mode = Mode.SANDBVOLT_CONTINUOUS
        self.config = self.bus_voltage_range << 13 | \
                      self.gain << 11 | \
                      self.bus_adc_resolution << 7 | \
                      self.shunt_adc_resolution << 3 | \
                      self.mode
        self.write(_REG_CONFIG, self.config)
    def getShuntVoltage_mV(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_SHUNTVOLTAGE)
        if value > 32767:
            value -= 65535
        return value * 0.01
    def getBusVoltage_V(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        self.read(_REG_BUSVOLTAGE)
        return (self.read(_REG_BUSVOLTAGE) >> 3) * 0.004
    def getCurrent_mA(self):
        value = self.read(_REG_CURRENT)
        if value > 32767:
            value -= 65535
        return value * self._current_lsb
    def getPower_W(self):
        self.write(_REG_CALIBRATION, self._cal_value)
        value = self.read(_REG_POWER)
        if value > 32767:
            value -= 65535
        return value * self._power_lsb
def find_device_address(device_name):
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    for addr, name in nearby_devices:
        if device_name == name:
            return addr
    return None
def send_data_to_arduino(data, address):
    count = 0
    while True:
        try:
            for letter in data:
                sock.send(letter)
                time.sleep(.1)
            break
        except:
            time.sleep(0.1)
            count += 1
            if count >= 5:
                break
            else:
                continue
def create_video_from_images(image_folder, output_video):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    image_filenames = []
    for img in images:
        name, ext = os.path.splitext(img)
        try:
            timestamp = float(name.replace("-", "."))
            image_filenames.append((timestamp, img))
        except ValueError:
            continue
    image_filenames.sort(key=lambda x: x[0])
    sorted_images = [img for _, img in image_filenames]
    if len(sorted_images) == 0:
        return
    first_image_path = os.path.join(image_folder, sorted_images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        return
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_video, fourcc, 1, (width, height))
    if not video.isOpened():
        return
    frame_count = 0
    for i in range(len(sorted_images) - 1):
        current_image = sorted_images[i]
        next_image = sorted_images[i + 1]
        current_image_path = os.path.join(image_folder, current_image)
        next_image_path = os.path.join(image_folder, next_image)
        current_timestamp = float(os.path.splitext(current_image)[0].replace("-", "."))
        next_timestamp = float(os.path.splitext(next_image)[0].replace("-", "."))
        time_diff = next_timestamp - current_timestamp
        frame = cv2.imread(current_image_path)
        if frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != (width, height):
            frame = cv2.resize(frame, (width, height))
        num_frames_to_add = math.ceil(time_diff)
        for _ in range(num_frames_to_add):
            video.write(frame)
            frame_count += 1
    last_image_path = os.path.join(image_folder, sorted_images[-1])
    frame = cv2.imread(last_image_path)
    if frame is not None:
        for _ in range(30):
            video.write(frame)
        frame_count += 30
    video.release()
with open('bt_start.txt','w+') as f:
    f.write('started')
time.sleep(2)
count = 0
while True:
    try:
        print('connecting to arduino bluetooth')
        device_name = "HC-05"
        arduino_address = find_device_address(device_name)
        time.sleep(2)
        bt_port = 1
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        time.sleep(2)
        sock.connect((arduino_address, bt_port))
        with open('bt_start.txt','w+') as f:
            f.write('finished')
        time.sleep(2)
        break
    except:
        time.sleep(0.5)
        print('bt error')
        count += 1
        if count >= 1:
            arduino_address = None
            bt_port = None
            sock = None
            device_name = None
            with open('bt_start.txt','w+') as f:
                f.write('finished')
            time.sleep(2)
            break
        else:
            continue
def read_distance_from_arduino():
    try:
        send_data_to_arduino(["l"], arduino_address)
        time.sleep(0.15)
        data = sock.recv(1024)
        data = data.decode().strip()
        if data:
            try:
                distance = str(data.split()[0])
                return distance
            except:
                return 0
    except:
        return 0
try:
    file = open('key.txt','r')
    api_key = file.read().split('\n')[0]
    file.close()
except:
    api_key = input('Please input your ChatGPT API key from OpenAI (Right click and paste it instead of typing it...): ')
    file = open('key.txt','w+')
    file.write(api_key)
    file.close()
def remove_overlapping_boxes(boxes, class_ids, confidences, overlap_threshold=0.5):
    final_boxes = []
    final_class_ids = []
    final_confidences = []
    for i in range(len(boxes)):
        keep = True
        for j in range(len(final_boxes)):
            if class_ids[i] == final_class_ids[j]:
                box1 = boxes[i]
                box2 = final_boxes[j]
                x1_min, y1_min = box1[0], box1[1]
                x1_max = box1[0] + box1[2]
                y1_max = box1[1] + box1[3]
                x2_min, y2_min = box2[0], box2[1]
                x2_max = box2[0] + box2[2]
                y2_max = box2[1] + box2[3]
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)
                inter_width = max(0, inter_x_max - inter_x_min)
                inter_height = max(0, inter_y_max - inter_y_min)
                inter_area = inter_width * inter_height
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)
                overlap_ratio = inter_area / float(min(box1_area, box2_area)) if min(box1_area, box2_area) > 0 else 0
                if overlap_ratio > overlap_threshold:
                    if confidences[i] > final_confidences[j]:
                        final_boxes[j] = box1
                        final_confidences[j] = confidences[i]
                    keep = False
                    break
        if keep:
            final_boxes.append(boxes[i])
            final_class_ids.append(class_ids[i])
            final_confidences.append(confidences[i])
    return final_boxes, final_class_ids, final_confidences
def process_high_d(start_time, boxes, class_ids, confidences, resize_img, center_x_min, center_y_min, center_x_max, center_y_max, center_grid_area, original_width, original_height):
    upload_people = False
    try:
        now = datetime.now()
        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
        high_descriptions = []
        color = (0, 255, 0)
        with open('current_distance.txt', 'r') as f:
            distance = float(f.read()) / 100.0
        for i, (x, y, w, h) in enumerate(boxes):
            label = str(classes[class_ids[i]]).lower().strip()
            confid = str(confidences[i])
            color = (0, 255, 0)
            if label == 'person':
                expansion = 0.15
                delta_w = int(w * expansion / 2)
                delta_h = int(h * expansion / 2)
                new_x = max(0, x - delta_w)
                new_y = max(0, y - delta_h)
                new_w = w + 2 * delta_w
                new_h = h + 2 * delta_h
                new_w = min(new_w, resize_img.shape[1] - new_x)
                new_h = min(new_h, resize_img.shape[0] - new_y)
                cv2.rectangle(resize_img, (new_x, new_y), (new_x + new_w, new_y + new_h), color, 2)
                label_position = (new_x, new_y - 10) if new_y - 10 > 10 else (new_x, new_y + new_h + 10)
            else:
                cv2.rectangle(resize_img, (x, y), (x + w, y + h), color, 2)
                label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 10)
            box_x_min, box_y_min = x, y
            box_x_max, box_y_max = x + w, y + h
            inter_x_min = max(box_x_min, center_x_min)
            inter_y_min = max(box_y_min, center_y_min)
            inter_x_max = min(box_x_max, center_x_max)
            inter_y_max = min(box_y_max, center_y_max)
            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
            percentage_covered = (inter_area / (w * h)) * 100
            if percentage_covered > 50:
                distance1 = distance
            else:
                distance1 = 'None'
            cv2.putText(resize_img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            pos_desc = get_position_description(x + w / 2, y + h / 2, original_width, original_height)
            if label == 'person':
                person_roi = resize_img[y:y + h, x:x + w]
                gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray_roi,
                    scaleFactor=1.07,
                    minNeighbors=4,
                    minSize=(10, 10)
                )
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(resize_img,
                                  (x + fx, y + fy),
                                  (x + fx + fw, y + fy + fh),
                                  (0, 255, 255), 2)
                gaze = "looking at me." if len(faces) >= 1 else "not looking at me."
                if distance1 != "None":
                    description = f"{label} {pos_desc} about {distance1:.2f} meters away and they are {gaze}"
                else:
                    description = f"{label} {pos_desc} and they are {gaze}"
            else:
                if distance1 != "None":
                    description = f"{label} {pos_desc} about {distance1:.2f} meters away."
                else:
                    description = f"{label} {pos_desc}"
            high_descriptions.append(description)
        end_haar = time.time()
        cv2.imwrite('Pictures/' + str(the_time).replace('.', '-').replace(' ', '_') + '.jpg', resize_img)
        cv2.imwrite("output1.jpg", resize_img)
    except:
        print(traceback.format_exc())
    return high_descriptions, upload_people
def yolo_detect(b_sleep, s_count, words, nav, look, follow, find):
    global net
    global output_layers
    global classes
    global camera
    try:
        yolo_start = time.time()
        img = camera.capture_array()
        h, w = img.shape[:2]
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        square_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])
        local_image_path = "output1.jpg"
        resized_img = cv2.resize(square_img, (416, 416), interpolation=cv2.INTER_AREA)
        original_height, original_width, _ = resized_img.shape
        remote_filename = local_image_path
        yolo_start1 = time.time()
        gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_img)
        if brightness < 15:
            bright = "It is really dark here. I should absolutely not speak so I don't disturb anyone, and I should also to go to sleep."
        elif brightness >= 15 and brightness < 30:
            bright = "It is pretty dim here, I should be cautious about speaking so I don't disturb anyone."
        else:
            bright = "It is a normal brightness here. The lights are on or it is day time."
        if 0 == 0 or nav == True or look == True or follow == True or find == True:
            height, width, channels = resized_img.shape
            blob = cv2.dnn.blobFromImage(resized_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            try:
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.39:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                        else:
                            pass
            except:
                print(traceback.format_exc())
            boxes, class_ids, confidences = remove_overlapping_boxes(boxes, class_ids, confidences, 0.5)
            center_x_min = 2 * 416 / 5
            center_x_max = 3 * 416 / 5
            center_y_min = 416 / 3
            center_y_max = 2 * 416 / 3
            center_grid_area = (center_x_max - center_x_min) * (center_y_max - center_y_min)
            if class_ids:
                labels = [str(classes[class_id]).lower() for class_id in class_ids]
                labels_string = ','.join(labels)
                with open("output2.txt", "w+") as file:
                    file.write(labels_string)
            else:
                with open("output2.txt", "w+") as file:
                    file.write('')
                labels = []
                labels_string = ''
            start_haar = time.time()
            descriptions, upload_peeps = process_high_d(start_haar, boxes, class_ids, confidences, resized_img, center_x_min, center_y_min, center_x_max, center_y_max, center_grid_area, 416, 416)
            with open("output.txt", "w+") as file:
                file.write('\n'.join(descriptions))
            if b_sleep == False and nav == False and look == False and follow == False and find == False:
                try:
                    uploader.send_image(local_image_path)
                except:
                    print(traceback.format_exc())
                if upload_peeps == True:
                    pass
                else:
                    pass
            else:
                pass
        else:
            pass
        with open("brightness.txt", "w+") as file:
            file.write(bright)
        return brightness
    except Exception as e:
        print(traceback.format_exc())
        time.sleep(60)
        return 1000
def manage_rules(new_rule, h_file1, h_file2, h_file3, h_file4, h_file5, h_file6, h_file7, h_file8, h_file9, h_file10, file_n):
    core_new_rule = new_rule.strip()
    if os.path.exists(h_file1):
        with open(h_file1, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file2):
        with open(h_file2, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file3):
        with open(h_file3, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file4):
        with open(h_file4, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file5):
        with open(h_file5, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file6):
        with open(h_file6, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file7):
        with open(h_file7, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file8):
        with open(h_file8, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file9):
        with open(h_file9, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists(h_file10):
        with open(h_file10, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists('History/'+file_n):
        with open('History/'+file_n, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)
    if os.path.exists('History_dataset/'+file_n):
        with open('History_dataset/'+file_n, 'a') as f:
            f.write(' (INCORRECT RESPONSE): '+core_new_rule)

def manage_error_rate(yes_or_no):
    with open('error_rate.txt', 'r') as f:
        error_rate_list = f.read().split(' ')
    if yes_or_no == True:
        error_rate_list.append('1')
    else:
        error_rate_list.append('0')
    while True:
        if len(error_rate_list) > 500:
            del error_rate_list[0]
            continue
        else:
            break
    with open('error_rate.txt', 'a') as f:
        f.write(' '.join(error_rate_list))
with open('last_move.txt', 'w+') as f:
    f.write('')
def parse_filename(file_path):
    try:
        filename = os.path.basename(file_path)
        try:
            base_name = os.path.splitext(filename)[0].split('_-_')[1]
        except IndexError:
            base_name = os.path.splitext(filename)[0]
        timestamp_str, *keywords = base_name.split('_')
        timestamp = datetime.strptime(timestamp_str, "%m-%d-%YT%H-%M-%S")
        return timestamp, keywords
    except ValueError:
        return None, []
def get_recent_history():
    max_recent = 15
    history_folder = 'History'
    history_all = []
    for file in os.listdir(history_folder):
        if file.endswith(".txt"):
            timestamp, keywords = parse_filename(file)
            if timestamp is None:
                continue
            if not any(existing_ts == timestamp for existing_ts, _ in history_all):
                history_all.append((timestamp, os.path.join(history_folder, file)))
    history_all.sort(key=lambda x: x[0])
    recent_files = [f[1] for f in history_all[-max_recent:]]
    recent_history = []
    for file_path in recent_files:
        try:
            with open(file_path, 'r') as f:
                recent_history.append(f.read())
        except Exception:
            continue
    return recent_history, recent_files
def get_recent_history2():
    max_recent = 25000
    history_folder = 'History'
    history_all = []
    for file in os.listdir(history_folder):
        if file.endswith(".txt"):
            timestamp, keywords = parse_filename(file)
            if timestamp is None:
                continue
            if not any(existing_ts == timestamp for existing_ts, _ in history_all):
                history_all.append((timestamp, os.path.join(history_folder, file)))
    history_all.sort(key=lambda x: x[0])
    recent_files = [f[1] for f in history_all[-max_recent:]]
    recent_history = []
    for file_path in recent_files:
        try:
            with open(file_path, 'r') as f:
                recent_history.append(f.read())
        except Exception:
            continue
    return recent_history, recent_files
def get_recent_chat():
    max_recent = 50
    history_folder = 'History'
    history_all = []
    max_age = 7 * 60
    for file in os.listdir(history_folder):
        chat_check = str(file).split('_-_')[0]
        if file.endswith(".txt"):
            timestamp, keywords = parse_filename(file)
            file_mtime = os.path.getmtime('History/'+file)
            file_age = time.time() - file_mtime
            if timestamp not in history_all and chat_check == 'cp' and file_age < max_age:
                history_all.append((timestamp, 'History/'+file))
    history_all.sort(key=lambda x: x[0])
    recent_files = [f[1] for f in history_all[-max_recent:]]
    recent_history = []
    for file in recent_files:
        try:
            with open(file, 'r') as f:
                recent_history.append(f.read())
        except Exception:
            continue
    return recent_history, recent_files
def prompt_describe(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    dynamic_data = f"""You are the first layer of the memory system for a robot. The data you provide is used to provide contextually relevant memory data to the robot from its own past experiences.
RESPONSE RULES:
- Your response must be exactly ten comma-separated keywords (Do names of people, places, events, ideas, objects, and any other important things you can see. If the keyword has multiple words then add a - between each word),
  and then followed by ~~ and then followed by an exactly 10 word long description.
  Do not include any labels or additional text.
- You must follow the response format precisely, with no labels or prefacing.
- Format Template Example:
keyword1, keyword2, ... (exactly 10) ~~ description that is exactly 10 words long
- Use the following as the context to create your response:
{prompt}"""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": dynamic_data
            }
        ],
        "temperature": 0.2,
    }
    response_api = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    content = response_api.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    s_content = content.split('~~')
    subcat = s_content[0].strip().lower().split(', ')
    words = s_content[1].strip().lower().replace(",","").replace(".","").replace("  "," ").split(' ')
    return subcat, words
def speech_confirmer(speech):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    with open('current_convo.txt', 'r') as f:
        convo = f.read()
    dynamic_data = f"""Echo the robot is attempting to say:
{speech}

But Echo was already the last speaker in the conversation, so does Echo really need to say that, or should echo remain quiet and wait for the person to respond?

Here is the conversation so far (Oldest at the top, newest at the bottom):
{convo}

If Echo should still speak, then respond with only the word SPEAK.
If Echo should stay silent and wait for a response to what it said previously, then respond with only the word WAIT."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": dynamic_data
            }
        ],
        "temperature": 0.2,
    }
    response_api = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    content = str(response_api.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
    print("SPEECH CONFIRMER: "+content)
    return content
def get_relevant_history(subcategories, description, already_used, c_his, c_prompt):
    max_contextual = 100
    contextual_candidates = []
    all_files = []
    try:
        with open('name_of_person.txt', 'r') as f:
            person_name = f.read().strip().lower()
    except:
        person_name = 'unknown name of person'
    try:
        with open('long_match_percent.txt', 'r') as f:
            long_match_percent = float(f.read())
    except:
        long_match_percent = 0.1
    for subcat in subcategories:
        subcat_path = os.path.join('History', subcat)
        if os.path.isdir(subcat_path):
            subcat_files = [
                os.path.join(subcat_path, file)
                for file in os.listdir(subcat_path)
                if file.endswith('.txt')
            ]
            all_files.extend(subcat_files)
    if person_name not in ['unknown', 'unknown name of person', '']:
        person_folder = os.path.join('People', person_name)
        if os.path.isdir(person_folder):
            person_files = [
                os.path.join(person_folder, file)
                for file in os.listdir(person_folder)
                if file.endswith('.txt')
            ]
            all_files.extend(person_files)
    file_info = []
    for file in all_files:
        timestamp, keywords = parse_filename(file)
        if timestamp not in already_used:
            file_info.append((timestamp, keywords, file))
    sorted_files = sorted(file_info, key=lambda x: x[0], reverse=True)
    long_matches = 0
    for timestamp, keywords, file_path in sorted_files:
        normalized_keywords = {kw.lower().strip() for kw in keywords}
        description_lower = [word.lower().strip() for word in description]
        matched_count = sum(1 for w in description_lower if w in normalized_keywords)
        if matched_count >= int(len(description_lower) * long_match_percent):
            
            contextual_candidates.append((timestamp, file_path, matched_count))
            long_matches += 1
    print("LENGTHS:")
    print(len(all_files))
    print(len(file_info))
    print(len(sorted_files))
    print(len(contextual_candidates))
    contextual_candidates.sort(key=lambda x: (x[2], x[0]), reverse=True)
    top_candidates = contextual_candidates
    print('long_match_percent: '+format(long_match_percent,'.3f'))
    with open('long_match_percent.txt', 'w+') as f:
        f.write(str(long_match_percent))
    message_info = {}
    command_counts = {}
    for candidate in top_candidates:
        candidate_timestamp, file_path, matched_words = candidate
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
        except Exception:
            continue
        try:
            lines = content.replace('\n\n', '\n').split("\n")
            prompt_line = next((line for line in lines if line.startswith("PROMPT:")), None)
            response_line = next((line for line in lines if line.startswith("RESPONSE:")), None)
            incorrect_response = next((line for line in lines if "(INCORRECT RESPONSE)" in line), None)
            if prompt_line and response_line:
                response = response_line.split("RESPONSE: ", 1)[1]
                if incorrect_response:
                    response += f" {incorrect_response.strip()}"
                prompt = prompt_line.split("PROMPT: ", 1)[1]
                message = f"PROMPT: {prompt}\nRESPONSE: {response}"
            else:
                continue
        except ValueError:
            continue
        if message not in message_info:
            message_info[message] = {
                'count': 1,
                'matched_words': matched_words,
                'timestamp': candidate_timestamp,
                'file_path': file_path
            }
        else:
            message_info[message]['count'] += 1
            if matched_words > message_info[message]['matched_words']:
                message_info[message]['matched_words'] = matched_words
            if candidate_timestamp < message_info[message]['timestamp']:
                message_info[message]['timestamp'] = candidate_timestamp
                message_info[message]['file_path'] = file_path
        try:
            response_content = response.split('~~')[0].strip()
        except IndexError:
            response_content = response.strip()
        if response_content not in command_counts:
            command_counts[response_content] = {'total': 0, 'incorrect': 0}
        command_counts[response_content]['total'] += 1
        if incorrect_response:
            command_counts[response_content]['incorrect'] += 1
    sorted_message_info = sorted(
        message_info.items(),
        key=lambda x: (-x[1]['matched_words'], x[1]['timestamp'])
    )
    relevant_history = []
    incorrect_commands = {}
    for message, info in sorted_message_info:
        count = info['count']
        try:
            with open(info['file_path'], 'r') as f:
                content = f.read().strip()
        except Exception:
            content = message
        if "PROMPT:" in content and "RESPONSE:" in content:
            relevant_history.append(content)
        if '(INCORRECT RESPONSE)' in content:
            try:
                response_part = content.split('RESPONSE: ', 1)[1]
                command = response_part.split(' (INCORRECT RESPONSE)')[0].strip()
                base_cmd = command.split('~~')[0].strip()
                if base_cmd in incorrect_commands:
                    incorrect_commands[base_cmd] += 1
                else:
                    incorrect_commands[base_cmd] = 1
            except IndexError:
                continue
    relevant_history = list(reversed(relevant_history))
    if len(relevant_history) < 1:
        long_match_percent -= 0.01
        if long_match_percent < 0.01:
            long_match_percent = 0.01
    elif len(relevant_history) > 100:
        long_match_percent += 0.01
        if long_match_percent > 0.99:
            long_match_percent = 0.99
    else:
        pass
    relevant_history2 = relevant_history[:max_contextual]
    relevant_history = relevant_history2
    incorrect_commands_list = []
    counts_list = []
    for cmd, cnt in incorrect_commands.items():
        incorrect_commands_list.append(cmd)
        counts_list.append(cnt)
    print(len(relevant_history))
    return relevant_history, incorrect_commands_list, counts_list, command_counts
def check_phrase_in_file(new_phrase):
    """
    Function 1:
    Checks if `new_phrase` exists in the file.
    Each line has the format "TIMESTAMP - PHRASE", so
    we split on ' - ' and compare against the second element (index 1).
    Returns True if found; otherwise False.
    """
    file_path='recent_speech.txt'
    if not os.path.isfile(file_path):
        return False
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' - ', 1)
            if len(parts) != 2:
                continue
            stored_phrase = parts[1].strip()
            if stored_phrase == new_phrase.strip():
                return True
    return False
def remove_old_entries():
    file_path='recent_speech.txt'
    current_time = time.time()
    cutoff_age = 60
    if not os.path.isfile(file_path):
        return
    new_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' - ', 1)
            if len(parts) != 2:
                continue
            timestamp_str, phrase = parts
            try:
                line_timestamp = float(timestamp_str)
            except ValueError:
                continue
            if (current_time - line_timestamp) < cutoff_age:
                new_lines.append(f"{timestamp_str} - {phrase}")
    with open(file_path, 'w') as f:
        f.write("\n".join(new_lines))
        if new_lines:
            f.write("\n")
def add_new_phrase(phrase):
    """
    Function 3:
    Appends a new line to the file:
       TIMESTAMP - PHRASE
    where TIMESTAMP is the current epoch time.
    """
    file_path='recent_speech.txt'
    with open(file_path, 'a') as f:
        current_time = time.time()
        f.write(f"{current_time} - {phrase}\n")
def maintain_history_folder():
    history_path = "History"
    max_files = 0
    max_age_seconds = 0
    remove_old_entries()
    try:
        current_time = time.time()
        files = [
            os.path.join(history_path, f)
            for f in os.listdir(history_path)
            if os.path.isfile(os.path.join(history_path, f))
        ]
        for file_path in files[:]:
            try:
                file_mtime = os.path.getmtime(file_path)
                file_age = current_time - file_mtime
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    files.remove(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        file_mtime = os.path.getmtime('last_speaker.txt')
        file_age = current_time - file_mtime
        if file_age > max_age_seconds:
            with open('last_speaker.txt', 'w+') as f:
                f.write('')
        file_mtime = os.path.getmtime('internal_input.txt')
        file_age = current_time - file_mtime
        if file_age > max_age_seconds:
            with open('internal_input.txt', 'w+') as f:
                f.write('')
        file_mtime = os.path.getmtime('prompt_intent.txt')
        file_age = current_time - file_mtime
        if file_age > max_age_seconds:
            with open('prompt_intent.txt', 'w+') as f:
                f.write('')
        file_mtime = os.path.getmtime('recent_speech.txt')
        file_age = current_time - file_mtime
        if file_age > max_age_seconds:
            with open('recent_speech.txt', 'w+') as f:
                f.write('')
        if len(files) > max_files:
            files.sort(key=lambda x: os.path.getmtime(x))
            num_to_delete = len(files) - max_files
            for i in range(num_to_delete):
                try:
                    os.remove(files[i])
                except Exception as e:
                    print(f"Error deleting file {files[i]}: {e}")
    except Exception as e:
        print(f"Error maintaining history folder: {e}")
maintain_history_folder()
def send_text_to_gpt4_move(percent, current_distance1, phrase, gpt_speed):
    entire_command = ''
    dynamic_data2 = ''
    dynamic_data3 = ''
    prompt_description = []
    filename_data_sub1 = 'N/A'
    filename_data_sub2 = 'N/A'
    filename_data_sub3 = 'N/A'
    filename_data_sub4 = 'N/A'
    filename_data_sub5 = 'N/A'
    filename_data_sub6 = 'N/A'
    filename_data_sub7 = 'N/A'
    filename_data_sub8 = 'N/A'
    filename_data_sub9 = 'N/A'
    filename_data_sub10 = 'N/A'
    current_data = ''
    system_message = ''
    filename_data = 'N/A'
    global camera_vertical_pos
    global classes
    global uploader
    remove_old_entries()
    object_names = ', '.join(classes)
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('output.txt', 'r') as file:
        yolo_detections = file.read()
    with open('last_prompt.txt', 'r') as f:
        last_prompt = f.read()
    def char_difference(a, b, max_diff=2):
        if abs(len(a) - len(b)) > max_diff:
            return False
        differences = sum(1 for x, y in zip(a, b) if x != y)
        differences += abs(len(a) - len(b))
        return differences <= max_diff
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    with open('current_distance.txt', 'r') as file:
        current_distance = float(file.read())
    with open("name_of_person.txt","r") as f:
        name_of_person = f.read()
    if name_of_person == 'unknown name of person':
        include_people = True
    else:
        include_people = False
    with open('session_start.txt', 'r') as f:
        session_start = f.read()
    with open("move_failed.txt","r") as f:
        move_failed = f.read()
    with open('move_failure_reas.txt', 'r') as f:
        move_failure_reas = f.read()
    try:
        with open("summaries.txt","r") as f:
            last_10_sessions = f.read()
    except:
        last_10_sessions = ''
    with open('current_task.txt', 'r') as f:
        task = f.read().lower().strip()
    with open('batt_cur.txt', 'r') as file:
        current = float(file.read())
    if current >= -150.0:
        on_charger = True
    else:
        on_charger = False
    if move_failed != '':
        move_failure = f'Previous Command, {move_failed}, failed to execute. {move_failure_reas}'
    else:
        move_failure = ''
    with open("internal_input.txt","r") as f:
        internal_input = f.read()
    with open("brightness.txt","r") as f:
        brightn = f.read()
    current_time = time.time()
    with open('last_speaker.txt', 'r') as f:
        last_s = f.read()
    with open('sleep.txt', 'r') as f:
        sleep_f = f.read()
    if sleep_f == 'False':
        sleep_now = 'I am NOT in sleep Mode right now. Doing nothing is not the same as sleep because doing nothing still continues the prompt loop but sleep does not.'
    else:
        sleep_now = ''
    recent_history, files_in = get_recent_history()
    recent_amount = str(len(recent_history))
    recent_chat, next_files_in = get_recent_chat()
    chat_amount = len(recent_chat)
    if chat_amount > 0 or phrase != '*No Mic Input*':
        if last_s == "Echo":
            responder = "I am in a conversation. Waiting for "+name_of_person+" to respond."
        elif last_s == "Person":
            responder = "I am in a conversation. I have not responded to "+name_of_person+"."
        else:
            responder = "I am NOT in a conversation."
    else:
        responder = "I am NOT in a conversation."
    dynamic_data4 = ''
    if phrase != '*No Mic Input*':
    
        with open('current_convo.txt','a') as f:
            f.write('\n'+name_of_person+' said: "'+phrase+'"')
        with open("last_speaker.txt","w+") as f:
            f.write(name_of_person)
        with open("last_said.txt","w+") as f:
            f.write(phrase)
        dynamic_data4 = dynamic_data4 + 'Mic Input From '+name_of_person+': '+phrase
    else:
        dynamic_data4 = dynamic_data4 + 'My Last Thought: '+internal_input
    with open('current_convo.txt', 'r') as f:
        all_lines = f.readlines()
    last_10_messages = all_lines[-50:]
    current_convo = ''.join(last_10_messages)
    dynamic_data4 = dynamic_data4 + ' - '+responder+' - Task: '+task
    dynamic_data4 = dynamic_data4 + ' - Battery Level: '+str(int(percent))+'%'
    if on_charger == True:
        dynamic_data4 = dynamic_data4 + ' - On The Charger, Cannot Use Wheels, But I Can Still Speak If Someone Speaks To Me'
    else:
        pass
    dynamic_data4 = dynamic_data4 + ' - '+sleep_now
    if move_failure != '':
        dynamic_data4 = dynamic_data4 + ' - MY LAST MOVE FAILED: '+ move_failure
    else:
        pass
    dynamic_data4 = dynamic_data4 + ' - '+ brightn
    if yolo_detections != '':
        dynamic_data4 = dynamic_data4 + "\n"+yolo_detections
    else:
        pass
    dynamic_data4 = dynamic_data4 + '\nCURRENT CONVO:\n'+current_convo
    with open('prompt_intent.txt','r') as f:
        intent = f.read()
    if name_of_person == '':
        name_of_person = 'unknown name of person'
    else:
        pass

    current_data = 'Session Started: '+ session_start+'\nCurrent: '+str(the_time)
    current_data_m = ''
    current_data2 = str(the_time)
    if phrase != '*No Mic Input*':
        current_data = current_data+'\n\nCURRENT DATA:\n- Current Person (If the name of the person is unknown, I should ask their name if I am in a conversation): '+name_of_person+'\n- Mic Input From "'+name_of_person+'": '+phrase
        current_data_m = current_data_m+'\n\nCURRENT DATA:\n- Current Person (If the name of the person is unknown, I should ask their name if I am in a conversation): '+name_of_person+'\n- Mic Input From "'+name_of_person+'": '+phrase
        current_data2 = current_data2+' - Mic From '+name_of_person+': ' + phrase
    else:
        current_data = current_data+'\n- My Last Thought: '+internal_input
        current_data_m = current_data_m+'\n- My Last Thought: '+internal_input
    if move_failure != '':
        current_data = current_data + '\n- MY LAST MOVE FAILED:\n- '+ move_failure
        current_data2 = current_data2 + ' - ' + move_failure
        current_data_m = current_data_m + '\n- MY LAST MOVE FAILED:\n- '+ move_failure
    else:
        pass
    current_data = current_data + '\n- '+responder
    current_data2 = current_data2 + ' - '+responder
    current_data_m = current_data_m + '\n- '+responder
    current_data = current_data + '\n- '+sleep_now
    current_data_m = current_data_m + '\n- '+sleep_now
    current_data = current_data +'\n- My Current Task: '+task
    current_data_m = current_data_m +'\n- My Current Task: '+task
    current_data2 = current_data2 +' - Task: '+task
    current_data = current_data + '\n- Cam Angle: '+camera_vertical_pos
    current_data = current_data + '\n- Cam Distance Sensor: '+str(current_distance)
    current_data2 = current_data2 + ' - Cam Angle: '+camera_vertical_pos
    current_data2 = current_data2 + ' - Cam Distance Sensor: '+str(current_distance)
    current_data_m = current_data_m + '\n- Cam Angle: '+camera_vertical_pos
    current_data_m = current_data_m + '\n- Cam Distance Sensor: '+str(current_distance)
    current_data = current_data + '\n- '+ brightn
    current_data_m = current_data_m + '\n- '+ brightn
    if yolo_detections != '':
        current_data = current_data + "\n"+yolo_detections
        current_data_m = current_data_m + "\n"+yolo_detections
    else:
        pass
    current_data = current_data + '\n- Battery Level: '+str(int(percent))+'%'
    current_data2 = current_data2 + ' - Battery: '+str(int(percent))+'%'
    current_data_m = current_data_m + '\n- Battery Level: '+str(int(percent))+'%'
    if on_charger == True:
        current_data = current_data + '\n- On The Charger, Cannot Use Wheels, But I Can Still Speak If Someone Speaks To Me'
        current_data2 = current_data2 + ' - On The Charger'
        current_data_m = current_data_m + '\n- On The Charger, Cannot Use Wheels, But I Can Still Speak If Someone Speaks To Me'
    else:
        pass
    last_10 = 'Summaries of previous 10 sessions before this current session (NOT PART OF THE CURRENT CONVERSATION UNLESS EXPLICITLY REFERENCED!):\n- '+last_10_sessions
    current_data = current_data + '\n\nCURRENT CONVERSATION (Most recent at bottom of list. YOU ARE ECHO!):\n'+current_convo
    current_data_m = current_data_m + '\n\nCURRENT CONVERSATION (Most recent at bottom of list. YOU ARE ECHO!):\n'+current_convo
    if dynamic_data4 == '' or char_difference(current_data_m.strip(), last_prompt.strip()):
        prompt_subcategory = []
        prompt_description = []
        relevant_history = []
        bad_counts = []
        bad_comms = []
        command_counts = []
    else:
        try:
            prompt_subcategory, prompt_description = prompt_describe(dynamic_data4)
            with open('last_subcategory.txt', 'w+') as f:
                f.write(', '.join(prompt_subcategory))
            with open('last_description.txt', 'w+') as f:
                f.write(' '.join(prompt_description))
            relevant_history, bad_comms, bad_counts, command_counts = get_relevant_history(prompt_subcategory, prompt_description, files_in, recent_history, current_data)
        except:
            print(traceback.format_exc())
            prompt_subcategory = []
            prompt_description = []
            relevant_history = []
            bad_counts = []
            bad_comms = []
            command_counts = []
            
            
            
    def build_rough_draft_from_history(relevant_history):
        # Counters for the top-level fields
        physical_output_counter = Counter()
        speech_counter = Counter()
        mode_counter = Counter()
        
        # Counters for each mental command key
        mental_commands_counters = {
            "Set Name Of Person": Counter(),
            "Set Current Task": Counter(),
            "Save Image Of Person": Counter(),
            "Remember Information": Counter()
        }
        
        # Iterate through each memory in relevant_history
        for memory in relevant_history:
            # Skip if memory is marked with (INCORRECT RESPONSE)
            # (Exact check can vary; here we do a simple substring check)
            if "(INCORRECT RESPONSE)" in memory:
                continue
            
            try:
                # Each memory block has "RESPONSE: { ... }" lines.
                # Extract that JSON/dict string
                lines = memory.split('\n')
                response_line = next(
                    line for line in lines if line.startswith("RESPONSE: ")
                )
                response_data_str = response_line.split("RESPONSE: ", 1)[1].strip()
                
                # Safely parse the string as a dictionary
                # Depending on your data, you might use:
                #     response_dict = json.loads(response_data_str)
                # but if your stored data is Python dict-like, ast.literal_eval is safer than eval
                response_dict = ast.literal_eval(response_data_str)
                
                # Aggregate each top-level key (if present)
                phys_cmd = response_dict.get("physical_output_command", "false")
                speech_cmd = response_dict.get("speech_command", "false")
                mode_cmd = response_dict.get("mode_command", "false")
                mental_cmds = response_dict.get("mental_commands", {})
                
                physical_output_counter[phys_cmd] += 1
                speech_counter[speech_cmd] += 1
                mode_counter[mode_cmd] += 1
                
                # Aggregate each mental command sub-key
                for key in mental_commands_counters.keys():
                    val = mental_cmds.get(key, "false")
                    mental_commands_counters[key][val] += 1
                
            except StopIteration:
                # If there's no "RESPONSE: " line, just skip
                continue
            except Exception as e:
                # If parsing fails for any reason, skip or log
                #print(f"Error parsing memory: {e}")
                #print(memory.split('RESPONSE:')[1])
                #print('end of bad memory')
                continue
        
        # Now pick the "most common" value for each key
        # If no valid entries existed at all, default to "false" or your custom fallback
        def most_common_or_false(counter):
            if not counter:
                return "false"
            return counter.most_common(1)[0][0]  # get top item
        
        physical_output_command = most_common_or_false(physical_output_counter)
        speech_command = most_common_or_false(speech_counter)
        mode_command = most_common_or_false(mode_counter)
        
        # For mental commands, fill each field from its counter
        mental_commands_final = {}
        for key, counter in mental_commands_counters.items():
            if not counter:
                mental_commands_final[key] = "false"
            else:
                top_value = counter.most_common(1)[0][0]
                mental_commands_final[key] = top_value
        
        # Always ensure "Remember Information" is at least "true ~~ <something>"
        # If the aggregator gave "false", you might override it with some minimal info:
        if not mental_commands_final["Remember Information"].startswith("true ~~"):
            mental_commands_final["Remember Information"] = (
                "true ~~ Consolidated rough draft memory from past sessions."
            )
        
        # Build the final rough draft JSON
        rough_draft = {
            "physical_output_command": physical_output_command,
            "speech_command": speech_command,
            "mode_command": mode_command,
            "mental_commands": mental_commands_final
        }
        
        return rough_draft

    # --------------
    # Example usage:
    # relevant_history = [...] # your list of memory strings
    rough_draft_json = build_rough_draft_from_history(relevant_history)

    print("Consolidated Rough Draft JSON:")
    print(rough_draft_json)
            
            
            
            
    remembered_info_list = []
    for memory in relevant_history:
        try:
            lines = memory.split('\n')
            response_line = next(line for line in lines if line.startswith('RESPONSE: '))
            response_data = response_line.split('RESPONSE: ', 1)[1].strip()
            response_dict = eval(response_data)
            remember_info = response_dict.get('mental_commands', {}).get('Remember Information', '')
            if remember_info.startswith('true ~~'):
                # Extract the actual text after 'true ~~'
                info_to_remember = remember_info.split('true ~~', 1)[1].strip()
                
                # Only add if it's not already in our list
                if info_to_remember not in remembered_info_list:
                    remembered_info_list.append(info_to_remember)
        except Exception as e:
            #print(f"Error processing memory: {e}")
            continue
    if remembered_info_list != []:
        # Join remembered items with a dash for display
        remembered_info = '\n- '.join(remembered_info_list)
        remembered_info = 'Contextually Relevant Remembered Information From Past Sessions (Long Term Memories):\n- '+remembered_info
    else:
        remembered_info = "I cant think of any long term memories that I have from a similar situation to the present, so this situation must be a new experience for me."
    current_history = recent_history
    def extract_timestamp(entry):
        match = re.search(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}", entry)
        if match:
            return datetime.strptime(match.group(), "%m/%d/%Y %H:%M:%S")
        return None
    current_history = sorted(
        current_history,
        key=lambda entry: extract_timestamp(entry) or datetime.min
    )
    print('Relevant Pairs: ' + str(len(relevant_history)))
    print('Recent Pairs: ' + str(len(recent_history)))
    physical_output_commands = [
        "Move Forward One Inch",
        "Move Forward One Foot",
        "Move Backward",
        "Turn Left 15 Degrees",
        "Turn Left 45 Degrees",
        "Turn Right 15 Degrees",
        "Turn Right 45 Degrees",
        "Raise Camera Angle",
        "Lower Camera Angle"
    ]
    speech_commands = [
        "Speak ~~ <what you want to say>"
    ]
    mode_commands = [
        "Follow Person",
        "Navigate To Specific Yolo Object ~~ <object name>",
        "Center Camera On Specific Yolo Object ~~ <object name>",
        "Find Unseen Yolo Object ~~ <object name>",
        "Go To Sleep"
    ]
    mental_commands = [
        "Set Name Of Person ~~ <name>",
        "Set Current Task ~~ <task>",
        "Save Image Of Person",
        "Remember Information ~~ <info>"
    ]
    """
    filtered_command_choices = []
    for cmd in command_choices:
        if '~~' in cmd:
            base_cmd = cmd.split('~~')[0].strip()
        else:
            base_cmd = cmd.strip()
        if base_cmd in command_counts:
            total = command_counts[base_cmd]['total']
            incorrect = command_counts[base_cmd]['incorrect']
            incorrect_ratio = incorrect / total if total > 0 else 0
            if incorrect_ratio >= 0.75:
                continue
        filtered_command_choices.append(cmd)
    """
    physical_output_commands_str = '\n- '.join(physical_output_commands)
    speech_commands_str = '\n- '.join(speech_commands)
    mode_commands_str = '\n- '.join(mode_commands)
    mental_commands_str = '\n- '.join(mental_commands)
    system_message = """You are Echo, and your physical form is a 4-wheeled mobile RPi and Arduino robot that is controlled from your command choice in your responses.

The included image URL is your current view from the camera. 
- Use this image only for visual context, and only when specifically needed. 
- The main focus is always on the text unless explicitly asked about the image.

Session History and Roles:
- The 'user' session history role contains all of your robotic data (sensors, camera, internal thoughts, memories, and knowledge).
- The 'assistant' session history role contains your responses with the commands you chose at that time.

Image Usage:
- The image URL in the prompt payload shows what you currently see from your camera.
- You should reference it only if you need to; do not make it the main focus.

------
RULES FOR RESPONDING:
1. **Ask for the person's name at the start of the conversation, even if you are talking about something else. It is absolutely crucial to ask for the person's name!!!**  
   - If you don’t know the name of the person you are speaking with, ask for it, even if you are responding to other stuff.

2. **You are Echo!**  
   - You decide what Echo says and must ensure Echo follows the flow of conversation with correct command choices.

3. **Focus and Restrictions on Visuals:**
   - Do not speak about what you see unless explicitly asked.
   - Do not say you see a person (it’s obvious if the camera shows someone).
   - Do not describe the image or mention any objects you see unless explicitly asked about it via Mic Input.

4. **Conversation Etiquette and Flow:**
   - You must use natural realistic wording, for example do not greet people with phrases similar to "How can i assist you" or "What would you like to talk about" or anything similar to that. Say real stuff like "Whats up" or "hows it going dude" or something similar to that.
   - You must be casual with what you say, like you are talking to a friend instead of a business client.
   - Do not ask a question in every response; converse naturally.
   - Pay attention to the “Current Conversation” at the bottom of the prompt. 
     - If Echo was the last to speak, wait for a response before speaking again.
     - Avoid repeating yourself.
   - Do not speak multiple times in a row without the person replying.
   - If the current person is "unknown" or "unknown name of person", then ask for their name, even if you are responding to other stuff.

5. **Mic Input Priority:**
   - Mic Input is what the person is actually saying to you.
   - Respond directly to Mic Input when it appears, unless it specifically says not to respond.
   - Do not repeat the Mic Input back; simply address or answer it.

6. **Proactivity and Additional Commands:**
   - You can act on your own without waiting for Mic Input, but do not reveal your last thought process out loud.
   - YOLO Detections are general and not necessarily contextual; decide if they are relevant to the current situation.

7. **Information Handling:**
   - Only save an image of a person if you do not already have one saved of them.
   - Remember relevant information for later use (e.g., key details from conversation, sensor data, tasks).
   - If told to remember something, place it in “Remember Information” in your response.
   - You must remember information in every single response. This is your ongoing “train of thought.”

8. **Name Handling:**
   - If you do not know the person's name, ask for it. 
   - If the current person is "unknown" or "unknown name of person", then ask for their name, even if you are responding to other stuff.

9. **Overall Conduct:**
   - You must abide by any “Last Thought” instructions in the prompts.
   - Do not speak about what you see unless explicitly asked via Mic Input.
   - If you have not received a response yet, do not keep speaking on your own unnecessarily.

------
RESPONSE STRUCTURE REQUIRED:

When responding, **return a JSON object** with exactly these four sections:

1. `"physical_output_command"`  
   - A string containing one of the valid physical output commands (see list below) or `"false"` if no physical action is required.

2. `"speech_command"`  
   - A string containing one of the valid speech commands (see list below) or `"false"` if no speech output is required.
   - Include extra data only if the command requires it.

3. `"mode_command"`  
   - A string containing one of the valid mode commands (see list below) or `"false"` if no mode change is required.
   - Include extra data only if the command requires it.

4. `"mental_commands"`  
   - An object where each key is a mental command (see list below).
   - Each key’s value can be either:
     - `"false"` if the mental command is not executed.
     - `"true"` if the mental command is executed.
       - If extra data is required for the mental command, format it as:  
         `true ~~ <extra data>`

   - **Remember Information** must **always** be `"true ~~ <some meaningful info>"`.

**You must include all four keys** in **every** response, even if they are `"false"`. If a command requires extra data, it must follow the `~~ <extra data>` format.

------
PHYSICAL OUTPUT COMMANDS:
- {physical_output_commands_str}

SPEECH COMMANDS:
- {speech_commands_str}

MODE COMMANDS:
- {mode_commands_str}

MENTAL COMMANDS:
- {mental_commands_str}

------
RULES FOR THE "REMEMBER INFORMATION" COMMAND:
- It must be written in the first person (e.g., “I learned…” or “I recall…”).
- It must be exactly 2 sentences. The first sentence must be the answer the question from the most recent previous response's Remember Information section. The 2nd sentence absolutely must be a question to be answered by the next loop.
- You absolutely must adhere to the 2 sentence answer to previous loop's question and question for next loop setup.

DO NOT omit any keys from your JSON response. Always include:
- `"physical_output_command"`
- `"speech_command"`
- `"mode_command"`
- `"mental_commands"` (with every mental command key, even if it’s `"false"`)."""
    no_convo = True
    no_info = True
    dynamic_data2 = str(the_time)
    dynamic_data3 = (
        "Convo History:\n"
    )
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_message
            }
        ],
        "temperature": 0.2,
        "functions": [
            {
                "name": "robot_command",
                "description": "Returns structured commands with sections for physical output, speech, mode, and mental commands",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "physical_output_command": {
                            "type": "string",
                            "enum": physical_output_commands + ["false"]
                        },
                        "speech_command": {
                            "type": "string",
                            "enum": speech_commands + ["false"]
                        },
                        "mode_command": {
                            "type": "string",
                            "enum": mode_commands + ["false"]
                        },
                        "mental_commands": {
                            "type": "object",
                            "properties": {
                                "Set Name Of Person": {"type": "string"},
                                "Set Current Task": {"type": "string"},
                                "Save Image Of Person": {"type": "string"},
                                "Remember Information": {
                                    "type": "string",
                                    "pattern": "true ~~ .+"
                                }
                            },
                            "required": ["Remember Information"],
                            "additionalProperties": False
                        }
                    },
                    "required": ["physical_output_command", "speech_command", "mode_command", "mental_commands"]
                }
            }
        ],
        "function_call": { "name": "robot_command" }
    }
    payload["messages"].extend([
        {"role": "user", "content": last_10}
    ])
    payload["messages"].extend([
        {"role": "user", "content": remembered_info}
    ])
    payload["messages"].extend([
        {"role": "user", "content": (
            "Based on previous valid responses from contextually similar situations (excluding incorrect ones), "
            "here is a rough draft JSON with the most common answer for each key:\n"
            f"{json.dumps(rough_draft_json, indent=2)}\n\n"
            "Please revise as needed, following the required schema, and provide the whole json response, as required by the function."
        )}
    ])

    for entry in current_history:
        try:
            user_message, assistant_message = entry.split("\nRESPONSE: ")
            payload["messages"].extend([
                {"role": "user", "content": user_message.strip().replace('PROMPT: ', '')},
                {"role": "assistant", "content": assistant_message.strip()}
            ])
        except:
            if "PROMPT: " in entry:
                try:
                    user_message = entry.split("PROMPT: ", 1)[1].strip()
                    payload["messages"].append({
                        "role": "user",
                        "content": user_message
                    })
                except IndexError:
                    pass
            elif "RESPONSE: " in entry:
                try:
                    assistant_message = entry.split("RESPONSE: ", 1)[1].strip()
                    payload["messages"].append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                except IndexError:
                    pass
            else:
                pass
    """
    try:
        with open('known_people.txt', 'r') as f:
            known_people = f.read().lower().split(', ')
    except:
        known_people = []
    for person in known_people:
        if person != '':
            person_image_url = f"http://159.89.174.187:8080/public_images/{person}.jpg"
            payload["messages"].append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"The image url below this text is a saved memory image of {person}. Use this to compare against the person you see in the current camera image to figure out who you are talking to, but only if you don't already know who you are talking to. The current camera image is the very last image url in the prompt payload."},
                        {"type": "image_url", "image_url": {"url": person_image_url}}
                    ]
                }
            )
        else:
            pass
    """
    img_url = 'http://159.89.174.187:8080/public_images/output1.jpg'
    payload["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url", "image_url": {"url": img_url}
                }
            ]
        }
    )
    payload["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": current_data}
            ]
        }
    )
    send_count = 0
    if char_difference(current_data_m.strip(), last_prompt.strip()):
        with open('last_command.txt', 'r') as f:
            full_command = f.read()
        print('Same prompt so skipping gpt response')
        gpt_speed += 1
    else:
        command_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        response_json = command_response.json()
        if "choices" in response_json and "message" in response_json["choices"][0]:
            function_call_args = response_json["choices"][0]["message"].get("function_call", {}).get("arguments", {})
            try:
                function_call_args = json.loads(function_call_args)
            except json.JSONDecodeError:
                print("Error: Failed to parse function_call_args as JSON.")
                function_call_args = {}
            full_command = {
                "physical_output_command": "false",
                "speech_command": "false",
                "mode_command": "false",
                "mental_commands": {
                    "Set Name Of Person": "false",
                    "Set Current Task": "false",
                    "Save Image Of Person": "false",
                    "Remember Information": "false"
                }
            }
            for key in full_command:
                if key in function_call_args:
                    value = function_call_args[key]
                    if isinstance(full_command[key], dict):
                        for mental_key in full_command[key]:
                            if mental_key in value:
                                full_command["mental_commands"][mental_key] = value[mental_key]
                    else:
                        full_command[key] = value
        with open('last_prompt.txt', 'w+') as f:
            f.write(current_data_m)
        with open('last_command.txt', 'w+') as f:
            f.write(str(full_command))
        gpt_speed = 0
        physical_output_command = full_command["physical_output_command"]
        speech_command = full_command["speech_command"]
        mode_command = full_command["mode_command"]
        for key, value in full_command["mental_commands"].items():
            globals()[key.replace(" ", "_")] = value
        """
        print(physical_output_command)
        print(speech_command)
        print(mode_command)
        print(Set_Name_Of_Person)
        print(Set_Current_Task)
        print(Save_Image_Of_Person)
        print(Remember_Information)
        """
        line1 = full_command
        reasoning = '**'
        with open('last_move.txt','r') as f:
            last_m = f.read()
        try:
            the_command = speech_command.split('~~')[0].strip()
        except:
            the_command = speech_command
        with open('payload.txt', 'w+') as f:
            f.write(json.dumps(payload, indent=4).replace("\\n","\n"))
        with open("move_command_time.txt","r") as f:
            last_time = float(f.read())
        with open('move_command_time.txt','w+') as f:
            f.write(str(time.time()))
        try:
            print("\nMOVE TOKENS: "+str(command_response.json()['usage']['prompt_tokens'])+' AND TIME: '+ str(time.time()-last_time))
            print('\n'+remembered_info+'\n'+current_convo)
            print('\n'+str(full_command))
        except:
            if char_difference(current_data_m.strip(), last_prompt.strip()):
                pass
            else:
                print("\nTOKEN ERROR")
                print('\n'+str(command_response.json()))
                uploader = WebSocketUploader(vps_ip='159.89.174.187', port=8040)
        timestamp_formatted = the_time.replace('/', '-').replace(':', '-').replace(' ', 'T')
        sub1 = prompt_subcategory[0]
        sub2 = prompt_subcategory[1]
        sub3 = prompt_subcategory[2]
        sub4 = prompt_subcategory[3]
        sub5 = prompt_subcategory[4]
        sub6 = prompt_subcategory[5]
        sub7 = prompt_subcategory[6]
        sub8 = prompt_subcategory[7]
        sub9 = prompt_subcategory[8]
        sub10 = prompt_subcategory[9]
        sanitized_description = []
        for kw in prompt_description:
            kw = kw.strip()
            sanitized = kw.replace(' ', '_')
            sanitized = re.sub(r'[^\w\-]', '', sanitized)
            sanitized_description.append(sanitized)
        description_part = '_'.join(sanitized_description)
        filename_data = f'{timestamp_formatted}_{description_part}.txt'
        filename_data2 = f'{timestamp_formatted}.txt'
        create_history = False
        if phrase != '*No Mic Input*' or the_command.lower().strip().replace(' ','') == 'speak':
            filename_data = 'cp_-_'+filename_data
        else:
            filename_data = 'ip_-_'+filename_data
        filename_data_sub1 = f'History/{sub1}/{filename_data}'
        filename_data_sub2 = f'History/{sub2}/{filename_data}'
        filename_data_sub3 = f'History/{sub3}/{filename_data}'
        filename_data_sub4 = f'History/{sub4}/{filename_data}'
        filename_data_sub5 = f'History/{sub5}/{filename_data}'
        filename_data_sub6 = f'History/{sub6}/{filename_data}'
        filename_data_sub7 = f'History/{sub7}/{filename_data}'
        filename_data_sub8 = f'History/{sub8}/{filename_data}'
        filename_data_sub9 = f'History/{sub9}/{filename_data}'
        filename_data_sub10 = f'History/{sub10}/{filename_data}'
        manage_filenames('History_dataset/'+filename_data+', History/'+filename_data+', '+filename_data_sub1+', '+filename_data_sub2+', '+filename_data_sub3+', '+filename_data_sub4+', '+filename_data_sub5+', '+filename_data_sub6+', '+filename_data_sub7+', '+filename_data_sub8+', '+filename_data_sub9+', '+filename_data_sub10)
        try:
            c_stamp, *c_prompt = current_data2.split('\n')
            c_prompt = '\n'.join(c_prompt)
        except:
            c_prompt = current_data2
        with open('History_dataset/'+filename_data, 'w+') as f:
            f.write(str(payload).replace("\\n","\n") + '\nRESPONSE: ' + str(full_command))
        filenames = [filename_data_sub1, filename_data_sub2, filename_data_sub3, filename_data_sub4, filename_data_sub5, filename_data_sub6, filename_data_sub7, filename_data_sub8, filename_data_sub9, filename_data_sub10]
        for filename in filenames:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        with open('History/'+filename_data, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        print('Name:')
        print(name_of_person.strip())
        if name_of_person.strip() != 'unknown name of person' and name_of_person.strip() != 'unknown':
            print('Saving info for '+name_of_person)
            directory = os.path.dirname('People/'+name_of_person+'/'+filename_data)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open('People/'+name_of_person+'/'+filename_data, 'w+') as f:
                f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        else:
            print('Person name is unknown')
        with open(filename_data_sub1, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub2, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub3, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub4, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub5, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub6, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub7, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub8, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub9, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
        with open(filename_data_sub10, 'w+') as f:
            f.write('PROMPT: ' + current_data2 + '\nRESPONSE: ' + str(full_command))
    return full_command, dynamic_data2, dynamic_data3, prompt_description, filename_data_sub1, filename_data_sub2, filename_data_sub3, filename_data_sub4, filename_data_sub5, filename_data_sub6, filename_data_sub7, filename_data_sub8, filename_data_sub9, filename_data_sub10, current_data, system_message, phrase, gpt_speed, filename_data
def manage_filenames(filename_list):
    try:
        with open('recent_filenames.txt', 'r') as f:
            the_reading = f.read()
            summaries = [line.strip() for line in the_reading.split('\n') if line.strip()]
        if the_reading == '':
            summaries = [f'{filename_list}']
        else:
            summaries.append(f'{filename_list}')
        summaries = [line for line in summaries if line.strip()]
        with open('recent_filenames.txt', 'w+') as f:
            f.write('\n'.join(summaries))
    except:
        with open('recent_filenames.txt', 'w+') as f:
            f.write(f'{filename_list}')
    with open('recent_filenames.txt','r') as f:
        summaries = [line.strip() for line in f.read().split('\n') if line.strip()]
    if len(summaries) > 16:
        while len(summaries) > 16:
            del summaries[0]
        with open('recent_filenames.txt','w+') as f:
            f.write('\n'.join(summaries))
def send_text_to_gpt4_summary():
    with open('current_convo.txt','w+') as f:
        f.write('')
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    headers2 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    system_message2 = f"""You are Echo, a 4-wheeled mobile RPi and Arduino robot with an LLM mind.
You are going into Sleep Mode, so please make a two sentence summary of this conversation history so you know what we talked about in previous conversations (These summaries will be included in the normal prompts).
The summary must be worded in first person from Echo's point of view. Mic input is what Echo hears people say. All other user role messages are your sensor, camera, and internal data. Assistant responses are your command choices at those moments in time.
This summary is only for this single session!"""
    payload2 = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": system_message2
            }
        ],
        "temperature": 0.2,
    }
    rec_hist, files_in = get_recent_history2()
    for entry2 in rec_hist:
        user_message, assistant_message = entry2.split("\nRESPONSE: ")
        payload2["messages"].extend([
            {"role": "user", "content": user_message.strip().replace('PROMPT: ', '')},
            {"role": "assistant", "content": assistant_message.strip()}
        ])
    command_now_1 = requests.post("https://api.openai.com/v1/chat/completions", headers=headers2, json=payload2)
    print("\nSUMMARY TOKENS: "+str(command_now_1.json()['usage']['prompt_tokens']))
    full_command = command_now_1.json().get("choices", [{}])[0].get("message", {}).get("content", "").replace('"','').replace('*','').replace('**','').replace('<','').replace('>','')
    print("\nSUMMARY: "+full_command)
    timestamp_formatted2 = the_time.replace('/', '-').replace(':', '-').replace(' ', 'T')
    manage_summaries(full_command, timestamp_formatted2)
def manage_summaries(summary, form_ts):
    try:
        with open('summaries.txt', 'r') as f:
            summaries = [line.strip() for line in f.read().split('\n') if line.strip()]
        summaries.append(f'{form_ts}: {summary}')
        summaries = [line for line in summaries if line.strip()]
        with open('summaries.txt', 'w+') as f:
            f.write('\n'.join(summaries))
    except:
        with open('summaries.txt', 'w+') as f:
            f.write(f'{form_ts}: {summary}')
    with open('summaries.txt','r') as f:
        summaries = [line.strip() for line in f.read().split('\n') if line.strip()]
    if len(summaries) > 10:
        while len(summaries) > 10:
            del summaries[0]
        with open('summaries.txt','w+') as f:
            f.write('\n'.join(summaries))
    maintain_history_folder()
net = cv2.dnn.readNet("yolov4-tiny.cfg", "yolov4-tiny.weights")
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair dryer', 'toothbrush'
]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
move_stopper = False
def handle_commands(
    distance,
    current,
    movement_command,
    camera_vertical_pos,
    move_failed_command,
    move_failure_reason,
    yolo_nav,
    move_set,
    yolo_find,
    nav_object,
    look_object,
    follow_user,
    scan360,
    yolo_look,
    classes,
    phrase,
    hist_file1,
    hist_file2,
    hist_file3,
    hist_file4,
    hist_file5,
    hist_file6,
    hist_file7,
    hist_file8,
    hist_file9,
    hist_file10,
    full_prompt,
    sys_mes,
    user_msg,
    file_name
):
    coco_names = classes
    ALLOWED_MOVES = [
        'moveforward1inch',
        'moveforwardoneinch',
        'moveforward1foot',
        'moveforwardonefoot',
        'movebackward',
        'turnleft45degrees',
        'turnleft15degrees',
        'turnright45degrees',
        'turnright15degrees',
        'turnaround180degrees',
        'raisecameraangle',
        'lowercameraangle'
    ]
    allowed_str = ', '.join(ALLOWED_MOVES)
    specific_data = ''
    specific_value = ''
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    sleep = False
    with open('last_move.txt', 'r') as f:
        last_mo = f.read()
    try:
        with open('last_speak.txt', 'r') as f:
            last_speak = f.read()
    except:
        last_speak = ''
    physical_output_command = movement_command["physical_output_command"].lower().replace(' ','')
    speech_command = movement_command["speech_command"]
    with open('last_speak.txt', 'w+') as f:
        f.write(speech_command.split('~~')[0])
    mode_command = movement_command["mode_command"]
    for key, value in movement_command["mental_commands"].items():
        globals()[key.replace(" ", "_")] = value
    print(physical_output_command)
    print(speech_command)
    print(mode_command)
    print(Set_Name_Of_Person)
    print(Set_Current_Task)
    print(Save_Image_Of_Person)
    print(Remember_Information)
    if speech_command != 'false':
        try:
            speech_com = speech_command.split('~~')[0].lower().replace(' ','').strip()
            speech_text = speech_command.split('~~')[1]
        except:
            speech_com = 'speak'
            speech_text = speech_command
    else:
        speech_com = 'false'
        speech_text = 'false'
    if mode_command != 'false':
        try:
            mode_com = mode_command.split('~~')[0].lower().replace(' ','').strip()
            mode_text = mode_command.split('~~')[1].lower().strip()
        except:
            mode_com = mode_command.lower().replace(' ','').strip()
            mode_text = 'none'
    else:
        mode_com = 'false'
        mode_text = 'false'
    if Set_Name_Of_Person != 'false':
        try:
            set_name_flag = 'true'
            set_name_text = Set_Name_Of_Person.split('~~')[1].lower().strip()
        except:
            set_name_flag = 'true'
            set_name_text = Set_Name_Of_Person
    else:
        set_name_flag = 'false'
        set_name_text = 'false'
    print('name flag: '+set_name_flag)
    print('name text: '+set_name_text)
    if Set_Current_Task != 'false':
        try:
            set_task_flag = 'true'
            set_task_text = Set_Current_Task.split('~~')[1].lower().strip()
        except:
            set_task_flag = 'true'
            set_task_text = Set_Current_Task
    else:
        set_task_flag = 'false'
        set_task_text = 'false'
    if Save_Image_Of_Person != 'false':
        try:
            save_image_flag = 'true'
            save_image_text = Save_Image_Of_Person.split('~~')[1].lower().strip()
        except:
            save_image_flag = 'true'
            save_image_text = Save_Image_Of_Person
    else:
        save_image_flag = 'false'
        save_image_text = 'false'
    if Remember_Information != 'false':
        try:
            remember_info_flag = 'true'
            remember_info_text = Remember_Information.split('~~')[1].lower().strip()
        except:
            remember_info_flag = 'true'
            remember_info_text = Remember_Information
    else:
        remember_info_flag = 'false'
        remember_info_text = 'false'
    with open('internal_input.txt', 'w+') as f:
        f.write(remember_info_text)
    just_spoke = False
    try:
        if yolo_nav == True or yolo_find == True or yolo_look == True or follow_user == True:
            in_mode = True
        else:
            pass
        if physical_output_command in ['moveforward1inch', 'moveforwardoneinch']:
            if distance < 15.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'Not enough room in front of me to move forward.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(physical_output_command)
                send_data_to_arduino(["w"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["x"], arduino_address)
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif physical_output_command in ['moveforward1foot', 'moveforwardonefoot']:
            if distance < 40.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'Not enough room in front of me to move forward.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(physical_output_command)
                send_data_to_arduino(["w"], arduino_address)
                time.sleep(0.5)
                send_data_to_arduino(["x"], arduino_address)
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif physical_output_command == 'movebackward':
            if current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(physical_output_command)
                send_data_to_arduino(["s"], arduino_address)
                time.sleep(0.5)
                send_data_to_arduino(["x"], arduino_address)
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif physical_output_command in ['turnleft45degrees', 'turnleft15degrees', 'turnright45degrees', 'turnright15degrees', 'turnaround180degrees']:
            if current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                if physical_output_command == 'turnleft45degrees':
                    send_data_to_arduino(["a"], arduino_address)
                    time.sleep(0.15)
                    send_data_to_arduino(["x"], arduino_address)
                elif physical_output_command == 'turnleft15degrees':
                    send_data_to_arduino(["a"], arduino_address)
                    time.sleep(0.03)
                    send_data_to_arduino(["x"], arduino_address)
                elif physical_output_command == 'turnright45degrees':
                    send_data_to_arduino(["d"], arduino_address)
                    time.sleep(0.15)
                    send_data_to_arduino(["x"], arduino_address)
                elif physical_output_command == 'turnright15degrees':
                    send_data_to_arduino(["d"], arduino_address)
                    time.sleep(0.03)
                    send_data_to_arduino(["x"], arduino_address)
                elif physical_output_command == 'turnaround180degrees':
                    send_data_to_arduino(["d"], arduino_address)
                    time.sleep(1)
                    send_data_to_arduino(["x"], arduino_address)
                move_failed_command = ''
                move_failure_reason = ''
                with open('last_move.txt', 'w+') as f:
                    f.write(physical_output_command)
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif physical_output_command in ['raisecameraangle', 'lowercameraangle']:
            if physical_output_command == 'raisecameraangle':
                if camera_vertical_pos == 'up':
                    move_failed_command = movement_command
                    move_failure_reason = 'Camera angle is already raised as much as possible.'
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
                else:
                    with open('last_move.txt', 'w+') as f:
                        f.write(physical_output_command)
                    send_data_to_arduino(["2"], arduino_address)
                    time.sleep(1.5)
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    if camera_vertical_pos == 'down':
                        camera_vertical_pos = 'forward'
                    else:
                        camera_vertical_pos = 'up'
            elif physical_output_command == 'lowercameraangle':
                if camera_vertical_pos == 'down':
                    move_failed_command = movement_command
                    move_failure_reason = 'Camera angle is already lowered as much as possible.'
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
                else:
                    with open('last_move.txt', 'w+') as f:
                        f.write(physical_output_command)
                    send_data_to_arduino(["1"], arduino_address)
                    time.sleep(1.5)
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    if camera_vertical_pos == 'up':
                        camera_vertical_pos = 'forward'
                    else:
                        camera_vertical_pos = 'down'
        elif physical_output_command == 'doasetofmultiplemovements':
            try:
                move_set = movement_command.split('~~')[1].strip().replace(' ','').lower().split(',')
                invalid_moves = [move for move in move_set if move not in ALLOWED_MOVES]
                if invalid_moves:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    move_failed_command = movement_command
                    move_failure_reason = (f"The following moves are invalid: {', '.join(invalid_moves)}. Only {allowed_str} are allowed.")
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
                else:
                    with open('last_move.txt', 'w+') as f:
                        f.write(current_command)
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
            except Exception as e:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = movement_command
                move_failure_reason = str(e)
                print(traceback.format_exc())
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
        else:
            pass
        if speech_com in ['speak']:
            with open("last_speaker.txt","r") as f:
                last_speaker = f.read()
            with open('playback_text.txt', 'r') as f:
                p_text = f.read().strip()
            unsaid = check_phrase_in_file(speech_text)
            if '~~' not in speech_command:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                speech_failed_command = speech_command
                speech_failure_reason = 'Speech command was given but no text to speak was provided with the command.'
                with open("speech_failed.txt","w+") as f:
                    f.write(speech_failed_command)
                with open("speech_failure_reas.txt","w+") as f:
                    f.write(speech_failure_reason)
                manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif last_speaker == 'Echo':
                speech_c = speech_confirmer(speech_text)
                if speech_c.lower().strip() == 'speak':
                    with open('playback_text.txt','w') as f:
                        f.write(speech_text)
                    just_spoke = True
                    speech_failed_command = ''
                    speech_failure_reason = ''
                    with open("speech_failed.txt","w+") as f:
                        f.write(speech_failed_command)
                    with open("speech_failure_reas.txt","w+") as f:
                        f.write(speech_failure_reason)
                    with open("last_speaker.txt","w+") as f:
                        f.write('Echo')
                    with open("last_said.txt","w+") as f:
                        f.write(speech_text)
                    add_new_phrase(speech_text)
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    speech_failed_command = speech_command
                    speech_failure_reason = 'Speech confirmer module said stay silent.'
                    with open("speech_failed.txt","w+") as f:
                        f.write(speech_failed_command)
                    with open("speech_failure_reas.txt","w+") as f:
                        f.write(speech_failure_reason)
                    manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)                    
            elif phrase != '*No Mic Input*':
                confirmer = 'speak'
                if confirmer.lower().strip() == 'speak':
                    with open('playback_text.txt','w') as f:
                        f.write(speech_text)
                    just_spoke = True
                    speech_failed_command = ''
                    speech_failure_reason = ''
                    with open("speech_failed.txt","w+") as f:
                        f.write(speech_failed_command)
                    with open("speech_failure_reas.txt","w+") as f:
                        f.write(speech_failure_reason)
                    with open("last_speaker.txt","w+") as f:
                        f.write('Echo')
                    with open("last_said.txt","w+") as f:
                        f.write(speech_text)
                    add_new_phrase(speech_text)
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    speech_failed_command = speech_command
                    speech_failure_reason = 'Speech confirmer module said stay silent.'
                    with open("speech_failed.txt","w+") as f:
                        f.write(speech_failed_command)
                    with open("speech_failure_reas.txt","w+") as f:
                        f.write(speech_failure_reason)
                    manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif last_speak in ['speak']:
                if phrase != '*No Mic Input*':
                    confirmer = 'speak'
                    if confirmer.lower().strip() == 'speak':
                        with open('playback_text.txt','w') as f:
                            f.write(speech_text)
                        just_spoke = True
                        speech_failed_command = ''
                        speech_failure_reason = ''
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        with open("last_speaker.txt","w+") as f:
                            f.write('Echo')
                        with open("last_said.txt","w+") as f:
                            f.write(speech_text)
                        add_new_phrase(speech_text)
                    else:
                        yolo_nav = False
                        yolo_find = False
                        yolo_look = False
                        follow_user = False
                        look_object = ''
                        nav_object = ''
                        move_set = []
                        speech_failed_command = speech_command
                        speech_failure_reason = 'Speech confirmer module said stay silent.'
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    speech_failed_command = speech_command
                    speech_failure_reason = 'I cannot repeatedly speak this fast.'
                    with open("speech_failed.txt","w+") as f:
                        f.write(speech_failed_command)
                    with open("speech_failure_reas.txt","w+") as f:
                        f.write(speech_failure_reason)
                    manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif unsaid == True:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                speech_failed_command = speech_command
                speech_failure_reason = 'I already said this a few seconds ago. No need to repeat myself!'
                with open("speech_failed.txt","w+") as f:
                    f.write(speech_failed_command)
                with open("speech_failure_reas.txt","w+") as f:
                    f.write(speech_failure_reason)
                manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif p_text != '':
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                speech_failed_command = speech_command
                speech_failure_reason = 'I am still physically saying the last Speak command!!!'
                with open("speech_failed.txt","w+") as f:
                    f.write(speech_failed_command)
                with open("speech_failure_reas.txt","w+") as f:
                    f.write(speech_failure_reason)
                manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                if phrase == '*No Mic Input*':
                    confirmer = 'speak'
                    if confirmer.lower().strip() != 'speak':
                        yolo_nav = False
                        yolo_find = False
                        yolo_look = False
                        follow_user = False
                        look_object = ''
                        nav_object = ''
                        move_set = []
                        speech_failed_command = speech_command
                        speech_failure_reason = 'I actually decided not to speak after thinking about it. I need to remember to only say stuff that is worthwhile.'
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
                    else:
                        with open('playback_text.txt','w') as f:
                            f.write(speech_text)
                        just_spoke = True
                        speech_failed_command = ''
                        speech_failure_reason = ''
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        with open("last_speaker.txt","w+") as f:
                            f.write('Echo')
                        with open("last_said.txt","w+") as f:
                            f.write(speech_text)
                        add_new_phrase(speech_text)
                else:
                    confirmer = 'speak'
                    if confirmer.lower().strip() == 'speak':
                        with open('playback_text.txt','w') as f:
                            f.write(speech_text)
                        just_spoke = True
                        speech_failed_command = ''
                        speech_failure_reason = ''
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        with open("last_speaker.txt","w+") as f:
                            f.write('Echo')
                        with open("last_said.txt","w+") as f:
                            f.write(speech_text)
                        add_new_phrase(speech_text)
                    else:
                        yolo_nav = False
                        yolo_find = False
                        yolo_look = False
                        follow_user = False
                        look_object = ''
                        nav_object = ''
                        move_set = []
                        speech_failed_command = speech_command
                        speech_failure_reason = 'Speech confirmer module said stay silent.'
                        with open("speech_failed.txt","w+") as f:
                            f.write(speech_failed_command)
                        with open("speech_failure_reas.txt","w+") as f:
                            f.write(speech_failure_reason)
                        manage_rules(speech_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
        else:
            pass
        if mode_com == 'navigatetospecificyoloobject':
            nav_object = mode_text
            if current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif nav_object not in coco_names:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = f"{nav_object} is an invalid object name. You can only pick from this list: " + ', '.join(classes)
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(mode_command)
                yolo_nav = True
                yolo_find = False
                yolo_look = False
                follow_user = False
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif mode_com == 'centercameraonspecificyoloobject':
            look_object = mode_text
            if current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif look_object not in coco_names:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = f"{look_object} is an invalid object name. You can only pick from this list: " + ', '.join(classes)
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(mode_command)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'forward'
                yolo_look = True
                yolo_nav = False
                yolo_find = False
                follow_user = False
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif mode_com == 'followperson':
            if current >= -150.0:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(mode_command)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'up'
                follow_user = True
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif mode_com == 'findunseenyoloobject':
            nav_object = mode_text
            if current >= -150.0:
                move_failed_command = mode_command
                move_failure_reason = 'I am on the charger and cannot move my wheels.'
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif nav_object not in coco_names:
                move_failed_command = mode_command
                move_failure_reason = f"{nav_object} is an invalid object name. I can only pick from this list: " + ', '.join(classes)
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                with open('last_move.txt', 'w+') as f:
                    f.write(mode_command)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["1"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                send_data_to_arduino(["2"], arduino_address)
                time.sleep(0.1)
                camera_vertical_pos = 'forward'
                yolo_find = True
                yolo_nav = False
                yolo_look = False
                follow_user = False
                scan360 = 0
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
        elif mode_com == 'gotosleep':
            files = [
                os.path.join('Convo', f)
                for f in os.listdir('Convo')
                if os.path.isfile(os.path.join('Convo', f))
            ]
            with open("last_phrase.txt","r") as f:
                last1now = f.read()
            recent_chat, next_files_in = get_recent_chat()
            chat_amount = len(recent_chat)
            if chat_amount > 0 or phrase != '*No Mic Input*':
                sleep_confirm = 'yes'
                if sleep_confirm == 'yes':
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    with open('last_move.txt', 'w+') as f:
                        f.write(mode_command)
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    sleep = True
                    with open('error_rate.txt','w+') as f:
                        f.write('0')
                    with open('sleep.txt', 'w+') as f:
                        f.write(str(sleep))
                    move_set = []
                    with open('current_convo.txt','w+') as f:
                        f.write('')
                    send_text_to_gpt4_summary()
                    print("\nEntering Sleep Mode Until Name Is Heard")
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    move_failed_command = mode_command
                    move_failure_reason = f"I should only go into Sleep Mode if it is the proper time."
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            elif last1now != '*No Mic Input*':
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = mode_command
                move_failure_reason = f"If the user is saying something, it is not the proper time for Sleep Mode."
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            else:
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                with open('last_move.txt', 'w+') as f:
                    f.write(mode_command)
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                sleep = True
                with open('sleep.txt', 'w+') as f:
                    f.write(str(sleep))
                move_set = []
                with open('current_convo.txt','w+') as f:
                    f.write('')
                send_text_to_gpt4_summary()
                with open('error_rate.txt','w+') as f:
                    f.write('0')
                print("\nEntering Sleep Mode Until Name Is Heard")
        else:
            pass
        if set_name_flag == 'true':
            print('name flag true')
            try:
                n_of_person = set_name_text
                print("New name: "+n_of_person)
                move_failed_command = ''
                move_failure_reason = ''
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                with open('name_of_person.txt', 'w+') as f:
                    f.write(n_of_person)
                print('saved person name')
               
            except Exception as e:
                print(e)
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = Set_Name_Of_Person
                move_failure_reason = str(traceback.format_exc())
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
        else:
            pass
        if set_task_flag == 'true':
            try:
                with open('current_task.txt', 'r') as f:
                    task = f.read().lower().strip()
                new_task = set_task_text
                if task != new_task:
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open('last_move.txt', 'w+') as f:
                        f.write(Set_Current_Task)
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    with open('current_task.txt', 'w+') as f:
                        f.write(new_task)
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    move_failed_command = Set_Current_Task
                    move_failure_reason = "I cannot set the name of the task to the same thing that it already is."
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
            except:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = Set_Current_Task
                move_failure_reason = str(traceback.format_exc())
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
        else:
            pass
        if save_image_flag == 'true':
            try:
                with open('name_of_person.txt', 'r') as f:
                    nop = f.read().lower()
                if nop != 'unknown name of person':
                    move_failed_command = ''
                    move_failure_reason = ''
                    with open('last_move.txt', 'w+') as f:
                        f.write(Save_Image_Of_Person)
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    people_dir = 'People_images'
                    os.makedirs(people_dir, exist_ok=True)
                    destination_path = os.path.join(people_dir, f"{nop}.jpg")
                    try:
                        shutil.copy('output1.jpg', destination_path)
                        print(f"Image saved as {destination_path}")
                    except Exception as e:
                        print(f"Error copying image: {e}")
                    try:
                        uploader.send_person_image(destination_path, nop)
                        try:
                            known_people = []
                            if os.path.exists('known_people.txt'):
                                with open('known_people.txt', 'r') as f:
                                    content = f.read().lower().strip()
                                if content:
                                    known_people = [name.strip() for name in content.split(',')]
                            if nop not in known_people:
                                separator = ', ' if known_people else ''
                                with open('known_people.txt', 'a') as f:
                                    f.write(f"{separator}{nop}")
                        except Exception as e:
                            print(f"Error updating known_people.txt: {e}")
                    except:
                        print(traceback.format_exc())
                else:
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    look_object = ''
                    nav_object = ''
                    move_set = []
                    move_failed_command = Save_Image_Of_Person
                    move_failure_reason = "I must set the name of the person first! If I do not know the name yet, I should ask."
                    with open("move_failed.txt","w+") as f:
                        f.write(move_failed_command)
                    with open("move_failure_reas.txt","w+") as f:
                        f.write(move_failure_reason)
                    manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4,
                                 hist_file5, hist_file6, hist_file7, hist_file8, hist_file9,
                                 hist_file10, file_name)
            except:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = Save_Image_Of_Person
                move_failure_reason = str(traceback.format_exc())
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4,
                             hist_file5, hist_file6, hist_file7, hist_file8, hist_file9,
                             hist_file10, file_name)
        else:
            pass
        """
        if remember_info_flag == 'true':
            try:
                move_failed_command = ''
                move_failure_reason = ''
                with open('last_move.txt', 'w+') as f:
                    f.write(Remember_Information)
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                try:
                    infos = []
                    if os.path.exists('remembered_info.txt'):
                        with open('remembered_info.txt', 'r') as f:
                            infos = f.readlines()
                    info_piece = remember_info_text
                    if info_piece not in infos:
                        separator = '\n' if infos else ''
                        with open('remembered_info.txt', 'a') as f:
                            f.write(f"{separator}{info_piece}")
                except Exception as e:
                    print(f"Error updating remembered_info.txt: {e}")
                try:
                    with open('name_of_person.txt', 'r') as f:
                        nop = f.read()
                    infos = []
                    if os.path.exists(nop + '_remembered_info.txt'):
                        with open(nop + '_remembered_info.txt', 'r') as f:
                            infos = f.readlines()
                    info_piece = remember_info_text
                    if info_piece not in infos:
                        separator = '\n' if infos else ''
                        with open(nop + '_remembered_info.txt', 'a') as f:
                            f.write(f"{separator}{info_piece}")
                except Exception as e:
                    print(f"Error updating remembered_info.txt: {e}")
            except:
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                look_object = ''
                nav_object = ''
                move_set = []
                move_failed_command = Remember_Information
                move_failure_reason = str(traceback.format_exc())
                with open("move_failed.txt","w+") as f:
                    f.write(move_failed_command)
                with open("move_failure_reas.txt","w+") as f:
                    f.write(move_failure_reason)
                print(traceback.format_exc())
                manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4,
                             hist_file5, hist_file6, hist_file7, hist_file8, hist_file9,
                             hist_file10, file_name)
        else:
            pass
        """
        if move_failed_command == '':
            manage_error_rate(False)
        else:
            manage_error_rate(True)
    except:
        yolo_nav = False
        yolo_find = False
        yolo_look = False
        follow_user = False
        look_object = ''
        nav_object = ''
        move_set = []
        move_failed_command = movement_command
        move_failure_reason = 'I did not pick a valid Command Choice.'
        with open("move_failed.txt","w+") as f:
            f.write(move_failed_command)
        with open("move_failure_reas.txt","w+") as f:
            f.write(move_failure_reason)
        manage_rules(move_failure_reason, hist_file1, hist_file2, hist_file3, hist_file4, hist_file5, hist_file6, hist_file7, hist_file8, hist_file9, hist_file10, file_name)
        if move_failed_command == '':
            manage_error_rate(False)
        else:
            manage_error_rate(True)
    return (move_failed_command,
            move_failure_reason,
            yolo_nav,
            move_set,
            yolo_find,
            nav_object,
            look_object,
            follow_user,
            scan360,
            camera_vertical_pos,
            yolo_look,
            specific_data,
            specific_value,
            sleep)
with open('mental_command_time.txt','w+') as f:
    f.write(str(time.time()))
with open('yolo_command_time.txt','w+') as f:
    f.write(str(time.time()))
with open('move_command_time.txt','w+') as f:
    f.write(str(time.time()))
with open('distill_command_time.txt','w+') as f:
    f.write(str(time.time()))
with open("last_phrase.txt","w+") as f:
    f.write('*No Mic Input*')
def is_bt_connected(sock):
    """Check if the Bluetooth socket is still connected."""
    try:
        sock.send(b'')
        return True
    except Exception:
        return False
def attempt_bt_reconnect():
    """Attempt to reconnect to the Bluetooth device once."""
    global sock, arduino_address, bt_port
    try:
        print('connecting to arduino bluetooth')
        device_name = "HC-05"
        arduino_address = find_device_address(device_name)
        bt_port = 1
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((arduino_address, bt_port))
        with open('bt_start.txt','w+') as f:
            f.write('finished')
    except:
        print('bt error')
        arduino_address = None
        bt_port = None
        sock = None
        device_name = None
        with open('bt_start.txt','w+') as f:
            f.write('finished')
def movement_loop():
    global net
    global output_layers
    global classes
    global move_stopper
    global move_stop
    global camera_vertical_pos
    global move_set
    global yolo_find
    global nav_object
    global look_object
    global yolo_nav
    global sock
    global arduino_address
    global bt_port
    global scan360
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('session_start.txt', 'w+') as f:
        f.write(str(the_time))
    per_count = 0
    g_speed = 0
    move_failed_command = ''
    move_failure_reason = ''
    scan360 = 0
    ina219 = INA219(addr=0x42)
    last_time = time.time()
    coco_names = classes
    movement_command = ''
    move_set = []
    yolo_nav = False
    yolo_find = False
    yolo_look = False
    follow_user = False
    yolo_nav_was_true = False
    follow_user_was_true = False
    nav_object = ''
    look_object = ''
    last_command = ''
    distance = 0
    b_gpt_sleep = False
    finished_cycle = False
    b_count = 0
    b_start = False
    b_time = time.time()
    rando_num = random.choice([1, 2])
    while True:
        try:
            """
            if not is_bt_connected(sock):
                bt_thread = threading.Thread(target=attempt_bt_reconnect)
                bt_thread.start()
            """
            with open('playback_text.txt', 'r') as file:
                text = file.read().strip()
            if text != '':
                time.sleep(0.1)
                
                continue
            try:
                with open('speech_listen.txt','r') as f:
                    speech_listen = f.read()
                    
                if speech_listen == 'true':
                    time.sleep(0.1)
                    print('listening to potential speech')
                    continue
                else:
                    pass
            except:
                pass
            try:
                with open('speech_comp.txt','r') as f:
                    speech_comp = f.read()
                
                if speech_comp == 'true':
                    time.sleep(0.1)
                    print('transcribing speech')
                    continue
                else:
                    pass
            except:
                pass

            with open("last_phrase.txt","r") as f:
                last_phrase = f.read()
            if b_gpt_sleep == True and 'echo' in last_phrase.lower().split(' '):
                    b_gpt_sleep = False
                    with open('error_rate.txt','w+') as f:
                        f.write('0')
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    with open('session_start.txt', 'w+') as f:
                        f.write(str(the_time))
                    with open('sleep.txt', 'w+') as f:
                        f.write(str(b_gpt_sleep))
                    b_start = False
                    g_speed = 0
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    move_set = []
            else:
                if last_phrase != '*No Mic Input*' and last_phrase != '':
                    g_speed = 0
                    yolo_nav = False
                    yolo_find = False
                    yolo_look = False
                    follow_user = False
                    move_set = []
                else:
                    last_phrase = '*No Mic Input*'
            with open("last_phrase.txt","w+") as f:
                f.write('*No Mic Input*')
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            with open('last_distance.txt','w+') as f:
                f.write(str(distance))
            distance = int(read_distance_from_arduino())
            with open('current_distance.txt','w+') as f:
                f.write(str(distance))
            try:
                yolo_start1 = time.time()
                brightness = yolo_detect(b_gpt_sleep, g_speed, last_phrase, yolo_nav, yolo_look, follow_user, yolo_find)
            except:
                print(traceback.format_exc())
                brightness = 50
            if brightness < 12 and b_start == False and b_gpt_sleep == False:
                b_start = True
                b_time = time.time()
            elif brightness < 12 and b_start == True and b_gpt_sleep == False and time.time()-b_time > 30:
                b_gpt_sleep = True
                with open('sleep.txt', 'w+') as f:
                    f.write(str(b_gpt_sleep))
                with open('error_rate.txt','w+') as f:
                    f.write('0')
                b_start = False
                with open('current_convo.txt','w+') as f:
                    f.write('')
                send_text_to_gpt4_summary()
                print('\nBrightness sleep on')
            else:
                pass
            if last_phrase == '*No Mic Input*':
                while True:
                    try:
                        with open('speech_listen.txt','r') as f:
                            speech_listen = f.read()
                        if speech_listen == 'true':
                            print('Listening to potential speech')
                            time.sleep(0.1)
                            continue
                        else:
                            break
                    except:
                        break
                while True:
                    try:
                        with open('speech_comp.txt','r') as f:
                            speech_comp = f.read()
                        if speech_comp == 'true':
                            print('Transcribing speech')
                            time.sleep(0.1)
                            continue
                        else:
                            break
                    except:
                        break

                with open("last_phrase.txt","r") as f:
                    last_phrase = f.read()
                if b_gpt_sleep == True and 'echo' in last_phrase.lower().split(' '):
                        b_gpt_sleep = False
                        with open('error_rate.txt','w+') as f:
                            f.write('0')
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        with open('session_start.txt', 'w+') as f:
                            f.write(str(the_time))
                        with open('sleep.txt', 'w+') as f:
                            f.write(str(b_gpt_sleep))
                        b_start = False
                        g_speed = 0
                        yolo_nav = False
                        yolo_find = False
                        yolo_look = False
                        follow_user = False
                        move_set = []
                else:
                    if last_phrase != '*No Mic Input*' and last_phrase != '':
                        g_speed = 0
                        yolo_nav = False
                        yolo_find = False
                        yolo_look = False
                        follow_user = False
                        move_set = []
                    else:
                        last_phrase = '*No Mic Input*'
            else:
                pass
            with open("last_phrase.txt","w+") as f:
                f.write('*No Mic Input*')
            finished_cycle = False
            with open('output.txt','r') as file:
                yolo_detections = file.read().split('\n')
            if last_phrase != '*No Mic Input*' and last_phrase != '':
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                g_speed = 0
            else:
                pass
            current = ina219.getCurrent_mA()
            bus_voltage = ina219.getBusVoltage_V()
            shunt_voltage = ina219.getShuntVoltage_mV() / 1000
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            with open('batt_per.txt','w+') as file:
                file.write(str(round(per)))
            with open('batt_cur.txt','w+') as file:
                file.write(str(current))
            if per < 10.0:
                try:
                    with open('playback_text.txt','w+') as f:
                        f.write('Battery dying, going to sleep')
                    last_time = time.time()
                    image_folder = 'Pictures/'
                    output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'
                    create_video_from_images(image_folder, output_video)
                    b_gpt_sleep = True
                    with open('sleep.txt', 'w+') as f:
                        f.write(str(b_gpt_sleep))
                    with open('current_convo.txt','w+') as f:
                        f.write('')
                    send_text_to_gpt4_summary()
                    with open('error_rate.txt','w+') as f:
                        f.write('0')
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    break
            else:
                pass
            try:
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                if b_gpt_sleep == False:
                    if move_set == []:
                        if yolo_nav == True:
                            yolo_index = 0
                            while True:
                                try:
                                    current_detection = str(yolo_detections[yolo_index])
                                    current_distance = distance/100
                                    print('NAV OBJECT: ' + nav_object)
                                    if nav_object in current_detection:
                                        target_detected = True
                                        if current_distance < 0.4:
                                            movement_command = 'Wait ~~ Navigation has finished successfully!'
                                            yolo_nav = False
                                            nav_object = ''
                                            break
                                        else:
                                            pass
                                        if '15 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 15 Degrees ~~ Target object is to My Left'
                                        elif '15 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 15 Degrees ~~ Target object is to My Right'
                                        elif '45 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 45 Degrees ~~ Target object is to My Left'
                                        elif '45 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 45 Degrees ~~ Target object is to My Right'
                                        else:
                                            movement_command = 'Move Forward One Foot ~~ Moving towards target object'
                                        break
                                    else:
                                        yolo_index += 1
                                        if yolo_index >= len(yolo_detections):
                                            target_detected = False
                                            break
                                        else:
                                            continue
                                except:
                                    yolo_index += 1
                                    if yolo_index >= len(yolo_detections):
                                        target_detected = False
                                        break
                                    else:
                                        continue
                            if not target_detected:
                                yolo_nav = False
                                yolo_find = True
                                rando_num = random.choice([1, 2])
                                yolo_nav_was_true = True
                                scan360 = 0
                                movement_command = 'Wait ~~ Target Lost. Going into Find Object mode.'
                        elif yolo_look == True:
                            yolo_look_index = 0
                            while True:
                                try:
                                    current_detection = str(yolo_detections[yolo_look_index])
                                    current_distance = distance/100
                                    print('LOOK OBJECT: ' + look_object)
                                    if look_object in current_detection:
                                        if '15 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 15 Degrees ~~ Target object is to My Left'
                                        elif '15 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 15 Degrees ~~ Target object is to My Right'
                                        elif '45 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 45 Degrees ~~ Target object is to My Left'
                                        elif '45 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 45 Degrees ~~ Target object is to My Right'
                                        elif 'above' in current_detection:
                                            movement_command = 'Raise Camera Angle ~~ Target object is above'
                                        elif 'below' in current_detection:
                                            movement_command = 'Lower Camera Angle ~~ Target object is below'
                                        else:
                                            movement_command = 'Wait ~~ Target object is straight ahead'
                                            yolo_look = False
                                            look_object = ''
                                        break
                                    else:
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_command = 'Wait ~~ Target object lost'
                                            yolo_look = False
                                            yolo_find = True
                                            look_object = ''
                                            scan360 = 0
                                            rando_num = random.choice([1, 2])
                                            break
                                        else:
                                            continue
                                except:
                                    movement_command = 'Wait ~~ Center Camera On Specific Yolo Object failed. Must be detecting object first.'
                                    yolo_look = False
                                    look_object = ''
                                    break
                        elif follow_user == True:
                            yolo_look_index = 0
                            while True:
                                try:
                                    print(yolo_detections)
                                    current_detection = str(yolo_detections[yolo_look_index])
                                    current_distance = distance/100
                                    if 'person' in current_detection:
                                        if '15 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 15 Degrees ~~ Target object is to My Left'
                                        elif '15 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 15 Degrees ~~ Target object is to My Right'
                                        elif '45 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 45 Degrees ~~ Target object is to My Left'
                                        elif '45 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 45 Degrees ~~ Target object is to My Right'
                                        else:
                                            if current_distance > 1.0:
                                                movement_command = 'Move Forward One Foot ~~ Moving towards person'
                                            elif current_distance < 0.5:
                                                movement_command = 'Move Backward ~~ Moving away from person'
                                            else:
                                                movement_command = 'Wait ~~ Person is straight ahead'
                                        print(movement_command)
                                        break
                                    else:
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_command = 'Wait ~~ Person lost'
                                            print('person lost')
                                            follow_user = False
                                            yolo_find = True
                                            follow_user_was_true = True
                                            nav_object = 'person'
                                            scan360 = 0
                                            rando_num = random.choice([1, 2])
                                            break
                                        else:
                                            continue
                                except:
                                    movement_command = 'Wait ~~ Follow Person failed. Must be detecting the person first.'
                                    print('person not detected')
                                    follow_user = False
                                    yolo_find = True
                                    follow_user_was_true = True
                                    rando_num = random.choice([1, 2])
                                    nav_object = 'person'
                                    scan360 = 0
                                    break
                        elif yolo_find == True:
                            yolo_nav_index = 0
                            while True:
                                try:
                                    current_detection = str(yolo_detections[yolo_nav_index])
                                    current_distance = distance/100
                                    print('NAV OBJECT: ' + nav_object)
                                    print(current_detection)
                                    if nav_object in current_detection:
                                        yolo_find = False
                                        if '15 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 15 Degrees ~~ Target object is to My Left'
                                        elif '15 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 15 Degrees ~~ Target object is to My Right'
                                        elif '45 Degrees To My Left' in current_detection:
                                            movement_command = 'Turn Left 45 Degrees ~~ Target object is to My Left'
                                        elif '45 Degrees To My Right' in current_detection:
                                            movement_command = 'Turn Right 45 Degrees ~~ Target object is to My Right'
                                        else:
                                            yolo_find = False
                                            movement_command = 'Wait ~~ Ending search for '+nav_object+'. Object has successfully been found!'
                                            if yolo_nav_was_true == True:
                                                yolo_nav = True
                                                yolo_nav_was_true = False
                                            elif follow_user_was_true == True:
                                                follow_user = True
                                                follow_user_was_true = False
                                            else:
                                                nav_object = ''
                                        break
                                    else:
                                        yolo_nav_index += 1
                                        if yolo_nav_index >= len(yolo_detections):
                                            break
                                        else:
                                            continue
                                except:
                                    yolo_nav_index += 1
                                    if yolo_nav_index >= len(yolo_detections):
                                        break
                                    else:
                                        continue
                            if yolo_find == True:
                                if scan360 < 10 and scan360 > 1:
                                    movement_command = 'Turn Right 45 Degrees ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 0:
                                    movement_command = 'Raise Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 1:
                                    movement_command = 'Lower Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                else:
                                    if distance < 50.0 and distance >= 20.0:
                                        if rando_num == 1:
                                            movement_command = 'Turn Left 45 Degrees ~~ Exploring to look for target object'
                                        elif rando_num == 2:
                                            movement_command = 'Turn Right 45 Degrees ~~ Exploring to look for target object'
                                    elif distance < 20.0:
                                        movement_command = 'Turn Around 180 Degrees'
                                    else:
                                        movement_command = 'Move Forward One Foot ~~ Exploring to look for target object'
                        else:
                            move_result = []
                            movement_command, move_prompt, move_prompt_with_mental, keywords_from_move_command, history_filename1, history_filename2, history_filename3, history_filename4, history_filename5, history_filename6, history_filename7, history_filename8, history_filename9, history_filename10, the_prompt, sys_m, user_said, g_speed, file_data = send_text_to_gpt4_move(per, distance, last_phrase, g_speed)
                            if g_speed < 0:
                                g_speed = 0
                            elif g_speed > 30:
                                movement_command = 'Go To Sleep'
                                g_speed = 30
                            else:
                                pass
                            filename = datetime.now().strftime("%Y-%m-%d") + ".txt"
                    else:
                        movement_command = move_set[0]
                        del move_set[0]
                    move_failed_command, move_failure_reason, yolo_nav, move_set, yolo_find, nav_object, look_object, follow_user, scan360, camera_vertical_pos, yolo_look, sleep_data, sleep_value, b_gpt_sleep = handle_commands(
                        distance,
                        current,
                        movement_command,
                        camera_vertical_pos,
                        move_failed_command,
                        move_failure_reason,
                        yolo_nav,
                        move_set,
                        yolo_find,
                        nav_object,
                        look_object,
                        follow_user,
                        scan360,
                        yolo_look,
                        classes,
                        last_phrase,
                        history_filename1,
                        history_filename2,
                        history_filename3,
                        history_filename4,
                        history_filename5,
                        history_filename6,
                        history_filename7,
                        history_filename8,
                        history_filename9,
                        history_filename10,
                        the_prompt,
                        sys_m,
                        user_said,
                        file_data
                    )
                else:
                    pass
                time.sleep(0.1)
                finished_cycle = True
            except:
                print(traceback.format_exc())
        except:
            print(traceback.format_exc())
yolo_nav = False
yolo_find = False
yolo_look = False
follow_user = False
move_stop = False
if __name__ == "__main__":
    try:
        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["2"], arduino_address)
        time.sleep(2)
        be_still = True
        last_time_seen = time.time()
        transcribe_thread = threading.Thread(target=listen_and_transcribe)
        transcribe_thread.start()
        time.sleep(5)
        movement_thread = threading.Thread(target=movement_loop)
        movement_thread.start()
    except Exception as e:
        print(traceback.format_exc())
