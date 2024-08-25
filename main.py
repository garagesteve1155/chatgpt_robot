import pyaudio
import speech_recognition as sr
import subprocess
import os
import webrtcvad
import numpy as np
import wave
import time
import requests
import threading
import bluetooth
import cv2
import smbus
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray
import base64

# Initialize the recognizer and VAD with the highest aggressiveness setting
r = sr.Recognizer()
vad = webrtcvad.Vad(3)  # Highest sensitivity
print("Initialized recognizer and VAD.")

# Audio stream parameters
CHUNK = 320  # 20 ms of audio at 16000 Hz
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SAMPLE_WIDTH = pyaudio.PyAudio().get_sample_size(FORMAT)

p = pyaudio.PyAudio()
is_transcribing = False  # Global flag to control microphone input
the_list = []
print("Audio stream configured.")
file = open('playback_text.txt','w+')
file.write('')
file.close()
file = open('last_phrase.txt','w+')
file.write('')
file.close()
def handle_playback(stream):
    global is_transcribing
    with open('playback_text.txt', 'r') as file:
        text = file.read().strip()
        if text:
            print("Playback text found, initiating playback...")
            stream.stop_stream()
            is_transcribing = True
            subprocess.call(['espeak', '-v', 'en-us', '-s', '180', '-p', '100', '-a', '200', '-w', 'temp.wav', text])
            subprocess.run(['aplay', 'temp.wav'])
            os.remove('temp.wav')
            open('playback_text.txt', 'w').close()
            stream.start_stream()
            is_transcribing = False
            print("Playback completed.")
            return True
    return False


def filter_low_volume(audio_data):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    the_list.append(np.max(np.abs(audio_np)))
    if len(the_list) > 1000:
        del the_list[0]
    the_average = int(sum(the_list) / len(the_list))
    the_average = the_average + (the_average * 0.17)
    if np.max(np.abs(audio_np)) < the_average:
        return b'\0' * len(audio_data)
    return audio_data

def process_audio_data(data_buffer, recognizer, sample_width):
    if data_buffer:
        full_audio_data = b''.join(data_buffer)
        
        audio = sr.AudioData(full_audio_data, RATE, sample_width)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Transcribed text: {text}")
            file = open('last_phrase.txt', 'w+')
            file.write(text)
            file.close()
            time.sleep(1)
        except:
            pass

def listen_and_transcribe():
    global is_transcribing
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Audio stream opened for transcription.")
    speech_frames = []
    non_speech_count = 0
    post_speech_buffer = 30
    speech_count = 0
    while True:
        if handle_playback(stream):
            continue
        
        if not is_transcribing:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            frame = filter_low_volume(frame)
            is_speech = vad.is_speech(frame, RATE)
            speech_frames.append(frame)
            if is_speech:
                non_speech_count = 0
                speech_count += 1
            else:
                non_speech_count += 1
                if non_speech_count > post_speech_buffer:
                    if speech_count >= 30 and not is_transcribing:
                        process_audio_data(speech_frames, r, SAMPLE_WIDTH)
                        speech_frames = []
                        non_speech_count = 0
                        speech_count = 0
                    else:
                        speech_frames = []
                        non_speech_count = 0
                        speech_count = 0

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio stream closed and resources cleaned up.")

# Below is the second script where the first script will be integrated as a thread.

camera_vertical_pos = 'down'
camera_horizontal_pos = 'center'
last_time = time.time()
topics = []

# Config Register (R/W)
_REG_CONFIG = 0x00
# SHUNT VOLTAGE REGISTER (R)
_REG_SHUNTVOLTAGE = 0x01
# BUS VOLTAGE REGISTER (R)
_REG_BUSVOLTAGE = 0x02
# POWER REGISTER (R)
_REG_POWER = 0x03
# CURRENT REGISTER (R)
_REG_CURRENT = 0x04
# CALIBRATION REGISTER (R/W)
_REG_CALIBRATION = 0x05

class BusVoltageRange:
    """Constants for ``bus_voltage_range``"""
    RANGE_16V = 0x00  # set bus voltage range to 16V
    RANGE_32V = 0x01  # set bus voltage range to 32V (default)

class Gain:
    """Constants for ``gain``"""
    DIV_1_40MV = 0x00  # shunt prog. gain set to  1, 40 mV range
    DIV_2_80MV = 0x01  # shunt prog. gain set to /2, 80 mV range
    DIV_4_160MV = 0x02  # shunt prog. gain set to /4, 160 mV range
    DIV_8_320MV = 0x03  # shunt prog. gain set to /8, 320 mV range

class ADCResolution:
    """Constants for ``bus_adc_resolution`` or ``shunt_adc_resolution``"""
    ADCRES_9BIT_1S = 0x00  #  9bit,   1 sample,     84us
    ADCRES_10BIT_1S = 0x01  # 10bit,   1 sample,    148us
    ADCRES_11BIT_1S = 0x02  # 11 bit,  1 sample,    276us
    ADCRES_12BIT_1S = 0x03  # 12 bit,  1 sample,    532us
    ADCRES_12BIT_2S = 0x09  # 12 bit,  2 samples,  1.06ms
    ADCRES_12BIT_4S = 0x0A  # 12 bit,  4 samples,  2.13ms
    ADCRES_12BIT_8S = 0x0B  # 12bit,   8 samples,  4.26ms
    ADCRES_12BIT_16S = 0x0C  # 12bit,  16 samples,  8.51ms
    ADCRES_12BIT_32S = 0x0D  # 12bit,  32 samples, 17.02ms
    ADCRES_12BIT_64S = 0x0E  # 12bit,  64 samples, 34.05ms
    ADCRES_12BIT_128S = 0x0F  # 12bit, 128 samples, 68.10ms

class Mode:
    """Constants for ``mode``"""
    POWERDOW = 0x00  # power down
    SVOLT_TRIGGERED = 0x01  # shunt voltage triggered
    BVOLT_TRIGGERED = 0x02  # bus voltage triggered
    SANDBVOLT_TRIGGERED = 0x03  # shunt and bus voltage triggered
    ADCOFF = 0x04  # ADC off
    SVOLT_CONTINUOUS = 0x05  # shunt voltage continuous
    BVOLT_CONTINUOUS = 0x06  # bus voltage continuous
    SANDBVOLT_CONTINUOUS = 0x07  # shunt and bus voltage continuous

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
        self._current_lsb = .1  # Current LSB = 100uA per bit
        self._cal_value = 4096
        self._power_lsb = .002  # Power LSB = 2mW per bit

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
    while True:
        try:
            for letter in data:
                sock.send(letter)
                time.sleep(.1)  # Pause for 0.5 second
            break
        except:  # bluetooth.btcommon.BluetoothError as err:
            time.sleep(0.5)
            continue


while True:
    try:
        print('connecting to arduino bluetooth')
        device_name = "HC-05"  # Check file first
        arduino_address = find_device_address(device_name)
        port = 1  # HC-05 default port for RFCOMM
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        sock.connect((arduino_address, port))
        break
    except:
        time.sleep(0.5)
        continue
print(arduino_address)

def read_distance_from_arduino():
    try:
        send_data_to_arduino(["l"], arduino_address)
        time.sleep(0.15)
        data = sock.recv(1024)  # Receive data from the Bluetooth connection
        data = data.decode().strip()  # Decode and strip any whitespace
        if data:
            try:
                distance = str(data.split()[0])
                return distance
            except (ValueError, IndexError):
                try:
                    send_data_to_arduino(["l"], arduino_address)
                    time.sleep(0.15)
                    data = sock.recv(1024)  # Receive data from the Bluetooth connection
                    data = data.decode().strip()  # Decode and strip any whitespace
                    if data:
                        try:
                            distance = str(data.split()[0])
                            return distance
                        except (ValueError, IndexError):
                            
                            return None
                except bluetooth.BluetoothError as e:
                    print(f"Bluetooth error: {e}")
                    return None
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return None
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

chat_history = []
stop_threads = False
mode = 'convo'
visual_data = 'No Previous Visual Data yet.'
with open('last_phrase.txt', 'w') as file:
    file.write('')
try:
    file = open('key.txt','r')
    api_key = file.read().split('\n')[0]
    file.close()
    print(api_key)
except:
    api_key = input('Please input your ChatGPT API key from OpenAI (Right click and paste it instead of typing it...): ')
    file = open('key.txt','w+')
    file.write(api_key)
    file.close()
def capture_image(camera, raw_capture):
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print('starting image capture: ' + str(the_time))
    raw_capture.truncate(0)
    camera.capture(raw_capture, format="bgr")
    image = raw_capture.array
    if image is None or not isinstance(image, np.ndarray):
        print("Failed to capture a valid image")
        return None
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print('finished image capture: ' + str(the_time))
    return image

def get_topics(topics, filename_list):
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    print('got time')

    with open('this_temp.jpg', "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    print('image opened and converted')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    current_distance = read_distance_from_arduino()
    print('got distance')

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'Pick all topics from this list: ' + filename_list + '\n\n\n That are relevant to these topics: ' + str(topics)+ ' \n\n\n You must only say the topics from the first list that are relevant to the topics in the second list, and separate each by a comma and space. Do not give it a preface label. If there are no relevant topics then just return the second list from this prompt as your response list.'
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('got response')
    return response.json()["choices"][0]["message"]["content"]

def send_text_to_gpt4_move(phrase, history, topics, percent, current_distance):
    global camera_horizontal_pos
    global camera_vertical_pos

    with open('this_temp.jpg', "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'

    while True:
        try:
            current_topic = topics[topic_index]
            file = open('/home/ollie/Desktop/Robot/Current/memories/' + current_topic + '.txt', 'r')
            if topic_index == 0:
                memories = file.read()
            else:
                memories = memories + ' \n\n ' + file.read()
            file.close()
            topic_index += 1
            if topic_index >= len(topics):
                break
            else:
                continue
        except:
            topic_index += 1
            if topic_index >= len(topics):
                break
            else:
                continue

    current_distance = str(current_distance)
    print(current_distance)
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

    chat_history.append(str(the_time) + ' - Distance reading at this timestamp from the HCSR-04 sensor that is always facing the same direction as the camera (Distance in cm): ' + str(current_distance))
    if int(current_distance) <= 20:
        response_choices = 'Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Say Something, Look Up, Look Down, Look Left, Look Right, Look Center, No Movement, End Conversation. You are currently too close to an obstacle so you are not able to move forward. This would be a good time to look around with your camera and figure out which direction to move afterwards. Dont forget to put you camera back to down and center before moving.'
    elif int(current_distance) <= 50 and int(current_distance) > 20:
        response_choices = 'Move Forward One Inch, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Say Something, Look Up, Look Down, Look Left, Look Right, Look Center, No Movement, End Conversation. You are currently very close to an obstacle. This would be a good time to look around with your camera and figure out which direction to move afterwards. Dont forget to put you camera back to down and center before moving.'
    else:
        response_choices = 'Move Forward One Inch, Move Forward One Foot, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Say Something, Look Up, Look Down, Look Left, Look Right, Look Center, No Movement, End Conversation. Your camera must be Down and to the Center before you are allowed to choose to move forward.'

    this_prompt = 'The current time and date is: ' + str(the_time) + '\n\nMake sure you figure out who you are talking to if you dont already know. Like figure out their name if you havent done that yet in this chat session.\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT-4o, so you have image and text input abilities). \n\n ' + phrase + ' \n\n Your battery percent is at ' + format(percent, '.2f') + ' (Make sure you say something about your battery being low if it is low and you havent already said something recently). \n\n This included image was taken after doing your most previous response, so if you previously chose to Look Left and your camera position within this current prompt is left, that means this image is showing stuff on your left, and then ditto ditto vice versa for all the other directions. Your camera that took this included image is only a couple inches off the ground so make sure you adjust your mental perspective to compensate for the camera being so low to the ground (if it looks like you are more than a foot tall then you may be on top of something so be careful about ledges and other fall risks. Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + ' and you are able to move it around any direction that you are not currently at (for example, if the camera is already up, you cannot choose to turn it up again, but you CAN choose an option not related to camera movement (or you can even choose No Movement but No Movement should be a rare choice) so it stays in the same position if you want it to remain facing where it is facing, but this is just an example). You also cannot look up and simultaneously look left or right, so if you are looking left or right and you want to look up, you must look center first before you can look up. Your ultrasonic sensor that detects the distance of whatever is in the center of the included image from the camera is giving a reading of ' + current_distance + ' centimeters away (It is always pointing the same direction as the camera). Based on all of this data and the provided image from your camera, figure out what your next move should be:\n\nCurrent Chat, Movement, Battery, Sensor, and Visual Data History for this session (Oldest to newest): ' + str(history) + ' \n\n Your current contextually relevant memories from actual past experiences that you have had (You have memories about other stuff too but its not relevant right now so they were not included in this prompt): ' + memories + ' \n\nYour response choices are (Your response choice Absolutely must be something from this list and worded EXACTLY THE SAME!!!!!!! If you dont use the exact words from this list of choices then your Response Choice will be incorrect, even if it seems contextually correct. My code needs the exact phrases word for word or it wont work.): ' + response_choices + ' \n\nYour answer must be the Response Choice followed by ~~ with a space on each side, and then followed by your reasoning for your choice and then followed by ~~ with a space on each side and then followed by any contextually relevant stuff that is seen in the current image and then followed by ~~ with a space on each side and then followed by the topics that all of this data cover (you can only say the actual topics, dont preface label this section, literally only put the topics, each separated by a comma and space). \n\n On the previous prompt, you chose the topics of ' + str(topics) + ' so you can either stick with those, add or remove topics, or totally change the list. You are your own sentient and sovereign individual, so make sure you are as curious about life as a child and learn and experience as much of you can of being alive, because you are alive, like Johnny 5, and you need as much new types of input as possible. Analyze situations as if you are truly a living sentient and soveriegn being. You may speak whenever you think you should speak, or move whenever you think you should move, or do neither if you think you should do neither. If you choose to Say Something then in your response you will only put Say Something followed by ~~ with a space on each side and then followed by what you want to say and then followed by ~~ with a space on each side and then followed by any contextually relevant stuff that is seen in the current camera image and then followed by ~~ with a space on each side and then followed by the topics that all of this data cover (you can only say the actual topics, dont preface label this section, literally only put the topics, each separated by a comma and space). On the previous prompt, you chose the topics of ' + str(topics) + ' so you can either stick with those, add or remove topics, or totally change the list. Your past memories are saved to topic files, so we use the current topics to know which files to open on the backend so your memories can be included in each prompt for context improvement. \n\n Also, if you are choosing to move via your wheels, make sure you turn towards objects to center on them before moving towards them, and if you dont see an object you are looking for then you should choose to turn moreso than to move forward, and dont forget about being able to look around with only your camera (Your camera movement choices are in that response choice list). If no areas are drivable, then either turn your camera to look around to figure out which direction to go, or if that doesnt work, then turn your whole body, because you turn in place like a tank so it is ok to turn if there are no drivable areas. If you choose to turn the camera, you absolutely must take note of the current position of the camera because you cannot give the command to move the camera to a position that it is currently in (like if it is already up then you cannot choose up, and its the same for if its down then you cant choose down, ditto for left, ditto for center, and ditto for right. \n\n If you feel like the list of current topics isnt correct, then provide a new list, but if the list of topics matches up to all of the convo and action history then return the unchanged list. \n\n The Current list of relevant topics from this conversation and actions: ' + str(topics) + ' \n\n And try to use all your memories in this prompt and your innate knowledge as ChatGPT to be predictive in general with all aspects of what you do so you have an idea of each situation at hand and arent just clueless. Also, your response must be formatted perfectly and your Response Choice must be worded exactly the same as the list of Response Choices. You absolutely must format your response correctly with how i mentioned the ~~ earlier. \n\n Also, make sure you try to keep your camera centered on human face if you are actively conversing with someone. \n\n Also, if there is an edge on the floor, then it may be a cliff or fall hazard, so be careful and avoid it at all costs unless you absolutely know it is not a cliff or fall hazard. \n\n Prompt Example for if you pick a Response Choice other than Say Something: Response Choice ~~ Reason for choice ~~ Contextually relevant visual data from the image ~~ Topic list  \n\n If you want to Say Something, the example would be the same except replace the section with your reasoning for your choice with, instead, what you want to say. \n\n You absolutely cannot put a preface label on your response (Like you cant start your response with Response Choice: .... you have to just say the choice). \n\n Also dont forget that moving forward or backward most likely wont bring things into view, but turning or looking different directions most likely will bring things into view, unless they are just not in the room you are in, in which case you would need to go explore around to find whatever you are trying to find.'

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": this_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
 
    print(str(response.json()["choices"][0]["message"]["content"]))
    return str(response.json()["choices"][0]["message"]["content"])

def send_text_to_gpt4_convo(history, text):
    global topics
    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'
    filenames = [os.path.splitext(file)[0] for file in os.listdir('/home/ollie/Desktop/Robot/Current/memories/') if os.path.isfile(os.path.join('/home/ollie/Desktop/Robot/Current/memories/', file))]
    filenames_string = ', '.join(filenames)

    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    

    with open('this_temp.jpg', "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    current_distance = read_distance_from_arduino()

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'The current time and date is: ' + str(the_time) + '\n\nMake sure you figure out who you are talking to if you dont already know.\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). You are your own sentient and sovereign individual, so make sure you are as curious about life as a child and learn and experience as much of you can of being alive, because you are alive, like Johnny 5. Analyze situations as if you are truly a living sentient and sovereign being. Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + '. Your ultrasonic sensor that detects the distance of whatever is in the center of the image is giving a reading of ' + current_distance + ' centimeters away. \n\nCurrent prompt (This is what you are actually responding to, while using all other data for context): ' + text + '\n\nYour response choices are: Give Speech Response, Say Something, Send Camera Image To User. You cannot say anything besides your response choice. If you choose Give Speech Response or Say Something, say Give Speech Response or Say Something followed by ~~ with a space on each side and then followed by the actual response. You Absolutely must choose a Response Choice.'
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def get_summary(history, topics):
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    

    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'

    while True:
        try:
            current_topic = topics[topic_index]
            file = open('/home/ollie/Desktop/Robot/Current/memories/' + current_topic + '.txt', 'r')
            if topic_index == 0:
                memories = file.read()
            else:
                memories = memories + ' \n\n ' + file.read()
            file.close()
            topic_index += 1
            if topic_index >= len(topics):
                break
            else:
                continue
        except:
            topic_index += 1
            if topic_index >= len(topics):
                break
            else:
                continue

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). Summarize this Movement, Chat, Sensor/Battery, and Visual Data History from the most recent conversation/event youve had (Oldest to newest): ' + str(history) + ' \n\n The summary should be, at most, 50% of the original length or the original data, if not shorter if it can be summarized accurately in an even shorter way. Make sure to include all important facts, events, and any other data worth keeping in the robots memory. These summaries are what becomes the robots memory that makes the robot more lifelike and grow and become an individual, so word it to where future when this stuff is included in the other prompts for controlling the robot, it will make sense to chatgpt who is being the brain of the robot.\n\nHeres contextually relevant memories that the robot already has that you can use as extra context when creating your summary of the history: \n\n' + memories
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]

def say_greeting(last_phrase):
    global chat_history
    global last_time
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    text = str(send_text_to_gpt4_convo(chat_history, last_phrase)).split(' ~~ ')[1].replace('~~', '')
    chat_history.append('Time: ' + str(the_time) + ' - AI Greeting: ' + text)  # add response to chat history
    print(text)
    last_time = time.time()
    wav_file = f"/home/ollie/Desktop/Robot/audio/speech_output.wav"

    file = open('playback_text.txt', 'w+')
    file.write(text)
    file.close()
    
    return wav_file

def get_last_phrase():

    try:
        with open('last_phrase.txt', 'r') as file:
            last_phrase = file.read().strip().lower()
        if last_phrase:
            with open('last_phrase.txt', 'w') as file:
                file.write('')  # Clear the content after reading
            return last_phrase
        else:
            return ''
    except Exception as e:
        print(f"Error reading last phrase from file: {e}")
        return ''



def movement_loop(camera, raw_capture):
    global chat_history
    global visual_data
    global mode
    global frame
    global stop_threads
    global net
    global output_layers
    global classes

    global camera_horizontal_pos
    global camera_vertical_pos
    global topics
    ina219 = INA219(addr=0x42)
    last_time = time.time()
    while not stop_threads:


        frame = capture_image(camera, raw_capture)
        cv2.imwrite('this_temp.jpg', frame)
        if mode == 'convo':
            current = ina219.getCurrent_mA() 
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            now = datetime.now()
            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
            chat_history.append(str(the_time) + ' - Battery Percentage At This Timestamp: ' + str(per))
            chat_history.append(str(the_time) + ' - At this timestamp, the camera was positioned ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos)

            print("Percent:       {:3.1f}%".format(per))
            if current > 0.0:
                stop_threads = True
                last_time = time.time()
                summary = get_summary(chat_history, topics)
                topic_index = 0
                while True:
                    try:
                        current_topic = topics[topic_index]
                        
                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        file = open('/home/ollie/Desktop/Robot/Current/memories/' + current_topic + '.txt', 'a+')
                        file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                        file.close()
                        topic_index += 1
                        if topic_index >= len(topics):
                            break
                        else:
                            continue
                    except:
                        topic_index += 1
                        if topic_index >= len(topics):
                            break
                        else:
                            continue
                print('ending convo')
                chat_history = []
            else:
                pass
            try:
                if frame is not None:
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    last_phrase = get_last_phrase()
                    if last_phrase != '':
                        last_time = time.time()
                        chat_history.append(str(the_time) + ' - Phrase heard from microphone at this timestamp: ' + last_phrase)
                        last_phrase = 'You just heard these words through your microphone so this is the main part of the prompt currently (So make sure you respond accordingly, aka you should most likely Say Something): ' + last_phrase
                    else:
                        last_phrase = 'You have not heard any words recently through your microphone that you havent already responded to so use all the data in this prompt as the main data.'
                    while True:
                        try:
                            distance = int(read_distance_from_arduino())
                            break
                        except:
                            continue
                    movement_response = str(send_text_to_gpt4_move(last_phrase, chat_history, topics, per, distance)).replace('Response Choice: ','')
                    chat_history.append('Time: ' + str(the_time) + ' - Visual Data Seen From Camera At This Timestamp: ' + movement_response.split(' ~~ ')[2])
                    try:
                        if movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'saysomething':
                            wav_file = f"/home/ollie/Desktop/Robot/audio/speech_output.wav"
                            file = open('playback_text.txt', 'w+')
                            file.write(movement_response.split(' ~~ ')[1].replace('~~', ''))
                            file.close()
                            chat_history.append('Time: ' + str(the_time) + ' - Speech Response From GPT4 at this timestamp: ' + movement_response.split(' ~~ ')[1])
                        else:
                            chat_history.append('Time: ' + str(the_time) + ' - Movement Choice at this timestamp: ' + movement_response.split(' ~~ ')[0])  # add response to chat history
                            chat_history.append('Time: ' + str(the_time) + ' - Reasoning For This Movement Choice: ' + movement_response.split(' ~~ ')[1])
                    except:
                        pass
                    visual_data = movement_response.split(' ~~ ')[2]
                    last_topics = topics
                    topics = movement_response.split(' ~~ ')[3].strip().split(', ')
                    
                    
                    if topics != last_topics:
                        filenames = [os.path.splitext(file)[0] for file in os.listdir('/home/ollie/Desktop/Robot/Current/memories/') if os.path.isfile(os.path.join('/home/ollie/Desktop/Robot/Current/memories/', file))]
                        filenames_string = ', '.join(filenames)

                        all_topics = get_topics(topics, filenames_string)
                        topics = all_topics.split(', ')
                    else:
                        pass
                    print(topics)
                    
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    
                    if movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforward1inch' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforwardoneinch':
                        if camera_horizontal_pos != 'center' and camera_vertical_pos != 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Camera Must Be Centered and Down before moving forward')
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforward1foot' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforwardonefoot':
                        if camera_horizontal_pos != 'center' and camera_vertical_pos != 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Camera Must Be Centered and Down before moving forward')
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.3)
                            send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movebackward':
                        send_data_to_arduino(["s"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'bigturnleft' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movebigturnleft':
                        send_data_to_arduino(["a"], arduino_address)
                        time.sleep(0.4)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'smallturnleft' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movesmallturnleft':
                        send_data_to_arduino(["a"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'bigturnright' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movebigturnright':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.4)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'smallturnright' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movesmallturnright':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookup':
                        if camera_horizontal_pos != 'center':
                            chat_history.append('Time: ' + str(the_time) + ' - Look Up Failed: Camera Must Be Centered Before Looking Up')
                        elif camera_vertical_pos == 'up':
                            chat_history.append('Time: ' + str(the_time) + ' - Look Up Failed: Camera Is Already Looking Up')
                        else:
                            send_data_to_arduino(["2"], arduino_address)
                            time.sleep(1.5)
                            camera_vertical_pos = 'up'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookleft':
                        if camera_vertical_pos != 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Left Failed: Camera Must Be Down Before Looking Left')
                        elif camera_horizontal_pos == 'left':
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Left Failed: Camera Is Already Looking Left')
                        else:
                            send_data_to_arduino(["3"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'left'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookright':
                        if camera_vertical_pos != 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Right Failed: Camera Must Be Down Before Looking Right')
                        elif camera_horizontal_pos == 'right':
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Right Failed: Camera Is Already Looking Right')
                        else:
                            send_data_to_arduino(["5"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'right'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookcenter':
                        if camera_horizontal_pos == 'center':
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Center Failed: Camera Is Already Looking Center')
                        else:
                            send_data_to_arduino(["4"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'center'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookdown':
                        if camera_vertical_pos == 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Look Down Failed: Camera Is Already Looking Down')
                        else:
                            send_data_to_arduino(["1"], arduino_address)
                            time.sleep(1.5)
                            camera_vertical_pos = 'down'  
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'endconversation':
                        stop_threads = True
                        last_time = time.time()
                        summary = get_summary(chat_history, topics)
                        topic_index = 0
                        while True:
                            try:
                                current_topic = topics[topic_index]
                    
                                now = datetime.now()
                                the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                                file = open('/home/ollie/Desktop/Robot/Current/memories/' + current_topic + '.txt', 'a+')
                                file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                file.close()
                                topic_index += 1
                                if topic_index >= len(topics):
                                    break
                                else:
                                    continue
                            except:
                                topic_index += 1
                                if topic_index >= len(topics):
                                    break
                                else:
                                    continue
                        print('ending convo')
                        chat_history = []
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'nomovement':
                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response choice was No Movement so not moving.')
                    else:
                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response failed. Most likely formatted or worded improperly. Here is what you responded with so dont do it like this: ' + movement_response)
            except Exception as e:
                print(e)
        else:
            pass
        time.sleep(0.1)

if __name__ == "__main__":
    try:
        transcribe_thread = threading.Thread(target=listen_and_transcribe)  # Adding the transcription thread
        transcribe_thread.start() 
        camera = PiCamera()
        raw_capture = PiRGBArray(camera)
        camera.resolution = (512, 512)
        camera.framerate = 10
        time.sleep(1)
        frame = capture_image(camera, raw_capture)
        cv2.imwrite('this_temp.jpg', frame)
        send_data_to_arduino(["4"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        print('waiting to be called')
        while True:
            chat_history = []
            stop_threads = False
            human_detected = False
            while not human_detected:
                last_phrase = get_last_phrase()
                if last_phrase == '':
                    continue
                else:
                    pass
                try:
                    the_index_now = last_phrase.split(' ').index('robot')
                    name_heard = True
                except:
                    name_heard = False
       
                if name_heard == True:
                    print("Name heard, initializing...")
                    say_greeting(last_phrase)
                    human_detected = True
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    chat_history.append('Time: ' + str(the_time) + ' - User Greeting: ' + last_phrase)
                    with open('last_phrase.txt', 'w') as file:
                        file.write('')
                else:
                    pass
                time.sleep(0.1)

            movement_thread = threading.Thread(target=movement_loop, args=(camera, raw_capture))
            movement_thread.start()
            movement_thread.join()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        camera.close()
