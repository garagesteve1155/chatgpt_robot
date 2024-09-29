import re
import pyaudio
import speech_recognition as sr
import subprocess
import os
import webrtcvad
import numpy as np
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
import signal
import traceback
import random
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
file = open('last_phrase2.txt','w+')
file.write('')
file.close()
file = open('self_response.txt','w+')
file.write('')
file.close()
def get_position_description(x, y, width, height):
    """Return a text description of the position based on coordinates."""
    if x < width / 3:
        horizontal = " Small Turn Left"
    elif x > 2 * width / 3:
        horizontal = " Small Turn Right"
    else:
        horizontal = " already centered between left and right"
    
    if y < height / 3:
        vertical = "look up "
    elif y > 2 * height / 3:
        vertical = "look forward "
    else:
        vertical = "already centered on the vertical "

    if horizontal == " already centered between left and right" and vertical == "already centered on the vertical ":
        return "already centered on object"
    else:
        return f"{vertical}-{horizontal} of the image"

def get_size_description(w, h, width, height):
    """Return a text description of the size based on bounding box dimensions."""
    box_area = w * h
    image_area = width * height

    size_ratio = box_area / image_area

    if size_ratio < 0.1:
        return "small"
    elif size_ratio < 0.3:
        return "medium"
    else:
        return "large"

def remove_overlapping_boxes(boxes, class_ids, confidences):
    """Remove overlapping boxes of the same class, keeping only the one with the highest confidence."""
    final_boxes = []
    final_class_ids = []
    final_confidences = []

    for i in range(len(boxes)):
        keep = True
        for j in range(len(final_boxes)):
            if class_ids[i] == final_class_ids[j]:
                box1 = boxes[i]
                box2 = final_boxes[j]

                x1_min, y1_min, x1_max, y1_max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
                x2_min, y2_min, x2_max, y2_max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]

                # Calculate the overlap area
                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                box1_area = (x1_max - x1_min) * (y1_max - y1_min)
                box2_area = (x2_max - x2_min) * (y2_max - y2_min)

                # Calculate overlap ratio
                overlap_ratio = inter_area / min(box1_area, box2_area)

                if overlap_ratio > 0.5:
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
def get_wm8960_card_number():
    result = subprocess.run(["aplay", "-l"], stdout=subprocess.PIPE, text=True)
    match = re.search(r"card (\d+): wm8960sound", result.stdout)
    if match:
        card_number = match.group(1)
        return card_number
    else:
        return None
def set_max_volume(card_number):
    subprocess.run(["amixer", "-c", card_number, "sset", 'Headphone', '100%'], check=True)
    subprocess.run(["amixer", "-c", card_number, "sset", 'Speaker', '100%'], check=True)

def handle_playback(stream):
    global is_transcribing
    with open('playback_text.txt', 'r') as file:
        text = file.read().strip()
        if text:
            print("Playback text found, initiating playback...")
            stream.stop_stream()
            is_transcribing = True

            # Generate speech from text
            subprocess.call(['espeak', '-v', 'en-us', '-s', '180', '-p', '100', '-a', '200', '-w', 'temp.wav', text])

            # Get the correct sound device for the WM8960 sound card
            wm8960_card_number = get_wm8960_card_number()
            if wm8960_card_number:
                # Set volume to maximum before playback
                set_max_volume(wm8960_card_number)

                # Play the sound file on the WM8960 sound card
                subprocess.call(["aplay", "-D", f"plughw:{wm8960_card_number}", 'temp.wav'])
            else:
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
          
            if text.strip().lower().replace(' ','') != '':
                file = open('last_phrase.txt', 'w+')
                file.write(text)
                file.close()

                file = open('playback_text.txt', 'w+')
                file.close()
                file = open('self_response.txt', 'w+')
                file.close()
            else:
                pass
        except Exception as e:
            print(e)

def listen_and_transcribe():
    global is_transcribing
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Audio stream opened for transcription.")
    speech_frames = []
    non_speech_count = 0
    post_speech_buffer = 30
    speech_count = 0
    while True:
        if speech_count == 0:
            #check self_response file
            with open('self_response.txt','r') as f:
                self_response = f.read()
            if self_response != '':
                with open('playback_text.txt', 'w') as f:
                    f.write(self_response)
                file = open('self_response.txt', 'w+')
                file.close()
            else:
                pass
            #if data, add data to playback text file
        else:
            pass
        if handle_playback(stream):
            continue
        else:
            pass

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

camera_vertical_pos = 'forward'
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
    POWERDOW = 0x00  # power forward
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
                            
                            return 0
                except bluetooth.BluetoothError as e:
                    print(f"Bluetooth error: {e}")
                    return 0
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return 0
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth error: {e}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

chat_history = []
stop_threads = False

try:
    file = open('key.txt','r')
    api_key = file.read().split('\n')[0]
    file.close()
except:
    api_key = input('Please input your ChatGPT API key from OpenAI (Right click and paste it instead of typing it...): ')
    file = open('key.txt','w+')
    file.write(api_key)
    file.close()
def capture_image(camera, raw_capture):
    raw_capture.truncate(0)
    camera.capture(raw_capture, format="bgr")
    image = raw_capture.array
    return image

def get_topics(topics, filename_list):
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
                        "text": 'Pick all topics from this list: ' + filename_list + '\n\n\n That are relevant to these topics: ' + str(topics)+ ' \n\n\n You must only say the topics from the first list that are relevant to the topics in the second list, and separate each by a comma and space. Do not give it a preface label. If there are no relevant topics then just return the second list from this prompt as your response list.'
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]



def yolo_detect():
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    global chat_history
    try:
        
        img = cv2.imread('this_temp.jpg')
        height, width, channels = img.shape

       
        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.35:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Remove overlapping boxes
        boxes, class_ids, confidences = remove_overlapping_boxes(boxes, class_ids, confidences)

        # Initialize a list for descriptions
        descriptions = []

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 10)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Generate and collect descriptions
            pos_desc = get_position_description(x + w/2, y + h/2, width, height)
            size_desc = get_size_description(w, h, width, height)
            descriptions.append(f"The {label} is {size_desc} and located at the {pos_desc}.")

        # Save descriptions to a file
        with open("output.txt", "w") as file:
            for description in descriptions:
                file.write(description + "\n")
        chat_history.append('Time: ' + str(the_time) + ' - YOLO Detections and which moves would be needed to make to center on each object at this timestamp: \n' + '\n'.join(descriptions))
        # Display and save the processed image

        cv2.imwrite("output.jpg", img)
    except Exception as e:
        print(e)
      

      
        

def get_gtp_visual():
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    global chat_history

    with open('this_temp.jpg', "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    this_prompt = 'Please describe, from the image, objects and their positions, as well as the general environment and what is happening. Your response can be no longer than a few sentences within a paragraph formatting.'
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
        "max_tokens": 200
    }
  
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    summary_response = str(response.json()["choices"][0]["message"]["content"])
    with open('visual.txt','w+') as file:
        file.write(summary_response)
    chat_history.append('Time: ' + str(the_time) + ' - GPT4-Vision Description of Camera Image at this timestamp: ' + summary_response)


def send_text_to_gpt4_move(history, topics, percent, current_distance, phrase):
    global camera_vertical_pos
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('output.txt','r') as file:
        yolo_detections = file.read()
    with open('visual.txt','r') as file:
        gpt_visual_data = file.read()
    

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'
    while True:
        try:
            current_topic = topics[topic_index].replace('.','')
            file = open('memories/'+current_topic + '.txt', 'r')
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
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

    

    #yolo_detections, the_time, percent, camera_vertical_pos, camera_horizontal_pos, current_distance, chat_history, memories, base64_image
    
    
    """
    OLD
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": 'You are a 4 wheeled mobile robot. \n\n '+last_phrase2+' \n\n SESSION HISTORY (it is all timestamped and in order from oldest to newest): '+'\n'.join(history)+' \n\n You also have a camera and an HCSR04 distance sensor pointing the same direction as the camera at whatever object that yolo says is in the absolute middle center of the image. Your camera is currently pointed ' + camera_vertical_pos + '. Here is the text description from YOLO and the movements that would be necessary to center your camera on each specific object: \n\n '+str(yolo_detections)+ ' \n\n Here is a general visual description from ChatGPT vision of what you currently see through you camera (this is in addition to YOLO data. Trust the YOLO data moreso than this as far as directional data is concerned.): '+gpt_visual_data+'. \n\n Memories from past events that are relevant to the current situation at hand: '+memories+' \n\n Your response choices are: Move Forward One Inch, Move Forward One Foot, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Look Up, Look Forward, No Movement, End Conversation \n\n Your response choice Absolutely must be something from this list and worded EXACTLY THE SAME!!!!!!! If you dont use the exact words from this list of choices then your Response Choice will be incorrect, even if it seems contextually correct. My code needs the exact phrases word for word or it wont work. \n\nYour answer must be the Response Choice (you can choose up to 5 at once, and you can use each Response choice multiple times in the list. Separate your response choices by a comma and space. The robot will execute each response choice you add to the list in the order that it is in the list, so it will the the 0 index first.) followed by ~~ with a space on each side, and then followed by your reasoning for your choice and then followed by ~~ with a space on each side and then followed by the topics that all of this data covers (you can only say the actual topics, dont preface label this section, literally only put the topics, each separated by a comma and space). \n\n On your previous response, you chose the topics of ' + str(topics) + ' so you can either stick with those, add or remove topics, or totally change the list. Your past memories are saved to topic files, so we use the current topics to know which files to open on the backend so your memories can be included in each prompt for context improvement. \n\n Also, if you want to move to an object, make sure you center on it first with YOLO before moving forward towards it, and if you dont see an object you are looking for then you should choose to turn moreso than to move forward. If no areas are drivable, then turn, because you turn in place like a tank so it is ok to turn if there are no drivable areas. \n\n Also, your response must be formatted perfectly and your Response Choice must be worded exactly the same as one of the options from the list of Response Choices. You absolutely must format your response correctly with how i mentioned in the example template earlier in this prompt. \n\n You absolutely cannot put a preface label on your response (Like you cant start your response with Response Choice: .... you have to just say the choice or choices). \n\n And as a last reminder, your response choice or choices have to be worded exactly the same as your choices from the provided list, you must use the exact same words on your response choice.'
                    }
                ]
            }
        ],
        "max_tokens": 250
    }
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # System message with initial context and detailed instructions
            {
                "role": "system",
                "content": (
                    "You are a 4-wheeled mobile robot.\n\n"
                    f"You have a camera and an HCSR04 distance sensor pointing in the same direction as the camera and the distance sensor detects the distance to whatever object that YOLO says is in the absolute middle center of the image. Here is the distance it currently says: {current_distance}"
                    f"Your camera is currently pointed {camera_vertical_pos}.\n\n"
                    "Here is the text description from YOLO and the movement choices necessary to center your camera on each specific object:\n\n"
                    f"{yolo_detections}\n\n"
                    "Here is a general visual description from ChatGPT Vision of what you currently see through your camera (this is in addition to YOLO data; trust the YOLO data more so than this as far as directional data is concerned):\n\n"
                    f"{gpt_visual_data}.\n\n"
                    "Memories from past events that are relevant to the current situation at hand:\n\n"
                    f"{memories}\n\n"
                    "Your software has 2 threads, the convo thread and the movement thread. You are the movement thread so you decide how to move the robot, or to say stuff even if you havent been directly prompted.\n\n"
                    "Your response choices are:\n"
                    "Move Forward One Inch, Move Forward One Foot, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Look Up, Look Forward, No Movement, Say Something, End Conversation.\n\n"
                    "Your response choice(s) absolutely must be something from this list and worded **exactly the same**. If you don't use the exact words from this list of choices, then your response choice will be incorrect, even if it seems contextually correct. My code needs the exact phrases word for word, or it won't work.\n\n"
                    "Your answer must be the response choice (you can choose up to 7 at once and can use each response choice multiple times in the list). If you choose Say Something, you cannot do multiple response choices and Say Something would be your only response choice. Only choose Say Something if you need to alert the user of something, such as some kind of data you are seeing that they want to know about, or a certain amount of time has passed for some purpose, or your battery is below 25 percent, or just something that needs to be said that hasnt been said yet and you havent been prompted about it.). Separate your response choices by a comma and space. The robot will execute each response choice you add to the list in the order they are listed, starting with index 0. It will add what it sees and what happens after each move to the history so you will know what happened during it executing the previous set of moves. You can choose up to 7 for your list.\n\n"
                    "After your response choice(s), include ' ~~ ' (with spaces on both sides), followed by your reasoning for your choices, then ' ~~ ' (with spaces on both sides), and finally, the topics that all of this data covers (only list the topics, separated by a comma and space; do not add any labels or additional text).\n\n"
                    f"In your previous response, you chose the topics: {', '.join(topics)}. You can stick with those, add or remove topics, or totally change the list. Your past memories are saved to topic files; we use the current topics to know which files to open on the backend so your memories can be included in each prompt for context improvement.\n\n"
                    "If you want to move to an object, make sure you center on it first with YOLO before moving forward towards it. If you don't see an object you're looking for, you should choose to turn rather than move forward. If no areas are drivable, then turn, because you turn in place like a tank, so it's okay to turn if there are no drivable areas.\n\n"
                    "If you choose Say Something, replace the Reasoning in your response with what you want to say instead. Also, you can only choose one response choice in your list if you choose Say Something.\n\n"
                    "Also, your response must be formatted perfectly, and your response choice must be worded exactly the same as one of the options from the list of response choices. You absolutely must format your response correctly as mentioned in the instructions.\n\n"
                    "You cannot include any preface labels in your response (for example, do not start your response with 'Response Choice: ...'; you should just state the choice or choices).\n\n"
                    "As a final reminder, your response choice or choices must be worded exactly the same as the choices from the provided list; you must use the exact same words in your response choice."
                )
            },
            # The session history will be added here as individual messages
        ],
        "max_tokens": 250
    }

    # Now, parse the session history and add messages accordingly
    for entry in history:
        timestamp_and_content = entry.split(" - ", 1)
        if len(timestamp_and_content) != 2:
            continue  # Skip entries that don't match the expected format

        timestamp, content = timestamp_and_content

        if "User Greeting:" in content or "Prompt heard from microphone" in content:
            # User message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "user",
                "content": message_content.strip()
            })
        elif "AI Greeting:" in content or "Speech Response From GPT4" in content:
            # Assistant message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": message_content.strip()
            })
        elif "Movement Choice at this timestamp:" in content:
            # Assistant's movement choice
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": content.strip()
            })
        elif "Reasoning For This Movement Choice:" in content:
            # Assistant's reasoning
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": content.strip()
            })
        else:
            # Other data (e.g., sensor readings, system messages)
            payload["messages"].append({
                "role": "system",
                "content": content.strip()
            })
    if phrase != "":
        # Finally, include the current prompt as the latest user message
        payload["messages"].append({
            "role": "user",
            "content": phrase.strip()
        })
    else:
        pass
    
    
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nYolo Detections: \n'+str(yolo_detections))
    return str(response.json()["choices"][0]["message"]["content"])

def send_text_to_gpt4_convo(history, text, vis_data):
    global topics
    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'
    home_directory = os.path.expanduser('~')
    filenames = [os.path.splitext(file)[0] for file in os.listdir(home_directory) if os.path.isfile(os.path.join(home_directory, file))]
    filenames_string = ', '.join(filenames)
    with open('output.txt','r') as file:
        yolo_detections = file.read()
    with open('visual.txt','r') as file:
        gpt_visual_data = file.read()
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'

    while True:
        try:
            current_topic = topics[topic_index].replace('.','')
            file = open('memories/'+current_topic + '.txt', 'r')
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


    with open('batt_per.txt','r') as file:
        percent = file.read()
        
    """
    OLD
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a 4 wheeled mobile raspberry pi and arduino robot that is fully controlled by ChatGPT (Specifically GPT4o). \n\n Your battery percent is: "+percent+" \n\n Here is chat history/robot internal workings history for this session (it is all timestamped and in order from oldest to newest): "+'\n'.join(history)+' \n\n Here is data from your memory from past conversations and experiences that is contextually relevant currently: '+memories+' \n\n The current time and date is: ' + str(the_time) + ' \n\n The current yolo Visual data from your camera and the movements that would be necessary to center your camera on each specific object: '+yolo_detections+' \n\n General description of what is currently seen in the camera image (This is in addition to the YOLO data): '+gpt_visual_data+' \n\n Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + '. \n\nCurrent prompt (This is what you are actually responding to, while using all other data for context. Your response cannot be longer than a few sentences at max): ' + text
                    }
                ]
            }
        ],
        "max_tokens": 77
    }
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # System message with initial context and relevant data
            {
                "role": "system",
                "content": (
                    f"You are a 4-wheeled mobile Raspberry Pi and Arduino robot fully controlled by ChatGPT (specifically GPT-4o).\n"
                    f"Your software has 2 threads, the convo thread and the movement thread. You are the convo thread. You can either respond with what you want to say or if you think you should remain silent and not respond to the prompt (like if you were told to remain quiet or just some other logical reason), only say the words STAYING SILENT (but only be silent if you actually shouldnt speak). If you do speak, dont say any asterisk stuff. Just speak normal.\n"
                    f"If the name of the current user has not been provided by the user within the current chat history, you must ask their name so you know who you are talking to. If you see multiple people, ask who everyone is if you dont already know.\n"
                    f"Your battery percentage is: {percent}%.\n"
                    f"The current time and date is: {the_time}.\n"
                    f"Your camera is currently pointed {camera_vertical_pos}.\n"
                )
            },
            # Include contextually relevant memories
            {
                "role": "system",
                "content": f"Here is data from your memory from past conversations and experiences that is contextually relevant currently: {memories}"
            },
            # Include YOLO visual data
            {
                "role": "system",
                "content": f"The current YOLO visual data from your camera and the movement choices necessary to center your camera on each specific object: {yolo_detections}"
            },
            # Include general description from GPT-4 Vision
            {
                "role": "system",
                "content": f"General description of what is currently seen in the camera image (in addition to the YOLO data): {gpt_visual_data}"
            }
        ]
    }

    # Now, parse the history and add messages accordingly
    for entry in history:
        timestamp_and_content = entry.split(" - ", 1)
        if len(timestamp_and_content) != 2:
            continue  # Skip if the entry doesn't have the expected format

        _, content = timestamp_and_content

        if "User Greeting:" in content or "Prompt heard from microphone" in content:
            # User message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "user",
                "content": message_content.strip()
            })
        elif "AI Greeting:" in content or "Speech Response From GPT4" in content:
            # Assistant message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": message_content.strip()
            })
        elif "Movement Choice at this timestamp:" in content:
            # Assistant's movement choice
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": content.strip()
            })
        elif "Reasoning For This Movement Choice:" in content:
            # Assistant's reasoning
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "assistant",
                "content": content.strip()
            })
        elif "Distance reading at this timestamp" in content:
            # System message for sensor readings
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "system",
                "content": content.strip()
            })
        elif "Failed" in content:
            # System message for failed actions
            payload["messages"].append({
                "role": "system",
                "content": content.strip()
            })
        else:
            # Other system messages
            payload["messages"].append({
                "role": "system",
                "content": content.strip()
            })

    # Finally, include the current prompt as the latest user message
    payload["messages"].append({
        "role": "user",
        "content": text.strip()  # This is the current prompt the assistant should respond to
    })

    # Include the max_tokens parameter
    payload["max_tokens"] = 77

    
    

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def get_summary(history, topics):
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    

    topic_index = 0
    memories = 'No memories yet for any contextually relevant topics.'

    while True:
        try:
            current_topic = topics[topic_index].replace('.','')
            file = open('memories/'+current_topic + '.txt', 'r')
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
                        "text": 'The summary of this history you create must be in story form, not data list form. Like how a human remembers situations and conversations. This summary will be a memory for my robot so you must summarize what happened and if the robot did correct or not in general and with specific stuff. Also say what it should have done if it didnt do correct. Really turn your summary into a story instead of just a list of specific stuff.\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). Summarize this Movement, Chat, Sensor/Battery, and Visual Data History from the most recent conversation/event youve had (Oldest to newest): ' + str(history) + ' \n\n The summary should be, at most, 50% of the original length of the original data, if not shorter if it can be summarized accurately in an even shorter way. Make sure to include all important facts, events, goals, conversational information, learned information about the world, and any other data worth keeping in the robots memory so it has useful information to use on future prompts (The prompts that choose the robots actions and when to speak are given relevant topic memories so this summary has to include good relevant knowledge). These summaries are what becomes the robots memory that makes the robot more lifelike and grow and become an individual, so word it as like a story to where on future prompts when this stuff is included in the prompts for controlling the robot, it will make sense to chatgpt who is being the brain of the robot.\n\nHeres contextually relevant memories that the robot already has that you can use as extra context when creating your summary of the history: \n\n' + memories 
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]
    

def say_greeting(last_phrase):
    global chat_history
    global last_time
    now = datetime.now()
    try:
        with open('visual.txt','r') as file:
            visual = file.read()
    except:
        visual = 'no visual data yet'
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    text = str(send_text_to_gpt4_convo(chat_history, last_phrase, visual))
    chat_history.append('Time: ' + str(the_time) + ' - User Greeting: ' + last_phrase)  # add response to chat history
    chat_history.append('Time: ' + str(the_time) + ' - AI Greeting: ' + text)  # add response to chat history
    last_time = time.time()

    file = open('playback_text.txt', 'w+')
    file.write(text)
    file.close()
    
    

def get_last_phrase():

    try:
        with open('last_phrase.txt', 'r') as file:
            last_phrase = file.read().strip().lower()
        if last_phrase != '':
            with open('last_phrase.txt', 'w') as file:
                file.write('')  # Clear the content after reading
            return last_phrase
        else:
            return ''
    except Exception as e:
        print(f"Error reading last phrase from file: {e}")
        return ''
def get_last_phrase2():

    try:
        with open('last_phrase2.txt', 'r') as file:
            last_phrase2 = file.read().strip().lower()
        if last_phrase2 != '':
            with open('last_phrase2.txt', 'w') as file:
                file.write('')  # Clear the content after reading
            return last_phrase2
        else:
            return ''
    except Exception as e:
        print(f"Error reading last phrase from file: {e}")
        return ''
# Load YOLOv4-tiny configuration and weights
net = cv2.dnn.readNet("yolov4-tiny.cfg", "yolov4-tiny.weights")

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# Adjust the index extraction to handle the nested array structure
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True

# Register the signal handler for SIGINT (Ctrl-C)
signal.signal(signal.SIGINT, signal_handler)


move_stopper = False
def convo_loop():
    print('convo thread start')
    global chat_history
    global stop_threads
    global camera_vertical_pos
    global topics
    global move_stopper
    while not stop_threads:


 


        try:
            now = datetime.now()
            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
            last_phrase = get_last_phrase()
            if last_phrase != '':
                move_stopper = True
                print('Prompt heard from microphone: ' + last_phrase)
                last_time = time.time()
                
                last_phrase_now = 'You just heard this prompt through your microphone: ' + last_phrase
                with open('visual_data.txt','r') as file:
                    visual = file.read()
                speech_response = str(send_text_to_gpt4_convo(chat_history, last_phrase_now, visual)).replace('Response Choice: ','')
                if speech_response.strip().lower().replace(' ','').replace('.','') != 'stayingsilent':
                    file = open('playback_text.txt', 'w+')
                    file.write(speech_response)
                    file.close()
                else:
                    pass
                print('Response From GPT4: ' + speech_response)
                chat_history.append(str(the_time) + ' - Prompt heard from microphone at this timestamp: ' + last_phrase)
                chat_history.append('Time: ' + str(the_time) + ' - Speech Response From GPT4 at this timestamp: ' + speech_response)
                file = open('last_phrase2.txt', 'w+')
                file.write(last_phrase)
                file.close()
            else:
                time.sleep(0.1)
                continue
              
                
        
        except Exception as e:
            print('convo loop outer error: ' + str(e))
            print(traceback.format_exc())
        time.sleep(0.1)

        
def movement_loop(camera, raw_capture):
    global chat_history
    global frame
    global stop_threads
    global net
    global output_layers
    global classes
    global move_stopper
    global camera_vertical_pos
    global topics
    ina219 = INA219(addr=0x42)
    last_time = time.time()
    total_topics_list = []
    print('movement thread start')
    while not stop_threads:
        try:
            with open('current_history.txt','w+') as file:
                file.write('\n'.join(chat_history))
            last_phrase2 = get_last_phrase2()
            if last_phrase2 != '':
                print('Last phrase now on movement loop: ' + last_phrase2)
                move_stopper = False
                last_phrase2 = 'You just heard this from your microphone so THIS IS THE MAIN PART OF THE PROMPT (You already responded with speech and your speech response was the most recent assistant response in the chat history, so now you must decide how to respond to this with movement): ' + last_phrase2
            else:
                pass
            if move_stopper == True:
                time.sleep(0.1)
                continue
            else:
                pass
            frame = capture_image(camera, raw_capture)
            cv2.imwrite('this_temp.jpg', frame)
            # Open the image file
            img = frame
            if img is None:
                print("Image not found or unable to load. Check the path and try again.")


            #do yolo and gpt visual threads
           
            gpt_visual_thread = threading.Thread(target=get_gtp_visual)
            yolo_thread = threading.Thread(target=yolo_detect)

            gpt_visual_thread.start()
            yolo_thread.start()
            gpt_visual_thread.join()
            yolo_thread.join()
            current = ina219.getCurrent_mA() 
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            with open('batt_per.txt','w+') as file:
                file.write(str(per))
            now = datetime.now()
            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    
            
            if current > 0.0 or per < 15.0 or interrupted:
                try:
                    stop_threads = True
                    last_time = time.time()
                    summary = get_summary(chat_history, total_topics_list)
                    topic_index = 0
                    while True:
                        try:
                            current_topic = total_topics_list[topic_index].replace('.','')
                            
                            now = datetime.now()
                            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                            file = open('memories/'+current_topic + '.txt', 'a+')
                            file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                            file.close()
                            topic_index += 1
                            if topic_index >= len(topics):
                                break
                            else:
                                continue
                        except Exception as e:
                            print(e)
                            topic_index += 1
                            if topic_index >= len(topics):
                                break
                            else:
                                continue
                    print('ending convo')
                    chat_history = []
                    if interrupted:
                        print("Exiting due to Ctrl-C")
                        os._exit(0)
                except Exception as e:
                    print(e)
            else:
                pass
            try:
                if frame is not None:
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

                    while True:
                        try:
                            distance = int(read_distance_from_arduino())

                            
                            break
                        except:
                            time.sleep(0.1)
                            continue
                    movement_response = str(send_text_to_gpt4_move(chat_history, topics, per, distance, last_phrase2)).replace('Response Choice: ','')
                    try:
                        print("\nPercent:       {:3.1f}%".format(per))
                        print('\nCurrent Distance: ' + str(distance) + ' cm')
                        print('\nResponse Choice: '+ movement_response.split('~~')[0].strip().replace('.',''))
                        print('\nReasoning: '+ movement_response.split('~~')[1].strip())
                        last_topics = topics
                        
                        topics = movement_response.split('~~')[2].strip().replace('.','').split(', ')
                        print('\nTopics: ' + str(topics))
                        chat_history.append(str(the_time) + ' - Distance reading at this timestamp from the HCSR-04 sensor that measures the distance of any object that yolo detects is in the absolute middle center of the image (Distance in cm): ' + str(distance))

                        chat_history.append('Time: ' + str(the_time) + ' - Movement Choice at this timestamp: ' + movement_response.split('~~')[0])  # add response to chat history
                        chat_history.append('Time: ' + str(the_time) + ' - Reasoning For This Movement Choice: ' + movement_response.split('~~')[1])
                    except:
                        pass
                    
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    
                    
                    
                    #RESPONSE CHOICES LOOP
                    response_list = movement_response.replace('Movement Choice at this timestamp:','').split('~~')[0].replace('.','').replace('Successful','').split(', ')
                    first_time = True
                    while True:
                        last_phrase_now = get_last_phrase2()
                        if last_phrase_now != '' or move_stopper == True:
                            with open('last_phrase2.txt','w') as f:
                                f.write(last_phrase_now)
                            break
                        else:
                            pass
                        time.sleep(0.25)

                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        if first_time != True:
                            frame = capture_image(camera, raw_capture)
                            cv2.imwrite('this_temp.jpg', frame)
                            # Open the image file
                            if frame is None:
                                print("Image not found or unable to load. Check the path and try again.")
                                time.sleep(1)
                                continue
                            print('got image')
                            gpt_visual_thread = threading.Thread(target=get_gtp_visual)
                            yolo_thread = threading.Thread(target=yolo_detect)

                            gpt_visual_thread.start()
                            yolo_thread.start()
                            gpt_visual_thread.join()
                            yolo_thread.join()
                            print('got visual stuff')
                            while True:
                                try:
                                    distance = int(read_distance_from_arduino())
                                    print('got distance')
                                    chat_history.append(str(the_time) + ' - Distance reading at this timestamp from the HCSR-04 sensor that measures the distance of any object that yolo detects is in the absolute middle center of the image (Distance in cm): ' + str(distance))
                                    break
                                except Exception as e:
                                    print(e)
                                    time.sleep(0.1)
                                    continue
                        else:
                            first_time = False
                        
                        current_response = response_list[0].strip().lower().replace(' ', '')
                        print(current_response)
                        if current_response == 'moveforward1inch' or current_response == 'moveforwardoneinch':
                            if camera_vertical_pos != 'forward':
                                print('move forward 1 inch failed. not looking forward and center')
                                chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Camera Must Be Centered and forward before moving forward. Deleting list of moves and prompting for new moves.')
                                response_list = []
                            elif distance < 20.0:
                                print('move forward 1 inch failed. Too close to obstacle to move forward anymore')
                                chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Too close to obstacle to move forward anymore. Deleting list of moves and prompting for new moves.')
                                response_list = []
                            else:
                                send_data_to_arduino(["w"], arduino_address)
                                time.sleep(0.1)
                                send_data_to_arduino(["x"], arduino_address)
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Inch')
                        elif current_response == 'moveforward1foot' or current_response == 'moveforwardonefoot':
                            if camera_vertical_pos != 'forward':
                                print('Move forward 1 foot Failed. Not looking forward and center')
                                chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Camera Must Be Centered and forward before moving forward. Deleting list of moves and prompting for new moves.')
                                response_list = []
                            elif distance < 35.0:
                                print('move forward 1 foot failed. Too close to obstacle to move forward that far')
                                chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Too close to obstacle to move forward that far. Deleting list of moves and prompting for new moves.')
                                response_list = []
                            else:
                                send_data_to_arduino(["w"], arduino_address)
                                time.sleep(0.3)
                                send_data_to_arduino(["x"], arduino_address)
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Foot')
                        elif current_response == 'movebackward':
                            if distance > 10.0:
                                print('Move Backward Failed. You are not close enough to need to move backward. You should turn instead.')
                                chat_history.append('Time: ' + str(the_time) + ' - Move Backward Failed. You are not close enough to need to move backward. You should turn instead. Deleting list of moves and prompting for new moves.')
                                response_list = []
                            else:
                                send_data_to_arduino(["s"], arduino_address)
                                time.sleep(0.2)
                                send_data_to_arduino(["x"], arduino_address)
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Backward')
                        elif current_response == 'bigturnleft' or current_response == 'movebigturnleft':
                            send_data_to_arduino(["a"], arduino_address)
                            time.sleep(0.3)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Big Turn Left')
                        elif current_response == 'smallturnleft' or current_response == 'movesmallturnleft':
                            send_data_to_arduino(["a"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Small Turn Left')
                        elif current_response == 'bigturnright' or current_response == 'movebigturnright':
                            send_data_to_arduino(["d"], arduino_address)
                            time.sleep(0.3)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Big Turn Right was successful')
                        elif current_response == 'smallturnright' or current_response == 'movesmallturnright':
                            send_data_to_arduino(["d"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Small Turn Right')
                        elif current_response == 'lookup':
                            if camera_vertical_pos == 'up':
                                chat_history.append('Time: ' + str(the_time) + ' - Look Up Failed: Camera Is Already Looking Up. Deleting list of moves and prompting for new moves.')
                                print('Look Up Failed. Already looking up')
                                response_list = []
                            else:
                                send_data_to_arduino(["2"], arduino_address)
                                time.sleep(1.5)
                                camera_vertical_pos = 'up'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully looked up so camera and distance sensor are currently looking up.')
                        elif current_response == 'lookforward':
                            if camera_vertical_pos == 'forward':
                                chat_history.append('Time: ' + str(the_time) + ' - Look Forward Failed: Camera Is Already Looking forward. Deleting list of moves and prompting for new moves.')
                                print('Look Forward failed. Already looking Forward')
                                response_list = []
                            else:
                                send_data_to_arduino(["1"], arduino_address)
                                time.sleep(1.5)
                                camera_vertical_pos = 'forward'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully looked forward.')
                        elif current_response == 'endconversation':
                            stop_threads = True
                            last_time = time.time()
                            summary = get_summary(chat_history, total_topics_list)
                            topic_index = 0
                            while True:
                                try:
                                    current_topic = total_topics_list[topic_index].replace('.','')
                        
                                    now = datetime.now()
                                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                                    file = open('memories/'+current_topic + '.txt', 'a+')
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
                            break
                        elif current_response == 'nomovement':
                            now = datetime.now()
                            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                            chat_history.append('Time: ' + str(the_time) + ' - Response choice was No Movement so not moving.')
                        elif current_response == 'saysomething':
                            now = datetime.now()
                            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                            chat_history.append('Time: ' + str(the_time) + ' - Attempting to say:'+movement_response.split('~~')[1])
                            with open('self_response.txt','w') as f:
                                f.write(movement_response.split('~~')[1])
                        else:
                            now = datetime.now()
                            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                            chat_history.append('Time: ' + str(the_time) + ' - Response failed. Did not choose an actual Response Choice from the list. Here is what you responded with so dont do it again: ' + current_response + '\n\nDeleting list of moves and prompting for new moves.')
                            print('failed response')
                            response_list = []
                        print(response_list)
                        if response_list != []:
                            del response_list[0]
                        else:
                            pass
                        print(response_list)
                        if response_list == []:
                            time.sleep(0.25)
                            break
                        else:
                            continue
                    
                    
                    if topics != last_topics and topics != []:
                        
                        last_time = time.time()

                    
                        
                        
                        home_directory = os.path.expanduser('~')
                        filenames = [os.path.splitext(file)[0] for file in os.listdir(home_directory) if os.path.isfile(os.path.join(home_directory, file))]
                        filenames_string = ', '.join(filenames)

                        all_topics = get_topics(topics, filenames_string).replace('.','')
                        topics = all_topics.split(', ')
                        #add to total_topics_list
                        topic_index = 0
                        while True:
                            total_topics_list.append(topics[topic_index])
                            topic_index += 1
                            if topic_index >= len(topics):
                                break
                            else:
                                continue
                        print('Updated topics: '+str(topics))
                    else:
                        topics = last_topics
                    
            
            except Exception as e:
                print(e)
            
            
        except:
            print(traceback.format_exc())
            while True:
                continue
        time.sleep(0.1)
if __name__ == "__main__":
    try:
        be_still = True
        last_time_seen = time.time()
        transcribe_thread = threading.Thread(target=listen_and_transcribe)  # Adding the transcription thread
        transcribe_thread.start() 
        camera = PiCamera()
        raw_capture = PiRGBArray(camera)
        camera.resolution = (416, 416)
        camera.framerate = 10
        time.sleep(1)

        send_data_to_arduino(["4"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        print('waiting to be called')
        ina219 = INA219(addr=0x42)
        while True:
            time.sleep(0.25)
            current = ina219.getCurrent_mA()
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            print('\nBattery Percent: ')
            print(per)
            with open('batt_per.txt','w+') as file:
                file.write(str(per))
            chat_history = []
            stop_threads = False
            frame = capture_image(camera, raw_capture)
            cv2.imwrite('this_temp.jpg', frame)
            last_phrase = get_last_phrase()
            
            try:
                the_index_now = last_phrase.split(' ').index('robot')
                name_heard = True
            except:
                name_heard = False
   
            if name_heard == True:

                print("Name heard, initializing...")
                #do yolo and gpt visual threads

                gpt_visual_thread = threading.Thread(target=get_gtp_visual)
                yolo_thread = threading.Thread(target=yolo_detect)
                gpt_visual_thread.start()
                yolo_thread.start()
                gpt_visual_thread.join()
                yolo_thread.join()
                print('Saying Greeting')
                with open('last_phrase.txt', 'w') as file:
                    file.write('')
                say_greeting(last_phrase)
                now = datetime.now()
                the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                movement_thread = threading.Thread(target=movement_loop, args=(camera, raw_capture))
                convo_thread = threading.Thread(target=convo_loop)

                movement_thread.start()
                convo_thread.start()
                print('threads started')
                movement_thread.join()
                convo_thread.join()
                
            else:
                
                #do yolo
                yolo_thread = threading.Thread(target=yolo_detect)
                yolo_thread.start()
                yolo_thread.join()
                with open('output.txt','r') as file:
                    yolo_detections = file.readlines()
                print("YOLO Detections:")
                print('\n'.join(yolo_detections))
                yolo_index = 0
                while True:
                    try:
                        current_detection = yolo_detections[yolo_index].lower()
                        if 'person' in current_detection:
                            last_time_seen = time.time()
                            be_still = False
                            break
                        else:
                            yolo_index += 1
                            if yolo_index >= len(yolo_detections):
                                break
                            else:
                                continue
                    except:
                        break
                if current >= 0.0 or be_still == True or 0==0:
                    pass
                else:
                    if 'person' in current_detection:
                        print('person seen')
                        #follow any human seen
                        if 'small turn left' in current_detection:
                            send_data_to_arduino(["a"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                        elif 'small turn right' in current_detection:
                            send_data_to_arduino(["d"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)       
                        else:
                            pass
                
                    else:
                        print('No person seen. Doing object avoidance mode.')
                        #do object avoidance if no human seen
                        if time.time()>last_time_seen+180.0:
                            be_still = True
                            continue
                        else:
                            be_still = False
                            
                            
                        distance = int(read_distance_from_arduino())
                        print('\nDistance sensor: ')
                        print(str(distance)+' cm')
                        if distance < 20.0 and distance >= 10.0:
                            rando_list = [1,2,3,4]
                            rando_index = random.randrange(len(rando_list))
                            rando_num = rando_list[rando_index]
                            if rando_num == 1:
                                send_data_to_arduino(["a"], arduino_address)
                                time.sleep(0.3)
                                send_data_to_arduino(["x"], arduino_address)
                            elif rando_num == 2:
                                send_data_to_arduino(["a"], arduino_address)
                                time.sleep(0.1)
                                send_data_to_arduino(["x"], arduino_address)
                            elif rando_num == 3:
                                send_data_to_arduino(["d"], arduino_address)
                                time.sleep(0.3)
                                send_data_to_arduino(["x"], arduino_address)
                            else:
                                send_data_to_arduino(["d"], arduino_address)
                                time.sleep(0.1)
                                send_data_to_arduino(["x"], arduino_address)
                        elif distance < 35.0 and distance >= 20.0:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                        elif distance < 10.0:
                            send_data_to_arduino(["s"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.3)
                            send_data_to_arduino(["x"], arduino_address)


                        
            time.sleep(0.1)


    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
    finally:
        camera.close()
