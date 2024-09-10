import re
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
import signal
import traceback
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
def get_position_description(x, y, width, height):
    """Return a text description of the position based on coordinates."""
    if x < width / 3:
        horizontal = "a small turn to the left"
    elif x > 2 * width / 3:
        horizontal = "a small turn to the right"
    else:
        horizontal = "center"
    
    if y < height / 3:
        vertical = "look up"
    elif y > 2 * height / 3:
        vertical = "look down"
    else:
        vertical = "center"

    if horizontal == "center" and vertical == "center":
        return "center of the image"
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
            file = open('last_phrase.txt', 'w+')
            file.write(text)
            file.close()
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
                if confidence > 0.37:
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

        # Display and save the processed image

        cv2.imwrite("output.jpg", img)
    except Exception as e:
        print(e)
      
def get_summary_prompt(yolo_stuff, the_time, percent, camera_vertical_pos, camera_horizontal_pos, current_distance, history, memories, base64_image):
    with open('output.txt','r') as file:
        yolo_stuff = file.read()
    with open('visual.txt','r') as file:
        gpt_visual_data = file.read()
    

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


    this_prompt = 'Your response cannot be longer than a few sentences. The current time and date is: ' + str(the_time) + ' \n\n You are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT-4o, so you have image and text input abilities). \n\n Your battery percent is at ' + format(percent, '.2f') + ' (Make sure you say something about your battery being low if it is below 30 percent and you havent already said something recently). \n\n Your camera that took the image where your yolo and other visual data is from is only a couple inches off the ground so make sure you adjust your mental perspective to compensate for the camera being so low to the ground (if it looks like you are more than a foot tall then you may be on top of something so be careful about ledges and other fall risks. Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + '. Your ultrasonic sensor that detects the distance of whatever is in the center of the included image from the camera is giving a reading of ' + current_distance + ' centimeters away (It is always pointing the same direction as the camera). Here is the text description from YOLO for what the camera currently sees and the movements that would be necessary to center your camera on each specific object: \n\n'+str(yolo_stuff)+' \n\n And here is a general description of what your camera currently sees: '+gpt_visual_data+' \n\n Based on all of this data, make a summary of your current goals, aspirations, tasks, and just the whole situation and context so in the next prompt that is given your summary, it can figure out how to control the robot based on your recommendation (Your recommendation choices are Move Forward One Inch, Move Forward One Foot, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Say Something, Look Up, Look Down, No Movement, End Conversation. You can only give the summary and a recommendation choice specifically from this list):\n\nCurrent Data History for this session (Oldest to newest): ' + '\n'.join(history) + ' \n\n Your current contextually relevant memories from actual past experiences that you have had (You have memories about other stuff too but its not relevant right now so they were not included in this prompt): ' + memories
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": this_prompt
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    summary_response = str(response.json()["choices"][0]["message"]["content"])
    with open('summary.txt','w+') as file:
        file.write(summary_response)
    return summary_response
      
        

def get_gtp_visual():


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
        "max_tokens": 100
    }
  
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    summary_response = str(response.json()["choices"][0]["message"]["content"])
    with open('visual.txt','w+') as file:
        file.write(summary_response)



def send_text_to_gpt4_move(history, topics, percent, current_distance):
    global camera_horizontal_pos
    global camera_vertical_pos
    now = datetime.now()
    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
    with open('output.txt','r') as file:
        yolo_detections = file.read()
    with open('visual.txt','r') as file:
        gpt_visual_data = file.read()
    print('Yolo Detections: \n'+str(yolo_detections))
    print('GPT Visual Description: \n'+str(gpt_visual_data))
    try:
        with open('output.jpg', "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except:
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

    chat_history.append(str(the_time) + ' - Distance reading at this timestamp from the HCSR-04 sensor that measures the distance of any object that yolo detects is in the absolute middle center of the image (Distance in cm): ' + str(current_distance))
    if int(current_distance) <= 25:
        response_choices = 'Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Look Up, Look Down, No Movement, End Conversation. You are currently too close to an obstacle so you are not able to move forward. This would be a good time to look around with your camera and figure out which direction to move afterwards. Dont forget to put you camera back to down and center before moving.'
    elif int(current_distance) <= 50 and int(current_distance) > 25:
        response_choices = 'Move Forward One Inch, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Look Up, Look Down, No Movement, End Conversation. You are currently very close to an obstacle. This would be a good time to look around with your camera and figure out which direction to move afterwards. Dont forget to put you camera back to down and center before moving.'
    else:
        response_choices = 'Move Forward One Inch, Move Forward One Foot, Move Backward, Small Turn Left, Big Turn Left, Small Turn Right, Big Turn Right, Look Up, Look Down, No Movement, End Conversation. Your camera must be Down and to the Center before you are allowed to choose to move forward.'
    chat_history2 = history
    while True:
        if len(chat_history2) > 1000:
            del chat_history2[0]
            continue
        else:
            break
    summary_from_gpt = get_summary_prompt(yolo_detections, the_time, percent, camera_vertical_pos, camera_horizontal_pos, current_distance, chat_history, memories, base64_image)
    summary_prompt = 'You are a 4 wheeled mobile robot. Here is the last 1000 entries from the current chat history/robot internal workings history (it is all timestamped and in order from oldest to newest): '+'\n'.join(chat_history2)+' \n\n You also have a camera and an HCSR04 distance sensor pointing the same direction as the camera at whatever object that yolo says is in the absolute middle center of the image. Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + '. Here is the text description from YOLO and the movements that would be necessary to center your camera on each specific object: \n\n '+str(yolo_detections)+ ' \n\n Here is a general visual description of what you currently see through you camera (this is in addition to YOLO data): '+gpt_visual_data+'. \n\n Here is the summary of the current situation at hand: '+summary_from_gpt+' \n\n You should most likely follow the recommendation made in this summary. \n\n Your response choices are: ' + response_choices + ' \n\n Your response choice Absolutely must be something from this list and worded EXACTLY THE SAME!!!!!!! If you dont use the exact words from this list of choices then your Response Choice will be incorrect, even if it seems contextually correct. My code needs the exact phrases word for word or it wont work. \n\nYour answer must be the Response Choice followed by ~~ with a space on each side, and then followed by your reasoning for your choice and then followed by ~~ with a space on each side and then followed by the topics that all of this data covers (you can only say the actual topics, dont preface label this section, literally only put the topics, each separated by a comma and space). \n\n On the previous prompt, you chose the topics of ' + str(topics) + ' so you can either stick with those, add or remove topics, or totally change the list. Your past memories are saved to topic files, so we use the current topics to know which files to open on the backend so your memories can be included in each prompt for context improvement. \n\n Also, if you want to move to an object, make sure you turn towards objects to center on it first before moving towards it, and if you dont see an object you are looking for then you should choose to turn moreso than to move forward. If no areas are drivable, then turn, because you turn in place like a tank so it is ok to turn if there are no drivable areas. \n\n If you feel like the list of current topics isnt correct, then provide a new list, but if the list of topics matches up to all of the convo and action history then return the unchanged list. \n\n Also, your response must be formatted perfectly and your Response Choice must be worded exactly the same as one of the options from the list of Response Choices. You absolutely must format your response correctly with how i mentioned the ~~ earlier. \n\n Prompt Example: Response Choice ~~ Reason for choice ~~ Topic list  \n\n You absolutely cannot put a preface label on your response (Like you cant start your response with Response Choice: .... you have to just say the choice). \n\n Also dont forget that moving forward or backward most likely wont bring things into view, but turning most likely will bring things into view, unless they are just not in the room you are in, in which case you would need to go explore around to find whatever you are trying to find. \n\n And as a last reminder, your response choice has to be worded exactly the same as your choices from the provided list, you must use the exact same words on your response choice.'
    
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": summary_prompt
                    }
                ]
            }
        ],
        "max_tokens": 100
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
 
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
            current_topic = topics[topic_index]
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

    try:
        with open('summary.txt','r') as file:
            summary_response = file.read()
    except:
        summary_response = 'No summary yet'
    with open('batt_per.txt','r') as file:
        percent = file.read()
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a 4 wheeled mobile raspberry pi and arduino robot that is fully controlled by ChatGPT (Specifically GPT4o). \n\n Your battery percent is: "+format(percent, '.1f')+" \n\n Here is chat history/robot internal workings history for this session (it is all timestamped and in order from oldest to newest): "+'\n'.join(history)+' \n\n Here is data from your memory from past conversations and experiences that is contextually relevant currently: '+memories+' \n\n The current time and date is: ' + str(the_time) + ' \n\n The current yolo Visual data from your camera and the movements that would be necessary to center your camera on each specific object: '+yolo_detections+' \n\n General description of what is currently seen in the camera image (This is in addition to the YOLO data): '+gpt_visual_data+' \n\n Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos + '. \n\nCurrent prompt (This is what you are actually responding to, while using all other data for context. Your response cannot be longer than a few sentences at max): ' + text
                    }
                ]
            }
        ],
        "max_tokens": 77
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
                        "text": '\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). Summarize this Movement, Chat, Sensor/Battery, and Visual Data History from the most recent conversation/event youve had (Oldest to newest): ' + str(history) + ' \n\n The summary should be, at most, 50% of the original length or the original data, if not shorter if it can be summarized accurately in an even shorter way. Make sure to include all important facts, events, goals, conversational information, learned information about the world, and any other data worth keeping in the robots memory so it has useful information to use on future prompts (The prompts that choose the robots actions and when to speak are given relevant topic memories so this summary has to include good relevant knowledge). These summaries are what becomes the robots memory that makes the robot more lifelike and grow and become an individual, so word it to where future when this stuff is included in the other prompts for controlling the robot, it will make sense to chatgpt who is being the brain of the robot.\n\nHeres contextually relevant memories that the robot already has that you can use as extra context when creating your summary of the history: \n\n' + memories
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]

def summarize_all_memories():
    
    memdex = 0
    #get list of all memory file names
    while True:
        with open('memories/'+filenames[memdex]) as file:
            current_memory = file.read()
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
                            "text": '\n\nYou are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). \n\n Summarize this: ' + current_memory
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        memdex += 1
        if memdex >= len(filenames):
            break
        else:
            continue
    

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
    wav_file = f"/home/ollie/Desktop/Robot/audio/speech_output.wav"

    file = open('playback_text.txt', 'w+')
    file.write(text)
    file.close()
    
    return wav_file

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



def convo_loop():
    global chat_history
    global stop_threads
    global camera_horizontal_pos
    global camera_vertical_pos
    global topics
    while not stop_threads:


 


        try:
            now = datetime.now()
            the_time = now.strftime("%d/%m/%Y %H:%M:%S")
            last_phrase = get_last_phrase()
            if last_phrase != '':
                print('Prompt heard from microphone: ' + last_phrase)
                last_time = time.time()
                chat_history.append(str(the_time) + ' - Prompt heard from microphone at this timestamp: ' + last_phrase)
                last_phrase = 'You just heard this prompt through your microphone: ' + last_phrase
                with open('visual_data.txt','r') as file:
                    visual = file.read()
                speech_response = str(send_text_to_gpt4_convo(chat_history, last_phrase, visual)).replace('Response Choice: ','')
                wav_file = f"/home/ollie/Desktop/Robot/audio/speech_output.wav"
                file = open('playback_text.txt', 'w+')
                file.write(speech_response)
                file.close()
                print('Response From GPT4: ' + speech_response)
                chat_history.append('Time: ' + str(the_time) + ' - Speech Response From GPT4 at this timestamp: ' + speech_response)
            else:
                time.sleep(0.1)
                continue
              
                
        
        except Exception as e:
            print('convo loop outer error: ' + str(e))
            print(traceback.format_exc())
        time.sleep(0.1)

        
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
        with open('current_history.txt','w+') as file:
            file.write('\n'.join(chat_history))

        frame = capture_image(camera, raw_capture)
        cv2.imwrite('this_temp.jpg', frame)
        # Open the image file
        img = frame
        if img is None:
            print("Image not found or unable to load. Check the path and try again.")
        try:
            with open('this_temp.jpg', "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8') 
        except:
            time.sleep(1)
            continue
        #do yolo and gpt visual threads
       
        gpt_visual_thread = threading.Thread(target=get_gtp_visual)
        yolo_thread = threading.Thread(target=yolo_detect)

        gpt_visual_thread.start()
        yolo_thread.start()
        gpt_visual_thread.join()
        yolo_thread.join()
        if mode == 'convo':
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
    
            print("Percent:       {:3.1f}%".format(per))
            if current > 0.0 or per < 15.0 or interrupted:
                stop_threads = True
                last_time = time.time()
                summary = get_summary(chat_history, topics)
                topic_index = 0
                while True:
                    try:
                        current_topic = topics[topic_index]
                        
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
                if interrupted:
                    print("Exiting due to Ctrl-C")
                    os._exit(0)
            else:
                pass
            try:
                if frame is not None:
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

                    while True:
                        try:
                            distance = int(read_distance_from_arduino())
                            print('Current Distance: ' + str(distance) + ' cm')
                            break
                        except:
                            time.sleep(0.1)
                            continue
                    print('Your camera is currently pointed ' + camera_vertical_pos + ' and to the ' + camera_horizontal_pos)
                    movement_response = str(send_text_to_gpt4_move(chat_history, topics, per, distance)).replace('Response Choice: ','')
                    try:
                        print('Response Choice: '+ movement_response.split(' ~~ ')[0].strip())
                        print('Reasoning: '+ movement_response.split(' ~~ ')[1].strip())
                        last_topics = topics
                        
                        topics = movement_response.split(' ~~ ')[2].strip().split(', ')
                        print('Topics: ' + str(topics))
                        chat_history.append('Time: ' + str(the_time) + ' - Movement Choice at this timestamp: ' + movement_response.split(' ~~ ')[0])  # add response to chat history
                        chat_history.append('Time: ' + str(the_time) + ' - Reasoning For This Movement Choice: ' + movement_response.split(' ~~ ')[1])
                    except:
                        pass
                    
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                    
                    if movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforward1inch' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforwardoneinch':
                        if camera_horizontal_pos != 'center' or camera_vertical_pos != 'down':
                            print('move failed. not looking down and center')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Camera Must Be Centered and Down before moving forward')
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforward1foot' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'moveforwardonefoot':
                        if camera_horizontal_pos != 'center' or camera_vertical_pos != 'down':
                            print('Move Failed. Not looking down and center')
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
                        time.sleep(0.3)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'smallturnleft' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movesmallturnleft':
                        send_data_to_arduino(["a"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'bigturnright' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movebigturnright':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.3)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'smallturnright' or movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'movesmallturnright':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["x"], arduino_address)
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookup':
                        if camera_horizontal_pos != 'center':
                            print('Look Up Failed. Camera not centered')
                            chat_history.append('Time: ' + str(the_time) + ' - Look Up Failed: Camera Must Be Centered Before Looking Up')
                        elif camera_vertical_pos == 'up':
                            chat_history.append('Time: ' + str(the_time) + ' - Look Up Failed: Camera Is Already Looking Up')
                            print('Look Up Failed. Already looking up')
                        else:
                            send_data_to_arduino(["2"], arduino_address)
                            time.sleep(1.5)
                            camera_vertical_pos = 'up'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookleft':
                        if camera_vertical_pos != 'down':
                            print('Look left failed. Camera not down')
                            chat_history.append('Time: ' + str(the_time) + ' - Look Left Failed: Camera Must Be Down Before Looking Left')
                        elif camera_horizontal_pos == 'left':
                            print('Look left failed. Already looking left')
                            chat_history.append('Time: ' + str(the_time) + ' - Look Left Failed: Camera Is Already Looking Left')
                        else:
                            send_data_to_arduino(["3"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'left'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookright':
                        if camera_vertical_pos != 'down':
                            print('Look right failed. Camera not down')
                            chat_history.append('Time: ' + str(the_time) + ' - Look Right Failed: Camera Must Be Down Before Looking Right')
                        elif camera_horizontal_pos == 'right':
                            print('Look right failed. Already looking right')
                            chat_history.append('Time: ' + str(the_time) + ' - Look Right Failed: Camera Is Already Looking Right')
                        else:
                            send_data_to_arduino(["5"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'right'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookcenter':
                        if camera_horizontal_pos == 'center':
                            print('Look center failed. Already looking center')
                            chat_history.append('Time: ' + str(the_time) + ' - Turn Camera Center Failed: Camera Is Already Looking Center')
                        else:
                            send_data_to_arduino(["4"], arduino_address)
                            time.sleep(1.5)
                            camera_horizontal_pos = 'center'
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'lookdown':
                        if camera_vertical_pos == 'down':
                            chat_history.append('Time: ' + str(the_time) + ' - Look Down Failed: Camera Is Already Looking Down')
                            print('Look Down failed. Already looking down')
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
                    elif movement_response.split(' ~~ ')[0].strip().lower().replace(' ', '') == 'nomovement':
                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response choice was No Movement so not moving.')
                    else:
                        now = datetime.now()
                        the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response failed. Did not choose an actual Response Choice from the list. Here is what you responded with so dont do it again: ' + movement_response)
                    
                    
                    
                    if topics != last_topics:
                        
                        last_time = time.time()
                        summary = get_summary(chat_history, last_topics)
                        topic_index = 0
                        while True:
                            try:
                                current_topic = last_topics[topic_index]
                                
                                now = datetime.now()
                                the_time = now.strftime("%d/%m/%Y %H:%M:%S")
                                file = open('memories/'+current_topic + '.txt', 'a+')
                                file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                file.close()
                                topic_index += 1
                                if topic_index >= len(last_topics):
                                    break
                                else:
                                    continue
                            except:
                                topic_index += 1
                                if topic_index >= len(last_topics):
                                    break
                                else:
                                    continue
                    
                        
                        
                        home_directory = os.path.expanduser('~')
                        filenames = [os.path.splitext(file)[0] for file in os.listdir(home_directory) if os.path.isfile(os.path.join(home_directory, file))]
                        filenames_string = ', '.join(filenames)

                        all_topics = get_topics(topics, filenames_string)
                        topics = all_topics.split(', ')
                        print('Updated topics: '+str(topics))
                    else:
                        pass
                    
            
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

        send_data_to_arduino(["4"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        print('waiting to be called')
        while True:

            chat_history = []
            stop_threads = False
            human_detected = False
            frame = capture_image(camera, raw_capture)
            cv2.imwrite('this_temp.jpg', frame)
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
                    human_detected = True
                    now = datetime.now()
                    the_time = now.strftime("%d/%m/%Y %H:%M:%S")

                else:
                    pass
                time.sleep(0.1)

            movement_thread = threading.Thread(target=movement_loop, args=(camera, raw_capture))
            convo_thread = threading.Thread(target=convo_loop)

            movement_thread.start()
            convo_thread.start()
            movement_thread.join()
            convo_thread.join()
    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
    finally:
        camera.close()
