import re
import subprocess
import os
import numpy as np
import time
import requests
import threading
import bluetooth
import cv2
import smbus
from datetime import datetime
import base64
import traceback
import random
import speech_recognition as sr
import webrtcvad
import pyaudio
import json
from picamera2 import Picamera2
import math
import heapq
import glob

def clear_files_in_folder(folder_path):
    # Pattern to match all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))

    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted file: {file}")

# Example usage
folder_path = 'Pictures/'
clear_files_in_folder(folder_path)
move_set = []
camera_horizontal_fov = 160.0


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
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


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

# Dictionary of known object heights (in meters)
known_object_heights = {
    'person': 1.7,
    'bicycle': 1.0,
    'car': 1.5,
    'motorcycle': 1.1,
    'airplane': 10.0,      # Average for small aircraft
    'bus': 3.0,
    'train': 4.0,           # Per car
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
    'sports ball': 0.22,    # e.g., soccer ball
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

# Default height for unknown classes (in meters)
default_height = 1.0

# -----------------------------------
# 2. Camera Calibration Parameters
# -----------------------------------

# Camera parameters (these should be obtained from calibration)
focal_length_px = 2050 

# -----------------------------------
# 3. Distance Estimation Function
# -----------------------------------

def estimate_distance(focal_length, real_height, pixel_height):
    """
    Estimate the distance from the camera to the object using the pinhole camera model.
    
    Parameters:
    - focal_length (float): Focal length of the camera in pixels.
    - real_height (float): Real-world height of the object in meters.
    - pixel_height (float): Height of the object's bounding box in the image in pixels.
    
    Returns:
    - float: Estimated distance in meters.
    """
    if pixel_height == 0:
        return float('inf')  # Avoid division by zero
    return ((focal_length * real_height) / pixel_height)/6

# -----------------------------------
# 4. Position and Size Description Functions
# -----------------------------------

def get_position_description(x, y, width, height):
    """
    Return a text description of the position based on coordinates in a 5x5 grid.
    
    Parameters:
    - x (float): X-coordinate of the object's center in the image.
    - y (float): Y-coordinate of the object's center in the image.
    - width (float): Width of the image.
    - height (float): Height of the image.
    
    Returns:
    - str: Description of the object's position.
    """
    # Determine horizontal position in 5 sections
    if x < width / 5:
        horizontal = "Turn Left 45 Degrees"
    elif x < 2 * width / 5:
        horizontal = "Turn Left 15 Degrees"
    elif x < 3 * width / 5:
        horizontal = "already centered between left and right"
    elif x < 4 * width / 5:
        horizontal = "Turn Right 15 Degrees"
    else:
        horizontal = "Turn Right 45 Degrees"
    
    # Determine vertical position in 5 sections
    if y < height / 5:
        vertical = "Raise Camera Angle"
    elif y < 2 * height / 5:
        vertical = "Raise Camera Angle"
    elif y < 3 * height / 5:
        vertical = "already centered on the vertical"
    elif y < 4 * height / 5:
        vertical = "Lower Camera Angle"
    else:
        vertical = "Lower Camera Angle"
    
    # Combine the horizontal and vertical descriptions
    if horizontal == "already centered between left and right" and vertical == "already centered on the vertical":
        return "already centered on object"
    else:
        return f"{vertical} and {horizontal}"



with open("last_phrase2.txt","w+") as f:
    f.write('')
# -----------------------------------
# 5. Overlap Removal Function
# -----------------------------------

def remove_overlapping_boxes(boxes, class_ids, confidences):
    """
    Remove overlapping boxes of the same class, keeping only the one with the highest confidence.
    
    Parameters:
    - boxes (list): List of bounding boxes [x, y, w, h].
    - class_ids (list): List of class IDs corresponding to each box.
    - confidences (list): List of confidence scores corresponding to each box.
    
    Returns:
    - tuple: Filtered lists of boxes, class_ids, and confidences.
    """
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

# -----------------------------------
# 6. YOLO Detection Function with Distance Estimation
# -----------------------------------



def yolo_detect():
    global chat_history
    """
    Perform YOLO object detection on an image, estimate distances to detected objects,
    and update chat history with descriptive information.
    
    Returns:
    - None
    """
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    
    # Dictionary to store times
    time_logs = {}
    
    # Start time tracking
    total_start = time.time()

    try:
        # Load image
        start = time.time()
        img = cv2.imread('this_temp.jpg')
        if img is None:
            print("Error: Image 'this_temp.jpg' not found.")
            return
        height, width, channels = img.shape
        time_logs['Load Image'] = time.time() - start

        # Prepare the image for YOLO
        start = time.time()
        blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        time_logs['YOLO Forward'] = time.time() - start

        # Extract bounding boxes and confidences
        start = time.time()
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
        time_logs['Extract Bounding Boxes'] = time.time() - start

        # Remove overlapping boxes
        start = time.time()
        boxes, class_ids, confidences = remove_overlapping_boxes(boxes, class_ids, confidences)
        time_logs['Remove Overlapping Boxes'] = time.time() - start

        # Annotate image and generate descriptions
        start = time.time()
        descriptions = []

        # Define center grid rectangle
        center_x_min = 2 * width / 5
        center_x_max = 3 * width / 5
        center_y_min = height / 3
        center_y_max = 2 * height / 3
        center_grid_area = (center_x_max - center_x_min) * (center_y_max - center_y_min)

        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]).lower()
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 10)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Get real-world height
            real_height = known_object_heights.get(label)
            if real_height is None:
                print(f"Warning: Real-world height for class '{label}' not found. Using default height.")
                real_height = default_height

            # Calculate the percentage of the center grid covered by the bounding box
            # Bounding box coordinates
            box_x_min = x
            box_y_min = y
            box_x_max = x + w
            box_y_max = y + h

            # Compute intersection with center grid
            inter_x_min = max(box_x_min, center_x_min)
            inter_y_min = max(box_y_min, center_y_min)
            inter_x_max = min(box_x_max, center_x_max)
            inter_y_max = min(box_y_max, center_y_max)

            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

            # Calculate percentage of center grid covered by bounding box
            percentage_covered = (inter_area / center_grid_area) * 100

            if percentage_covered > 30:
                # Read distance from current_distance.txt
                try:
                    with open('current_distance.txt', 'r') as f:
                        distance = float(f.read())/100.0
                except Exception as e:
                    print(f"Error reading current_distance.txt: {e}")
                    # Fall back to estimating distance
                    distance = estimate_distance(focal_length_px, real_height, h)
            else:
                # Estimate distance
                distance = estimate_distance(focal_length_px, real_height, h)

            # Generate and collect descriptions with distance
            pos_desc = get_position_description(x + w/2, y + h/2, width, height)
            if float(distance) < 0.8:
                description = f"You are close to a {label} that is about {distance:.2f} meters away. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            else:
                description = f"There is a {label} about {distance:.2f} meters away. You are not close to it. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            descriptions.append(description)

            # Optional: Annotate distance on the image
            cv2.putText(img, f"{distance:.2f}m", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        time_logs['Annotate Image & Generate Descriptions'] = time.time() - start

        # Save descriptions to a file
        start = time.time()
        if descriptions != []:
            with open("output.txt", "w") as file:
                file.write('\n'.join(descriptions))
        else:
            with open("output.txt", "w") as file:
                file.write('')
        time_logs['Save Descriptions'] = time.time() - start

        # Display and save the processed image
        start = time.time()
        cv2.imwrite("output.jpg", img)
        cv2.imwrite('Pictures/' + str(time.time()).replace('.', '-') + '.jpg', img)
        time_logs['Save Images'] = time.time() - start

        # Print YOLO detections
        print('\nYOLO Detections:')
        for desc in descriptions:
            print(desc)

        # Print the times for each step
        total_time = time.time() - total_start
        time_logs['Total Time'] = total_time
        
        # Find the step that took the longest
        longest_step = max(time_logs, key=time_logs.get)
        print(f"\nLongest step: {longest_step} took {time_logs[longest_step]:.4f} seconds.")

        # Print all steps and their times
        for step, duration in time_logs.items():
            print(f"{step}: {duration:.4f} seconds")

    except Exception as e:
        print(f"Error in yolo_detect: {e}")





def parse_yolo_detections_from_file(filename):
    yolo_detections = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line:
                detection_info = parse_detection_sentence(line)
                if detection_info:
                    yolo_detections.append(detection_info)
    return yolo_detections

def parse_detection_sentence(sentence):
    try:
        # Extract label
        label_match = re.search(r'\b(?:a|an|the)\s+(\w+)', sentence)
        if label_match:
            label = label_match.group(1).lower()
        else:
            return None

        # Extract distance
        distance_match = re.search(r'about\s+([0-9.]+)\s+meters', sentence)
        if distance_match:
            distance = float(distance_match.group(1)) * 100  # Convert to cm
        else:
            return None

        # Extract angle from movement suggestion
        angle = 0.0  # Default angle
        if 'Turn Left' in sentence:
            angle_match = re.search(r'Turn Left\s+(\d+)\s+Degrees', sentence)
            if angle_match:
                angle = -float(angle_match.group(1))  # Negative for left
        elif 'Turn Right' in sentence:
            angle_match = re.search(r'Turn Right\s+(\d+)\s+Degrees', sentence)
            if angle_match:
                angle = float(angle_match.group(1))

        detection_info = {
            'label': label,
            'distance': distance,
            'angle': angle
        }
        return detection_info

    except Exception as e:
        print(f"Error parsing detection sentence: {e}")
        return None










def get_wm8960_card_number():
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
# Get the correct sound device for the WM8960 sound card
wm8960_card_number = get_wm8960_card_number()
#set_max_volume(wm8960_card_number)
print(wm8960_card_number)
def handle_playback(stream):
    global move_stopper
    global is_transcribing
    global wm8960_card_number
    with open('playback_text.txt', 'r') as file:
        text = file.read().strip()
        if text:
            print("Playback text found, initiating playback...")
            stream.stop_stream()
            is_transcribing = True

            # Generate speech from text
            subprocess.call(['espeak', '-v', 'en-us', '-s', '180', '-p', '100', '-a', '200', '-w', 'temp.wav', text])

            set_max_volume(wm8960_card_number)
            subprocess.check_call(["aplay", "-D", "plughw:{}".format(wm8960_card_number), 'temp.wav'])
            os.remove('temp.wav')
            open('playback_text.txt', 'w').close()
            stream.start_stream()
            is_transcribing = False
            print("Playback completed.")
            move_stopper = False
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

camera_vertical_pos = 'forward'
last_time = time.time()


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


import math

def create_video_from_images(image_folder, output_video):
    # Get a list of all image filenames in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]

    # Prepare a list to hold tuples of (timestamp, filename)
    image_filenames = []
    for img in images:
        # Remove the file extension to extract the timestamp
        name, ext = os.path.splitext(img)
        try:
            # Convert the dash back to a decimal and then to a float timestamp
            timestamp = float(name.replace("-", "."))
            image_filenames.append((timestamp, img))
        except ValueError:
            print(f"Skipping file '{img}': filename is not a valid timestamp.")
            continue

    # Sort the images by the float timestamp
    image_filenames.sort(key=lambda x: x[0])

    # Extract the sorted filenames
    sorted_images = [img for _, img in image_filenames]

    # Check if there are images
    if len(sorted_images) == 0:
        print("No valid images found in the folder.")
        return

    # Read the first image to get its dimensions
    first_image_path = os.path.join(image_folder, sorted_images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image '{first_image_path}'.")
        return
    height, width, layers = frame.shape

    print(f"First image dimensions: width={width}, height={height}, layers={layers}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'XVID' codec
    video = cv2.VideoWriter(output_video, fourcc, 1, (width, height))  # Setting fps to 1, we'll control duration manually
    if not video.isOpened():
        print("Error: VideoWriter not opened.")
        return

    frame_count = 0

    # Loop over each image and calculate the duration for each frame
    for i in range(len(sorted_images) - 1):
        current_image = sorted_images[i]
        next_image = sorted_images[i + 1]

        current_image_path = os.path.join(image_folder, current_image)
        next_image_path = os.path.join(image_folder, next_image)

        # Get the timestamps from the filenames
        current_timestamp = float(os.path.splitext(current_image)[0].replace("-", "."))
        next_timestamp = float(os.path.splitext(next_image)[0].replace("-", "."))

        # Calculate the time difference (in seconds) between the two frames
        time_diff = next_timestamp - current_timestamp

        # Read the current frame
        frame = cv2.imread(current_image_path)
        if frame is None:
            print(f"Warning: Could not read image '{current_image_path}'. Skipping.")
            continue

        # Resize frame if necessary
        if (frame.shape[1], frame.shape[0]) != (width, height):
            print(f"Resizing frame '{current_image}' from {frame.shape[1]}x{frame.shape[0]} to {width}x{height}")
            frame = cv2.resize(frame, (width, height))

        # Add the frame to the video, repeated according to the time difference
        num_frames_to_add = math.ceil(time_diff)  # You can adjust this to control how precise the time gap is represented
        for _ in range(num_frames_to_add):
            video.write(frame)
            frame_count += 1
            print(f"Added frame '{current_image}' to video for {num_frames_to_add} frames.")

    # Handle the last image (since it won't have a next image for time comparison)
    last_image_path = os.path.join(image_folder, sorted_images[-1])
    frame = cv2.imread(last_image_path)
    if frame is not None:
        for _ in range(30):  # Display last frame for a fixed 30 frames (arbitrary choice)
            video.write(frame)
        frame_count += 30

    # Release the video writer
    video.release()
    print(f"Video saved as '{output_video}' with {frame_count} frames.")


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

try:
    file = open('key.txt','r')
    api_key = file.read().split('\n')[0]
    file.close()
except:
    api_key = input('Please input your ChatGPT API key from OpenAI (Right click and paste it instead of typing it...): ')
    file = open('key.txt','w+')
    file.write(api_key)
    file.close()


def capture_image(camera):
    # Capture full-resolution image to a temporary file
    temp_image_path = 'temp_full_image.jpg'
    camera.capture_file(temp_image_path)
    
    # Load the captured image with OpenCV
    image = cv2.imread(temp_image_path)
    height, width = image.shape[:2]
    max_width = 512
    
    # Resize the image to max width of 416 pixels, maintaining aspect ratio
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = image

    # Calculate padding to make the image square (416x416)
    padded_image = np.full((max_width, max_width, 3), (0, 0, 0), dtype=np.uint8)  # Black square canvas

    # Center the resized image on the black canvas
    y_offset = (max_width - resized_image.shape[0]) // 2
    padded_image[y_offset:y_offset+resized_image.shape[0], 0:resized_image.shape[1]] = resized_image
    
    # Delete the temporary full-resolution image
    os.remove(temp_image_path)
    
    # Return the padded, square image
    return padded_image

def describe_image():
    


    # Read the original image
    image = cv2.imread('this_temp.jpg')

    # Resize the image to 320x320 pixels
    resized_image = cv2.resize(image, (320, 320))

    # Encode the resized image to JPEG format in memory
    success, buffer = cv2.imencode('.jpg', resized_image)
    if success:
        base64_image = base64.b64encode(buffer).decode('utf-8')
    else:
        print("Failed to encode image.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    with open("last_phrase2.txt","r") as f:
        phrase_now = f.read()
    with open("last_phrase2.txt","w+") as f:
        f.write('')


    # Initialize the payload with the system message with static instructions
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # The session history will be added here as individual messages
        ],
    }

    if phrase_now != '':
        phrase_now = 'You have just heard this from your microphone so use this as additional context for your response: ' + phrase_now
    else:
        pass
    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": "This image is from the front camera on a small robot. The camera is from a view of only a few inches above the ground. Please describe the scene in 3 sentences or less so the robot knows what it sees. "+phrase_now
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
        
    })
                

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nImage Description: \n' + str(response.json()["choices"][0]["message"]["content"]))
    with open('scene.txt','w+') as f:
        f.write(str(response.json()["choices"][0]["message"]["content"]))

def send_text_to_gpt4_move(history,percent, current_distance1, phrase, failed):
    global camera_vertical_pos
    
    from datetime import datetime
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('output.txt', 'r') as file:
        yolo_detections = file.read()
    with open('scene.txt', 'r') as file:
        scene_description = file.read()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }



    with open('current_distance.txt','r') as file:
        current_distance = float(file.read())
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")



    response_choices = ("Follow User, Say Something, Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, Do A Set Of Multiple Movements, Raise Camera Angle, Lower Camera Angle, Find Unseen Yolo Object, Focus Camera On Specific Yolo Object, Navigate To Specific Yolo Object, Alert User, No Movement, End Conversation, Good Bye.\n\n")
    if failed != '':
        response_choices = response_choices.replace(failed, '')
    else:
        pass



    # Initialize the payload with the system message with static instructions
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # The session history will be added here as individual messages
        ],
    }

    # Now, parse the session history and add messages accordingly
    for entry in history:
        timestamp_and_content = entry.split(" - ", 1)
        if len(timestamp_and_content) != 2:
            continue  # Skip entries that don't match the expected format

        timestamp, content = timestamp_and_content

        if "User Greeting:" in content or "You just heard this prompt" in content:
            # User message
            message_content = content.split(": ", 1)[-1]
            payload["messages"].append({
                "role": "user",
                "content": message_content.strip()
            })
        elif "Robot response at this timestamp:" in content or "Robot Greeting:" in content:
            # Assistant message
            message_content = content.split(": ", 1)[-1].strip()
            payload["messages"].append({
                "role": "assistant",
                "content": message_content
            })
        else:
            # System message or other data
            message_content = content.strip()
            payload["messages"].append({
                "role": "system",
                "content": message_content
            })

    navi = 'If you are choosing Navigate To Specific Yolo Object, then use Navigate To Specific Yolo Object as your response choice, then followed by ~~ and then replace your Reasoning with the closest relevant standard yolo coco name that goes with the object (you can only put the coco name, not a whole sentence or any adjectives on the name. You can only choose from this standard list of coco objects: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, tv monitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair dryer, toothbrush). The robot will automatically navigate to whichever yolo object you choose. You cannot choose to Navigate to an object if you just successfully navigated to it in the session history recently (Like for real, you cannot just repeatedly choose this if it was already successful. Only choose Navigate To Specific Yolo Object if you need to move to that object. Do not choose Navigate To Specific Yolo Object if you see in the history that you have already successfully navigated to the target object. You cannot choose to navigate to a specific object if the distance to that object is less than 0.6 meters or if you recently finished navigating to that object. If the user wants you to come to them or you want to go to the user, then navigate to a person object. And once, again, make sure you arent choosing to navigate to an object if you already navigated to it successfully.\n\n'
    object_focus = 'If you are choosing Focus Camera On Specific Yolo Object, then use Focus Camera On Specific Yolo Object as your response choice, then followed by ~~ and then replace your Reasoning with the closest relevant standard yolo coco name that goes with whatever type of object you are looking for (you can only put the coco name, not a whole sentence). The robot will automatically focus the camera on whichever yolo object you choose. Only choose Focus Camera On Specific Yolo Object if you need to constantly focus on that object. You can only choose to focus the camera on a yolo object that you are currently detecting. If the user wants you to look at them, then choose the person class.\n\n'
    object_find = 'If you are choosing Find Unseen Yolo Object, then use Find Unseen Yolo Object as your response choice, then followed by ~~ and then replace your Reasoning with absolutely only the yolo coco name of the object. The robot will automatically try to find whichever yolo object you choose so you absolutely must choose an actual yolo coco object from the standard list of 80, just choose the closest standard coco yolo object to what you are looking for (you can only put the coco name, not a whole sentence. You can only choose from this list of objects: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, tv monitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair dryer, toothbrush). Only choose Find Unseen Yolo Object if you need to find an object that you do not currently see with Yolo.\n\n'
    multi_moves = 'If you are choosing Do A Set Of Multiple Movements instead of choosing an individual command, then use Do A Set Of Multiple Movements as your response choice, then followed by ~~ and then replace your Reasoning with the list of moves you want the robot to make (separate the moves by a comma and space). They will happen one after another, starting at index 0. If the robot experiences any issues trying to do the set of moves, it will cancel doing the rest of the set and prompt you again to let you know there is an issue. You can only choose movements that make you move forward, backwards, or turn. You cannot choose camera adjustment commands or any of the other commands. Only movement commands are allowed.\n\n'
    user_follow = 'If you are choosing Follow User, then use Follow User as your response choice, then followed by ~~ and then your Reasoning. The robot will automatically follow the person YOLO object.\n\n'

    # Prepare the dynamic data to include in the last user message
    dynamic_data = (
        "You are a 4-wheeled mobile robot and your name is Echo because you get fine tuned with the data from your interactions.  You are real and have an actual physical robotic body.\n\n"
        f"Your response choices are: {response_choices}\n\n"
        "Your response choice absolutely must be something from this list above and worded **exactly the same**. If you don't use the exact words from this list of choices, then your response choice will be incorrect, even if it seems contextually correct. My code needs the exact phrase word for word, or it won't work because the robot code takes your direct response as the response choice.\n\n"
        "Your answer must be a response choice followed by ~~ (with a space on each side), followed by your reasoning for your response choice (Or if you are choosing Say Something or Alert User then this is where you put what you want to say).\n\n"
        "If you want to move to a particular object, make sure you turn and center on it between left and right first before moving forward towards it. If you cannot see what you are trying to see, turning most likely is a better option than going forward, but not all the time.\n\n"
        "The part of your response where you put either your reasoning or what you want to say or your alert can be no longer than like 2 to 4 sentences, but ideally less, like as few words as possible unless you have to say more.\n\n"
        "Your response must be formatted perfectly according to the template I gave, and your response choice must be worded exactly the same as one of the options from the list of response choices. You absolutely must format your response correctly as mentioned in the instructions.\n\n"
        "You cannot include any preface labels in your response (for example, do not start your response with 'Response Choice: ...'; you should just state the choice).\n\n"
        "As a final reminder, your response choice must be worded exactly the same as the choices from the provided Response Choices list; you must use the exact same words in your response choice. And if you Say Something or Alert User, replace the Reasoning area with what you want to say (And you must speak normally and realistically like a person).\n\n"
        "And your response must be formatted exactly according to the template I mentioned.\n\n"
        "Only choose Good Bye or End Conversation if the user says good bye or to end the conversation. If you are told goodbye, do not say something, just choose goodbye. You still must follow the response template correctly and do your response choice, followed by ~~ with a space on each side, followed by your reasoning.\n\n"
        f"{user_follow}"
        f"{multi_moves}"
        f"{object_find}"
        f"{navi}"
        f"{object_focus}"
        "If your most recent navigation has finished successfully then say something about how the navigation was succesful and you are done navigating to the object now.\n\n"
        "If you are going to say something, do not just repeat what you hear from your microphone.\n\n"
        "If you hear something from your microphone, you should most likely Say Something, unless you are explicitly told to do one of your Response Choices.\n\n"
        f"You have a camera and an HCSR04 distance sensor pointing in the same direction as the camera, and the distance sensor detects the distance to whatever object or obstacle that the visual description says you are centered on. Here is the distance it currently says: {current_distance}\n\n"
        f"Your camera is currently pointed {camera_vertical_pos}.\n\n"
        "Here is the current visual data from your camera (YOLO detections, and a scene description from GPT4 Vision):\n\n"
        f"CURRENT YOLO DETECTIONS:\n{yolo_detections}\n\n"
        f"CURRENT SCENE DESCRIPTION:\n{scene_description}\n\n"
        "You must make connections between all these different visual descriptions when choosing your response."
        f"{phrase}\n\n"
    )
    dynamic_data2 = (
        "You are a 4-wheeled mobile robot and your name is Echo because you get fine tuned with the data from your interactions.  You are real and have an actual physical robotic body.\n\n"
        f"Your response choices are: {response_choices}\n\n"
        f"You have a camera and an HCSR04 distance sensor pointing in the same direction as the camera, and the distance sensor detects the distance to whatever object or obstacle that the visual description says you are centered on. Here is the distance it currently says: {current_distance}\n\n"
        f"Your camera is currently pointed {camera_vertical_pos}.\n\n"
        "Here is the current visual data from your camera (YOLO detections, and a scene description from GPT4 Vision):\n\n"
        f"CURRENT YOLO DETECTIONS:\n{yolo_detections}\n\n"
        f"CURRENT SCENE DESCRIPTION:\n{scene_description}\n\n"
        f"{phrase}\n\n"
    )
    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data
        }]
        
    })
                

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nRESPONSE: \n' + str(response.json()))
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('memories/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.txt','w+') as f:
        f.write("PROMPT:\n\n\n"+str(payload).replace(dynamic_data,dynamic_data2)+"\n\n\n\nRESPONSE:\n\n\n"+str(response.json()["choices"][0]["message"]["content"]))
    return str(response.json()["choices"][0]["message"]["content"])


def send_text_to_gpt4_convo(history, text):
    
    print('getting yolo')
    with open('output.txt','r') as file:
        yolo_detections = file.read()
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")

    print('headers')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print('battery')
    with open('batt_per.txt','r') as file:
        percent = file.read()
    print('distance')
    while True:
        try:
            with open('current_distance.txt','r') as file:
                distance = float(file.read())
            print('got distance')
            
            break
        except Exception as e:
            print(e)
            time.sleep(0.1)
            continue
    print('putting together payload')
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            # System message with initial context and relevant data
            {
            "role": "system",
            "content": (
                f"You are a 4-wheeled mobile Raspberry Pi and Arduino robot fully controlled by ChatGPT (specifically GPT-4o).\n"
                f"Your software has 2 threads, the convo thread and the movement thread. You are the convo thread. You can either respond with what you want to say. Dont say any asterisk stuff. Just speak normal.\n"
                f"Keep your response no longer than one paragraph and just respond like a normal person. Don't narrate your moves and whatnot.\n"
                f"If the name of the current user has not been provided by the user within the current chat history, you must ask their name so you know who you are talking to. If you see multiple people, ask who everyone is if you dont already know.\n"
                f"Do not narrate movements and whatnot. Just respond with an actual response like a person would say and then let the movement thread handle the actual movement commands.\n"

                
            
        
            )}
        ]
    }
    print('parsing history')
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
        else:
            # Assistant message or other data
            message_content = content.strip()
            payload["messages"].append({
                "role": "assistant",
                "content": message_content
            })


    




    print('dynamic data')
    # Finally, include the current prompt as the latest user message
    payload["messages"].append({
        "role": "user",
        "content": f"Your battery percentage is: {percent}%.\n\nThe current time and date is: {the_time}.\n\nYour camera is currently pointed {camera_vertical_pos}.\n\nThe current YOLO visual data from your camera and the movement choices necessary to center your camera on each specific object (This is what you currently see. If its blank then no YOLO detections have been made on this frame): {yolo_detections}\n\nThe current distance in CM to any yolo object in the center of the image: {str(distance)}\n\n{text.strip()}"
    })

    # Include the max_tokens parameter
    payload["max_tokens"] = 200

    
    
    print('getting response')
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('returning response')
    return response.json()["choices"][0]["message"]["content"]





    

    


def say_greeting(last_phrase):
    global chat_history
    global last_time
    now = datetime.now()
    
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print('got time')
   
    text = str(send_text_to_gpt4_convo(chat_history, last_phrase))
    print('got gpt response')
    chat_history.append('Time: ' + str(the_time) + ' - User Greeting: ' + last_phrase)  # add response to chat history
    chat_history.append('Time: ' + str(the_time) + ' - Robot Greeting: ' + text)  # add response to chat history
    print('added to history')
    last_time = time.time()

    file = open('playback_text.txt', 'w+')
    file.write(text)
    file.close()
    print('playback text saved')
    

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
# Adjust the index extraction to handle the nested array structure
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]






move_stopper = False




 
def movement_loop(camera):
    global chat_history
    global frame
    global net
    global output_layers
    global classes
    global move_stopper
    global camera_vertical_pos

    global move_set
    global yolo_find
    global nav_object
    global yolo_nav  
    global scan360
    scan360 = 0

    ina219 = INA219(addr=0x42)
    last_time = time.time()
    failed_response = ''
    movement_response = ' ~~ '
    move_set = []
    yolo_nav = False
    yolo_find = False
    yolo_look = False
    follow_user = False
    nav_object = ''
    look_object = ''
    last_response = ''
    while True:
        try:

            distance = int(read_distance_from_arduino())
            with open('current_distance.txt','w+') as f:
                f.write(str(distance))
            break
        except:
            print(traceback.format_exc())
            time.sleep(0.1)
            continue
    #print('movement thread start')
    while True:
        try:
            with open('current_history.txt','w+') as file:
                file.write('\n'.join(chat_history))
            last_phrase2 = get_last_phrase()
            if last_phrase2 != '':
                print('Last phrase now on movement loop: ' + last_phrase2)
                with open("last_phrase2.txt","w+") as f:
                    f.write(last_phrase2)
                last_phrase2 = 'You just heard this prompt from your microphone. Do not repeat this prompt, actually respond. DO NOT SAY THIS, RESPOND TO IT INSTEAD WITH EITHER SPEECH OR ANOTHER OF THE AVAILABLE RESPONSE CHOICES. Respond with either Say Something or the correct Response Choice: ' + last_phrase2
                yolo_nav = False
                yolo_find = False
                yolo_look = False
                follow_user = False
                move_set = []
            else:
                pass
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            with open('last_distance.txt','w+') as f:
                f.write(str(distance))
            while True:
                try:

                    distance = int(read_distance_from_arduino())
                    with open('current_distance.txt','w+') as f:
                        f.write(str(distance))
                    break
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
                    continue
            try:
                frame = capture_image(camera)
                cv2.imwrite('this_temp.jpg', frame)
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
                continue
            if follow_user == True or yolo_find == True or yolo_nav == True or yolo_look == True:
                yolo_detect()
            else:
                thread1 = threading.Thread(target=describe_image)
                thread2 = threading.Thread(target=yolo_detect)
                thread1.start()
                thread2.start()
                thread1.join()
                thread2.join()
            
            current = ina219.getCurrent_mA() 
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            with open('batt_per.txt','w+') as file:
                file.write(str(per))

            
            if current > 0.0 or per < 10.0:
                try:
                    last_time = time.time()
                    image_folder = 'Pictures/'  # Replace with the path to your image folder
                    output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                    create_video_from_images(image_folder, output_video)
          
                    print('ending convo')
                    chat_history = []
                    
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    break
            else:
                pass
            if move_stopper == True:
                #print('moves stopped')
                continue
            else:
                pass
            try:
                if frame is not None:
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    if move_set == []:
                        if yolo_nav == True:
                            yolo_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if nav_object in current_detection:
                                        target_detected = True
                                        if current_distance < 0.6:
                                            movement_response = 'No Movement ~~ Navigation has finished successfully!'
                                            yolo_nav = False
                                            nav_object = ''
                                            break
                                        else:
                                            pass
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        else:
                                            movement_response = 'Move Forward One Foot ~~ Moving towards target object'
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
                                # Object lost, switch to route planning
                                print(f"Cannot see '{nav_object}'. Going into Find Object mode.")
                                yolo_nav = False
                                yolo_find = True
                                scan360 = 0
                                movement_response = 'No Movement ~~ Target Lost. Going into Find Object mode.'
                        elif yolo_look == True:
                            print('Looking at object')
                            #do yolo navigation to specific object
                            yolo_look_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_look_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if look_object in current_detection:
                                        print('object seen')
                                        #follow any human seen
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        elif 'Raise Camera Angle' in current_detection:
                                            movement_response = 'Raise Camera Angle ~~ Target object is above'
                                        elif 'Lower Camera Angle' in current_detection:
                                            movement_response = 'Lower Camera Angle ~~ Target object is below'
                                        else:
                                            movement_response = 'No Movement ~~ Target object is straight ahead'
                                        
                                        break
                                    else:
                                        
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_response = 'No Movement ~~ Target object lost'
                                            yolo_look = False
                                            yolo_find = True
                                            look_object = ''
                                            scan360 = 0
                                            break
                                        else:
                                            continue
                                except:
                                    movement_response = 'No Movement ~~ Focus Camera On Specific Yolo Object failed. Must be detecting object first.'
                                    yolo_look = False
                                    yolo_find = True
                                    look_object = ''
                                    scan360 = 0
                                    break
                                    
                        elif follow_user == True:
                            print('Looking at object')
                            #do yolo navigation to specific object
                            yolo_look_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_look_index]
                                    current_distance1 = current_detection.split(' ') #extract distance
                                    current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                    if 'person' in current_detection:
                                        print('User seen')
                                        #follow any human seen
                                        if 'Turn Left 15 Degrees' in current_detection:
                                            movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 15 Degrees' in current_detection:
                                            movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right'
                                        elif 'Turn Left 45 Degrees' in current_detection:
                                            movement_response = 'Turn Left 45 Degrees ~~ Target object is to the left'
                                        elif 'Turn Right 45 Degrees' in current_detection:
                                            movement_response = 'Turn Right 45 Degrees ~~ Target object is to the right'
                                        
                                        else:
                                            #if outside of distance range, move forward or backward, otherwise:
                                            if current_distance > 1.3:
                                                movement_response = 'Move Forward One Foot ~~ Moving towards user'
                                            elif current_distance < 1.0:
                                                movement_response = 'Move Backward ~~ Moving away from user'
                                            else:
                                                movement_response = 'No Movement ~~ Target object is straight ahead'
                                            
                                        break
                                    else:
                                        
                                        yolo_look_index += 1
                                        if yolo_look_index >= len(yolo_detections):
                                            movement_response = 'No Movement ~~ User lost'
                                            follow_user = False
                                            yolo_find = True
                                            nav_object = 'person'
                                            scan360 = 0
                                            break
                                        else:
                                            continue
                                except:
                                    movement_response = 'No Movement ~~ Follow User failed. Must be detecting person first.'
                                    follow_user = False
                                    yolo_find = True
                                    nav_object = 'person'
                                    scan360 = 0
                                    break
                                    
                        elif yolo_find == True:
                            #check if robot sees target object with yolo
                            yolo_nav_index = 0
                            with open('output.txt','r') as file:
                                yolo_detections = file.readlines()
                          
                                
                            while True:
                                try:
                                    current_detection = yolo_detections[yolo_nav_index]
                                    if nav_object in current_detection:
                                        yolo_find = False
                                        
                                        movement_response = 'No Movement ~~ Ending search for '+nav_object+'. Object has successfully been found!'
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
                                #do 360 scan
                                if scan360 < 10 and scan360 > 1:
                                    movement_response = 'Turn Right 45 Degrees ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 0:
                                    movement_response = 'Raise Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                elif scan360 == 1:
                                    movement_response = 'Lower Camera Angle ~~ Doing 360 scan for target object'
                                    scan360 += 1
                                else:
                                    #do object avoidance
                                    #if object not found in scan then start doing object avoidance until object is found
                                    
                                    print('\nDistance sensor: ')
                                    print(str(distance)+' cm')
                                    if distance < 50.0 and distance >= 20.0:

                                        if rando_num == 1:
                                            movement_response = 'Turn Left 45 Degrees ~~ Exploring to look for target object'
                                        
                                        elif rando_num == 2:
                                            movement_response = 'Turn Right 45 Degrees ~~ Exploring to look for target object'
                                    
                                    elif distance < 20.0:
                                        movement_response = 'Do A Set Of Multiple Movements ~~ Move Backward, Turn Left 45 Degrees, Turn Left 45 Degrees'
                                    else:
                                        movement_response = 'Move Forward One Foot ~~ Exploring to look for target object'
                       
                        else:
                            movement_response = str(send_text_to_gpt4_move(chat_history, per, distance, last_phrase2, failed_response)).replace('Response Choice: ','').replace('Movement Choice at this timestamp: ','').replace('Response Choice at this timestamp: ','').replace('Attempting to do movement response choice: ','')
                    else:
                        movement_response = move_set[0] + ' ~~ Doing move from list of moves'
                        del move_set[0]
                    if last_phrase2 != '':
                        chat_history.append('Time: ' + str(the_time) + ' - ' + last_phrase2)
                    else:
                        pass
                    chat_history.append('Time: ' + str(the_time) + ' - Robot response at this timestamp: ' + movement_response)  # add response to chat history                   
                    
                    try:
                        print("\nPercent:       {:3.1f}%".format(per))
                        print('\nCurrent Distance: ' + str(distance) + ' cm')
                        print('\nResponse Choice: '+ movement_response.split('~~')[0].strip().replace('.',''))
                        print('\nReasoning: '+ movement_response.split('~~')[1].strip())
                       



                    except:
                        print(traceback.format_exc())
                    
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")

                    current_response = movement_response.split('~~')[0].strip().replace('.','')
                    
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    last_response = current_response
                    current_response = current_response.lower().replace(' ', '')
                    if current_response == 'moveforward1inch' or current_response == 'moveforwardoneinch':
                        if distance < 20.0:
                            print('move forward 1 inch failed. Too close to obstacle to move forward anymore')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Too close to obstacle to move forward anymore. You cannot choose Move Forward One Inch right now.')
                            failed_response = 'Move Forward One Inch, '
                            yolo_nav = False
                            move_set = []
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            #if yolo_nav == False and yolo_find == False:
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Inch')
                            failed_response = ''
                            
                    elif current_response == 'moveforward1foot' or current_response == 'moveforwardonefoot':
                        if distance < 50.0:
                            print('move forward 1 foot failed. Too close to obstacle to move forward that far')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Too close to obstacle to move forward that far. You cannot choose Move Forward One Foot right now.')
                            failed_response = 'Move Forward One Foot, '
                            yolo_nav = False
                            move_set = []
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            #if yolo_nav == False and yolo_find == False:
                            time.sleep(0.5)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Foot')
                            failed_response = ''
                    elif current_response == 'movebackward':
                        send_data_to_arduino(["s"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.5)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Backward')
                        failed_response = ''
                    elif current_response == 'turnleft45degrees' or current_response == 'moveleft45degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.15)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Left 45 Degrees')
                        failed_response = ''
                    elif current_response == 'turnleft15degrees' or current_response == 'moveleft15degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.03)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Left 15 Degrees')
                        failed_response = ''
                    elif current_response == 'turnright45degrees' or current_response == 'moveright45degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.15)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Right 45 Degrees')
                        failed_response = ''
                    elif current_response == 'turnright15degrees' or current_response == 'moveright15degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        #if yolo_nav == False and yolo_find == False:
                        time.sleep(0.03)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Right 15 Degrees')
                        failed_response = ''
                    elif current_response == 'turnaround180degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(1)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Around 180 Degrees')
                        failed_response = ''
                    elif current_response == 'doasetofmultiplemovements':
                        move_set = movement_response.split('~~')[1].strip().split(', ')
                        chat_history.append('Time: ' + str(the_time) + ' - Initiating this set of moves: '+movement_response.split('~~')[1].strip())
                        failed_response = ''

                    elif current_response == 'raisecameraangle':
                        if camera_vertical_pos == 'up':
                            chat_history.append('Time: ' + str(the_time) + ' - Raise Camera Angle Failed: Camera angle is already raised as much as possible. You cannot choose Raise Camera Angle right now.')
                            print('Raise Camera Angle Failed. Camera angle is already raised as much as possible.')
                            failed_response = 'Raise Camera Angle, '
                        else:
                            send_data_to_arduino(["2"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            
                            camera_vertical_pos = 'up'
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Raised the Camera Angle to the max upward angle.')
                    elif current_response == 'lowercameraangle':
                        if camera_vertical_pos == 'forward':
                            chat_history.append('Time: ' + str(the_time) + ' - Lower Camera Angle Failed: Camera angle is already lowered as much as possible. You cannot choose Lower Camera Angle right now.')
                            print('Lower Camera Angle failed. Camera angle is already lowered as much as possible.')
                            failed_response = 'Lower Camera Angle, '
                        else:
                            send_data_to_arduino(["1"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            
                            camera_vertical_pos = 'forward'
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Lowered the Camera Angle to the lowest angle, which is looking directly forward.')

                    elif current_response == 'endconversation' or current_response == 'goodbye':
                        with open('playback_text.txt', 'w') as f:
                            f.write(movement_response.split('~~')[1].strip())
                    
                        last_time = time.time()
                        image_folder = 'Pictures/'  # Replace with the path to your image folder
                        output_video = 'Videos/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.avi'  # Change the extension to .avi
                        create_video_from_images(image_folder, output_video)                        
                        print('ending convo')
                        chat_history = []
                        break
                    elif current_response == 'nomovement':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response choice was No Movement so not moving.')
                    elif current_response == 'saysomething' or current_response == 'alertuser':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        with open('playback_text.txt','w') as f:
                            f.write(movement_response.split('~~')[1])
                    elif current_response == 'navigatetospecificyoloobject':
                        nav_object = movement_response.split('~~')[1].strip().lower()
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Starting navigation to ' + nav_object)
                        yolo_nav = True
                    elif current_response == 'focuscameraonspecificyoloobject':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        camera_vertical_pos = 'forward'
                        yolo_look = True
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        look_object = movement_response.split('~~')[1]
                        chat_history.append('Time: ' + str(the_time) + ' - Starting to look at '+look_object)
                    elif current_response == 'followuser':
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
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Starting to follow user.')
                    elif current_response == 'findunseenyoloobject':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.1)
                        send_data_to_arduino(["2"], arduino_address)
                        time.sleep(0.1)
                        camera_vertical_pos = 'forward'
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        yolo_find = True
                        scan360 = 0
                        nav_object = movement_response.split('~~')[1]
                        chat_history.append('Time: ' + str(the_time) + ' - Starting to explore around to look for '+movement_response.split('~~')[1])
                        rando_list = [1,2]
                        rando_index = random.randrange(len(rando_list))
                        rando_num = rando_list[rando_index]
                    else:
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response failed. You didnt follow my instructions properly for how you should respond. Here is what you responded with so dont do it again: ' + movement_response)
                        print('failed response')
                        
                    
                    
                    time.sleep(0.1)
            except:
                print(traceback.format_exc())
                
            
            
        except:
            print(traceback.format_exc())
            
if __name__ == "__main__":
    try:
        be_still = True
        last_time_seen = time.time()
        transcribe_thread = threading.Thread(target=listen_and_transcribe)  # Adding the transcription thread
        transcribe_thread.start() 
        camera = Picamera2()
        camera_config = camera.create_still_configuration()
        camera.configure(camera_config)
        camera.start()
        time.sleep(1)

        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["1"], arduino_address)
        send_data_to_arduino(["2"], arduino_address)
                        
        print('waiting to be called')
        ina219 = INA219(addr=0x42)
        while True:
            try:

                distance = int(read_distance_from_arduino())
                with open('current_distance.txt','w+') as f:
                    f.write(str(distance))
                break
            except:
                print(traceback.format_exc())
                time.sleep(0.1)
                continue
        while True:
            time.sleep(0.25)
            current = ina219.getCurrent_mA()
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            per = max(0, min(per, 100))  # Clamp percentage to 0-100 range
            per = (per * 2) - 100
            print('\nBattery Percent: ')
            print(per)
            with open('batt_per.txt', 'w+') as file:
                file.write(str(per))
            chat_history = []
            
            # Capture and resize the image, returning it as an array
            frame = capture_image(camera)
            cv2.imwrite('this_temp.jpg', frame)  # Save if needed for reference
            last_phrase = get_last_phrase()
            
            try:
                the_index_now = last_phrase.split(' ').index('echo')
                name_heard = True
            except ValueError:
                name_heard = False
            try:
                the_index_now = last_phrase.split(' ').index('robot')
                name_heard = True
            except ValueError:
                pass
            if name_heard:
                print("Name heard, initializing...")
                

                
                with open('last_phrase.txt', 'w') as file:
                    file.write(last_phrase)
                
          
                print('starting thread')
                movement_loop(camera)
                
            time.sleep(0.05)

    except Exception as e:
        print(traceback.format_exc())
        print(f"An error occurred: {e}")
    finally:
        camera.close()
