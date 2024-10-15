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
in_task = False
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
file = open('self_response.txt','w+')
file.write('')
file.close()
file = open('memory_summary.txt','w+')
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
    Return a text description of the position based on coordinates.
    
    Parameters:
    - x (float): X-coordinate of the object's center in the image.
    - y (float): Y-coordinate of the object's center in the image.
    - width (float): Width of the image.
    - height (float): Height of the image.
    
    Returns:
    - str: Description of the object's position.
    """
    if x < width / 3:
        horizontal = "Turn Left 15 Degrees"
    elif x > 2 * width / 3:
        horizontal = "Turn Right 15 Degrees"
    else:
        horizontal = "already centered between left and right"
    
    if y < height / 3:
        vertical = "Raise Camera Angle"
    elif y > 2 * height / 3:
        vertical = "Lower Camera Angle"
    else:
        vertical = "already centered on the vertical"
    
    if horizontal == "already centered between left and right" and vertical == "already centered on the vertical":
        return "already centered on object"
    else:
        return f"{vertical} and {horizontal}"



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
    
    Parameters:
    - net (cv2.dnn_Net): Pre-loaded YOLO network.
    - output_layers (list): List of output layer names for YOLO.
    - classes (list): List of class names corresponding to COCO dataset.
    - chat_history (list): List to store chat history for logging.
    
    Returns:
    - None
    """
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    try:
        img = cv2.imread('this_temp.jpg')
        if img is None:
            print("Error: Image 'this_temp.jpg' not found.")
            return
        height, width, channels = img.shape

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        # Extract bounding boxes and confidences
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

            # Estimate distance
            distance = estimate_distance(focal_length_px, real_height, h)

            # Generate and collect descriptions with distance
            pos_desc = get_position_description(x + w/2, y + h/2, width, height)
            if float(distance) < 0.5:
                description = f"You are close to a {label} that is about {distance:.2f} meters away. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            else:
                description = f"There is a {label} about {distance:.2f} meters away. You are not close to it. You can center your camera on it by doing these moves (you should only center on this object if it is what you are looking for): {pos_desc}."
            descriptions.append(description)

            # Optional: Annotate distance on the image
            cv2.putText(img, f"{distance:.2f}m", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if descriptions != []:
            # Save descriptions to a file
            with open("output.txt", "w") as file:
                file.write('\n'.join(descriptions))
        else:
            with open("output.txt", "w") as file:
                file.write('')            
        # Update chat history with descriptions
        
        # Display and save the processed image
        cv2.imwrite("output.jpg", img)
        cv2.imwrite('Pictures/'+str(int(time.time()))+".jpg", img)
        print('\nYOLO Detections:')

        for desc in descriptions:
            print(desc)
        
    except Exception as e:
        print(f"Error in yolo_detect: {e}")



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
    global move_stopper
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

task = 'No task is currently set.'
task_steps = ''
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





                
      
def get_column_height():
    image_filename = 'output.jpg'
    frame = cv2.imread(image_filename)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    height, width = frame.shape[:2]

    # Define the custom width for the center column: center 50%
    left_width = width // 4
    center_width = width // 2
    x1, x2 = left_width, left_width + center_width  # Coordinates for the center column

    # Initialize the heights as the number of squares from the bottom
    edge_height = 0  # Start counting at 0 for edge detection
    color_height = 0  # Start counting at 0 for color/brightness change detection

    # Get the color/brightness of the bottom square
    bottom_square = gray[8 * height // 9: height, x1:x2]
    bottom_avg_brightness = np.mean(bottom_square)

    # Edge detection loop
    for row in range(9):  # Iterate from the bottom square to the top for edge detection
        y1, y2 = (8 - row) * height // 9, (9 - row) * height // 9  # Divide into 9 sections
        square_edges = edges[y1:y2, x1:x2]

        # Check for edges in the current square
        if np.any(square_edges):
            break  # Stop at the first square with an edge (don't count it)
        # Increment the edge height if no edge is detected
        edge_height += 1

    # Color/brightness detection loop
    for row in range(9):  # Iterate from the bottom square to the top for color detection
        y1, y2 = (8 - row) * height // 9, (9 - row) * height // 9  # Divide into 9 sections
        square_brightness = np.mean(gray[y1:y2, x1:x2])  # Calculate the average brightness of the current square

        # Check for drastic change in brightness compared to the bottom square
        if abs(square_brightness - bottom_avg_brightness) > 50:  # 50 is an arbitrary threshold for drastic change
            break  # Stop at the first square with a drastic color/brightness change (don't count it)
        # Increment the color height if no drastic brightness change is detected
        color_height += 1

    # The final height is the minimum of the edge height and color height
    final_height = min(edge_height, color_height)

    # Save the final height to a file
    with open('column_height.txt', 'w+') as f:
        f.write(str(color_height) + '\n')
  


    
    """
    #UNCOMMENT THIS SECTION TO SHOW IMAGE
    for i in range(3):
        print(f"Column {i} is {heights[i]:.2f} squares high before encountering an edge.")
        x_position = i * width // 3 + 20
        cv2.putText(frame, f"Height: {heights[i]:.2f}",
                    (x_position, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    print("Drawing green edges on the original frame...")
    frame[edges != 0] = [0, 255, 0]
    print("Displaying the processed frame...")
    cv2.imshow("Edge Detection & Grid Highlighting", frame)
    cv2.waitKey(1000)  # Wait for a key press to close the window
    print("Closing all windows...")
    cv2.destroyAllWindows()
    print("Resources released and windows closed successfully.")
    """






def send_text_to_gpt4_move(history,percent, current_distance, phrase, task, task_steps, user_name, user_data, mems, failed):
    global camera_vertical_pos
    global in_task
    from datetime import datetime
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    with open('output.txt', 'r') as file:
        yolo_detections = file.read()
    with open('output.jpg', "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }



    current_distance = str(current_distance)
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")



    if in_task:
        if yolo_detections != '':
            response_choices = (
                "Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Around 180 Degrees, "
                "Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, "
                "Raise Camera Angle, Lower Camera Angle, Navigate To Specific Yolo Object, Say Something, Alert User, No Movement, End Conversation, Good Bye, "
                "End Task, Mark Off A Completed Task Step\n\n"
            )
        else:
            response_choices = (
                "Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Around 180 Degrees, "
                "Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, "
                "Raise Camera Angle, Lower Camera Angle, Say Something, Alert User, No Movement, End Conversation, Good Bye, "
                "End Task, Mark Off A Completed Task Step\n\n"
            )
        if failed != '':
            response_choices = response_choices.replace(failed, '')
        else:
            pass
        task_data = (
            f"Your current task: {task}\n\nSteps To Complete Task\n{task_steps}\n\n"
            "If all steps have been completed, choose End Task. If you need to mark a completed step off the list of steps, choose Mark Off A Completed Task Step."
        )
    else:
        if yolo_detections != '':
            response_choices = (
                "Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Around 180 Degrees, "
                "Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, "
                "Raise Camera Angle, Lower Camera Angle, Navigate To Specific Yolo Object, Say Something, Alert User, No Movement, End Conversation, Good Bye, Start Task\n\n"
            )
        else:
            response_choices = (
                "Move Forward One Inch, Move Forward One Foot, Move Backward, Turn Around 180 Degrees, "
                "Turn Left 15 Degrees, Turn Left 45 Degrees, Turn Right 15 Degrees, Turn Right 45 Degrees, "
                "Raise Camera Angle, Lower Camera Angle, Say Something, Alert User, No Movement, End Conversation, Good Bye, Start Task\n\n"
            )            
        if failed != '':
            response_choices = response_choices.replace(failed, '')
        else:
            pass
        task_data = task

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
    if yolo_detections != '':
        navi = 'If you are choosing Navigate To Specific Yolo Object, then give it as your response choice, then followed by ~~ and then replace your Reasoning with the yolo coco name of the object, and then followed by - with a space on each side, and then followed by the desired distance to be from the object after successful navigation. The robot will automatically navigate to whichever yolo object you choose. Only choose Navigate To Specific Yolo Object if you need to move to that object. You can only choose to navigate to a yolo object that you are currently detecting.\n\n'
    else:
        navi = ''
    # Prepare the dynamic data to include in the last user message
    dynamic_data = (
        "You are a 4-wheeled mobile robot.\n\n"
        f"Your response choices are: {response_choices}\n\n"
        "Your response choice absolutely must be something from this list above and worded **exactly the same**. If you don't use the exact words from this list of choices, then your response choice will be incorrect, even if it seems contextually correct. My code needs the exact phrase word for word, or it won't work because the robot code takes your direct response as the response choice.\n\n"
        "Your answer must be a response choice followed by ~~ (with a space on each side), followed by your reasoning for your response choice (Or if you are choosing Say Something or Alert User then this is where you put what you want to say), and then followed by ~~ with a space on each side and then followed by the name of the person or people you are talking to right now (If multiple people then separate each name by a comma and a space).\n\n"
        "If you want to move to a particular object, make sure you turn and center on it between left and right first with YOLO before moving forward towards it. If you cannot see what you are trying to see, turning most likely is a better option than going forward, but not all the time.\n\n"
        "The part of your response where you put either your reasoning or what you want to say or your alert can be no longer than like 2 to 4 sentences, but ideally less, like as few words as possible unless you have to say more.\n\n"
        "If no areas are drivable, then turn, because you turn in place like a tank, so it's okay to turn if there are no drivable areas.\n\n"
        "Your response must be formatted perfectly according to the template I gave, and your response choice must be worded exactly the same as one of the options from the list of response choices. You absolutely must format your response correctly as mentioned in the instructions.\n\n"
        "You cannot include any preface labels in your response (for example, do not start your response with 'Response Choice: ...'; you should just state the choice).\n\n"
        "As a final reminder, your response choice must be worded exactly the same as the choices from the provided Response Choices list; you must use the exact same words in your response choice. And if you Say Something or Alert User, replace the Reasoning area with what you want to say (And you must speak normally and realistically like a person).\n\n"
        "And your response must be formatted exactly according to the template I mentioned.\n\n"
        "Only choose Good Bye or End Conversation if the user says good bye or to end the conversation.\n\n"
        "Do not just turn back and forth repeatedly. That will get you nowhere.\n\n"
        "If you are not in a task and you are told to do something, then start task but only start task if you are explicitely told to do something, otherwise do not choose this response choice.\n\n"
        f"{navi}"
        "Make sure to pay attention the chat history and what has been said and what actions the robot has taken so you have the full context and can make better decisions.\n\n"
        "If you are going to say something, do not just repeat what you hear from your microphone.\n\n"
        f"You have a camera and an HCSR04 distance sensor pointing in the same direction as the camera, and the distance sensor detects the distance to whatever object that YOLO says is in the absolute middle center of the image. Here is the distance it currently says: {current_distance}\n\n"
        f"Your camera is currently pointed {camera_vertical_pos}.\n\n"
        "Here is the currently detected YOLO objects and the movement choices necessary to center your camera on each specific object\n"
        "(THIS IS THE ABSOLUTE CURRENT YOLO DATA):\n"
        f"{yolo_detections}\n\n"
        #f"{edge_detect}\n\n"
        f"{task_data}\n\n"
        f"You are currently talking to {', '.join(user_name)}. Here is some info about them:\n\n"
        f"{user_data}\n\n"
        f"Your personal relevant memories from your past experiences as a robot:\n{mems}\n\n"
        "The image included in this prompt is what you, the robot, currently see from your camera. It has the YOLO bounding boxes around any detected objects from the normal 80 object dataset.\n\n"
        f"{phrase}\n\n"
    )

    # Append the dynamic data as the last user message
    payload["messages"].append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": dynamic_data
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }]
        
    })
                

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('\n\n\n\nRESPONSE: \n' + str(response.json()))
    return str(response.json()["choices"][0]["message"]["content"])


def send_text_to_gpt4_convo(history, text, task, task_steps, mems):
    global in_task
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
    print('task')
    if in_task == True:
        task_data = f"Your current task: {task}\n\nSteps To Complete Task\n{task_steps}\n\n"
    else:
        task_data = task
    print('battery')
    with open('batt_per.txt','r') as file:
        percent = file.read()
    print('distance')
    while True:
        try:
            distance = int(read_distance_from_arduino())
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
        "content": f"Your personal relevant memories from your past experiences as a robot:\n{mems}\n\nYour battery percentage is: {percent}%.\n\nThe current time and date is: {the_time}.\n\nYour camera is currently pointed {camera_vertical_pos}.\n\n{task_data}\n\nThe current YOLO visual data from your camera and the movement choices necessary to center your camera on each specific object (This is what you currently see. If its blank then no YOLO detections have been made on this frame): {yolo_detections}\n\nThe current distance in CM to any yolo object in the center of the image: {str(distance)}\n\n{text.strip()}"
    })

    # Include the max_tokens parameter
    payload["max_tokens"] = 200

    
    
    print('getting response')
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print('returning response')
    return response.json()["choices"][0]["message"]["content"]

def get_chat_summary(history, mem_sum):
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    


   



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
                        "text": 'You are a 4wd mobile arduino and raspberry pi robot that is controlled by ChatGPT. Give a summary of this session history (Oldest to newest): ' + str(history) + ' \n\n Make sure to include important facts, events, goals, conversational information, learned information, and any other data worth keeping in the robot memory so it has useful information to use on future prompts (The prompts that choose the robot actions and when to speak are given relevant memories so this has to include good relevant knowledge). \n\n\nHeres contextually relevant memories that the robot already has that you can use as extra context when creating your overview of the current history with the person: \n\n' + mem_sum+'\n\nKeep your response as short as possible but include all important information so it can be longer if absolutely necessary. Your summary will be given to chatgpt for helping chatgpt decide how to control the robot so make sure you word it in a way where it is focused on enhancing the understanding and ability of chatgpt to control the robot.'
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]

def get_memory_summary(): #to be used on memory thread
    global chat_history
    while not stop_threads:

        memory_dir = 'memories/'

        # Initialize an empty string to hold the contents of all memory files
        memory_file_data = ''

        # Get a list of all files in the memories directory
        memory_files = os.listdir(memory_dir)

        # Loop through each file in the directory
        while memory_files:
            # Open the file at index 0 of the list
            file_path = os.path.join(memory_dir, memory_files[0])
            with open(file_path, 'r') as file:
                # Add file content to memory_file_data with a new line
                memory_file_data += file.read() + '\n'
            
            # Remove the processed file from the list, but leave it on the hard drive
            del memory_files[0]

        # Print or use the final memory_file_data string
        print('got memory file data')
        

        while True:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                

                #get chatgpt to summarize the memory file data using the chat history as the context of what information to put in the summary
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": 'You are a 4wd mobile arduino and raspberry pi robot that is controlled by ChatGPT. Summarize the memory file data using the chat history as the context of what information to put in the summary. \n\n MEMORY FILE DATA:\n'+memory_file_data+'\n\nChat and Movement History that should be the context of what to put in your summary of the memory file data:\n'+'\n'.join(chat_history)
                                }
                            ]
                        }
                    ]
                }
                print('got response')
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                print('got response')
                #save response to file
                with open('memory_summary.txt','w+') as f:
                    f.write(response.json()["choices"][0]["message"]["content"])
                break
            except:
                #if prompt is too long, get chatgpt to summarize each file and save the new summary to the file and then start this process over again until the memory data isnt too much
                print(traceback.format_exc())
                continue


    
def get_person_summary(history, user):
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    


   
    try:
        file = open('people/'+user + '.txt', 'r')
        memories = file.read()
        file.close()
    except:
        memories = 'No memories yet for this person.'


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
                        "text": 'You are a 4wd mobile arduino and raspberry pi robot that is controlled by ChatGPT. Give a summary of this person and your interaction with them. Current History for Movement, Chat, Sensor/Battery, and Visual Data from the most recent conversation/event youve had as a robot (Oldest to newest): ' + str(history) + ' \n\n Make sure to include important facts, events, goals, conversational information, learned information about the person, and any other data worth keeping in the robot memory so it has useful information to use on future prompts (The prompts that choose the robots actions and when to speak are given relevant memories so this has to include good relevant knowledge). \n\n\nHeres contextually relevant memories that the robot already has that you can use as extra context when creating your overview of the current history with the person: \n\n' + memories+'\n\n\n\n\nThis is who your summary is about: '+user+'\n\nKeep your response as short as possible.'
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]
    
def get_task(history, phrase):
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    

    

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
                        "text": 'You are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). Based on this Movement, Chat, Sensor/Battery, and Visual Data History from the most recent conversation/event youve had (Oldest to newest): ' + str(history) + '\n\n'+phrase+'\n\n You have been told to start a task so say what the task is that you are going to start, then followed by ~~, then followed by the list of steps you will need to take to accomplish the task.' 
                    }
                ]
            }
        ]
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()["choices"][0]["message"]["content"]
def mark_off_a_completed_step(history, phrase, task, task_steps):
    now = datetime.now()
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    

    

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
                        "text": 'You are a 4 wheeled mobile robot that is fully controlled by ChatGPT (Specifically GPT4o). You are currently doing this task: '+task+' You have been told a new task step has been completed. Based on this Movement, Chat, Sensor/Battery, and Visual Data History from the most recent conversation/event youve had (Oldest to newest): ' + str(history) + '\n\n'+phrase+'\n\n choose which step from this list that has just been completed (only say literally what the step is, word for word how it is on the list) and then followed by ~~ and then followed by which step is now the current step to work on (You must word it EXACTLY the same as it is on this list):\n'+task_steps 
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
    
    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print('got time')
    #get memory summary from file (file gets its data from memory thread)
    with open('memory_summary.txt','r') as f:
        memories = f.read()
    print('got memory summary')
    text = str(send_text_to_gpt4_convo(chat_history, last_phrase, task, task_steps, memories))
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

person = []
person_data = 'No data about person yet.'



move_stopper = False

        
 
def movement_loop(camera, raw_capture):
    global chat_history
    global frame
    global stop_threads
    global net
    global output_layers
    global classes
    global move_stopper
    global camera_vertical_pos
    global in_task
    global task
    global task_steps
    global person
    global person_data
    ina219 = INA219(addr=0x42)
    last_time = time.time()
    failed_response = ''
    movement_response = ' ~~  ~~ '
    yolo_nav = False
    #print('movement thread start')
    while not stop_threads:
        try:
            with open('current_history.txt','w+') as file:
                file.write('\n'.join(chat_history))
            last_phrase2 = get_last_phrase()
            if last_phrase2 != '':
                print('Last phrase now on movement loop: ' + last_phrase2)
                
                last_phrase2 = 'You just heard this from your microphone so THIS IS THE MAIN PART OF THE PROMPT (You need to respond to this by saying something most likely): ' + last_phrase2
                
            else:
                pass

            frame = capture_image(camera, raw_capture)
            cv2.imwrite('this_temp.jpg', frame)
        
      
            

            #do yolo and gpt visual threads
            
            yolo_thread = threading.Thread(target=yolo_detect)
            yolo_thread.start()
            yolo_thread.join()
            columns_thread = threading.Thread(target=get_column_height)
            columns_thread.start()
            columns_thread.join()
            
            
            current = ina219.getCurrent_mA() 
            bus_voltage = ina219.getBusVoltage_V()
            per = (bus_voltage - 6) / 2.4 * 100
            if per > 100: per = 100
            if per < 0: per = 0
            per = (per * 2) - 100
            with open('batt_per.txt','w+') as file:
                file.write(str(per))
            now = datetime.now()
            the_time = now.strftime("%m/%d/%Y %H:%M:%S")
            while True:
                try:
                    distance = int(read_distance_from_arduino())

                    
                    break
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
                    continue
            #get memory summary from file (file gets its data from memory thread)
            with open('memory_summary.txt','r') as f:
                memories = f.read()
            if current > 0.0 or per < 10.0:
                try:
                    stop_threads = True
                    last_time = time.time()
                    
                    #create summary of history
                    chat_summary = get_chat_summary(chat_history, memories)
                    #save to file
                    with open('memories/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.txt','w+') as f:
                        f.write(chat_summary)
                    print(chat_summary)
                    person_index = 0
                    while True:
                        try:
                            print('\nCURRENT PERSON')
                            print(person[person_index].replace('.',''))
                            if len(person[person_index].replace('.',''))>1:
                                summary = get_person_summary(chat_history, person[person_index].replace('.',''))
                                print('Got Summary')
                                now = datetime.now()
                                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                                try:
                                    file = open('people/'+person[person_index].replace('.','') + '.txt', 'r')
                                    file.close()
                                    file = open('people/'+person[person_index].replace('.','') + '.txt', 'a')
                                    file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                    file.close()
                                    print('Appended to memory file')
                                except:
                                    print(traceback.format_exc())
                                    try:
                                        file = open('people/'+person[person_index].replace('.','') + '.txt', 'w+')
                                        file.write('Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                        file.close()
                                        print('Created memory file')
                                    except:
                                        print(traceback.format_exc())
                            else:
                                pass
                            person_index += 1
                            if person_index >= len(person):
                                break
                            else:
                                continue
                        except Exception as e:
                            print(traceback.format_exc())
                            person_index += 1
                            if person_index >= len(person):
                                break
                            else:
                                continue
                    #print('ending convo')
                    chat_history = []
                except Exception as e:
                    print(traceback.format_exc())
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

                    if yolo_nav == True:
                        print('navigating')
                        #do yolo navigation to specific object
                        yolo_nav_index = 0
                        with open('output.txt','r') as file:
                            yolo_detections = file.readlines()
                        while True:
                            try:
                                current_detection = yolo_detections[yolo_nav_index]
                                current_distance1 = current_detection.split(' ') #extract distance
                                current_distance = float(current_distance1[current_distance1.index('meters')-1])
                                if nav_object in current_detection and current_distance > float(nav_distance.split(' ')[0]):
                                    print('object seen')
                                    #follow any human seen
                                    if 'Turn Left 15 Degrees' in current_detection:
                                        movement_response = 'Turn Left 15 Degrees ~~ Target object is to the left ~~ '
                                    elif 'Turn Right 15 Degrees' in current_detection:
                                        movement_response = 'Turn Right 15 Degrees ~~ Target object is to the right ~~ '
                                    else:
                                        movement_response = 'Move Forward One Foot ~~ Target object is straight ahead ~~ '
                                    
                                    break
                                else:
                                    if current_distance <= float(nav_distance.split(' ')[0]):
                                        movement_response = 'No Movement ~~ Navigation has finished Successfully! ~~ '
                                        yolo_nav = False
                                        break
                                    else:
                                        yolo_nav_index += 1
                                        if yolo_nav_index >= len(yolo_detections):
                                            movement_response = 'No Movement ~~ Target object lost ~~ '
                                            yolo_nav = False
                                            break
                                        else:
                                            continue
                            except:
                                movement_response = 'No Movement ~~ Yolo navigation failed. Must be detecting object first. ~~ '
                                break
                    else:
                        movement_response = str(send_text_to_gpt4_move(chat_history, per, distance, last_phrase2, task, task_steps, person, person_data, memories, failed_response)).replace('Response Choice: ','').replace('Movement Choice at this timestamp: ','').replace('Response Choice at this timestamp: ','').replace('Attempting to do movement response choice: ','')
                        
                    chat_history.append('Time: ' + str(the_time) + ' - Robot action response at this timestamp: ' + movement_response)  # add response to chat history                   
                    
                    try:
                        print("\nPercent:       {:3.1f}%".format(per))
                        print('\nCurrent Distance: ' + str(distance) + ' cm')
                        print('\nCurrent Task: ' + task)
                        print('\nTask Steps:\n'+task_steps)
                        print('\nResponse Choice: '+ movement_response.split('~~')[0].strip().replace('.',''))
                        print('\nReasoning: '+ movement_response.split('~~')[1].strip())
                        person_list = movement_response.split('~~')[2].strip().replace('.','').split(', ')
                        person_dex = 0
                        while True:
                            if person_list[person_dex] not in person:
                                person.append(person_list[person_dex])
                            else:
                                pass
                                person_dex += 1
                                if person_dex >= len(person_list):
                                    break
                                else:
                                    continue
                                
                        print(person)
                        person_index = 0
                        person_data = ''
                        while True:
                            try:
                                with open('people/'+person[person_index]+'.txt','r') as f:
                                    if person_index == 0:
                                        person_data = f'Memory data for {person[person_index]}:\n\n'+f.read()
                                    else:
                                        person_data = person_data + f'\n\n\n\nMemory data for {person[person_index]}:\n\n'+f.read()
                                person_index += 1
                                if person_index >= len(person):
                                    break
                                else:
                                    continue
                            except:
                                print(traceback.format_exc())
                                person_data = person_data + f'\n\n\n\nNo data about {person[person_index]} yet.'
                                person_index += 1
                                if person_index >= len(person):
                                    break
                                else:
                                    continue



                    except:
                        print(traceback.format_exc())
                    
                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    
                    current_response = movement_response.split('~~')[0].strip().replace('.','')
                    
                    #RESPONSE CHOICES LOOP

         

                    now = datetime.now()
                    the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                    
                    current_response = current_response.lower().replace(' ', '')
                    if current_response == 'moveforward1inch' or current_response == 'moveforwardoneinch':
                        if camera_vertical_pos != 'forward':
                            print('move forward 1 inch failed. not looking forward and center')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Camera angle must be lowered to the lowest angle before moving forward. You cannot choose Move Forward One Inch right now.')
                            failed_response = 'Move Forward One Inch, '
                            yolo_nav = False
                        elif distance < 20.0:
                            print('move forward 1 inch failed. Too close to obstacle to move forward anymore')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Inch Failed: Too close to obstacle to move forward anymore. You cannot choose Move Forward One Inch right now.')
                            failed_response = 'Move Forward One Inch, '
                            yolo_nav = False
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.1)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Inch')
                            failed_response = ''
                    elif current_response == 'moveforward1foot' or current_response == 'moveforwardonefoot':
                        if camera_vertical_pos != 'forward':
                            print('Move forward 1 foot Failed. Not looking forward and center')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Camera angle must be lowered to the lowest angle before moving forward. You cannot choose Move Forward One Foot right now.')
                            failed_response = 'Move Forward One Foot, '
                            yolo_nav = False
                        elif distance < 35.0:
                            print('move forward 1 foot failed. Too close to obstacle to move forward that far')
                            chat_history.append('Time: ' + str(the_time) + ' - Move Forward 1 Foot Failed: Too close to obstacle to move forward that far. You cannot choose Move Forward One Foot right now.')
                            failed_response = 'Move Forward One Foot, '
                            yolo_nav = False
                        else:
                            send_data_to_arduino(["w"], arduino_address)
                            time.sleep(0.4)
                            send_data_to_arduino(["x"], arduino_address)
                            chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Forward 1 Foot')
                            failed_response = ''
                    elif current_response == 'movebackward':
                        send_data_to_arduino(["s"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Moved Backward')
                        failed_response = ''
                    elif current_response == 'turnleft45degrees' or current_response == 'moveleft45degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Left 45 Degrees')
                        failed_response = ''
                    elif current_response == 'turnleft15degrees' or current_response == 'moveleft15degrees':
                        send_data_to_arduino(["a"], arduino_address)
                        time.sleep(0.05)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Left 15 Degrees')
                        failed_response = ''
                    elif current_response == 'turnright45degrees' or current_response == 'moveright45degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.2)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Right 45 Degrees')
                        failed_response = ''
                    elif current_response == 'turnright15degrees' or current_response == 'moveright15degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.05)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Right 15 Degrees')
                        failed_response = ''
                    elif current_response == 'turnaround180degrees':
                        send_data_to_arduino(["d"], arduino_address)
                        time.sleep(0.8)
                        send_data_to_arduino(["x"], arduino_address)
                        chat_history.append('Time: ' + str(the_time) + ' - Successfully Turned Around 180 Degrees')
                        failed_response = ''
                    elif current_response == 'starttask' and in_task == False:
                        print('\nStarting Task')
                        task_response = get_task(chat_history, last_phrase2)
                        in_task = True
                        task = task_response.split('~~')[0].strip()
                        task_steps = task_response.split('~~')[1].strip()
                    elif current_response == 'endtask' and in_task == True:
                        in_task = False
                        print('\nEnding Task')
                        task = 'No task is currently set.'
                        task_steps = ''
                    elif current_response == 'markoffacompletedtaskstep' and in_task == True:
                        print('Marking off a completed task step')
                        step_response = mark_off_a_completed_step(chat_history, last_phrase2, task, task_steps).split('~~')[0]
                        current_step = mark_off_a_completed_step(chat_history, last_phrase2, task, task_steps).split('~~')[1]
                        task_steps = task_steps.replace(step_response.strip(),'STEP IS COMPLETE: '+step_response.strip()).replace(current_step.strip(),'CURRENT STEP: '+current_step.strip())
                    elif current_response == 'raisecameraangle':
                        if camera_vertical_pos == 'up':
                            chat_history.append('Time: ' + str(the_time) + ' - Raise Camera Angle Failed: Camera angle is already raised as much as possible. You cannot choose Raise Camera Angle right now.')
                            print('Raise Camera Angle Failed. Camera angle is already raised as much as possible.')
                            failed_response = 'Raise Camera Angle, '
                        else:
                            send_data_to_arduino(["2"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            if camera_vertical_pos == 'middle':
                                camera_vertical_pos = 'up'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Raised the Camera Angle to the max upward angle.')
                            else:
                                camera_vertical_pos = 'middle'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Raised the Camera Angle to the middle upwards angle.')
                    elif current_response == 'lowercameraangle':
                        if camera_vertical_pos == 'forward':
                            chat_history.append('Time: ' + str(the_time) + ' - Lower Camera Angle Failed: Camera angle is already lowered as much as possible. You cannot choose Lower Camera Angle right now.')
                            print('Lower Camera Angle failed. Camera angle is already lowered as much as possible.')
                            failed_response = 'Lower Camera Angle, '
                        else:
                            send_data_to_arduino(["1"], arduino_address)
                            time.sleep(1.5)
                            failed_response = ''
                            if camera_vertical_pos == 'middle':
                                camera_vertical_pos = 'forward'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Lowered the Camera Angle to the lowest angle, which is looking directly forward.')
                            else:
                                camera_vertical_pos = 'middle'
                                chat_history.append('Time: ' + str(the_time) + ' - Successfully Raised the Camera Angle to the middle upwards angle.')
                    elif current_response == 'endconversation' or current_response == 'goodbye':
                        with open('playback_test.txt', 'w') as f:
                            f.write('Good bye!')
                        last_time = time.time()
                        #create summary of history
                        chat_summary = get_chat_summary(chat_history, memories)
                        #save to file
                        with open('memories/'+str(the_time).replace('/','-').replace(':','-').replace(' ','_')+'.txt','w+') as f:
                            f.write(chat_summary)
                        person_index = 0
                        while True:
                            try:
                                print('\nCURRENT PERSON')
                                print(person[person_index].replace('.',''))
                                summary = get_person_summary(chat_history, person[person_index].replace('.',''))
                                print('Got Summary')
                                now = datetime.now()
                                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                                try:
                                    file = open('people/'+person[person_index].replace('.','') + '.txt', 'r')
                                    file.close()
                                    file = open('people/'+person[person_index].replace('.','') + '.txt', 'a')
                                    file.write('\n\n Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                    file.close()
                                    print('Appended to memory file')
                                except:
                                    print(traceback.format_exc())
                                    try:
                                        file = open('people/'+person[person_index].replace('.','') + '.txt', 'w+')
                                        file.write('Convo and Action Summary from ' + the_time + ':\n\n' + summary)
                                        file.close()
                                        print('Created memory file')
                                    except:
                                        print(traceback.format_exc())
                                person_index += 1
                                if person_index >= len(person):
                                    break
                                else:
                                    continue
                            except Exception as e:
                                print(traceback.format_exc())
                                person_index += 1
                                if person_index >= len(person):
                                    break
                                else:
                                    continue
                        print('ending convo')
                        chat_history = []
                        stop_threads = True
                        break
                    elif current_response == 'nomovement':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response choice was No Movement so not moving.')
                    elif current_response == 'saysomething' or current_response == 'alertuser':
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Attempting to say:'+movement_response.split('~~')[1])
                        with open('self_response.txt','w') as f:
                            f.write(movement_response.split('~~')[1])
                    elif current_response == 'navigatetospecificyoloobject':
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(0.5)
                        send_data_to_arduino(["1"], arduino_address)
                        time.sleep(1.5)
                        camera_vertical_pos = 'forward'
                        yolo_nav = True
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        nav_object = movement_response.split('~~')[1].split(' - ')[0]
                        nav_distance = movement_response.split('~~')[1].split(' - ')[1]
                        chat_history.append('Time: ' + str(the_time) + ' - Starting navigation to'+movement_response.split('~~')[1].split(' - ')[0])
                        
                    else:
                        now = datetime.now()
                        the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                        chat_history.append('Time: ' + str(the_time) + ' - Response failed. You didnt follow my instructions properly for how you should respond. Here is what you responded with so dont do it again: ' + movement_response)
                        print('failed response')
                        
                    
                    
                    
            except:
                print(traceback.format_exc())
                
            
            
        except:
            print(traceback.format_exc())
            
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

                yolo_thread = threading.Thread(target=yolo_detect)
                yolo_thread.start()
                yolo_thread.join()
                print('Saying Greeting')
                with open('last_phrase.txt', 'w') as file:
                    file.write('')
                say_greeting(last_phrase)
                now = datetime.now()
                the_time = now.strftime("%m/%d/%Y %H:%M:%S")
                print('starting thread')
                movement_thread = threading.Thread(target=movement_loop, args=(camera, raw_capture))
                memory_thread = threading.Thread(target=get_memory_summary)
                memory_thread.start()
                time.sleep(1)
                movement_thread.start()
                print('threads started')
                movement_thread.join()
                memory_thread.join()
            else:
                

                if current >= 0.0 or be_still == True or 0==0:
                    pass
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
