import subprocess
import sys
import os
import re
import argparse
import time

def modify_file(file_path, search_exp, replace_exp):
    with open(file_path, 'r') as file:
        data = file.read()
    modified_data = re.sub(search_exp, replace_exp, data)
    with open(file_path, 'w') as file:
        file.write(modified_data)

def set_default_audio_device(card_number):
    config = f'''
pcm.!default {{
    type hw
    card {card_number}
}}
ctl.!default {{
    type hw
    card {card_number}
}}
'''
    with open('/etc/asound.conf', 'w') as file:
        file.write(config)
    print(f"Default audio device set to WM8960 Audio HAT on card {card_number}.")

def check_and_modify_boot_config():
    boot_config = "/boot/config.txt"
    try:
        with open(boot_config, 'r') as file:
            lines = file.readlines()
        
        modifications = {
            "dtparam=audio=on": "#dtparam=audio=on",  # Disable onboard audio
            "dtoverlay=wm8960-soundcard": "dtoverlay=wm8960-soundcard"  # Ensure WM8960 is enabled
        }
        
        modified = False
        with open(boot_config, 'w') as file:
            for line in lines:
                for key, value in modifications.items():
                    if key in line:
                        line = f"{value}\n"
                        modified = True
                file.write(line)
        
        if not modified:
            with open(boot_config, 'a') as file:
                file.write("dtoverlay=wm8960-soundcard\n")
        
        print(f"{boot_config} modified successfully.")
    except Exception as e:
        print(f"Failed to modify {boot_config}: {e}")

def check_and_modify_modules():
    modules_file = "/etc/modules"
    required_modules = ["snd_soc_wm8960", "snd_soc_bcm2835_i2s", "i2c-dev"]

    try:
        with open(modules_file, 'r') as file:
            lines = file.readlines()

        with open(modules_file, 'w') as file:
            for line in lines:
                file.write(line)
            
            for module in required_modules:
                if module not in lines:
                    file.write(f"{module}\n")

        print(f"{modules_file} updated successfully.")
    except Exception as e:
        print(f"Failed to update {modules_file}: {e}")

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

def install_package(package):
    subprocess.check_call(["sudo", "pip3", "install", package])

def install_apt_package(package):
    subprocess.check_call(["sudo", "apt", "install", "-y", package])

def setup_waveshare_audio_hat():
    print("Setting up Waveshare WM8960 Audio HAT...")
    
    subprocess.check_call(["git", "clone", "https://github.com/waveshare/WM8960-Audio-HAT"])
    
    # Replace files with the ones from the specified URLs
    wm8960_file_path = "WM8960-Audio-HAT/wm8960.c"
    wm8960_soundcard_file_path = "WM8960-Audio-HAT/wm8960-soundcard.c"
    
    # Download and replace the wm8960.c file
    subprocess.check_call(["wget", "-O", wm8960_file_path, 
                           "https://raw.githubusercontent.com/garagesteve1155/chatgpt_robot/main/wm8960.c"])
    
    # Download and replace the wm8960-soundcard.c file
    subprocess.check_call(["wget", "-O", wm8960_soundcard_file_path, 
                           "https://raw.githubusercontent.com/garagesteve1155/chatgpt_robot/main/wm8960-soundcard.c"])
    
    install_script_path = "WM8960-Audio-HAT/install.sh"
    
    if os.path.exists(install_script_path):
        modify_install_sh(install_script_path)
        subprocess.check_call(["sudo", "./install.sh"], cwd="WM8960-Audio-HAT")
        print("Rebooting to apply audio HAT setup...")
        card_number = get_wm8960_card_number()
        set_default_audio_device(card_number)
        subprocess.check_call(["sudo", "reboot"])
    else:
        print("Error: install.sh script not found!")

import wave
import audioop

import subprocess

def test_audio(card_number):
    print("Testing audio playback using espeak...")

    try:
        # Generate an audio file using espeak
        espeak_text = "Audio playback test successful. This is a test of the WM8960 audio HAT."
        espeak_output_file = "espeak_test.wav"
        
        subprocess.check_call([
            "espeak",
            espeak_text,
            "--stdout",
            "-w", espeak_output_file  # Write output to a WAV file
        ])

        # Play the generated audio file through the WM8960
        subprocess.check_call([
            "aplay",
            "-D", "plughw:{}".format(card_number),
            espeak_output_file
        ])
        
        print("Audio test passed using espeak.")
        
        # Record audio from the microphone at the same quality as espeak output
        mic_test_file = "mic_test.wav"
        print("Recording audio from microphone...")

        subprocess.check_call([
            "arecord",
            "-D", "plughw:{}".format(card_number),
            "-f", "S16_LE",
            "-c", "1",  # Mono channel
            "-r", "44100",  # Sample rate 44100 Hz
            "-d", "5",  # Record for 5 seconds
            mic_test_file
        ])

        # Play the recorded microphone audio
        print("Playing back the recorded audio...")
        subprocess.check_call([
            "aplay",
            "-D", "plughw:{}".format(card_number),
            mic_test_file
        ])

        print("Microphone test completed.")
        
    except subprocess.CalledProcessError as e:
        print("Audio test failed with error: ", e)

def modify_install_sh(script_path):
    print(f"Modifying {script_path} to use correct folder paths...")
    with open(script_path, 'r') as file:
        script_content = file.read()
    
    script_content = script_content.replace('/boot/firmware/config.txt', '/boot/config.txt')

    with open(script_path, 'w') as file:
        file.write(script_content)
    print("Modification of install.sh completed.")

def test_packages():
    print("Testing installed packages...")
    packages = ["pyaudio", "numpy", "speech_recognition", "webrtcvad", "requests", "bluetooth", "cv2"]
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"{package} is installed correctly.")
        except ImportError:
            print(f"Error: {package} is not installed.")

def test_camera():
    print("Testing camera capture...")
    try:
        # Capturing the image
        subprocess.check_call(["raspistill", "-o", "test.jpg"])
        print("Check test.jpg to verify camera capture.")
        import cv2
        import numpy as np
        # Load class labels from coco.names file
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Load YOLOv4-tiny configuration and weights
        net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # Load image
        image = cv2.imread("test.jpg")
        height, width, channels = image.shape

        # Convert image to blob
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Perform the detection
        outs = net.forward(output_layers)

        # Check for any object detected
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Print detected objects
        for i, box in enumerate(boxes):
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            print(f"Detected: {label} with confidence {confidence}")

        if boxes:
            print(f"Total objects detected: {len(boxes)}")
        else:
            print("No objects detected.")

        print("Camera test passed.")
    except subprocess.CalledProcessError:
        print("Camera test failed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def test_bluetooth():
    print("Testing Bluetooth connection to HC-05...")
    import bluetooth
    address = find_hc05_address()
    if not address:
        print("HC-05 module not found.")
        return

    port = 1
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    try:
        sock.connect((address, port))
        print(f"Connected to HC-05 at {address}")
        sock.send("l")
        data = sock.recv(1024)
        print(f"Received response from HC-05: {data.decode('utf-8')}")
    except bluetooth.BluetoothError as e:
        print(f"Bluetooth test failed: {e}")
    finally:
        sock.close()

def find_hc05_address():
    import bluetooth
    print("Scanning for HC-05 Bluetooth module...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True)
    for address, name in nearby_devices:
        if name == "HC-05":
            print(f"Found HC-05 module at address: {address}")
            return address
    print("HC-05 module not found.")
    return None
def pair_with_hc05(address, passkey):
    try:
        print(f"Trying to pair with HC-05 using passkey: {passkey}")
        subprocess.check_call([
            "expect", "-c",
            f'''
            spawn sudo bluetoothctl
            expect "#"
            send "pair {address}\r"
            expect "Passkey: "
            send "{passkey}\r"
            expect "#"
            send "trust {address}\r"
            expect "#"
            send "connect {address}\r"
            expect "#"
            send "quit\r"
            expect eof
            '''
        ])
        print(f"Paired and connected to HC-05 module at address: {address}")
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to pair with HC-05 using passkey: {passkey}")
        return False
def setup_bluetooth():
    print("Setting up Bluetooth...")
    subprocess.check_call(["sudo", "systemctl", "start", "bluetooth.service"])
    subprocess.check_call(["sudo", "bluetoothctl", "power", "on"])
    subprocess.check_call(["sudo", "bluetoothctl", "agent", "on"])

    address = find_hc05_address()
    if address:
        if not pair_with_hc05(address, "0000"):
            pair_with_hc05(address, "1234")
    else:
        print("HC-05 module not found. Please ensure the device is in pairing mode and try again.")
def install():
    
    print("Updating and upgrading system packages...")
    subprocess.check_call(["sudo", "apt", "update"])
    print("Installing system utilities and development packages...")
    install_apt_package("python3-pip")
    install_apt_package("git")
    install_apt_package("bluetooth")
    install_apt_package("bluez")
    install_apt_package("bluez-tools")
    install_apt_package("libportaudio2")
    install_apt_package("libatlas-base-dev")  # Required for NumPy
    install_apt_package("libopencv-dev")      # OpenCV dependencies and OpenCV installation
    install_apt_package("python3-opencv")     # OpenCV Python bindings
    install_apt_package("libbluetooth-dev")   # Bluetooth dependencies
    install_apt_package("libi2c-dev")         # For SMBus/I2C (corrected package name)
    install_apt_package("i2c-tools")
    install_apt_package("expect")             # Required for the interactive Bluetooth pairing
    install_apt_package("espeak")             # espeak installation for TTS
    
    print('Downloading YOLO files for object recognition')
    subprocess.check_call(["sudo", "wget", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"])
    subprocess.check_call(["sudo", "wget", "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"])
    subprocess.check_call(["sudo", "wget", "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"])
    print("Installing Python packages...")
    install_package("pyaudio")
    install_package("numpy")
    install_package("SpeechRecognition")
    install_package("webrtcvad")
    install_package("requests")
    install_package("pybluez")  # For Bluetooth

    setup_bluetooth()  # Set up Bluetooth before installing the Audio HAT to avoid reboot interruption
    
    subprocess.check_call(["wget", "https://raw.githubusercontent.com/garagesteve1155/chatgpt_robot/main/main.py"])
    setup_waveshare_audio_hat()  # Install and setup the audio HAT

def main():
    parser = argparse.ArgumentParser(description='Setup script for Raspberry Pi project.')
    parser.add_argument('--mode', type=str, choices=['install', 'test'], required=True,
                        help='Select mode: "install" to setup or "test" to verify installation.')
    args = parser.parse_args()

    if args.mode == 'install':
        install()
    elif args.mode == 'test':
        wm8960_card_number = get_wm8960_card_number()
        time.sleep(5)
        if wm8960_card_number:
            test_audio(wm8960_card_number)
            time.sleep(5)
        test_packages()
        time.sleep(5)
        test_camera()
        time.sleep(5)
        test_bluetooth()
        time.sleep(5)

if __name__ == "__main__":
    main()
