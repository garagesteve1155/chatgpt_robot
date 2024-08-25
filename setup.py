import subprocess
import sys
import argparse
import cv2

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_apt_package(package):
    subprocess.check_call(["sudo", "apt", "install", "-y", package])

def setup_waveshare_audio_hat():
    print("Setting up Waveshare WM8960 Audio HAT...")
    subprocess.check_call(["git", "clone", "https://github.com/waveshare/WM8960-Audio-HAT"])
    subprocess.check_call(["sudo", "./WM8960-Audio-HAT/install.sh"], cwd="WM8960-Audio-HAT")
    print("Rebooting to apply audio HAT setup...")
    subprocess.check_call(["sudo", "reboot"])

def setup_bluetooth():
    print("Setting up Bluetooth...")
    subprocess.check_call(["sudo", "systemctl", "start", "bluetooth.service"])
    subprocess.check_call(["sudo", "bluetoothctl", "power", "on"])
    subprocess.check_call(["sudo", "bluetoothctl", "agent", "on"])
    subprocess.check_call(["sudo", "bluetoothctl", "scan", "on"])
    print("Scanning for HC-05 Bluetooth module...")
    try:
        output = subprocess.check_output(["sudo", "bluetoothctl", "devices"], universal_newlines=True)
        for line in output.split('\n'):
            if "HC-05" in line:
                address = line.split()[1]
                print(f"Found HC-05 module at address: {address}")
                subprocess.check_call(["sudo", "bluetoothctl", "pair", address])
                subprocess.check_call(["sudo", "bluetoothctl", "trust", address])
                subprocess.check_call(["sudo", "bluetoothctl", "connect", address])
                print(f"Paired and connected to HC-05 module at address: {address}")
                break
        else:
            print("No HC-05 module found.")
    finally:
        subprocess.check_call(["sudo", "bluetoothctl", "scan", "off"])

def install():
    print("Updating and upgrading system packages...")
    subprocess.check_call(["sudo", "apt", "update"])
    subprocess.check_call(["sudo", "apt", "upgrade", "-y"])

    print("Installing Python3 pip and other system utilities...")
    install_apt_package("python3-pip")
    install_apt_package("git")
    install_apt_package("bluetooth")
    install_apt_package("bluez")
    install_apt_package("bluez-tools")

    print("Installing development packages...")
    install_apt_package("libportaudio2")
    install_apt_package("libportaudiocpp0")
    install_apt_package("portaudio19-dev")
    install_apt_package("libatlas-base-dev")  # Required for NumPy
    install_apt_package("libopencv-dev")      # OpenCV dependencies
    install_apt_package("python3-opencv")     # Install OpenCV
    install_apt_package("libbluetooth-dev")   # Bluetooth dependencies
    install_apt_package("libsmbus-dev")       # For SMBus/I2C
    install_apt_package("i2c-tools")

    print("Enabling I2C and Camera interfaces...")
    subprocess.check_call(["sudo", "raspi-config", "nonint", "do_i2c", "0"])
    subprocess.check_call(["sudo", "raspi-config", "nonint", "do_camera", "0"])

    print("Installing Python packages...")
    install_package("pyaudio")
    install_package("numpy")
    install_package("SpeechRecognition")
    install_package("opencv-python")
    install_package("opencv-python-headless")
    install_package("webrtcvad")
    install_package("requests")
    install_package("PyBluez")  # For Bluetooth

    setup_bluetooth()  # Set up Bluetooth before installing the Audio HAT to avoid reboot interruption
    setup_waveshare_audio_hat()

def test_audio():
    print("Testing audio recording and playback...")
    subprocess.call(["arecord", "-d", "5", "-f", "cd", "test-mic.wav"])
    subprocess.call(["aplay", "test-mic.wav"])

def test_camera():
    print("Testing camera capture...")
    subprocess.call(["raspistill", "-o", "test.jpg"])
    print("Check test.jpg to verify camera capture.")

    # Load and display the image using OpenCV
    image = cv2.imread("test.jpg")
    if image is not None:
        print("Displaying the captured image...")
        cv2.imshow("Captured Image", image)
        cv2.waitKey(0)  # Wait for a key press to close
        cv2.destroyAllWindows()
    else:
        print("Failed to load the captured image.")

def main():
    parser = argparse.ArgumentParser(description='Setup script for Raspberry Pi project.')
    parser.add_argument('--mode', type=str, choices=['install', 'test'], required=True,
                        help='Select mode: "install" to setup or "test" to verify installation.')
    args = parser.parse_args()

    if args.mode == 'install':
        install()
    elif args.mode == 'test':
        test_audio()
        test_camera()

if __name__ == "__main__":
    main()
