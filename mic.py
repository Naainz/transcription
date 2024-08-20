# USE THIS SCRIPT TO LIST ALL MICROPHONES / INPUT DEVICES
# CONNECTED TO YOUR SYSTEM

import pyaudio

def list_microphones():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        print(f"Device Index {i}: {device_info['name']}")

if __name__ == "__main__":
    list_microphones()
