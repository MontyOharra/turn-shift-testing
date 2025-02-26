import pyaudio

def get_audio_devices():
    p = pyaudio.PyAudio()
    
    print("Available devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        max_in_channels = dev_info.get('maxInputChannels', 0)
        print(f"  Device {i}: {dev_info['name']} (max input channels = {max_in_channels})")

if __name__ == "__main__":
    get_audio_devices()