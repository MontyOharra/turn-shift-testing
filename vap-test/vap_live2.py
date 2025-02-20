import pyaudio
import wave
import json
import VoiceActivityProjection.run2

def getPnowPfuture(json_file):
    with open(json_file) as f:
        data = json.load(f)
    pnow = data['p_now']
    pnow = pnow[-1]
    pnow = pnow[0]
    pfuture = data['p_future']
    pfuture = pfuture[-1]
    pfuture = pfuture[0]
    return pnow, pfuture

def record_audio(
    output_filename="output.wav",
    record_seconds=5,
    rate=16000,
    chunk=1024,
    channels=1,
    input_device_index=None,
):
    """
    Records audio for `record_seconds` and saves it to `output_filename`.
    
    :param output_filename: Name of the WAV file to save (e.g. "output.wav").
    :param record_seconds: Number of seconds to record.
    :param rate: Sample rate in Hz.
    :param chunk: Buffer size; number of frames per buffer.
    :param channels: Number of audio channels.
    :param input_device_index: (Optional) Device index if you want a specific input device.
    """

    p = pyaudio.PyAudio()

    # Print available input devices (for debugging)
    print("Available devices:")
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        max_in_channels = dev_info.get('maxInputChannels', 0)
        print(f"  Device {i}: {dev_info['name']} (max input channels = {max_in_channels})")

    # Open the audio stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=input_device_index,
    )

    print(f"\nRecording for {record_seconds} seconds...")

    frames = []
    for _ in range(int(rate / chunk * record_seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)

    print("Finished recording!")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data to a WAV file
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Saved to {output_filename}")


if __name__ == "__main__":
    record_audio(
        output_filename="output.wav",
        record_seconds=5,         # Adjust as desired
        rate=48000,               # Common sample rate
        chunk=1024,
        channels=1,
        input_device_index=0,  # Set to an integer if you want a specific mic
    )
    #VoiceActivityProjection.run2.runVAP("-a output.wav", "-sd VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt", "-f test.json")
    VoiceActivityProjection.run2.runVAP()
    pnow, pfuture = getPnowPfuture("test.json")
    print("pnow: ", pnow)
    print("pfuture: ", pfuture)