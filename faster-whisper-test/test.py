from faster_whisper import WhisperModel
import pyaudio
from src.utils import recordChunk, transcribeChunk

def main():
 # Test with the default audio device

    # model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")
    # model = WhisperModel(model_size_or_path="medium.en", device="cpu", compute_type="float32")
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"Device {i}: {info['name']}")

    # Input device index 9 worked for my airpods. I'm not sure if they're recognized as 9 or if that is generally the OS
    # detecting an audio device. Input device index 8 worked as well
    # snowball mic is 2
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=9)

    try:
        while True:
            # Open temp_chunk.wav to see if audio is being properly recorded
            chunk_file = "temp_chunk.wav"
            # For some reason, in periods of silence, this thinks it is hearing "Thank you very much"
            # Chunk size lower than 1 makes it hard to not get "Thank you very much" as a transcription
            recordChunk(p, stream, chunk_file, chunk_length=1)
            transcription = transcribeChunk(model, chunk_file)
            print(transcription)
    except KeyboardInterrupt: #ctrl + c
        print("stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()