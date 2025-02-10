from utils import transcribeChunk, recordChunk  
import pyaudio
from faster_whisper import WhisperModel
import os
import torch

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

def main2():
    print("starting up")
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float32") 
    print("Whisper model loaded.")

    print("check1")
    p = pyaudio.PyAudio()
    print("check2")

    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
        
    device_index = p.get_default_input_device_info()['index']
    default_sample_rate = p.get_device_info_by_index(device_index)['defaultSampleRate']
    print(f"Default sample rate: {default_sample_rate}")
    print(f"Using input device {device_index}")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=int(default_sample_rate), input=True, frames_per_buffer=65536)
    print("Stream opened successfully.")
    
    try:
        while True:
            print("Recording...")
            chunk_file = "temp_chunk.wav"
            recordChunk(p, stream, chunk_file)  
            transcription = transcribeChunk(model, chunk_file)  
            print(f"Transcription: {transcription}")
            with open('transcription.txt', 'a') as file:
                if transcription != "" and transcription != " Thank you for watching!" and transcription != " Thanks for watching!":
                    file.write(transcription + "\n") 
            os.remove(chunk_file)
            print(f"Removed temporary file {chunk_file}")
    except KeyboardInterrupt:  # ctrl + c
        print("stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main2()
