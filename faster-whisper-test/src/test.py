import pyaudio
from faster_whisper import WhisperModel
import os
import wave
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device.')

def transcribeChunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def recordChunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(8000/ 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def main2():
    print("starting up")
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # print("device: ", device)

    print("check1")
    p = pyaudio.PyAudio()
    print("check2")
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    try:
        while True:
            print("Recording...")
            chunk_file = "temp_chunk.wav"
            recordChunk(p, stream, chunk_file)
            transcription = transcribeChunk(model, chunk_file)
            print(transcription)
            os.remove(chunk_file)
    except KeyboardInterrupt: #ctrl + c
        print("stopping...")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()