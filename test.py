import wave
from faster_whisper import WhisperModel
import pyaudio

def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0. int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

def main2():
    model_size = "medium.en"
    model = WhisperModel(model_size, device="cude", compute_type="float16")

    p = pyaudio.PyAudio()