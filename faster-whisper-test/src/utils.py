import pyaudio
import wave

def transcribeChunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def recordChunk(p,
                stream, 
                format,
                rate,
                channels,
                file_path, 
                chunk_size,
                chunk_length=1):
    frames = []
    for _ in range(0, int(rate / chunk_size * chunk_length)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
