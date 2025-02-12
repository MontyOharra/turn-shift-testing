import pyaudio
import wave
import json
import VoiceActivityProjection.run

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

def getPnowPfuture(json_file):
    with open(json_file) as f:
        data = json.load(f)
    pnow = data['pnow']
    pnow = pnow[-1]
    pnow = pnow[0]
    pfuture = data['pfuture']
    pfuture = pfuture[-1]
    pfuture
    return pnow, pfuture

def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=0)
    try:
        while True:
            chunk_file = "temp_chunk.wav"
            # chunk_file = "small_talk.wav"
            recordChunk(p, stream, chunk_file, chunk_length=2)
            VoiceActivityProjection.run("-a temp_chunk.wav", "-sd VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt", "-f test.json")
            #extract last pnow pfuture from json file?
            pnow, pfuture = getPnowPfuture("test.json")
            if pnow > .5 and pfuture > .5:
                print("turn")

    except KeyboardInterrupt: #ctrl + c
        print("stopping...")
    finally:
        stream.stop_stream()
        stream.close()
