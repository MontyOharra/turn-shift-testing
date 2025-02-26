import pyaudio
import wave
import json
import VoiceActivityProjection.run2
from faster_whisper import WhisperModel
from pydub import AudioSegment

import sys
import os

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/faster-whisper-with-turn-gpt/applications"))
import record_audio

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/faster-whisper-with-turn-gpt/src"))
import utils

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/turn-gpt-test"))
import test as TurnGPTTest

whisper_model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")

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

def create_empty_wav(filename='output.wav', num_channels=1, sample_width=2, frame_rate=48000, num_frames=0):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)  
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)    
        wav_file.setnframes(num_frames)   
        wav_file.writeframes(b'')

create_empty_wav('output.wav')

def combine_wav(file1, file2, output_filename="output.wav", max_duration_ms=2 * 60 * 1000):
    # Load the audio files
    audio1 = AudioSegment.from_wav(file1)
    audio2 = AudioSegment.from_wav(file2)
    
    # Append the two audio segments
    combined_audio = audio1 + audio2
    
    if len(combined_audio) > max_duration_ms:
        combined_audio = combined_audio[-max_duration_ms:]
    
    combined_audio.export(output_filename, format="wav")
    
    print(f"Audio saved to {output_filename}. Total duration: {len(combined_audio) / 1000:.2f} seconds.")


if __name__ == "__main__":
    try:
        while True:
            record_audio.record_audio(
                output_filename="chunk.wav",
                record_seconds=.25,         # adjust this -- lower for more "live" predictions but this interferes with transcription (repeated segments/words)
                rate=48000,               # Common sample rate
                chunk=1024,
                channels=1,
                input_device_index=0,  # Set to an integer if you want a specific mic
            )
            combine_wav('output.wav', 'chunk.wav')
            if AudioSegment.from_wav('output.wav').duration_seconds > 5:
                VoiceActivityProjection.run2.runVAP(output_filename="output.wav", output_json="test.json")    
                pnow, pfuture = getPnowPfuture("test.json")
                print("pnow: ", pnow)
                print("pfuture: ", pfuture)
                # TurnGPTTest.main()
                # TurnGPTTest.onInput(transcription)
                if(pnow[0] < .4 and pfuture[0] < .4):
                    print("Turn detected")

    except KeyboardInterrupt:
        transcription = utils.transcribeChunk(
                model=whisper_model,
                file_path="output.wav"
            )
        print("Transcription: ", transcription)
        os.remove("output.wav")
        os.remove("test.json")
        print("\nLoop interrupted by user.")
    