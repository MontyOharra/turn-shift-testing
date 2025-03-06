import pyaudio
import wave
import json
from faster_whisper import WhisperModel
from pydub import AudioSegment
from threading import Thread
from queue import Queue
from turngpt.model import TurnGPT
from typing import Optional

import sys
import os

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/full-turn-shift/src/"))
import utils
from utils import calculateTurnShiftFromTranscription, printTranscription

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/vap-test/VoiceActivityProjection/"))
import run2
from run2 import runVAP

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


def transcribeChunk(
        model : TurnGPT, 
        file_path : str
    ) -> str :
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def getAudioTranscription2(
    liveTranscription: Queue,
    input_device_index: int,
    channels: int = 1,
    chunk_size: int = 1024,
    segment_duration: float = 2,   # seconds per segment for transcription
    overlap_duration: float = 0.5,   # seconds to overlap between segments
    rate: Optional[int] = None,
    save_duration: float = 5  # Save audio every 'save_duration' seconds (e.g., every 10 seconds)
):
    # Initialize the Whisper model
    model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(input_device_index)
    if rate is None:
        rate = int(device_info["defaultSampleRate"])
    default_format = pyaudio.paInt16

    # Open the audio stream
    stream = p.open(
        format=default_format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index,
    )

    # Calculate how many frames (chunks) constitute one segment and the overlap.
    frames_per_segment = int(rate / chunk_size * segment_duration)
    frames_per_overlap = int(rate / chunk_size * overlap_duration)

    audio_buffer = []  # will hold the incoming audio chunks
    total_frames = 0  # To track total frames for saving every `save_duration` seconds
    output_filename = "chunk.wav"  # Fixed filename to save audio chunks

    print("Recording continuously (press Ctrl+C to stop)...")
    try:
        while True:
            # Read one chunk from the stream
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_buffer.append(data)
            total_frames += chunk_size

            # When the buffer has accumulated enough frames for one segment...
            if len(audio_buffer) >= frames_per_segment:
                # Save audio to a fixed file ("chunk.wav") every 'save_duration' seconds
                if total_frames >= rate * save_duration:
                    with wave.open(output_filename, 'wb') as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(p.get_sample_size(default_format))
                        wf.setframerate(rate)
                        wf.writeframes(b''.join(audio_buffer))
                    combine_wav("output.wav", output_filename)
                    print(f"Saved audio chunk to '{output_filename}'.")
                    runVAP(output_filename="output.wav", output_json="test.json")
                    pnow, pfuture = getPnowPfuture("test.json")
                    print("pnow: ", pnow)
                    print("pfuture: ", pfuture)
                    if pnow[0] < 0.4 and pfuture[0] < 0.4:
                        print("Turn detected")
                    # Optionally, transcribe the saved chunk (this can be done asynchronously)
                    transcription = transcribeChunk(model, output_filename)
                    liveTranscription.put(transcription)


                    # Reset the buffer and frame counter after saving
                    audio_buffer = []
                    total_frames = 0

                # Remove old frames and keep only the last N frames (the overlap)
                audio_buffer = audio_buffer[-frames_per_overlap:]
    except KeyboardInterrupt:
        print("Stopping continuous transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    turnGptQueue = Queue()
    liveTranscription = Thread(
        target=getAudioTranscription2,
        args=(
            turnGptQueue,
            4,
            1,
            1024,
            2,      # record 2-second segments
            0.1,    # with 0.5-second overlap
            48000               # set the sample rate your mic supports
        ),
        daemon=True      
    )

    outputTranscription = Thread(
        target=calculateTurnShiftFromTranscription,
        args=(turnGptQueue,),
        daemon=True
    )

    liveTranscription.start()
    outputTranscription.start()
    try:
        while True:
            pass

    except KeyboardInterrupt:
        transcription = utils.transcribeChunk(
                model=whisper_model,
                file_path="output.wav"
            )
        print("Transcription: ", transcription)

        if os.path.exists("output.wav"):
            os.remove("output.wav")

        if os.path.exists("test.json"):
            os.remove("test.json")
        print("\nLoop interrupted by user.")
    