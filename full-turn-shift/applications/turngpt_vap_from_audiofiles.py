import os
import sys
import json
import io
from queue import Queue
from threading import Thread
from pydub import AudioSegment
from google.cloud import speech


import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials/google-api-key.json"

#the following envrion is being used for ~/turn-shift/full-turn-shift/applications$
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../credentials/google-api-key.json")
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/utils")))

from transcription_utils import calculateTurnShiftFromTranscription

# -------- CONFIG --------
SPEAKER1_FILE = 'josh_audio_1.wav'
SPEAKER2_FILE = 'rakesh_audio_1.wav'
CHUNK_DURATION_MS = 2000   # 2 seconds
OVERLAP_MS = 100           # 0.1 second overlap
SAMPLE_RATE = 48000        # Must match your WAV file
LANGUAGE_CODE = "en-US"

# -------- Transcription Utility --------
client = speech.SpeechClient()

def transcribe_chunk_google(audio_chunk):
    audio_chunk = audio_chunk.set_channels(1).set_sample_width(2).set_frame_rate(SAMPLE_RATE)
    buffer = io.BytesIO()
    audio_chunk.export(buffer, format="wav")
    buffer.seek(0)

    audio = speech.RecognitionAudio(content=buffer.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
    )

    response = client.recognize(config=config, audio=audio)
    transcripts = [result.alternatives[0].transcript for result in response.results]
    return " ".join(transcripts)

# -------- Chunking and Transcription Logic --------
def process_audio_file(filename, speaker_label, queue):
    audio = AudioSegment.from_wav(filename)
    step = CHUNK_DURATION_MS - OVERLAP_MS
    for start in range(0, len(audio) - CHUNK_DURATION_MS + 1, step):
        chunk = audio[start:start + CHUNK_DURATION_MS]
        transcript = transcribe_chunk_google(chunk)
        if transcript.strip():  # Ignore silent segments
            queue.put({
                "text": transcript,
                "speaker": speaker_label,
                "start_time": start / 1000,
                "end_time": (start + CHUNK_DURATION_MS) / 1000
            })

    print(f"Finished processing {speaker_label}")

# -------- Main Function --------
def main():
    turnGptQueue = Queue()

    # Start turn shift detection thread
    turn_shift_thread = Thread(
        target=calculateTurnShiftFromTranscription,
        args=(turnGptQueue,),
        daemon=True
    )
    turn_shift_thread.start()

    # Process both speakers (sequentially here, but can be parallelized)
    process_audio_file(SPEAKER1_FILE, "Speaker 1", turnGptQueue)
    process_audio_file(SPEAKER2_FILE, "Speaker 2",turnGptQueue)

    # Signal completion
    turnGptQueue.put(None)
    turn_shift_thread.join()
    print("Turn shift detection complete.")

if __name__ == "__main__":
    main()
    