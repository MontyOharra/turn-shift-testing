import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import tempfile
import os
import time
import whisper


# Constants
SAMPLE_RATE = 44100  # 44.1 kHz sample rate
DURATION = 5  # Duration in seconds for recording

model = whisper.load_model("small")

def transcribe(model, file_path):
    print("in here")
    result = model.transcribe(file_path)
    print(len(result["text"]))
    print(result["text"])
    return result["text"]

# Callback to record audio
def record_audio():
    print("Recording...")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording finished.")
    return audio_data.flatten()

# Function to save audio to an MP3 file and then delete it
def save_and_delete_audio(audio_data):
    # Convert the numpy array to an AudioSegment (from pydub)
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=SAMPLE_RATE,
        sample_width=audio_data.dtype.itemsize,
        channels=1
    )

    # Create a temporary file to save the MP3
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_filename = "howdy.mp3"
        
        # Export the audio as an MP3 file
        audio_segment.export(temp_filename, format="mp3")
        print(f"Saved audio to {temp_filename}")
        # Wait for 2 seconds before deleting
        time.sleep(2)  # Just to ensure it's saved and processed
        print(transcribe(model, temp_filename))
        # os.remove(temp_filename)  # Delete the file
        # print(f"Deleted {temp_filename}")

# Main function
def main():
    # Record the audio
    audio_data = record_audio()

    # Save and delete the audio file
    save_and_delete_audio(audio_data)

if __name__ == "__main__":
    main()

