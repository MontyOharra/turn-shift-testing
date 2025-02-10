import numpy as np
import sounddevice as sd
import whisper
import threading

command = ""

# OpenAI's Whisper model
model = whisper.load_model("small")

SAMPLE_RATE = 48000  # 16 kHz sample rate
BUFFER_DURATION = 2  # how long in seconds it waits before transcribing

audio_buffer = np.array([], dtype=np.float32)

stop_recording = False
finished_processing = False

# Lock to synchronize access to audio_buffer
audio_buffer_lock = threading.Lock()

# Processes audio
def audio_processing():
    global audio_buffer
    while not stop_recording:
        with audio_buffer_lock:
            if len(audio_buffer) >= SAMPLE_RATE * BUFFER_DURATION:
                # Transcribe current chunk of audio
                audio_np = audio_buffer.copy() 
                audio_buffer = np.array([], dtype=np.float32)  # clear buffer
                
        if len(audio_np) > 0:  # If there was a chunk to process
            result = model.transcribe(audio_np, temperature=0)
            print("I heard:", result['text'])
            global command
            command = result['text']

        # Sleep for a short time to prevent tight CPU usage in the loop
        threading.Event().wait(0.1)

def callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(status)

    # Convert captured audio to float32
    audio_chunk = indata.copy().flatten().astype(np.float32)
    
    with audio_buffer_lock:
        audio_buffer = np.append(audio_buffer, audio_chunk)

def main():
    # Starts live recording
    with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1, dtype='float32'):
        print("Listening for a command...")

        processing_thread = threading.Thread(target=audio_processing)
        processing_thread.start()
        input("Press Enter to stop recording...")
        global stop_recording, finished_processing
        stop_recording = True
        print("Processing command...")
        # Wait for the audio processing thread to finish
        processing_thread.join()
        finished_processing = True
        print(command)

if __name__ == "__main__":
    main()
