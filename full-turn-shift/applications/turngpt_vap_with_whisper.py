import pyaudio
from src.utils import getAudioTranscription, printTranscription, calculateTurnShiftFromTranscription
from threading import Thread
from queue import Queue


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



def main():
    turnGptQueue = Queue()
    liveTranscription = Thread(
        target=getAudioTranscription,
        args=(
            turnGptQueue,
            9,
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
            pass  # Main thread stays alive, waiting for interrupt

    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping threads...")
        turnGptQueue.put(None)  # Gracefully stop both threads
        liveTranscription.join()  # Ensure thread cleanup
        outputTranscription.join()  # Ensure thread cleanup
        print("All threads stopped.")

if __name__ == "__main__":
    main()