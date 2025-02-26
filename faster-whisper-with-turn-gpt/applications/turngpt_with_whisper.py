import pyaudio
from src.utils import getAudioTranscription, printTranscription, calculateTurnShiftFromTranscription
from threading import Thread
from queue import Queue

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