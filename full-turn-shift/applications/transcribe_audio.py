import pyaudio
from src.utils import getAudioTranscription, printTranscription
from threading import Thread
from queue import Queue
import signal
import sys

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
            48000   # set the sample rate your mic supports
        ),
        daemon=False
    )

    outputTranscription = Thread(
        target=printTranscription,
        args=(turnGptQueue,),
        daemon=False
    )

    # Start threads
    liveTranscription.start()
    outputTranscription.start()

    # Gracefully handle keyboard interrupt
    def exit_gracefully(signum, frame):
        print("Exiting gracefully...")
        turnGptQueue.put(None)
        liveTranscription.join()
        outputTranscription.join()
        sys.exit(0)

    # Set up signal handler
    signal.signal(signal.SIGINT, exit_gracefully)

    # Block main thread until KeyboardInterrupt
    signal.pause()

if __name__ == "__main__":
    main()