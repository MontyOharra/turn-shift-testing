import pyaudio
from src.utils import getAudioTranscription, calculateTurnShiftFromTranscription
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
        daemon=False      
    )

    outputTranscription = Thread(
        target=calculateTurnShiftFromTranscription,
        args=(turnGptQueue,),
        daemon=False
    )

    liveTranscription.start()
    outputTranscription.start()

    while True:
        try:
            pass
        except KeyboardInterrupt:
            turnGptQueue.put(None)
            break

    liveTranscription.join()
    outputTranscription.join()

if __name__ == "__main__":
    main()