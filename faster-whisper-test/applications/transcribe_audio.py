import pyaudio
from src.utils import getAudioTranscription
from threading import Thread
from queue import Queue


def main():
    turnGptQueue = Queue()
    turnGptThread = Thread(
        target=getAudioTranscription,
        args=(
            9,
            1,
            1024,
            2,      # record 2-second segments
            0.1,    # with 0.5-second overlap
            48000               # set the sample rate your mic supports
        ),
        daemon=False      
    )
    turnGptThread.start()

    





if __name__ == "__main__":
    main()