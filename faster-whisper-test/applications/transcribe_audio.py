import pyaudio
from src.utils import getAudioTranscription, printTranscription
from threading import Thread, Event
from queue import Queue

def main():
    # This queue will be used for sending recognized text to the printing thread.
    transcription_queue = Queue()
    
    # An Event that signals the transcription thread to stop
    stop_event = Event()

    # Set up the transcription thread
    transcription_thread = Thread(
        target=getAudioTranscription,
        args=(
            transcription_queue,  # The queue to write to
            stop_event,           # Stop event
            9,                    # input_device_index
            1,                    # channels
            1024,                 # chunk_size
            2.0,                  # segment_duration
            0.5,                  # overlap_duration
            48000                 # rate
        ),
        daemon=False
    )
    
    # Set up the printing thread
    printing_thread = Thread(
        target=printTranscription,
        args=(transcription_queue,),
        daemon=False
    )

    # Start threads
    transcription_thread.start()
    printing_thread.start()

    try:
        # Keep the main thread alive until Ctrl+C
        while True:
            pass
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt in main, shutting down...")
        # Signal the transcription thread to stop
        stop_event.set()
        
        # Tell printing thread to exit by sending `None`
        transcription_queue.put(None)
        
        # Wait for both threads to finish
        transcription_thread.join()
        printing_thread.join()
        
        print("All threads stopped. Exiting.")

if __name__ == "__main__":
    main()
