from threading import Thread, Event
from queue import Queue

from faster_whisper import WhisperModel

from src.utils.audio_utils import get_live_audio_transcription
from src.utils.turngpt_utils import initialize_turngpt_model, calculate_turn_shift_from_live_transcription

def main():
    whisper_model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")
    turn_gpt_model = initialize_turngpt_model()

    transcription_queue = Queue()
    turn_gpt_probs_queue = Queue()
    stop_event = Event()

    live_transcription_thread = Thread(
        target=get_live_audio_transcription,
        args=(
            whisper_model,
            transcription_queue,
            stop_event, 
            9,
        ),
        daemon=True
    )

    turn_shift_thread = Thread(
        target=calculate_turn_shift_from_live_transcription,
        args=(
            turn_gpt_model,
            transcription_queue,
            turn_gpt_probs_queue,
            stop_event,  # pass the event here as well
        ),
        daemon=True
    )

    live_transcription_thread.start()
    turn_shift_thread.start()

    try:
        while True:
            print(f'Current Transcription: {transcription_queue.get()}')
            # print(f'Turn Shift Probabilities: {turn_gpt_probs_queue.get()}')
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping threads...")
        stop_event.set()  # signal threads to exit
        live_transcription_thread.join()
        turn_shift_thread.join()
        print("All threads stopped.")

if __name__ == "__main__":
    main()
