from src.utils import printLiveTranscription

if __name__ == "__main__":
    # Use the appropriate input_device_index for your microphone.
    # For example, if device 9 is your Blue Snowball (or default via PulseAudio), use that.
    printLiveTranscription(
        input_device_index=9,
        channels=1,
        chunk_size=1024,
        segment_duration=2,    # record 2-second segments
        overlap_duration=0.1,    # with 0.5-second overlap
        rate=48000             # set the sample rate your mic supports
    )