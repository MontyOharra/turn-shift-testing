import pyaudio
import os
import tempfile
import wave

from queue import Queue
from faster_whisper import WhisperModel
from typing import Optional
import threading

def transcribeChunk(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def recordChunk(p,
                stream, 
                format,
                rate,
                channels,
                file_path, 
                chunk_size,
                chunk_length=1):
    """
    (This is not used in the continuous-transcription approach in getAudioTranscription,
     but can be used for single-chunk capture.)
    """
    frames = []
    for _ in range(0, int(rate / chunk_size * chunk_length)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)
    
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def getAudioTranscription(
    liveTranscription : Queue,
    stop_event : threading.Event,
    input_device_index : int,
    channels : int = 1,
    chunk_size : int = 1024,
    segment_duration : float = 2.0,   # seconds per segment for transcription
    overlap_duration : float = 0.5,   # seconds to overlap between segments
    rate : Optional[int] = None
):
    """
    Continuously record audio, transcribe in chunks, and put the transcription
    in liveTranscription queue. Will exit gracefully if stop_event is set.
    """

    model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")
    
    p = pyaudio.PyAudio()
    device_info = p.get_device_info_by_index(input_device_index)
    if rate is None:
        rate = int(device_info["defaultSampleRate"])
    default_format = pyaudio.paInt16

    # Open the audio stream.
    stream = p.open(
        format=default_format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index,
    )

    # Calculate how many frames (chunks) constitute one segment and the overlap.
    frames_per_segment = int(rate / chunk_size * segment_duration)
    frames_per_overlap = int(rate / chunk_size * overlap_duration)

    audio_buffer = []  # will hold the incoming audio chunks
    tmp_filename = None

    print("Recording continuously (press Ctrl+C to stop)...")
    try:
        while not stop_event.is_set():
            # Read one chunk from the stream.
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_buffer.append(data)

            # When the buffer has accumulated enough frames for one segment...
            if len(audio_buffer) >= frames_per_segment:
                # Write the current buffer to a temporary WAV file.
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmp_filename = tmpfile.name
                wf = wave.open(tmp_filename, 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(default_format))
                wf.setframerate(rate)
                wf.writeframes(b''.join(audio_buffer))
                wf.close()

                # Transcribe the temporary file.
                transcription = transcribeChunk(model, tmp_filename)
                liveTranscription.put(transcription)

                # Remove the temporary file.
                os.remove(tmp_filename)

                # Keep only the last N frames (the overlap) for the next segment.
                audio_buffer = audio_buffer[-frames_per_overlap:]

    finally:
        # Cleanup
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stopped continuous transcription thread.")


def printTranscription(liveTranscription : Queue):
    """
    Continuously read from liveTranscription queue and print.
    Exits when it reads a None sentinel.
    """
    while True:
        item = liveTranscription.get()
        if item is None:
            break
        print(item)
    print("Stopped printing thread.")
