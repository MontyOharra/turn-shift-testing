import pyaudio
import tempfile
import wave
import os

from faster_whisper import WhisperModel
from turngpt import TurnGPT
from queue import Queue, Empty
from threading import Event

from typing import Optional

def get_audio_chunk_transcription(
    faster_whisper_model : WhisperModel,
    audio_file_path : str
) -> str:
    """
    Transcribe an audio chunk using the faster_whisper model
    
    Args:
        faster_whisper_model (WhisperModel): The faster_whisper model
        audio_file_path (str): The path to the audio file
    
    Returns:
        str: The transcribed text
    """
    segments, info = faster_whisper_model.transcribe(audio_file_path)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def get_live_audio_transcription(
    faster_whisper_model,
    transcription_queue: Queue,
    stop_event: Event,
    input_device_index: int,
    stream_num_channels: int = 1,
    stream_format: type = pyaudio.paInt16,
    chunk_size: int = 1024,
    segment_duration: float = 2,
    overlap_duration: float = 0.5,
    audio_device_rate: Optional[int] = None,
) -> None:
    """
    Get the transcription of the audio stream from the input device
    using faster_whisper model
    
    Args:
        faster_whisper_model (WhisperModel): The faster_whisper model
        transcription_queue (Queue): A queue to store the transcribed text
        stop_event (Event): An event to signal the thread to stop
        input_device_index (int): The index of the input device
        stream_format (type): The format of the audio stream
        stream_num_channels (int): The number of channels in the audio stream
        chunk_size (int): The size of the audio chunk
        segment_duration (float): The duration of each segment for transcription
        overlap_duration (float): The duration to overlap between segments
        rate (Optional[int]): The sample rate of the audio stream
    """
    p = pyaudio.PyAudio()
    if audio_device_rate is None:
        device_info = p.get_device_info_by_index(input_device_index)
        audio_device_rate = int(device_info["defaultSampleRate"])

    frames_per_segment = int(audio_device_rate / chunk_size * segment_duration)
    frames_per_overlap = int(audio_device_rate / chunk_size * overlap_duration)

    stream = p.open(
        format=stream_format,
        channels=stream_num_channels,
        rate=audio_device_rate,
        input=True,
        frames_per_buffer=chunk_size,
        input_device_index=input_device_index,
    )

    audio_buffer = []
    try:
        while not stop_event.is_set():
            data: bytes = stream.read(chunk_size, exception_on_overflow=False)
            audio_buffer.append(data)

            if len(audio_buffer) >= frames_per_segment:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmpfile_name = tmpfile.name
                    wf = wave.open(tmpfile_name, 'wb')
                    wf.setnchannels(stream_num_channels)
                    wf.setsampwidth(p.get_sample_size(stream_format))
                    wf.setframerate(audio_device_rate)
                    wf.writeframes(b''.join(audio_buffer))
                    wf.close()

                transcription = get_audio_chunk_transcription(faster_whisper_model, tmpfile_name)
                transcription_queue.put(transcription)
                os.remove(tmpfile_name)
                audio_buffer = audio_buffer[-frames_per_overlap:]
    except KeyboardInterrupt:
        print("Stopping continuous transcription...")
    finally:
        # Clean up resources
        if os.path.exists(tmpfile_name):
            os.remove(tmpfile_name)
        stream.stop_stream()
        stream.close()
        p.terminate()