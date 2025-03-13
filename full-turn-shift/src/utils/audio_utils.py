import pyaudio
import tempfile
import wave
import os
import webrtcvad
import collections
import librosa
import numpy as np

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

def get_live_audio_transcription_vad(
    faster_whisper_model : WhisperModel,
    transcription_queue : Queue,
    stop_event : Event,
    input_device_index: int,
    stream_num_channels: int = 1,
    stream_format: int = pyaudio.paInt16,
    frame_duration_ms: int = 30,       # Allowed values: 10, 20, or 30
    silence_duration_ms: int = 150,    # Duration to consider as silence
    audio_device_rate: int = None,
) -> None:
    """
    Get the transcription of the audio stream from the input device
    using faster_whisper model

    Args:
        faster_whisper_model (WhisperModel): The faster_whisper model
        transcription_queue (Queue): A queue to store the transcribed text
        stop_event (Event): An event to signal the thread to stop
        input_device_index (int): The index of the input device
        stream_format (type): The format of
        stream_num_channels (int): The number of channels in the audio stream
        chunk_size (int): The size of the audio chunk
        segment_duration (float): The duration of each segment for transcription
        overlap_duration (float): The duration to overlap between segments
        rate (Optional[int]): The sample rate of the audio stream
    """    

    p = pyaudio.PyAudio()
    if audio_device_rate is None:
        device_info = p.get_device_info_by_index(input_device_index)
        audio_device_rate = int(device_info["defaultSampleRate"])  # e.g. 44100

    # Define target sample rate for VAD (must be one of 8000, 16000, 32000, 48000)
    target_sample_rate = 16000

    # Open the audio stream using the device's native sample rate
    stream = p.open(
        format=stream_format,
        channels=stream_num_channels,
        rate=audio_device_rate,
        input=True,
        # For reading, we use the number of samples for frame_duration_ms at the device rate:
        frames_per_buffer=int(audio_device_rate * (frame_duration_ms / 1000.0)),
        input_device_index=input_device_index,
    )

    vad = webrtcvad.Vad(2)  # VAD aggressiveness mode (0-3)
    sample_width = p.get_sample_size(stream_format)

    # Calculate how many samples to read from the device for a frame_duration at the native rate
    num_samples_original = int(audio_device_rate * (frame_duration_ms / 1000.0))
    # And how many samples are expected after resampling to the target rate:
    expected_target_samples = int(target_sample_rate * (frame_duration_ms / 1000.0))
    # Expected bytes after resampling:
    expected_bytes = expected_target_samples * sample_width

    # Set up buffers for accumulating audio frames
    ring_buffer = collections.deque(maxlen=int(silence_duration_ms / frame_duration_ms))
    voiced_frames = []
    
    try:
        while not stop_event.is_set():
            # Read a frame from the microphone at the original sample rate
            frame = stream.read(num_samples_original, exception_on_overflow=False)
            if len(frame) != num_samples_original * sample_width:
                print(f"Warning: Received frame of {len(frame)} bytes; expected {num_samples_original * sample_width}. Skipping frame.")
                continue

            # Convert the raw bytes to a numpy array of int16
            audio_np = np.frombuffer(frame, dtype=np.int16)

            # Resample from the original sample rate (e.g., 44100) to the target (e.g., 16000)
            resampled_audio = librosa.resample(audio_np.astype(np.float32), orig_sr=audio_device_rate, target_sr=target_sample_rate)
            
            # Ensure the resampled frame has exactly the expected number of samples
            if len(resampled_audio) < expected_target_samples:
                resampled_audio = np.pad(resampled_audio, (0, expected_target_samples - len(resampled_audio)), mode='constant')
            elif len(resampled_audio) > expected_target_samples:
                resampled_audio = resampled_audio[:expected_target_samples]
            
            resampled_audio = resampled_audio.astype(np.int16)
            resampled_frame = resampled_audio.tobytes()

            if len(resampled_frame) != expected_bytes:
                print(f"Warning: Resampled frame size {len(resampled_frame)} bytes does not match expected {expected_bytes}. Skipping frame.")
                continue

            # Now use the resampled frame with the supported sample rate for VAD
            try:
                is_speech = vad.is_speech(resampled_frame, sample_rate=target_sample_rate)
            except webrtcvad.Error as e:
                print(f"VAD error: {e}; skipping this frame.")
                continue

            # Add the frame and its VAD decision to the ring buffer
            ring_buffer.append((resampled_frame, is_speech))
            
            if is_speech:
                # If speech is detected, add all buffered frames and the current frame to voiced_frames
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
                voiced_frames.append(resampled_frame)
            else:
                # If there's been enough continuous silence, process the accumulated voiced frames
                if len(ring_buffer) == ring_buffer.maxlen and all(not s for _, s in ring_buffer):
                    if voiced_frames:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                            tmpfile_name = tmpfile.name
                            wf = wave.open(tmpfile_name, 'wb')
                            wf.setnchannels(stream_num_channels)
                            wf.setsampwidth(sample_width)
                            wf.setframerate(target_sample_rate)
                            wf.writeframes(b''.join(voiced_frames))
                            wf.close()
                        
                        transcription = get_audio_chunk_transcription(faster_whisper_model, tmpfile_name)
                        transcription_queue.put(transcription)
                        os.remove(tmpfile_name)
                        voiced_frames = []
                    ring_buffer.clear()
    except KeyboardInterrupt:
        print("Stopping live transcription (VAD)...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
