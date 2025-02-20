from faster_whisper import WhisperModel
import pyaudio
import os
import tempfile
import wave
import pyaudio
from src.utils import recordChunk, transcribeChunk

def transcribe_audio_continuous(
    input_device_index,
    channels=1,
    chunk_size=1024,
    segment_duration=2,   # seconds per segment for transcription
    overlap_duration=0.5,   # seconds to overlap between segments
    rate=None
):
    
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

    print("Recording continuously (press Ctrl+C to stop)...")
    try:
        while True:
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
                print(transcription)

                # Remove the temporary file.
                os.remove(tmp_filename)

                # Keep only the last N frames (the overlap) for the next segment.
                audio_buffer = audio_buffer[-frames_per_overlap:]
    except KeyboardInterrupt:
        print("Stopping continuous transcription...")
    finally:
        os.remove(tmp_filename)
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # Use the appropriate input_device_index for your microphone.
    # For example, if device 9 is your Blue Snowball (or default via PulseAudio), use that.
    transcribe_audio_continuous(
        input_device_index=9,
        channels=1,
        chunk_size=1024,
        segment_duration=2,    # record 2-second segments
        overlap_duration=0.1,    # with 0.5-second overlap
        rate=48000             # set the sample rate your mic supports
    )