import pyaudio
import os
import tempfile
import wave
import torch
import matplotlib.pyplot as plt
from turngpt.model import TurnGPT
from queue import Queue, Empty
from faster_whisper import WhisperModel
from typing import Optional

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
    input_device_index : int,
    channels : int = 1,
    chunk_size : int =1024,
    segment_duration : float = 2,   # seconds per segment for transcription
    overlap_duration : float = 0.5,   # seconds to overlap between segments
    rate : Optional[int] = None
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
                liveTranscription.put(transcription)

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


def printTranscription(liveTranscription : Queue):
    while True:
        try:
            message = liveTranscription.get(timeout=1)

        except Empty:
            continue

        if message == None:
            break  
        print(liveTranscription.get())


plt.ion()

# Initialize the TurnGPT model
def initialize_turngpt_model():
    # Argument parser setup, would normally be done outside the loop
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser = TurnGPT.add_model_specific_args(parser)
    args = parser.parse_args()

    # Initialize model
    model = TurnGPT(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        trp_projection_steps=args.trp_projection_steps,
        trp_projection_type=args.trp_projection_type,
        weight_loss=args.weight_loss,
        weight_eos_token=args.weight_eos_token,
        weight_regular_token=args.weight_regular_token,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        pretrained=args.pretrained,
        no_train_first_n=args.no_train_first_n,
        omit_dialog_states=args.omit_dialog_states,
    )

    model.init_tokenizer()
    model.initialize_special_embeddings()
    
    return model

def plot_trp(P, text):
    fig, ax = plt.subplots(1, 1)
    x = torch.arange(len(P))
    ax.bar(x, P)
    ax.set_xticks(x)
    ax.set_xticklabels(text, rotation=60)
    ax.set_ylim([0, 1])  # type: ignore
    fig.savefig("output.png")  # Save the figure instead of displaying it
    return fig, ax

def calculateTurnShiftFromTranscription(liveTranscription : Queue,):
    model = initialize_turngpt_model()
    currTranscription = []
    with open('transcription_and_turngpt_output.txt', 'w') as output_file:
        while True:
            try:
                message = liveTranscription.get(timeout=1)
            except Empty:
                continue  # If queue is empty, continue waiting for new messages

            if message is None:  # If we receive None, stop processing
                break

            currTranscription.append(message)  # Append the message to the current transcription list

            # Now use the transcribed text in currTranscription to generate the turn shift prediction
            full_turn_list = " ".join(currTranscription)  # Combine the list of transcriptions into one string
            
            # Process the transcribed text with TurnGPT
            out = model.string_list_to_trp(full_turn_list)
            print("TRP for transcribed example:", out["trp_probs"])

            for prob in out["trp_probs"][0]:
                prob *= 10  # Adjusting the probabilities for visualization

            # Write transcriptions and TurnGPT outputs to the file
            output_file.write("Transcription: " + full_turn_list + "\n")
            output_file.write("TurnGPT Output (TRP Probs): " + str(out["trp_probs"]) + "\n\n")

            # Optionally, plot the TRP probabilities
            fig, ax = plot_trp(out["trp_probs"][0], out["tokens"][0])

    print("Final transcription: ", currTranscription)
    print("All output has been written to 'transcription_and_turngpt_output.txt'.")