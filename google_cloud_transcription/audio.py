import time
from playsound import playsound
from gtts import gTTS
import os
import threading
import wave
from google.cloud import texttospeech
from google.cloud import speech
from pocketsphinx import *
import audioop
import pyaudio
import io
import numpy as np
from functools import partial
import math
import re
from collections import deque

# generate a dictionary file with http://www.speech.cs.cmu.edu/tools/lextool.html

input_file_count = 0

# Set up audio input stream
class Recorder:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 512
    WAVE_OUTPUT_FILENAME = "google_cloud_transcription/input_audio.wav"

    speech_line_queue = list()
    speech_file_queue = list()
    file_count = 0
    generating_audio = False

    def __init__(self):
        modeldir = get_model_path()
        self.decoder = Decoder(
            keyphrase="dobby",
            hmm=os.path.join(modeldir, "en-us/en-us"),
            dict="google_cloud_transcription/keyword.dict",
            kws_threshold=1e-9,
        )
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.talking_threshold = 40
        self.stop_recording_event = threading.Event()
        self.stop_listening_event = threading.Event()
        self.stop_speaking_event = threading.Event()
        self.speaking = False
        self.recieving_response = False
        self.recording = False

    def calibrate_microphone(self, threshold=2):
        print("Calibrating microphone for 5 chunks... Stay silent")

        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        chunks = 0
        rms_array = []
        while chunks < 10:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            cur_rms = audioop.rms(data, 2)
            print(cur_rms)
            rms_array.append(cur_rms)
            chunks += 1
        a = np.array(rms_array)
        background_rms = np.median(a)
        print("Median background noise: " + str(background_rms))
        self.talking_threshold = background_rms * threshold

        stream.stop_stream()
        stream.close()

    def start_recording(self, finished_callback, auto_stop=False):
        # start recording thread
        self.recording = True
        self.stop_recording_event = threading.Event()
        self.record_thread = threading.Thread(
            target=self.record_audio, args=(finished_callback, auto_stop), daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        # stop recording thread
        self.stop_recording_event.set()
        while self.recording:
            time.sleep(0.1)

    def record_audio(self, finished_callback=None, auto_stop=False):
        global input_file_count
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        max_silent_chunks = 12
        max_silent_chunks_convo = 6
        silent_chunks = 0
        last_rms = 0
        total_rms = 0
        chunks = 0
        speaking = False
        speaking_threshold = 0

        print("Recording...")
        while not self.stop_recording_event.is_set() and (
            not auto_stop or silent_chunks < max_silent_chunks
        ):
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            self.frames.append(data)
            chunks += 1
            rms = audioop.rms(data, 2)
            total_rms += rms

            if chunks > 10:
                rms = total_rms / chunks
                chunks = 0
                total_rms = 0
                print("Audio Level: " + str(rms))
                # spike in volume means we started speaking
                if last_rms != 0 and rms - last_rms > last_rms * 0.7:
                    if not speaking:
                        speaking = True
                        #set threshold halfway between silent and speaking 
                        speaking_threshold = last_rms + (rms - last_rms) * 0.5
                        print ("Started speaking " + str(speaking_threshold))
                        max_silent_chunks = max_silent_chunks_convo
                    silent_chunks = 0
                if rms < speaking_threshold or speaking_threshold == 0:
                    speaking = False
                    silent_chunks += 1
                last_rms = rms


        print("Stopping recording...")

        stream.stop_stream()
        stream.close()

        # Save recorded audio to file
        input_file_count += 1
        file_name = self.WAVE_OUTPUT_FILENAME + ".wav"
        waveFile = wave.open(file_name, "wb")
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(
            b"".join(self.frames[: len(self.frames) - (min(0, silent_chunks - 1))])
        )
        waveFile.close()
        print("Saved recording to", file_name)
        self.frames = []

        #recorder.add_audio_clip(file_name, time.time())

        if not self.stop_recording_event.is_set() and auto_stop:
            finished_callback()

        self.recording = False

    def start_listening(self, callback):
        self.stop_listening_event = threading.Event()
        self.listen_thread = threading.Thread(
            target=self.listen_keyword, args=(callback,), daemon=True
        )
        self.listen_thread.start()

    def stop_listening(self):
        # stop listening thread
        self.stop_listening_event.set()

    def listen_keyword(self, callback):
        stream_single = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        self.decoder.start_utt()
        # Loop until keyword is detected
        while not self.stop_listening_event.is_set():
            data = stream_single.read(self.CHUNK, exception_on_overflow=False)
            self.frames = [data]
            self.decoder.process_raw(data, False, False)
            if self.decoder.hyp() != None:
                print(
                    [
                        (seg.word, seg.prob, seg.start_frame, seg.end_frame)
                        for seg in self.decoder.seg()
                    ]
                )
                if not self.stop_listening_event.is_set():
                    callback("front", 0)
        self.decoder.end_utt()
        stream_single.stop_stream()
        stream_single.close()

    def enqueue_speech_line(self, line):
        self.speech_line_queue.append(line)
    
    def set_recieving_response(self, recieving):
        self.recieving_response = recieving

    def start_speaking(self, finished_callback):
        self.stop_speaking_event = threading.Event()
        if not self.speaking:
            self.recieving_response = True
            self.file_count = 0
            self.speaking = True
            self.speak_thread = threading.Thread(
                target=self.speak_lines, args=(finished_callback,), daemon=True
            )
            self.speak_thread.start()
            self.tts_thread = threading.Thread(target=self.text_to_speech, daemon=True)
            self.tts_thread.start()

    def text_to_speech(self):
        pattern = re.compile(r"\((.*?)\)")
        #voice = Voice(voice_id="e1m53KLIEGzPB8wotn3d")

        while not self.stop_speaking_event.isSet():
            if len(self.speech_line_queue) > 0:
                self.generating_audio = True
                text = self.speech_line_queue.pop(0)
                match = pattern.search(text)
                if match:
                    # an emotion was specified
                    filename = match.group(1)
                else:
                    self.file_count += 1
                    filename = f"Data/audio/output_audio{self.file_count}.mp3"
                    synthesis_input = texttospeech.SynthesisInput(text=text)
                    response = client.synthesize_speech(
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )
                    with open(os.path.join(os.path.dirname(__file__), "..", filename), "wb") as out_file:
                        out_file.write(response.audio_content)

                self.speech_file_queue.append(filename)
                self.generating_audio = False

    def speak_lines(self, finished_callback):
        started = False
        while not self.stop_speaking_event.isSet():
            if len(self.speech_file_queue) > 0:
                filename = self.speech_file_queue.pop(0)
                #recorder.add_audio_clip(filename, time.time())
                playsound(os.path.join(os.path.dirname(__file__), "..", filename), True)
                os.remove(os.path.join(os.path.dirname(__file__), "..", filename))
                started = True
            elif (
                started and
                not self.recieving_response
                and not self.generating_audio
                and len(self.speech_line_queue) == 0
                and finished_callback is not None
            ):  # no more audio in both queues
                finished_callback()
                started = False

    def stop_speaking(self):
        self.file_count = 0
        self.stop_speaking_event.set()
        self.speaking = False

    
# Google API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_cloud_transcription/google_api.json"
client = texttospeech.TextToSpeechClient()
speechClient = speech.SpeechClient()
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.MALE
)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

speech_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    enable_automatic_punctuation=True,
    audio_channel_count=1,
    language_code="en-US",
)

def transcribe_into_text(filename="google_cloud_transcription/input_audio"):
    with io.open(filename + ".wav", "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)

    response = speechClient.recognize(request={"config": speech_config, "audio": audio})
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


# Old one
def gTTS_speak(mytext):
    global count
    count += 1
    filename = f"output_audio{count}.mp3"
    language = "en"
    tts = gTTS(text=mytext, lang=language, slow=False)
    tts.save(filename)
    playsound(os.path.join(os.path.dirname(__file__), "..", filename), True)