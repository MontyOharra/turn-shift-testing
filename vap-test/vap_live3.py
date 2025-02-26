import pyaudio
import wave
import json
import VoiceActivityProjection.run2
from faster_whisper import WhisperModel

import sys
import os

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/faster-whisper-with-turn-gpt/applications"))
import record_audio

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/faster-whisper-with-turn-gpt/src"))
import utils

sys.path.append(os.path.abspath("/home/bwilab/turn-shift/turn-gpt-test"))
import test as TurnGPTTest

whisper_model = WhisperModel(model_size_or_path="medium.en", device="cuda", compute_type="float16")

def getPnowPfuture(json_file):
    with open(json_file) as f:
        data = json.load(f)
    pnow = data['p_now']
    pnow = pnow[-1]
    pnow = pnow[0]    
    pfuture = data['p_future']
    pfuture = pfuture[-1]
    pfuture = pfuture[0]
    return pnow, pfuture

if __name__ == "__main__":
    try:
        while True:
            record_audio.record_audio(
                output_filename="output.wav",
                record_seconds=5,         # Adjust as desired
                rate=48000,               # Common sample rate
                chunk=1024,
                channels=1,
                input_device_index=0,  # Set to an integer if you want a specific mic
            )
            transcription = utils.transcribeChunk(
                model=whisper_model,
                file_path="output.wav"
            )
            print("Transcription: ", transcription)
            VoiceActivityProjection.run2.runVAP(output_filename="output.wav", output_json="test.json")    
            pnow, pfuture = getPnowPfuture("test.json")
            print("pnow: ", pnow)
            print("pfuture: ", pfuture)
            # TurnGPTTest.main()
            # TurnGPTTest.onInput(transcription)
            if(pnow[0] < .4 and pfuture[0] < .4):
                print("Turn detected")

    except KeyboardInterrupt:
        os.remove("output.wav")
        os.remove("test.json")
        print("\nLoop interrupted by user.")
    

    # def onInput(turn_list):
    # model = 
    #()

    # model.init_tokenizer()
    # model.initialize_special_embeddings()

    # out = model.string_list_to_trp(turn_list)
    # print("TRP for input example:", out["trp_probs"])
    # fig, ax = plot_trp(out["trp_probs"][0], out["tokens"][0])
    # plt.show()  # Show the plot in interactive mode