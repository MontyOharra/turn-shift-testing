import whisper

model = whisper.load_model("small")
result = model.transcribe("Small Talk Everyday English.mp3")
print(result["text"])