import os
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import curses

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please check the path and try again.")
        return None
    return Model(model_path)

def start_audio_stream(rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=8000)
    stream.start_stream()
    return stream

def transcribe_live_audio(model, stream):
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)
    
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get('text', '').strip()
            if text:
                yield text

def live_transcription(screen, model):
    stream = start_audio_stream()
    
    
    live_text = ""

    for text in transcribe_live_audio(model, stream):
        live_text += " " + text
        live_text = live_text.strip()

        
        screen.clear()
        screen.addstr(0, 0, f"Live Transcription: {live_text}")
        screen.refresh()

def main():
    model_path = "vosk-en-big"  

    model = load_model(model_path)
    if model is None:
        return

    
    curses.wrapper(live_transcription, model)

if __name__ == "__main__":
    main()
