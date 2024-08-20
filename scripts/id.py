# DIFFERS FROM RTI VERSION 
# BY REPLACING REALTIME WITH AN AUDIO FILE

import whisper
import curses
import numpy as np
import librosa
import soundfile as sf

# CHANGE THIS TO THE PATH OF THE AUDIO FILE
file_path = 'multilingual.mp3'

def load_whisper_model():
    model = whisper.load_model("base")
    return model

def load_audio_file(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def identify_and_transcribe(model, audio_segment):
    audio_segment = audio_segment.astype(np.float32) / np.max(np.abs(audio_segment))
    result = model.transcribe(audio_segment, fp16=False, language=None)
    language = result['language']
    transcription = result['text']
    return language, transcription

def transcribe_audio_file(model, file_path, screen):
    audio_segment = load_audio_file(file_path)
    language, transcription = identify_and_transcribe(model, audio_segment)
    
    screen.clear()
    live_text = f"[{language}] {transcription.strip()}"
    screen.addstr(0, 0, live_text)
    screen.refresh()
    
    while True:
        key = screen.getch()
        if key == ord('q'):
            break

def main(stdscr):
    model = load_whisper_model()
    transcribe_audio_file(model, file_path, stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
