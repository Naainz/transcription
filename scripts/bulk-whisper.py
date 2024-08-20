import whisper
import curses
import numpy as np
import librosa

# CHANGE THIS TO A LIST OF PATHS TO AUDIO FILES
file_paths = [
    'audio/el.mp3',
    'audio/en.mp3',
    'audio/jp.mp3'
]

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

def transcribe_audio_files(model, file_paths, screen):
    transcriptions = []

    for idx, file_path in enumerate(file_paths):
        screen.clear()
        screen.addstr(0, 0, f"Transcribing file {idx+1}/{len(file_paths)}: {file_path}")
        screen.refresh()

        audio_segment = load_audio_file(file_path)
        language, transcription = identify_and_transcribe(model, audio_segment)
        
        live_text = f"[{language}] {transcription.strip()}"
        transcriptions.append(live_text)

        screen.clear()
        for i, trans in enumerate(transcriptions):
            screen.addstr(i, 0, trans) 
        screen.refresh()

    while True:
        key = screen.getch()
        if key == ord('q'):
            break

def main(stdscr):
    model = load_whisper_model()
    transcribe_audio_files(model, file_paths, stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
