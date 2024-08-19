import pyaudio
import numpy as np
import whisper
import curses

def load_whisper_model():
    
    model = whisper.load_model("base")
    return model

def start_audio_stream(rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=1024)
    return stream

def capture_audio_segment(stream, duration=3):
    frames = []
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    return np.concatenate(frames)

def identify_and_transcribe(model, audio_segment):
    
    audio_segment = audio_segment.astype(np.float32) / 32768.0

    
    result = model.transcribe(audio_segment, fp16=False)

    language = result['language']
    transcription = result['text']
    
    return language, transcription

def segment_and_transcribe(model, stream, screen):
    live_text = ""
    screen.clear()
    screen.refresh()
    
    while True:
        audio_segment = capture_audio_segment(stream)
        language, transcription = identify_and_transcribe(model, audio_segment)
        
        live_text += f"[{language}] {transcription.strip()} "
        
        screen.clear()
        screen.addstr(0, 0, live_text)
        screen.refresh()

def main(stdscr):
    model = load_whisper_model()
    stream = start_audio_stream()
    segment_and_transcribe(model, stream, stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
