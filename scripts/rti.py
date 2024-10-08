import pyaudio
import numpy as np
import whisper
import curses

# THIS MODEL IS VERY UNSTABLE AND MAY NOT WORK PROPERLY
# SOMETIMES GENERATES TRUE TRANSCRIPTIONS, SOMETIMES NOT

def load_whisper_model():
    model = whisper.load_model("base")
    return model

# CHANGE THE INPUT DEVICE INDEX TO MATCH YOUR MICROPHONE'S INDEX
def start_audio_stream(rate=16000, input_device_index=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    input_device_index=input_device_index, 
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
    result = model.transcribe(audio_segment, fp16=False, language=None)
    language = result['language']
    transcription = result['text']
    return language, transcription

def segment_and_transcribe(model, stream, screen):
    live_text = ""
    last_language = ""
    screen.clear()
    screen.refresh()
    
    while True:
        audio_segment = capture_audio_segment(stream)
        language, transcription = identify_and_transcribe(model, audio_segment)
        
        if transcription.strip():
            if language != last_language:
                live_text += f"\n[{language}] "
                last_language = language
            live_text += transcription.strip() + " "
        
        screen.clear()
        screen.addstr(0, 0, live_text)
        screen.refresh()

def main(stdscr):
    model = load_whisper_model()
    stream = start_audio_stream()
    segment_and_transcribe(model, stream, stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
