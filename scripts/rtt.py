import os
import json
from vosk import Model, KaldiRecognizer
import pyaudio
import time
from pydub import AudioSegment
from io import BytesIO
import wave
import curses

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please check the path and try again.")
        return None
    return Model(model_path)

def start_audio_stream(rate=16000, device_index=None):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=4000,
                    input_device_index=device_index)
    return stream

def transcribe_audio_segment(model, audio_data):
    recognizer = KaldiRecognizer(model, 16000)
    audio = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
    
    
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    wf = wave.open(wav_io, 'rb')
    recognizer = KaldiRecognizer(model, wf.getframerate())
    transcription = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription.append(result.get('text', '').strip())

    final_result = json.loads(recognizer.FinalResult())
    transcription.append(final_result.get('text', '').strip())
    
    return ' '.join(transcription)

def capture_audio_segment(stream, duration=3):
    frames = []
    for _ in range(0, int(16000 / 4000 * duration)):
        data = stream.read(4000, exception_on_overflow=False)
        frames.append(data)
    return b''.join(frames)

def segment_and_transcribe(model, stream, screen):
    live_text = ""
    screen.clear()
    screen.refresh()
    
    while True:
        audio_data = capture_audio_segment(stream)
        
        audio_segment = AudioSegment(data=audio_data, sample_width=2, frame_rate=16000, channels=1)
        mp3_data = BytesIO()
        audio_segment.export(mp3_data, format="mp3")

        transcription = transcribe_audio_segment(model, mp3_data.getvalue())
        live_text += " " + transcription.strip()
        
        screen.clear()
        screen.addstr(0, 0, live_text)
        screen.refresh()

def main(stdscr):
    model_path = "vosk-model-small-ru-0.22"  
    device_index = 2  

    model = load_model(model_path)
    if model is None:
        return

    stream = start_audio_stream(device_index=device_index)
    segment_and_transcribe(model, stream, stdscr)

if __name__ == "__main__":
    curses.wrapper(main)
