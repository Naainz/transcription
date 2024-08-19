import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    """Extract audio from a video file and save it as a WAV file."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def transcribe_audio(file_path, model_path):
    
    model = Model(model_path)
    
    
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")
    
    
    wf = wave.open(wav_path, "rb")
    
    
    rec = KaldiRecognizer(model, wf.getframerate())
    
    
    transcription = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            transcription.append(json.loads(result))
    
    transcription.append(json.loads(rec.FinalResult()))
    
    
    os.remove(wav_path)
    
    return transcription


model_path = "vosk-en"
file_path = "transcription.mp3"  


if file_path.endswith((".mp4", ".mkv", ".avi")):
    audio_file = "extracted_audio.wav"
    extract_audio(file_path, audio_file)
    file_path = audio_file

transcription_result = transcribe_audio(file_path, model_path)


for segment in transcription_result:
    print(segment["text"])
