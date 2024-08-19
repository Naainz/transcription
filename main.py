import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def gentle_normalize(audio_segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = gentle_normalize(audio)  # Apply gentle normalization
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav")
    return processed_audio_path

def transcribe_audio(file_path, model_path):
    model = Model(model_path)
    wav_path = preprocess_audio(file_path)
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
    return transcription, wav_path

model_path = "vosk-en"
file_path = "transcription.mp3"

if file_path.endswith((".mp4", ".mkv", ".avi")):
    audio_file = "extracted_audio.wav"
    extract_audio(file_path, audio_file)
    file_path = audio_file

transcription_result, processed_audio_path = transcribe_audio(file_path, model_path)

for segment in transcription_result:
    print(segment["text"])

print(f"Processed audio saved at: {processed_audio_path}")
