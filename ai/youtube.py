# THIS SCRIPT ALLOWS DOWNLOADING OF THE YOUTUBE AUDIO
# TRANSCRIPTION, USING WHISPER [X]
# SUMMARY USING OPENAI GPT-4o-MINI
# READING ALOUD USING ELEVENLABS API

import os
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import whisper
import openai
import numpy as np
import librosa
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# CHANGE THIS TO THE URL OF THE YOUTUBE VIDEO
# OR LEAVE IT AS IT IS TO ALLOW USER INPUT
video_url = input("Enter the YouTube video URL: ")

# CHANGE THIS TO THE PATH OF THE AUDIO / VIDEO
file_path = 'downloads/audio.mp3'

def download_youtube_audio(url, output_path='downloads'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_path, 'audio.%(ext)s'),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_whisper_model():
    model = whisper.load_model("base")
    return model

def load_audio_file(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio

def transcribe_audio(model, audio_segment):
    audio_segment = audio_segment.astype(np.float32) / np.max(np.abs(audio_segment))
    result = model.transcribe(audio_segment, fp16=False, language=None)
    return result['text']

def summarize_transcription(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following transcription:\n\n{transcription}"}
        ]
    )
    summary = response['choices'][0]['message']['content']
    return summary

def summarize_audio_video():
    model = load_whisper_model()
    audio_segment = load_audio_file(file_path)
    transcription = transcribe_audio(model, audio_segment)
    summary = summarize_transcription(transcription)
    return summary

if __name__ == "__main__":
    download_youtube_audio(video_url)
    summary = summarize_audio_video()
    print(summary)
