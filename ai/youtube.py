# THIS SCRIPT ALLOWS DOWNLOADING OF THE YOUTUBE AUDIO
# TRANSCRIPTION, USING WHISPER [X]
# SUMMARY USING OPENAI GPT-4o-MINI
# READING ALOUD USING ELEVENLABS API

import os
from yt_dlp import YoutubeDL
from pydub import AudioSegment

# CHANGE THIS TO THE URL OF THE YOUTUBE VIDEO
# OR LEAVE IT AS IT IS TO ALLOW USER INPUT
video_url = input("Enter the YouTube video URL: ")

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
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_youtube_audio(video_url)
