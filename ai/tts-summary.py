# THIS FILE SERVES THE SAME PURPOSE AS SUMMARY.PY
# ITS ONLY DIFFERENCE IS THAT IT USES TTS TO READ THE SUMMARY
# INSTEAD OF PRINTING IT TO THE CONSOLE

import whisper
import openai
import numpy as np
import librosa
import os
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# CHANGE THIS TO THE PATH OF THE AUDIO / VIDEO
file_path = 'audio/el.mp3' 

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
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following transcription:\n\n{transcription}"}
        ]
    )
    summary = response['choices'][0]['message']['content']
    return summary

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("summary.mp3")
    playsound("summary.mp3")
    os.remove("summary.mp3")

def summarize_and_speak():
    model = load_whisper_model()
    audio_segment = load_audio_file(file_path)
    transcription = transcribe_audio(model, audio_segment)
    summary = summarize_transcription(transcription)
    
    print("Summary:", summary)
    
    text_to_speech(summary)

if __name__ == "__main__":
    summarize_and_speak()
