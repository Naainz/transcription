import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from pydub.effects import normalize
from moviepy.editor import VideoFileClip
from termcolor import colored
from textblob import TextBlob

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

def color_word(word, confidence):
    if confidence < 0.2:
        return colored(word, 'red')
    elif confidence < 0.6:
        return colored(word, 'yellow')
    else:
        return colored(word, 'green')

def transcribe_audio(file_path, model_path):
    model = Model(model_path)
    wav_path = preprocess_audio(file_path)
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    transcription = []
    full_text = ""  # Variable to store the complete transcription text

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            print("Intermediate Result:", result)  # Debug output
            result_dict = json.loads(result)
            if 'text' in result_dict:
                full_text += " " + result_dict['text']  # Add the text to full transcription
                for item in result_dict.get('result', []):
                    word = item['word']
                    confidence = item['conf']
                    colored_word = color_word(word, confidence)
                    transcription.append(colored_word)
        else:
            partial_result = rec.PartialResult()
            print("Partial Result:", partial_result)  # Debug output

    final_result = json.loads(rec.FinalResult())
    print("Final Result:", final_result)  # Debug output
    if 'text' in final_result:
        full_text += " " + final_result['text']
        for item in final_result.get('result', []):
            word = item['word']
            confidence = item['conf']
            colored_word = color_word(word, confidence)
            transcription.append(colored_word)

    return transcription, full_text.strip(), wav_path

def correct_grammar(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

model_path = "vosk-en-big"  # Ensure this path points to your model directory
file_path = "transcription.mp3"

if file_path.endswith((".mp4", ".mkv", ".avi")):
    audio_file = "extracted_audio.wav"
    extract_audio(file_path, audio_file)
    file_path = audio_file

transcription_result, transcribed_text, processed_audio_path = transcribe_audio(file_path, model_path)

if transcribed_text.strip():  # Check if any transcription was actually captured
    corrected_text = correct_grammar(transcribed_text)

    print("Transcription with Color Coding:")
    print(" ".join(transcription_result))

    print("\nCorrected Grammar Text:")
    print(corrected_text)
else:
    print("No transcription was captured. Please check the audio input or model output.")

print(f"Processed audio saved at: {processed_audio_path}")
