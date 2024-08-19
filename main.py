import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import language_tool_python
from termcolor import colored

def gentle_normalize(audio_segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = gentle_normalize(audio)
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav")
    return processed_audio_path

def transcribe_audio(file_path, model_path):
    model = Model(model_path)
    wav_path = preprocess_audio(file_path)
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    transcription = []
    full_text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            if 'result' in result_dict:
                for item in result_dict['result']:
                    word = item['word']
                    confidence = item['conf']
                    transcription.append((word, confidence))
                    full_text += " " + word
        else:
            partial_result = rec.PartialResult()
            result_dict = json.loads(partial_result)
            if 'partial' in result_dict:
                full_text += " " + result_dict['partial']
                print(f"Partial Transcription: {result_dict['partial']}")

    final_result = json.loads(rec.FinalResult())
    if 'result' in final_result:
        for item in final_result['result']:
            word = item['word']
            confidence = item['conf']
            transcription.append((word, confidence))
            full_text += " " + word

    print(f"Final Transcription: {full_text.strip()}")  

    return transcription, full_text.strip()

def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def color_code_transcription(transcription, corrected_text):
    words = corrected_text.split()
    colored_text = ""
    word_index = 0

    for word, confidence in transcription:
        if word_index < len(words):
            if confidence < 0.2:
                colored_word = colored(words[word_index], 'red')
            elif confidence < 0.6:
                colored_word = colored(words[word_index], 'yellow')
            else:
                colored_word = colored(words[word_index], 'green')

            colored_text += colored_word + " "
            word_index += 1

    return colored_text.strip()

model_path = "vosk-en-big"
file_path = "transcription.mp3"

transcription_result, transcribed_text = transcribe_audio(file_path, model_path)

if transcribed_text.strip():  
    corrected_text = correct_grammar(transcribed_text)
    colored_output = color_code_transcription(transcription_result, corrected_text)
    print(colored_output)
else:
    print("No transcription was captured.")
