import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import language_tool_python

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
    transcription = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            if 'text' in result_dict:
                transcription += " " + result_dict['text']

    final_result = json.loads(rec.FinalResult())
    if 'text' in final_result:
        transcription += " " + final_result['text']

    return transcription.strip()

def correct_grammar(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

model_path = "vosk-en-big"  
file_path = "transcription.mp3"

transcription_result = transcribe_audio(file_path, model_path)

if transcription_result:
    corrected_text = correct_grammar(transcription_result)
    print(corrected_text)
else:
    print("No transcription was captured.")
