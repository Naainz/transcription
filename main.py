import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import language_tool_python
from colorama import Fore, Style, init
from difflib import SequenceMatcher

# HEY YOU! CHANGE THESE PATHS TO RESEBMLE YOUR OWN VOSK PATHS
# IF YOU CHOOSE TO LEAVE ONE UNDEFINED, ENSURE YOU
# UPDATE THE CODE TO HANDLE THE CASE WHERE THE MODEL
# PATH IS NOT DEFINED
small_model_path = "vosk-en"
big_model_path = "vosk-en-big"

# CHANGE THIS TO THE PATH OF THE AUDIO FILE
file_path = "transcription.mp3"

init(autoreset=True)

def gentle_normalize(audio_segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - audio_segment.dBFS
    return audio_segment.apply_gain(change_in_dBFS)

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = gentle_normalize(audio)
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav")
    print(f"Processed audio saved at: {processed_audio_path}")
    return processed_audio_path

def transcribe_audio(model_path, audio_path):
    if not os.path.exists(model_path):
        print(Fore.RED + f"Model path '{model_path}' does not exist. Please check the path and try again.")
        return []
    
    model = Model(model_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    transcription = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text_segment = result.get('text', '').strip()
            if text_segment:
                transcription.extend(text_segment.split())
        else:
            partial_result = json.loads(rec.PartialResult())
            if 'partial' in partial_result and partial_result['partial']:
                print(Fore.YELLOW + f"Partial result: {partial_result['partial']}")

    final_result = json.loads(rec.FinalResult())
    text_segment = final_result.get('text', '').strip()
    if text_segment:
        transcription.extend(text_segment.split())

    return transcription

def calculate_word_confidence(word1, word2):
    matcher = SequenceMatcher(None, word1, word2)
    return matcher.ratio()

def colorize_word(word, confidence):
    color = Fore.GREEN if confidence > 0.6 else Fore.YELLOW if confidence > 0.3 else Fore.RED
    return color + word + Style.RESET_ALL

def main():
    processed_audio_path = preprocess_audio(file_path)

    print(Fore.BLUE + f"\nTranscribing with {big_model_path} model...")
    transcription_big = transcribe_audio(big_model_path, processed_audio_path)

    print(Fore.BLUE + f"\nTranscribing with {small_model_path} model...")
    transcription_en = transcribe_audio(small_model_path, processed_audio_path)

    max_len = max(len(transcription_big), len(transcription_en))
    transcription_big += [''] * (max_len - len(transcription_big))
    transcription_en += [''] * (max_len - len(transcription_en))

    print(Fore.CYAN + "\nFinal transcription with word-level confidence-based coloring:\n")
    final_colored_text = []

    for word_big, word_en in zip(transcription_big, transcription_en):
        confidence = calculate_word_confidence(word_big, word_en)
        colored_word = colorize_word(word_big, confidence)
        final_colored_text.append(colored_word)
    
    print(" ".join(final_colored_text))

    if transcription_big:
        corrected_text = " ".join(transcription_big)
        corrected_text = language_tool_python.utils.correct(corrected_text, language_tool_python.LanguageTool('en-US').check(corrected_text))
        print(Fore.CYAN + f"\nCorrected transcription {big_model_path}:\n")
        print(corrected_text)
    else:
        print(Fore.RED + f"No transcription was captured from {big_model_path}.")

if __name__ == "__main__":
    main()