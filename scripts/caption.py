import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# HEY YOU! CHANGE THIS TO RESEMBLE YOUR OWN VOSK PATH
# FROM THE LOCATION OF YOUR CURRENT PATH
model_path = "vosk-en-big"  

# CHANGE THIS TO THE PATH OF THE AUDIO FILE
file_path = "transcription.mp3"


def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    processed_audio_path = "processed_audio.wav"
    audio.export(processed_audio_path, format="wav")
    return processed_audio_path

def transcribe_audio_with_timestamps(model_path, audio_path, interval=5):
    if not os.path.exists(model_path):
        print(f"Model path '{model_path}' does not exist. Please check the path and try again.")
        return []
    
    model = Model(model_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    current_time = 0
    interval_results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if 'result' in result:
                for word in result['result']:
                    start_time = word['start']
                    text = word['word']

                    
                    if start_time > current_time + interval:
                        results.append((current_time, current_time + interval, " ".join(interval_results)))
                        interval_results = []
                        current_time += interval

                    interval_results.append(text)

    
    if interval_results:
        results.append((current_time, current_time + interval, " ".join(interval_results)))

    return results

def save_captions_to_file(transcription_intervals, output_file):
    with open(output_file, "w") as f:
        for start, end, text in transcription_intervals:
            f.write(f"{start}-{end}s\n{text}\n\n")

def main():
    processed_audio_path = preprocess_audio(file_path)
    transcription_intervals = transcribe_audio_with_timestamps(model_path, processed_audio_path, interval=5)
    
    
    output_file = os.path.splitext(file_path)[0] + ".txt"
    save_captions_to_file(transcription_intervals, output_file)
    
    print(f"Captions saved to {output_file}")

if __name__ == "__main__":
    main()
