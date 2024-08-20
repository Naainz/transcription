# Audio Transcription & Summary Tool
This project provides tools for offline transcription, speech recognition, and summarization using advanced machine learning models. The core functionalities include downloading YouTube videos, transcribing audio using OpenAI Whisper, summarizing transcriptions using GPT-4o-mini, and converting summaries into speech using the ElevenLabs API.

## Features

- **Offline Transcription**: Utilize OpenAI Whisper / VOSK Models for transcribing audio from video or audio files.
- **YouTube Audio Downloading**: Download audio from YouTube videos directly in MP3 format using `yt-dlp`.
- **Transcription Summarization**: Generate concise summaries of transcriptions using OpenAI's GPT-4o-mini.
- **Text-to-Speech (TTS)**: Convert summaries to speech using ElevenLabs' natural-sounding voice engine.

### Prerequisites

- Python 3.7 or higher
- `ffmpeg` installed
- the following python libraries:
```bash 
  pip install yt-dlp pydub whisper openai librosa requests python-dotenv
  ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Naainz/transcription.git
cd transcription
```

2. Create a .env file in the root directory and add the following:
```bash
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

## Usage

Basic usage of the transcription tool (VOSK)
```bash
python main.py
```

Basic usage of the audio summary tool
```bash
python ai/summary.py
```

Basic usage of the YouTube video summary tool
```bash
python ai/youtube.py
```

> **NOTE**: Please only use the YouTube video summary tool for personal educational uses, and only download the video if you have explicit permission from the video creator.

**I (Naainz) am not responsible for any misuse of any of the tools in this project.**

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).





https://www.youtube.com/watch?v=JhU0yO43b6o
Video used for basic transcription testing