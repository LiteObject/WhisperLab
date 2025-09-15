# WhisperLab

A Python application that captures real-time audio and transcribes it using OpenAI Whisper. The transcribed text is displayed on the console with intelligent audio buffering for optimal speech recognition.

## Project Structure

```
WhisperLab/
├── .git/
├── .venv/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── audio_capture.py
│   ├── transcription.py
│   ├── config.py
│   └── test_microphone.py
├── requirements.txt
├── run.py
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LiteObject/WhisperLab.git
   cd WhisperLab
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

**Option 1: Using the runner script (Recommended)**
```bash
python run.py
```

**Option 2: Running from src directory**
```bash
cd src
python main.py
```

### How It Works

The application uses intelligent audio buffering for better transcription:

1. **Loads the Whisper model** (this may take a moment on first run)
2. **Starts capturing audio** from your default microphone
3. **Buffers audio for 3 seconds** to ensure quality transcription
4. **Transcribes every 2 seconds** when sufficient audio is detected
5. **Displays transcribed text** with clear indicators
6. **Press Ctrl+C** to stop the application

### Expected Output

When working correctly, you'll see output like:
```bash
Loading Whisper model: base
Whisper model 'base' loaded successfully
Starting WhisperLab...
Press Ctrl+C to stop
Starting audio recording...
Recording started at 16000 Hz, 1 channel(s)
Transcription thread started...
🎤 Transcribing 3.0s of audio (level: 0.0456)...
✅ Transcribed #1: Hello, this is a test of the whisper transcription system.
🎤 Transcribing 3.0s of audio (level: 0.0523)...
✅ Transcribed #2: How are you doing today?
```

### Tips for Best Results

- **Speak clearly** for 3-5 seconds at a time
- **Wait 2-3 seconds** between phrases for processing
- **Ensure good microphone placement** and volume
- **Minimize background noise** for better accuracy

## Dependencies

This project requires the following Python packages:
- `openai-whisper` - OpenAI's Whisper model for speech recognition
- `sounddevice` - Cross-platform audio I/O library
- `soundfile` - Audio file I/O
- `numpy` - Numerical computing library
- `torch` - PyTorch deep learning framework
- `librosa` - Audio analysis library

See `requirements.txt` for the complete list of dependencies.

## Features

- **Real-time audio capture** with intelligent buffering
- **Speech-to-text transcription** using OpenAI Whisper
- **Smart audio processing** with 3-second buffers for better accuracy
- **Console output** with clear status indicators
- **Multithreaded architecture** for smooth audio processing
- **Built-in microphone testing** tool for troubleshooting

## Troubleshooting

### Test Your Microphone
If you're not seeing transcriptions, first test your microphone:
```bash
cd src
python test_microphone.py
```

This will:
- Show available audio devices
- Record 5 seconds of audio
- Display audio levels and detection status
- Help diagnose microphone issues

## Requirements

- Python 3.7 or higher
- Microphone access
- Compatible audio drivers
