"""Audio transcription module using OpenAI Whisper."""

import os
import tempfile

import whisper
import numpy as np
import soundfile as sf
import librosa

from config import SAMPLE_RATE  # pylint: disable=import-error


class Transcription:
    """Handles audio transcription using OpenAI Whisper models."""

    def __init__(self, model_name="base"):
        """Initialize the transcription with a Whisper model.

        Args:
            model_name (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        print(f"Loading Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name)
            print(f"Whisper model '{model_name}' loaded successfully")
        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error loading Whisper model: {e}")
            # Fallback to tiny model
            print("Falling back to 'tiny' model...")
            self.model = whisper.load_model("tiny")

        self.sample_rate = SAMPLE_RATE

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using OpenAI Whisper.

        Args:
            audio_data (numpy.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        try:
            # Ensure audio data is in the right format
            if isinstance(audio_data, np.ndarray):
                # Normalize audio data if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Ensure audio is in the range [-1, 1]
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

            # Create a temporary file for Whisper (it expects file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                temp_file_path = temp_file.name

            # Transcribe the audio
            result = self.model.transcribe(temp_file_path, language="en")

            # Clean up temporary file
            os.unlink(temp_file_path)

            # Extract text from result
            text = result.get("text", "") if isinstance(result, dict) else ""
            return text.strip() if isinstance(text, str) else ""

        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error during transcription: {e}")
            return ""

    def transcribe_audio_direct(self, audio_data):
        """Alternative method that passes audio directly to Whisper without temp file.

        Args:
            audio_data (numpy.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        try:
            # Ensure audio data is properly formatted
            if isinstance(audio_data, np.ndarray):
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Ensure mono audio
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # Normalize to [-1, 1] range
                if np.max(np.abs(audio_data)) > 1.0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

            # Whisper expects audio at 16kHz, resample if necessary
            if self.sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, orig_sr=self.sample_rate, target_sr=16000
                )

            # Transcribe directly
            result = self.model.transcribe(audio_data, language="en")
            text = result.get("text", "") if isinstance(result, dict) else ""
            return text.strip() if isinstance(text, str) else ""

        except (OSError, RuntimeError, ValueError) as e:
            print(f"Error during direct transcription: {e}")
            return ""
