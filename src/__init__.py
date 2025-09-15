"""WhisperLab package for real-time audio transcription."""

from .audio_capture import AudioCapture
from .transcription import Transcription

__all__ = ["AudioCapture", "Transcription"]
