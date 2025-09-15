"""
Configuration settings for WhisperLab application.

This module contains all configuration parameters for audio processing,
Whisper model settings, VAD configuration, and transcription options.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512  # For Silero VAD compatibility
    buffer_duration: float = 3.0  # seconds
    device: Optional[int] = None  # None for default device
    dtype: str = "float32"


@dataclass
class WhisperConfig:
    """Whisper model configuration."""

    model_name: str = "base"
    device: str = "auto"  # "auto", "cpu", or "cuda"
    language: Optional[str] = None  # None for auto-detection
    task: str = "transcribe"  # "transcribe" or "translate"
    temperature: float = 0.0
    best_of: int = 5
    beam_size: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    suppress_tokens: str = "-1"
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True
    fp16: bool = True
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    enabled: bool = True
    vad_type: str = "silero"  # "silero" or "threshold"
    confidence_threshold: float = 0.5
    min_speech_duration: float = 0.25  # seconds
    min_silence_duration: float = 0.5  # seconds
    speech_pad_before: float = 0.1  # seconds
    speech_pad_after: float = 0.1  # seconds
    fallback_threshold: float = 0.01  # for threshold-based fallback


@dataclass
class TranscriptionConfig:
    """Transcription behavior configuration."""

    max_transcription_length: int = 1000
    min_transcription_length: int = 10
    enable_timestamps: bool = False
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'¿([{-"
    append_punctuations: str = '"\'.。,，!！?？:：")]}、'


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "whisperlab.log"
    console_enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AnimationConfig:
    """Console animation configuration."""

    enabled: bool = True
    update_interval: float = 0.1  # seconds
    meter_width: int = 50
    show_waveform: bool = True
    show_confidence: bool = True
    show_transcription_status: bool = True
    colors_enabled: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    model_cache_enabled: bool = True
    audio_buffer_size: int = 1000  # max items in deque
    thread_pool_size: int = 2
    memory_cleanup_interval: float = 60.0  # seconds
    gc_threshold: int = 700  # objects before garbage collection


class Config:
    """Main configuration class for WhisperLab application."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file override."""
        self.audio = AudioConfig()
        self.whisper = WhisperConfig()
        self.vad = VADConfig()
        self.transcription = TranscriptionConfig()
        self.logging = LoggingConfig()
        self.animation = AnimationConfig()
        self.performance = PerformanceConfig()

        # Load from environment variables if available
        self._load_from_env()

        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Audio configuration
        sample_rate_env = os.getenv("WHISPERLAB_SAMPLE_RATE")
        if sample_rate_env:
            self.audio.sample_rate = int(sample_rate_env)

        audio_device_env = os.getenv("WHISPERLAB_AUDIO_DEVICE")
        if audio_device_env:
            self.audio.device = int(audio_device_env)

        # Whisper configuration
        model_env = os.getenv("WHISPERLAB_MODEL")
        if model_env:
            self.whisper.model_name = model_env

        device_env = os.getenv("WHISPERLAB_DEVICE")
        if device_env:
            self.whisper.device = device_env

        language_env = os.getenv("WHISPERLAB_LANGUAGE")
        if language_env:
            self.whisper.language = language_env

        # VAD configuration
        vad_enabled_env = os.getenv("WHISPERLAB_VAD_ENABLED")
        if vad_enabled_env:
            self.vad.enabled = vad_enabled_env.lower() == "true"

        vad_threshold_env = os.getenv("WHISPERLAB_VAD_THRESHOLD")
        if vad_threshold_env:
            self.vad.confidence_threshold = float(vad_threshold_env)

        # Logging configuration
        log_level_env = os.getenv("WHISPERLAB_LOG_LEVEL")
        if log_level_env:
            self.logging.level = log_level_env

        log_file_env = os.getenv("WHISPERLAB_LOG_FILE")
        if log_file_env:
            self.logging.file_path = log_file_env

        # Animation configuration
        animation_enabled_env = os.getenv("WHISPERLAB_ANIMATION_ENABLED")
        if animation_enabled_env:
            self.animation.enabled = animation_enabled_env.lower() == "true"

    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON or YAML file."""
        try:
            import json

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            self._update_from_dict(config_data)
        except Exception:  # pylint: disable=broad-except
            # Could add YAML support here if needed
            pass

    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "audio": self.audio.__dict__,
            "whisper": self.whisper.__dict__,
            "vad": self.vad.__dict__,
            "transcription": self.transcription.__dict__,
            "logging": self.logging.__dict__,
            "animation": self.animation.__dict__,
            "performance": self.performance.__dict__,
        }

    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        import json

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> bool:
        """Validate configuration settings."""
        # Audio validation
        if self.audio.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.audio.channels not in [1, 2]:
            raise ValueError("Channels must be 1 or 2")
        if self.audio.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")

        # Whisper validation
        valid_models = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]
        if self.whisper.model_name not in valid_models:
            raise ValueError(f"Invalid Whisper model. Must be one of: {valid_models}")

        # VAD validation
        if not 0.0 <= self.vad.confidence_threshold <= 1.0:
            raise ValueError("VAD confidence threshold must be between 0.0 and 1.0")

        return True


# Module-level configuration instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    if _config_instance is None:
        return Config()
    return _config_instance


def load_config(config_file: Optional[str] = None) -> Config:
    """Load and return configuration instance."""
    global _config_instance  # pylint: disable=global-statement
    _config_instance = Config(config_file)
    return _config_instance


def save_config(config_file: str) -> None:
    """Save current configuration to file."""
    current_config = get_config()
    current_config.save_to_file(config_file)


# Legacy constants for backward compatibility
AUDIO_FORMAT = "wav"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 512
