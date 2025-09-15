"""Audio transcription module using OpenAI Whisper."""

import os
import tempfile
import logging
import gc
from typing import Dict, Optional, Any
import threading
import time

import whisper
import numpy as np
import soundfile as sf
import librosa

from config import get_config  # pylint: disable=import-error


class ModelCache:
    """Thread-safe cache for Whisper models."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_access: Dict[str, float] = {}
        self.logger = logging.getLogger("whisperlab.transcription.cache")

    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        with self._lock:
            if model_name in self._cache:
                self._last_access[model_name] = time.time()
                self.logger.debug("Model '%s' retrieved from cache", model_name)
                return self._cache[model_name]
            return None

    def put(self, model_name: str, model: Any) -> None:
        """Put model in cache."""
        with self._lock:
            self._cache[model_name] = model
            self._last_access[model_name] = time.time()
            self.logger.debug("Model '%s' stored in cache", model_name)

    def clear_old_models(self, max_age_seconds: float = 300) -> None:
        """Clear models older than max_age_seconds."""
        current_time = time.time()
        with self._lock:
            to_remove = []
            for model_name, last_access in self._last_access.items():
                if current_time - last_access > max_age_seconds:
                    to_remove.append(model_name)

            for model_name in to_remove:
                del self._cache[model_name]
                del self._last_access[model_name]
                self.logger.info("Removed old model '%s' from cache", model_name)

            if to_remove:
                gc.collect()  # Force garbage collection after removing models

    def clear_all(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._last_access.clear()
            gc.collect()
            self.logger.info("Cleared all models from cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_models": list(self._cache.keys()),
                "cache_size": len(self._cache),
                "last_access_times": dict(self._last_access),
            }


# Global model cache instance
_model_cache = ModelCache()


class Transcription:
    """Handles audio transcription using OpenAI Whisper models."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the transcription with a Whisper model.

        Args:
            model_name (str, optional): Whisper model size. If None, uses config default.
        """
        self.config = get_config()
        self.logger = logging.getLogger("whisperlab.transcription")

        # Use provided model name or get from config
        self.model_name = model_name or self.config.whisper.model_name
        self.sample_rate = self.config.audio.sample_rate

        # Load model (with caching)
        self.model = self._load_model_cached(self.model_name)

        # Performance tracking
        self._transcription_count = 0
        self._total_transcription_time = 0.0

        self.logger.info("Transcription initialized with model '%s'", self.model_name)

    def _load_model_cached(self, model_name: str) -> Any:
        """Load Whisper model with caching support."""
        self.logger.debug("Loading Whisper model: %s", model_name)

        # Check cache first if caching is enabled
        if self.config.performance.model_cache_enabled:
            cached_model = _model_cache.get(model_name)
            if cached_model is not None:
                self.logger.info("Using cached Whisper model '%s'", model_name)
                return cached_model

        # Load model from disk
        self.logger.info("Loading Whisper model '%s' from disk", model_name)
        try:
            model = whisper.load_model(
                model_name,
                device=(
                    self.config.whisper.device
                    if self.config.whisper.device != "auto"
                    else None
                ),
            )
            self.logger.info("Whisper model '%s' loaded successfully", model_name)

            # Cache the model if caching is enabled
            if self.config.performance.model_cache_enabled:
                _model_cache.put(model_name, model)

            return model

        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error("Error loading Whisper model '%s': %s", model_name, e)
            # Fallback to tiny model
            fallback_model = "tiny"
            if model_name != fallback_model:
                self.logger.warning("Falling back to '%s' model...", fallback_model)
                return self._load_model_cached(fallback_model)
            else:
                self.logger.critical("Failed to load fallback model")
                raise

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data using OpenAI Whisper.

        Args:
            audio_data (numpy.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        start_time = time.time()

        try:
            # Ensure audio data is in the right format
            audio_data = self._prepare_audio_data(audio_data)

            # Create a temporary file for Whisper (it expects file input)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                temp_file_path = temp_file.name

            # Transcribe the audio with config parameters
            transcribe_options = self._get_transcribe_options()
            result = self.model.transcribe(temp_file_path, **transcribe_options)

            # Clean up temporary file
            os.unlink(temp_file_path)

            # Extract text from result
            text = result.get("text", "") if isinstance(result, dict) else ""
            text = text.strip() if isinstance(text, str) else ""

            # Update performance metrics
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(elapsed_time)

            self.logger.debug(
                "Transcription completed in %.2fs: '%s'", elapsed_time, text[:50]
            )

            return text

        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error("Error during transcription: %s", e)
            return ""

    def transcribe_audio_direct(self, audio_data: np.ndarray) -> str:
        """Alternative method that passes audio directly to Whisper without temp file.

        Args:
            audio_data (numpy.ndarray): Audio data as numpy array

        Returns:
            str: Transcribed text
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        start_time = time.time()

        try:
            # Prepare audio data
            audio_data = self._prepare_audio_data(audio_data)

            # Whisper expects audio at 16kHz, resample if necessary
            if self.sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, orig_sr=self.sample_rate, target_sr=16000
                )

            # Transcribe directly with config parameters
            transcribe_options = self._get_transcribe_options()
            result = self.model.transcribe(audio_data, **transcribe_options)

            text = result.get("text", "") if isinstance(result, dict) else ""
            text = text.strip() if isinstance(text, str) else ""

            # Update performance metrics
            elapsed_time = time.time() - start_time
            self._update_performance_metrics(elapsed_time)

            self.logger.debug(
                "Direct transcription completed in %.2fs: '%s'", elapsed_time, text[:50]
            )

            return text

        except (OSError, RuntimeError, ValueError) as e:
            self.logger.error("Error during direct transcription: %s", e)
            return ""

    def _prepare_audio_data(self, audio_data: np.ndarray) -> np.ndarray:
        """Prepare audio data for transcription."""
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

        return audio_data

    def _get_transcribe_options(self) -> Dict[str, Any]:
        """Get transcription options from config."""
        options = {}

        if self.config.whisper.language:
            options["language"] = self.config.whisper.language
        if self.config.whisper.task != "transcribe":
            options["task"] = self.config.whisper.task
        if self.config.whisper.temperature != 0.0:
            options["temperature"] = self.config.whisper.temperature
        if self.config.whisper.initial_prompt:
            options["initial_prompt"] = self.config.whisper.initial_prompt
        if not self.config.whisper.condition_on_previous_text:
            options["condition_on_previous_text"] = False
        if not self.config.whisper.fp16:
            options["fp16"] = False

        return options

    def _update_performance_metrics(self, elapsed_time: float) -> None:
        """Update performance tracking metrics."""
        self._transcription_count += 1
        self._total_transcription_time += elapsed_time

        if self._transcription_count % 10 == 0:  # Log every 10 transcriptions
            avg_time = self._total_transcription_time / self._transcription_count
            self.logger.info(
                "Performance: %d transcriptions, avg %.2fs each",
                self._transcription_count,
                avg_time,
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = (
            self._total_transcription_time / self._transcription_count
            if self._transcription_count > 0
            else 0.0
        )

        return {
            "transcription_count": self._transcription_count,
            "total_time": self._total_transcription_time,
            "average_time": avg_time,
            "model_name": self.model_name,
            "cache_info": _model_cache.get_cache_info(),
        }

    def cleanup_cache(self) -> None:
        """Clean up old cached models."""
        if self.config.performance.model_cache_enabled:
            _model_cache.clear_old_models()

    @staticmethod
    def clear_all_cache() -> None:
        """Clear all cached models (static method)."""
        _model_cache.clear_all()
