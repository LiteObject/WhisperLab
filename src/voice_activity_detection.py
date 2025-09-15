"""
Voice Activity Detection (VAD) module using sophisticated algorithms.

This module provides both simple threshold-based VAD and advanced Silero VAD
for distinguishing speech from other audio signals like music, noise, etc.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

# Optional imports for advanced VAD
try:
    import torch
    import torch.nn  # pylint: disable=import-error,unused-import

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torchaudio  # pylint: disable=import-error

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    torchaudio = None


class VADBase(ABC):
    """Abstract base class for Voice Activity Detection implementations."""

    @abstractmethod
    def is_speech(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """
        Detect if audio contains speech.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the VAD implementation."""
        pass


class ThresholdVAD(VADBase):
    """Simple amplitude threshold-based VAD implementation."""

    def __init__(self, threshold: float = 0.01):
        """
        Initialize threshold-based VAD.

        Args:
            threshold: Amplitude threshold for speech detection
        """
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def is_speech(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """
        Detect speech using amplitude threshold.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio (unused in this implementation)

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        if audio_data is None or len(audio_data) == 0:
            return False, 0.0

        try:
            # Calculate RMS energy for more stable detection
            rms_energy = float(np.sqrt(np.mean(audio_data**2)))
            max_amplitude = float(np.max(np.abs(audio_data)))

            # Use RMS energy as primary metric with max amplitude as backup
            audio_level = max(rms_energy, max_amplitude * 0.5)

            is_speech = audio_level > self.threshold
            confidence = (
                min(audio_level / self.threshold, 1.0)
                if is_speech
                else audio_level / self.threshold
            )

            return is_speech, confidence

        except (ValueError, TypeError) as e:
            self.logger.warning("Error in threshold VAD: %s", e)
            return False, 0.0

    def get_name(self) -> str:
        """Get the name of this VAD implementation."""
        return f"ThresholdVAD(threshold={self.threshold})"


class SileroVAD(VADBase):
    """Silero VAD implementation using pre-trained neural network."""

    def __init__(self, threshold: float = 0.5):
        """
        Initialize Silero VAD.

        Args:
            threshold: Confidence threshold for speech detection (0.0-1.0)
        """
        self.threshold = threshold
        self.model = None  # Will be loaded lazily
        self.logger = logging.getLogger(__name__)
        self._model_loaded = False

        # Check dependencies
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for Silero VAD. Install with: pip install torch"
            )

        if not TORCHAUDIO_AVAILABLE:
            self.logger.warning(
                "torchaudio not available. Some audio preprocessing may be limited."
            )

    def _load_model(self) -> bool:
        """
        Load the Silero VAD model lazily.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._model_loaded:
            return True

        try:
            self.logger.info("Loading Silero VAD model...")

            # Check if torch is available before using
            if not TORCH_AVAILABLE or torch is None:
                raise ImportError("PyTorch not available")

            # Download and load Silero VAD model
            model, _ = torch.hub.load(  # pylint: disable=no-member
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )

            self.model = model
            self.model.eval()  # Set to evaluation mode
            self._model_loaded = True

            self.logger.info("Silero VAD model loaded successfully")
            return True

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Failed to load Silero VAD model: %s", e)
            self._model_loaded = False
            return False

    def is_speech(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """
        Detect speech using Silero VAD model.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        if audio_data is None or len(audio_data) == 0:
            return False, 0.0

        # Ensure model is loaded
        if not self._load_model():
            self.logger.warning("Silero VAD model not available, using fallback")
            return False, 0.0

        try:
            # Convert numpy array to PyTorch tensor
            if not TORCH_AVAILABLE or torch is None:
                raise ImportError("PyTorch not available")

            audio_tensor = torch.from_numpy(
                audio_data.astype(np.float32)
            )  # pylint: disable=no-member

            # Ensure audio is 1D
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.mean(dim=1)

            # Resample to 16kHz if needed (Silero VAD expects 16kHz)
            if sample_rate != 16000 and TORCHAUDIO_AVAILABLE and torchaudio is not None:
                resampler = torchaudio.transforms.Resample(  # pylint: disable=no-member
                    orig_freq=sample_rate, new_freq=16000
                )
                audio_tensor = resampler(audio_tensor)
            elif sample_rate != 16000:
                # Simple linear interpolation fallback
                target_length = int(len(audio_tensor) * 16000 / sample_rate)
                indices = np.linspace(0, len(audio_tensor) - 1, target_length)
                audio_tensor = torch.from_numpy(  # pylint: disable=no-member
                    np.interp(
                        indices, np.arange(len(audio_tensor)), audio_tensor.numpy()
                    )
                )

            # Silero VAD expects exactly 512 samples for 16kHz
            chunk_size = 512
            total_samples = len(audio_tensor)

            if total_samples < chunk_size:
                # Pad with zeros if too short
                padding = torch.zeros(
                    chunk_size - total_samples
                )  # pylint: disable=no-member
                audio_tensor = torch.cat(
                    [audio_tensor, padding]
                )  # pylint: disable=no-member
                speech_confidences = [self._get_chunk_confidence(audio_tensor)]
            else:
                # Process in chunks and take the maximum confidence
                speech_confidences = []
                for start in range(0, total_samples, chunk_size):
                    end = min(start + chunk_size, total_samples)
                    chunk = audio_tensor[start:end]

                    # Pad chunk if needed
                    if len(chunk) < chunk_size:
                        padding = torch.zeros(
                            chunk_size - len(chunk)
                        )  # pylint: disable=no-member
                        chunk = torch.cat([chunk, padding])  # pylint: disable=no-member

                    confidence = self._get_chunk_confidence(chunk)
                    speech_confidences.append(confidence)

            # Use maximum confidence across all chunks
            max_confidence = max(speech_confidences) if speech_confidences else 0.0
            is_speech = max_confidence > self.threshold

            return is_speech, max_confidence

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error in Silero VAD inference: %s", e)
            return False, 0.0

    def _get_chunk_confidence(self, audio_chunk):
        """Get speech confidence for a single 512-sample chunk."""
        try:
            if self.model is None or not TORCH_AVAILABLE or torch is None:
                raise RuntimeError("Model or PyTorch not available")

            with torch.no_grad():  # pylint: disable=no-member
                speech_prob = self.model(audio_chunk, 16000).item()

            return float(speech_prob)

        except Exception as e:  # pylint: disable=broad-except
            self.logger.debug("Error processing audio chunk: %s", e)
            return 0.0

    def get_name(self) -> str:
        """Get the name of this VAD implementation."""
        status = "loaded" if self._model_loaded else "not_loaded"
        return f"SileroVAD(threshold={self.threshold}, status={status})"


class VADManager:
    """
    Manager class for Voice Activity Detection with fallback support.

    Automatically selects the best available VAD implementation and provides
    fallback to simpler methods if advanced VAD fails.
    """

    def __init__(
        self,
        primary_vad: Optional[VADBase] = None,
        fallback_vad: Optional[VADBase] = None,
        prefer_silero: bool = True,
    ):
        """
        Initialize VAD manager.

        Args:
            primary_vad: Primary VAD implementation to use
            fallback_vad: Fallback VAD if primary fails
            prefer_silero: Whether to prefer Silero VAD if available
        """
        self.logger = logging.getLogger(__name__)

        # Set up VAD implementations
        if primary_vad is None:
            if prefer_silero and TORCH_AVAILABLE:
                try:
                    primary_vad = SileroVAD()
                    self.logger.info("Using Silero VAD as primary")
                except ImportError:
                    primary_vad = ThresholdVAD()
                    self.logger.info("Silero VAD unavailable, using Threshold VAD")
            else:
                primary_vad = ThresholdVAD()
                self.logger.info("Using Threshold VAD as primary")

        if fallback_vad is None:
            fallback_vad = ThresholdVAD()

        self.primary_vad = primary_vad
        self.fallback_vad = fallback_vad
        self.primary_failures = 0
        self.max_failures = 3  # Switch to fallback after 3 consecutive failures

        self.logger.info(
            "VAD Manager initialized: primary=%s, fallback=%s",
            self.primary_vad.get_name(),
            self.fallback_vad.get_name(),
        )

    def is_speech(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Tuple[bool, float, str]:
        """
        Detect speech using the best available VAD method.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Tuple of (is_speech: bool, confidence: float, vad_method: str)
        """
        # Try primary VAD first
        if self.primary_failures < self.max_failures:
            try:
                is_speech, confidence = self.primary_vad.is_speech(
                    audio_data, sample_rate
                )
                self.primary_failures = 0  # Reset failure counter on success
                return is_speech, confidence, self.primary_vad.get_name()

            except Exception as e:  # pylint: disable=broad-except
                self.primary_failures += 1
                self.logger.warning(
                    "Primary VAD failed (attempt %d/%d): %s",
                    self.primary_failures,
                    self.max_failures,
                    e,
                )

        # Use fallback VAD
        try:
            is_speech, confidence = self.fallback_vad.is_speech(audio_data, sample_rate)
            return is_speech, confidence, self.fallback_vad.get_name()

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Fallback VAD failed: %s", e)
            return False, 0.0, "error"

    def get_status(self) -> dict:
        """
        Get status information about the VAD manager.

        Returns:
            Dictionary with VAD status information
        """
        return {
            "primary_vad": self.primary_vad.get_name(),
            "fallback_vad": self.fallback_vad.get_name(),
            "primary_failures": self.primary_failures,
            "using_fallback": self.primary_failures >= self.max_failures,
            "torch_available": TORCH_AVAILABLE,
            "torchaudio_available": TORCHAUDIO_AVAILABLE,
        }


def create_vad_manager(vad_type: str = "auto", **kwargs) -> VADManager:
    """
    Factory function to create a VAD manager.

    Args:
        vad_type: Type of VAD to create ("auto", "silero", "threshold")
        **kwargs: Additional arguments for VAD initialization

    Returns:
        Configured VAD manager
    """
    if vad_type == "auto":
        return VADManager(prefer_silero=True, **kwargs)
    elif vad_type == "silero":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Silero VAD")
        primary_vad = SileroVAD(**kwargs)
        return VADManager(primary_vad=primary_vad)
    elif vad_type == "threshold":
        primary_vad = ThresholdVAD(**kwargs)
        return VADManager(primary_vad=primary_vad, fallback_vad=primary_vad)
    else:
        raise ValueError(f"Unknown VAD type: {vad_type}")
