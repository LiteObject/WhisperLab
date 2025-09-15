"""Main entry point for WhisperLab real-time audio transcription application."""

import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position,import-error
from audio_capture import AudioCapture
from transcription import Transcription
from console_animation import get_animator, start_animation, stop_animation
from voice_activity_detection import create_vad_manager
from config import get_config


def setup_logging() -> logging.Logger:
    """Set up comprehensive logging configuration."""
    config = get_config()

    # Create logger
    logger = logging.getLogger("whisperlab")
    logger.setLevel(getattr(logging, config.logging.level.upper()))

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    simple_formatter = logging.Formatter(config.logging.format)

    # Console handler
    if config.logging.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if config.logging.file_enabled:
        # Ensure log directory exists
        log_path = Path(config.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Log startup information
    logger.info("WhisperLab logging initialized")
    logger.debug("Log level: %s", config.logging.level)
    logger.debug("Console logging: %s", config.logging.console_enabled)
    logger.debug("File logging: %s", config.logging.file_enabled)
    if config.logging.file_enabled:
        logger.debug("Log file: %s", config.logging.file_path)

    return logger


class WhisperLab:
    """Main application class for real-time audio transcription using Whisper."""

    def __init__(self):
        # Set up logging first
        self.logger = setup_logging()
        self.logger.info("Initializing WhisperLab application")

        self.audio_capture = AudioCapture()
        self.transcription = Transcription()
        self.running = False
        self.animator = get_animator()

        # Initialize sophisticated VAD manager
        self.vad_manager = create_vad_manager(vad_type="auto")
        vad_status = self.vad_manager.get_status()
        self.logger.info("VAD initialized: %s", vad_status["primary_vad"])
        print(f"🎯 VAD initialized: {vad_status['primary_vad']}")

    def transcribe_audio(self):
        """Continuously get audio data and transcribe it."""
        self.logger.info("Transcription thread started")
        print("Transcription thread started...")

        # Use deque for efficient memory management with maxlen
        buffer_duration = 3.0  # seconds
        sample_rate = self.audio_capture.sample_rate
        max_buffer_samples = int(buffer_duration * sample_rate)

        # Use deque with maxlen for automatic memory management
        audio_buffer = deque(maxlen=max_buffer_samples)

        last_transcription_time = time.time()
        transcription_interval = 2.0  # transcribe every 2 seconds
        transcription_count = 0

        self.logger.debug(
            "Audio buffer configuration - duration: %s, max_samples: %s",
            buffer_duration,
            max_buffer_samples,
        )
        self.logger.debug(
            "Using deque with maxlen=%s for efficient memory management",
            max_buffer_samples,
        )

        while self.running:
            try:
                audio_data = self.audio_capture.get_audio_data()
                if audio_data is not None and len(audio_data) > 0:
                    # Add audio to deque (automatically handles size limit)
                    audio_buffer.extend(audio_data.flatten())

                    # Calculate current audio level for animation
                    current_level = (
                        float(np.max(np.abs(audio_data)))
                        if len(audio_data) > 0
                        else 0.0
                    )
                    self.animator.update_audio_level(current_level)

                    # Check if it's time to transcribe
                    current_time = time.time()
                    time_condition = (
                        current_time - last_transcription_time >= transcription_interval
                    )
                    buffer_condition = len(audio_buffer) >= sample_rate

                    if (
                        time_condition and buffer_condition
                    ):  # At least 1 second of audio
                        # Convert deque to numpy array efficiently
                        audio_array = np.fromiter(audio_buffer, dtype=np.float32)

                        # Use sophisticated VAD instead of simple threshold
                        speech_detected, vad_confidence, vad_method = (
                            self.vad_manager.is_speech(audio_array, sample_rate)
                        )

                        self.logger.debug(
                            "VAD result - speech: %s, confidence: %.3f, method: %s",
                            speech_detected,
                            vad_confidence,
                            vad_method,
                        )

                        # Update animation with VAD results
                        self.animator.set_speech_detected(speech_detected)
                        self.animator.set_vad_confidence(
                            vad_confidence
                        )  # Will add this method
                        self.animator.set_vad_method(vad_method)  # Will add this method

                        # Only transcribe if speech is detected
                        if speech_detected:
                            transcription_count += 1
                            self.animator.set_transcribing(True)

                            buffer_seconds = len(audio_buffer) / sample_rate
                            self.logger.info(
                                "Starting transcription #%d (%.1fs of audio, VAD confidence: %.3f)",
                                transcription_count,
                                buffer_seconds,
                                vad_confidence,
                            )

                            self.animator.display_static_message(
                                f"🎤 Transcribing {buffer_seconds:.1f}s of audio "
                                f"(VAD confidence: {vad_confidence:.3f}, method: {vad_method})...",
                                "info",
                            )

                            text = self.transcription.transcribe_audio_direct(
                                audio_array
                            )

                            self.animator.set_transcribing(False)

                            if text and text.strip():
                                self.logger.info(
                                    "Transcription #%d successful: %s",
                                    transcription_count,
                                    text.strip()[:100],
                                )
                                self.animator.set_transcription_result(
                                    text.strip(), transcription_count
                                )
                                self.animator.display_static_message(
                                    f"✅ Transcribed #{transcription_count}: {text}",
                                    "success",
                                )
                            else:
                                self.logger.warning(
                                    "Transcription #%d returned empty result",
                                    transcription_count,
                                )
                                self.animator.display_static_message(
                                    f"❌ Transcription #{transcription_count} "
                                    f"returned empty result",
                                    "warning",
                                )
                        else:
                            self.animator.set_speech_detected(False)
                            self.logger.debug(
                                "Skipping transcription - no speech detected"
                            )

                        last_transcription_time = current_time

                        # For deque, we need to manually manage overlap since maxlen auto-trims
                        # Keep some overlap for continuity by rotating the deque
                        overlap_samples = int(0.5 * sample_rate)  # 0.5 second overlap
                        if len(audio_buffer) > overlap_samples:
                            # Create new deque with overlap samples from the end
                            overlap_data = list(audio_buffer)[-overlap_samples:]
                            audio_buffer.clear()
                            audio_buffer.extend(overlap_data)
                            self.logger.debug(
                                "Buffer overlap maintained: %d samples",
                                len(overlap_data),
                            )

                else:
                    # Update animation with zero level when no audio
                    self.animator.update_audio_level(0.0)
                    # Small sleep to avoid busy waiting
                    time.sleep(0.01)

            except KeyboardInterrupt:
                self.logger.info("Transcription thread interrupted by user")
                break
            except (RuntimeError, OSError) as e:
                self.logger.error("Error in transcription thread: %s", e)
                print(f"Error in transcription thread: {e}")
                time.sleep(0.1)

    def signal_handler(self, _sig, _frame):
        """Handle Ctrl+C gracefully."""
        self.logger.info("Received interrupt signal - shutting down gracefully")
        print("\nStopping WhisperLab...")
        stop_animation()
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the audio capture and transcription."""
        self.logger.info("Starting WhisperLab application")
        print("Starting WhisperLab...")
        print("Press Ctrl+C to stop")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        self.running = True

        try:
            # Use context manager for automatic resource cleanup
            with self.audio_capture:
                # Start console animation
                self.logger.debug("Starting console animation")
                start_animation()

                # Start transcription in a separate thread
                self.logger.debug("Starting transcription thread")
                transcribe_thread = threading.Thread(
                    target=self.transcribe_audio, daemon=True
                )
                transcribe_thread.start()

                try:
                    # Keep the main thread alive
                    self.logger.debug("Main loop started")
                    while self.running:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    self.logger.info("Keyboard interrupt received in main loop")
                    self.stop()
        except Exception as e:
            self.logger.error("Error in audio capture context: %s", e)
            raise

    def stop(self):
        """Stop audio capture and transcription."""
        self.logger.info("Stopping WhisperLab application")
        self.running = False
        stop_animation()
        # Note: Audio capture cleanup is handled by context manager
        self.logger.info("WhisperLab stopped successfully")
        print("WhisperLab stopped.")


def main():
    """Main entry point."""
    try:
        app = WhisperLab()
        app.start()
    except (ImportError, OSError, RuntimeError) as e:
        # Try to log the error, but fallback to print if logging isn't set up
        try:
            logger = logging.getLogger("whisperlab")
            logger.critical("Failed to start WhisperLab application: %s", e)
        except (AttributeError, ImportError):  # pylint: disable=broad-except
            print(f"Failed to start WhisperLab application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
