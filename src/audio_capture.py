"""Audio capture module for real-time microphone input."""

import queue
import logging
from contextlib import contextmanager

import sounddevice as sd
import numpy as np

from config import SAMPLE_RATE, CHANNELS, CHUNK_SIZE  # pylint: disable=import-error


class AudioCapture:
    """Handles real-time audio capture from microphone using sounddevice."""

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.chunk_size = CHUNK_SIZE
        self.audio_queue = queue.Queue()
        self.recording = False
        self.stream = None
        self.logger = logging.getLogger("whisperlab.audio")

    def __enter__(self):
        """Context manager entry."""
        self.logger.debug("Entering AudioCapture context")
        self.start_recording()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.logger.debug(
            "Exiting AudioCapture context - type: %s, value: %s", exc_type, exc_val
        )
        self.stop_recording()

        # Clean up any remaining audio data in queue
        queue_size = self.audio_queue.qsize()
        if queue_size > 0:
            self.logger.debug("Clearing %d items from audio queue", queue_size)
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

        return False  # Don't suppress exceptions

    @contextmanager
    def recording_session(self):
        """Context manager for a recording session with automatic cleanup."""
        self.logger.info("Starting recording session")
        try:
            self.start_recording()
            yield self
        finally:
            self.logger.info("Ending recording session")
            self.stop_recording()

    def audio_callback(self, indata, _frames, _time, status):
        """Callback function called by sounddevice when audio data is available."""
        if status:
            self.logger.warning("Audio callback status: %s", status)
            print(f"Audio callback status: {status}")

        if self.recording:
            # Convert to mono if stereo and add to queue
            if self.channels == 1:
                audio_data = indata[:, 0].copy()
            else:
                audio_data = indata.copy()

            # Check queue size to prevent memory buildup
            if self.audio_queue.qsize() > 100:  # Configurable threshold
                self.logger.warning(
                    "Audio queue size exceeded threshold, dropping oldest frame"
                )
                try:
                    self.audio_queue.get_nowait()  # Remove oldest
                except queue.Empty:
                    pass

            self.audio_queue.put(audio_data)

    def start_recording(self):
        """Start audio recording in a separate thread."""
        if self.recording:
            self.logger.warning("Recording already started")
            return

        self.recording = True
        self.logger.info("Starting audio recording...")
        print("Starting audio recording...")

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32,
            )
            self.stream.start()
            self.logger.info(
                "Recording started at %d Hz, %d channel(s)",
                self.sample_rate,
                self.channels,
            )
            print(
                f"Recording started at {self.sample_rate} Hz, {self.channels} channel(s)"
            )

        except (OSError, sd.PortAudioError) as e:
            self.logger.error("Error starting audio recording: %s", e)
            print(f"Error starting audio recording: {e}")
            self.recording = False
            raise

    def stop_recording(self):
        """Stop audio recording with proper cleanup."""
        if not self.recording:
            self.logger.debug("Recording already stopped")
            return

        self.logger.info("Stopping audio recording...")
        self.recording = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.logger.debug("Audio stream stopped and closed successfully")
            except Exception as e:  # pylint: disable=broad-except
                self.logger.error("Error stopping audio stream: %s", e)
            finally:
                self.stream = None

        print("Audio recording stopped.")

    def get_audio_data(self):
        """Get audio data from the queue. Returns None if no data available."""
        try:
            # Get audio data with a short timeout to avoid blocking
            audio_data = self.audio_queue.get(timeout=0.1)
            return audio_data
        except queue.Empty:
            return None

    def is_recording(self):
        """Check if currently recording."""
        return self.recording

    def get_queue_size(self):
        """Get current audio queue size for monitoring."""
        return self.audio_queue.qsize()

    def clear_queue(self):
        """Clear all audio data from the queue."""
        queue_size = self.audio_queue.qsize()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.logger.debug("Cleared %d items from audio queue", queue_size)
        return queue_size
