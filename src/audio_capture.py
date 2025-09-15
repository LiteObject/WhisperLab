"""Audio capture module for real-time microphone input."""

import queue

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

    def audio_callback(self, indata, _frames, _time, status):
        """Callback function called by sounddevice when audio data is available."""
        if status:
            print(f"Audio callback status: {status}")

        if self.recording:
            # Convert to mono if stereo and add to queue
            if self.channels == 1:
                audio_data = indata[:, 0].copy()
            else:
                audio_data = indata.copy()

            self.audio_queue.put(audio_data)

    def start_recording(self):
        """Start audio recording in a separate thread."""
        self.recording = True
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
            print(
                f"Recording started at {self.sample_rate} Hz, {self.channels} channel(s)"
            )

        except (OSError, sd.PortAudioError) as e:
            print(f"Error starting audio recording: {e}")
            self.recording = False

    def stop_recording(self):
        """Stop audio recording."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
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
