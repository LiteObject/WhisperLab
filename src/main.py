"""Main entry point for WhisperLab real-time audio transcription application."""

import os
import signal
import sys
import threading
import time

import numpy as np

# Add current directory to Python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=wrong-import-position,import-error
from audio_capture import AudioCapture
from transcription import Transcription
from console_animation import get_animator, start_animation, stop_animation


class WhisperLab:
    """Main application class for real-time audio transcription using Whisper."""

    def __init__(self):
        self.audio_capture = AudioCapture()
        self.transcription = Transcription()
        self.running = False
        self.animator = get_animator()

    def transcribe_audio(self):
        """Continuously get audio data and transcribe it."""
        print("Transcription thread started...")

        audio_buffer = []
        buffer_duration = 3.0  # seconds
        sample_rate = self.audio_capture.sample_rate
        max_buffer_samples = int(buffer_duration * sample_rate)

        last_transcription_time = time.time()
        transcription_interval = 2.0  # transcribe every 2 seconds
        transcription_count = 0

        while self.running:
            try:
                audio_data = self.audio_capture.get_audio_data()
                if audio_data is not None and len(audio_data) > 0:
                    # Add audio to buffer
                    audio_buffer.extend(audio_data.flatten())

                    # Keep buffer at maximum size
                    if len(audio_buffer) > max_buffer_samples:
                        audio_buffer = audio_buffer[-max_buffer_samples:]

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
                        # Convert buffer to numpy array
                        audio_array = np.array(audio_buffer, dtype=np.float32)
                        audio_level = float(np.max(np.abs(audio_array)))

                        # Update VAD status
                        speech_detected = audio_level > 0.01
                        self.animator.set_speech_detected(speech_detected)

                        # Only transcribe if there's sufficient audio activity
                        if speech_detected:
                            transcription_count += 1
                            self.animator.set_transcribing(True)

                            buffer_seconds = len(audio_buffer) / sample_rate
                            self.animator.display_static_message(
                                f"🎤 Transcribing {buffer_seconds:.1f}s of audio "
                                f"(level: {audio_level:.4f})...",
                                "info",
                            )

                            text = self.transcription.transcribe_audio_direct(
                                audio_array
                            )

                            self.animator.set_transcribing(False)

                            if text and text.strip():
                                self.animator.set_transcription_result(
                                    text.strip(), transcription_count
                                )
                                self.animator.display_static_message(
                                    f"✅ Transcribed #{transcription_count}: {text}",
                                    "success",
                                )
                            else:
                                self.animator.display_static_message(
                                    f"❌ Transcription #{transcription_count} "
                                    f"returned empty result",
                                    "warning",
                                )
                        else:
                            self.animator.set_speech_detected(False)
                            # Don't display the "skipping" message as animation shows VAD status

                        last_transcription_time = current_time
                        # Keep some overlap for continuity
                        overlap_samples = int(0.5 * sample_rate)  # 0.5 second overlap
                        audio_buffer = (
                            audio_buffer[-overlap_samples:]
                            if len(audio_buffer) > overlap_samples
                            else []
                        )

                else:
                    # Update animation with zero level when no audio
                    self.animator.update_audio_level(0.0)
                    # Small sleep to avoid busy waiting
                    time.sleep(0.01)

            except KeyboardInterrupt:
                break
            except (RuntimeError, OSError) as e:
                print(f"Error in transcription thread: {e}")
                time.sleep(0.1)

    def signal_handler(self, _sig, _frame):
        """Handle Ctrl+C gracefully."""
        print("\nStopping WhisperLab...")
        stop_animation()
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the audio capture and transcription."""
        print("Starting WhisperLab...")
        print("Press Ctrl+C to stop")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        self.running = True

        # Start audio capture
        self.audio_capture.start_recording()

        # Start console animation
        start_animation()

        # Start transcription in a separate thread
        transcribe_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        transcribe_thread.start()

        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop audio capture and transcription."""
        self.running = False
        stop_animation()
        self.audio_capture.stop_recording()
        print("WhisperLab stopped.")


def main():
    """Main entry point."""
    app = WhisperLab()
    app.start()


if __name__ == "__main__":
    main()
