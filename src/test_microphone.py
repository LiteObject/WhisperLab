"""Simple audio test script to verify microphone is working."""

import numpy as np
import sounddevice as sd


def test_microphone():
    """Test if microphone is detecting audio."""
    print("🎤 Testing microphone...")
    print("Available audio devices:")
    print(sd.query_devices())

    print(f"\nDefault input device: {sd.query_devices(kind='input')}")

    duration = 5  # seconds
    sample_rate = 16000

    print(f"\n🔴 Recording for {duration} seconds... Please speak!")

    try:
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()  # Wait until recording is finished

        # Analyze the recorded audio
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data**2))

        print("✅ Recording completed!")
        print(f"📊 Max amplitude: {max_amplitude:.4f}")
        print(f"📊 RMS level: {rms_level:.4f}")

        if max_amplitude > 0.01:
            print("✅ Good! Audio detected - your microphone is working")
        elif max_amplitude > 0.001:
            print("⚠️ Weak audio signal - check microphone volume/placement")
        else:
            print("❌ No audio detected - check microphone permissions/connection")

        # Test if audio contains speech-like patterns
        if rms_level > 0.005:
            print("🗣️ Audio level suggests speech was detected")
        else:
            print("🔇 Audio level too low for speech detection")

    except (OSError, sd.PortAudioError, ValueError) as e:
        print(f"❌ Error during microphone test: {e}")


if __name__ == "__main__":
    test_microphone()
