"""Comprehensive audio test script to verify microphone functionality and quality."""

import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
import sounddevice as sd
from scipy import signal

# Try to import our config
try:
    from config import get_config  # pylint: disable=import-error

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class MicrophoneAnalyzer:
    """Comprehensive microphone testing and analysis."""

    def __init__(self):
        self.logger = logging.getLogger("whisperlab.microphone_test")

        # Audio configuration
        if CONFIG_AVAILABLE:
            config = get_config()  # type: ignore
            self.sample_rate = config.audio.sample_rate
            self.channels = config.audio.channels
        else:
            self.sample_rate = 16000
            self.channels = 1

        # Analysis parameters
        self.test_duration = 5.0  # seconds
        self.noise_floor_duration = 2.0  # seconds for noise floor measurement

    def list_audio_devices(self) -> None:
        """List all available audio devices with detailed information."""
        print("🎤 Audio Device Information:")
        print("=" * 60)

        try:
            # Use simple approach to avoid type issues
            print("Available audio devices:")
            devices = sd.query_devices()
            print(devices)
            print()

            default_input = sd.query_devices(kind="input")
            print(f"Default input device: {default_input}")

            default_output = sd.query_devices(kind="output")
            print(f"Default output device: {default_output}")
            print()

        except (OSError, ValueError, RuntimeError) as e:
            print(f"Error listing audio devices: {e}")
            self.logger.error("Failed to list audio devices: %s", e)

    def measure_noise_floor(self) -> Tuple[float, float, np.ndarray]:
        """Measure the noise floor of the microphone."""
        print(f"🔇 Measuring noise floor for {self.noise_floor_duration} seconds...")
        print("   Please remain silent!")

        try:
            # Record silence to measure noise floor
            noise_samples = int(self.noise_floor_duration * self.sample_rate)
            noise_data = sd.rec(
                noise_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()

            if self.channels == 1:
                noise_data = noise_data.flatten()
            else:
                noise_data = noise_data.mean(axis=1)

            noise_rms = np.sqrt(np.mean(noise_data**2))
            noise_max = np.max(np.abs(noise_data))

            return noise_rms, noise_max, noise_data

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("Error measuring noise floor: %s", e)
            return 0.0, 0.0, np.array([])

    def analyze_frequency_spectrum(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Perform frequency analysis on audio data."""
        if len(audio_data) == 0:
            return {}

        # Compute FFT using numpy
        n = len(audio_data)
        fft_values = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(n, 1 / self.sample_rate)

        # Take only positive frequencies - use integer indexing instead of boolean
        n_positive = n // 2
        positive_freqs = frequencies[1:n_positive]  # Skip DC component at index 0
        positive_magnitudes = np.abs(fft_values[1:n_positive])

        # Find dominant frequencies
        magnitude_db = 20 * np.log10(positive_magnitudes + 1e-10)  # Avoid log(0)

        # Find peaks
        peaks, _ = signal.find_peaks(magnitude_db, height=-60, distance=20)

        # Frequency band analysis
        freq_bands = {
            "sub_bass": (20, 60),  # Sub-bass
            "bass": (60, 250),  # Bass
            "low_mid": (250, 500),  # Low midrange
            "mid": (500, 2000),  # Midrange (important for speech)
            "high_mid": (2000, 4000),  # High midrange (speech clarity)
            "presence": (4000, 6000),  # Presence (speech intelligibility)
            "brilliance": (6000, 20000),  # Brilliance
        }

        band_powers = {}
        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            if np.any(band_mask):
                band_power = np.mean(positive_magnitudes[band_mask] ** 2)
                band_powers[band_name] = band_power
            else:
                band_powers[band_name] = 0.0

        # Find dominant frequency
        dominant_freq_idx = np.argmax(positive_magnitudes)
        dominant_frequency = positive_freqs[dominant_freq_idx]

        return {
            "dominant_frequency": dominant_frequency,
            "frequency_peaks": positive_freqs[peaks] if len(peaks) > 0 else [],
            "peak_magnitudes": magnitude_db[peaks] if len(peaks) > 0 else [],
            "band_powers": band_powers,
            "total_power": np.sum(positive_magnitudes**2),
            "spectral_centroid": np.sum(positive_freqs * positive_magnitudes)
            / np.sum(positive_magnitudes),
            "max_frequency": positive_freqs[np.argmax(positive_magnitudes)],
        }

    def detect_speech_characteristics(
        self, audio_data: np.ndarray, spectrum_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze audio for speech-like characteristics."""
        if len(audio_data) == 0 or not spectrum_analysis:
            return {"is_likely_speech": False, "confidence": 0.0, "reasons": []}

        reasons = []
        confidence_factors = []

        # Check RMS level (speech typically has moderate energy)
        rms = np.sqrt(np.mean(audio_data**2))
        if 0.01 < rms < 0.5:
            reasons.append(f"Good RMS level for speech ({rms:.3f})")
            confidence_factors.append(0.3)
        elif rms > 0.001:
            reasons.append(f"Detectable audio level ({rms:.3f})")
            confidence_factors.append(0.1)

        # Check frequency content
        band_powers = spectrum_analysis.get("band_powers", {})

        # Speech energy is typically concentrated in 85Hz-8kHz, especially 300Hz-3.4kHz
        speech_bands = ["bass", "low_mid", "mid", "high_mid"]
        speech_power = sum(band_powers.get(band, 0) for band in speech_bands)
        total_power = spectrum_analysis.get("total_power", 1)

        if total_power > 0:
            speech_ratio = speech_power / total_power
            if speech_ratio > 0.6:
                reasons.append(f"High speech frequency content ({speech_ratio:.2f})")
                confidence_factors.append(0.4)
            elif speech_ratio > 0.3:
                reasons.append(
                    f"Moderate speech frequency content ({speech_ratio:.2f})"
                )
                confidence_factors.append(0.2)

        # Check for fundamental frequency in typical speech range (85-300 Hz)
        dominant_freq = spectrum_analysis.get("dominant_frequency", 0)
        if 85 <= dominant_freq <= 300:
            reasons.append(
                f"Dominant frequency in speech range ({dominant_freq:.0f} Hz)"
            )
            confidence_factors.append(0.3)

        # Check spectral centroid (speech typically 1-4 kHz)
        spectral_centroid = spectrum_analysis.get("spectral_centroid", 0)
        if 1000 <= spectral_centroid <= 4000:
            reasons.append(
                f"Spectral centroid in speech range ({spectral_centroid:.0f} Hz)"
            )
            confidence_factors.append(0.2)

        # Calculate overall confidence
        confidence = min(1.0, sum(confidence_factors))
        is_likely_speech = confidence > 0.4

        return {
            "is_likely_speech": is_likely_speech,
            "confidence": confidence,
            "reasons": reasons,
            "rms_level": rms,
            "speech_frequency_ratio": (
                speech_power / total_power if total_power > 0 else 0
            ),
        }

    def comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive microphone test with analysis."""
        print("🔬 Comprehensive Microphone Analysis")
        print("=" * 50)

        results = {
            "timestamp": time.time(),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "success": False,
        }

        try:
            # Step 1: List devices
            self.list_audio_devices()

            # Step 2: Measure noise floor
            noise_rms, noise_max, noise_data = self.measure_noise_floor()
            noise_spectrum = self.analyze_frequency_spectrum(noise_data)

            print("✅ Noise floor measured:")
            print(f"   RMS: {noise_rms:.6f}")
            print(f"   Max: {noise_max:.6f}")
            if noise_spectrum:
                print(
                    f"   Dominant frequency: {noise_spectrum.get('dominant_frequency', 0):.0f} Hz"
                )
            print()

            results["noise_floor"] = {
                "rms": noise_rms,
                "max": noise_max,
                "spectrum": noise_spectrum,
            }

            # Step 3: Record test audio
            print(f"🔴 Recording for {self.test_duration} seconds...")
            print("   Please speak clearly!")

            test_samples = int(self.test_duration * self.sample_rate)
            audio_data = sd.rec(
                test_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()

            if self.channels == 1:
                audio_data = audio_data.flatten()
            else:
                audio_data = audio_data.mean(axis=1)

            print("✅ Recording completed!")
            print()

            # Step 4: Analyze recorded audio
            max_amplitude = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            snr = (
                20 * np.log10(rms_level / noise_rms) if noise_rms > 0 else float("inf")
            )

            # Frequency analysis
            spectrum_analysis = self.analyze_frequency_spectrum(audio_data)

            # Speech detection
            speech_analysis = self.detect_speech_characteristics(
                audio_data, spectrum_analysis
            )

            # Step 5: Generate report
            print("📊 Audio Analysis Results:")
            print(f"   Max amplitude: {max_amplitude:.4f}")
            print(f"   RMS level: {rms_level:.4f}")
            print(f"   Signal-to-Noise Ratio: {snr:.1f} dB")
            print(
                f"   Dynamic range: {20 * np.log10(max_amplitude / noise_max):.1f} dB"
            )
            print()

            if spectrum_analysis:
                print("🎵 Frequency Analysis:")
                print(
                    f"   Dominant frequency: {spectrum_analysis['dominant_frequency']:.0f} Hz"
                )
                print(
                    f"   Spectral centroid: {spectrum_analysis['spectral_centroid']:.0f} Hz"
                )

                # Show frequency band powers
                band_powers = spectrum_analysis["band_powers"]
                print("   Frequency band distribution:")
                for band, power in band_powers.items():
                    percentage = (
                        (power / spectrum_analysis["total_power"] * 100)
                        if spectrum_analysis["total_power"] > 0
                        else 0
                    )
                    print(f"     {band}: {percentage:.1f}%")
                print()

            print("🗣️ Speech Analysis:")
            if speech_analysis["is_likely_speech"]:
                print(
                    f"   ✅ Speech detected (confidence: {speech_analysis['confidence']:.2f})"
                )
            else:
                print(
                    f"   ❌ No speech detected (confidence: {speech_analysis['confidence']:.2f})"
                )

            for reason in speech_analysis["reasons"]:
                print(f"   • {reason}")
            print()

            # Step 6: Overall assessment
            print("🎯 Overall Assessment:")

            if max_amplitude < 0.001:
                print("   ❌ CRITICAL: No significant audio detected")
                print("   → Check microphone connection and permissions")
                assessment = "critical"
            elif rms_level < noise_rms * 2:
                print("   ⚠️  WARNING: Audio level too close to noise floor")
                print("   → Check microphone placement and volume settings")
                assessment = "warning"
            elif snr < 10:
                print("   ⚠️  WARNING: Poor signal-to-noise ratio")
                print("   → Consider using in a quieter environment")
                assessment = "warning"
            elif speech_analysis["is_likely_speech"] and snr > 15:
                print("   ✅ EXCELLENT: High quality speech detected")
                assessment = "excellent"
            elif speech_analysis["is_likely_speech"]:
                print("   ✅ GOOD: Speech detected with adequate quality")
                assessment = "good"
            elif max_amplitude > 0.01:
                print("   ⚠️  WARNING: Audio detected but may not be speech")
                print("   → Try speaking more clearly during recording")
                assessment = "warning"
            else:
                print("   ❌ POOR: Audio quality insufficient for speech recognition")
                assessment = "poor"

            results.update(
                {
                    "success": True,
                    "assessment": assessment,
                    "max_amplitude": max_amplitude,
                    "rms_level": rms_level,
                    "snr_db": snr,
                    "spectrum_analysis": spectrum_analysis,
                    "speech_analysis": speech_analysis,
                }
            )

            return results

        except (OSError, ValueError, RuntimeError) as e:
            print(f"❌ Error during microphone test: {e}")
            self.logger.error("Microphone test failed: %s", e)
            results["error"] = str(e)
            return results

    def quick_test(self) -> bool:
        """Quick microphone functionality test."""
        print("🚀 Quick Microphone Test (3 seconds)")

        try:
            duration = 3
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()

            if self.channels > 1:
                audio_data = audio_data.mean(axis=1)
            else:
                audio_data = audio_data.flatten()

            max_amplitude = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))

            print(f"Max: {max_amplitude:.4f}, RMS: {rms_level:.4f}")

            if max_amplitude > 0.01:
                print("✅ Microphone working!")
                return True
            else:
                print("❌ No audio detected")
                return False

        except (OSError, ValueError, RuntimeError) as e:
            print(f"❌ Test failed: {e}")
            return False


def test_microphone():
    """Simple compatibility test function."""
    analyzer = MicrophoneAnalyzer()
    return analyzer.quick_test()


def main():
    """Main function for comprehensive testing."""
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)

    analyzer = MicrophoneAnalyzer()

    print("WhisperLab Microphone Diagnostic Tool")
    print("=" * 40)
    print("Choose test type:")
    print("1. Quick test (3 seconds)")
    print("2. Comprehensive analysis (7+ seconds)")

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            analyzer.quick_test()
        elif choice == "2":
            results = analyzer.comprehensive_test()
            if results.get("success"):
                print("\n📄 Test completed successfully!")
                print(f"Assessment: {results['assessment'].upper()}")
            else:
                print(f"\n❌ Test failed: {results.get('error', 'Unknown error')}")
        else:
            print("Invalid choice. Running quick test...")
            analyzer.quick_test()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except (OSError, ValueError, RuntimeError, ImportError) as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
