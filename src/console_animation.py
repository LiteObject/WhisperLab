"""Console animation utilities for real-time audio visualization and VAD feedback."""

import sys
import time
import threading
from collections import deque

try:
    from colorama import init, Fore, Style

    init(autoreset=True)  # Initialize colorama for Windows
    COLORAMA_AVAILABLE = True
except ImportError:
    # Fallback if colorama is not available
    COLORAMA_AVAILABLE = False

    class _ForeColors:
        GREEN = RED = YELLOW = CYAN = MAGENTA = BLUE = WHITE = RESET = ""

    class _StyleColors:
        BRIGHT = DIM = RESET_ALL = ""

    Fore = _ForeColors()
    Style = _StyleColors()


class ConsoleAnimator:
    """Handles real-time console animations for audio visualization."""

    def __init__(self):
        self.running = False
        self.animation_thread = None
        self.lock = threading.Lock()

        # Animation state
        self.audio_level = 0.0
        self.is_speech_detected = False
        self.is_transcribing = False
        self.last_transcription = ""
        self.transcription_count = 0

        # Enhanced VAD state
        self.vad_confidence = 0.0
        self.vad_method = "Unknown"

        # Visual elements
        self.audio_history = deque(maxlen=50)  # Keep last 50 audio levels for waveform
        self.animation_frame = 0

    def start(self):
        """Start the animation thread."""
        if not self.running:
            self.running = True
            self.animation_thread = threading.Thread(
                target=self._animation_loop, daemon=True
            )
            self.animation_thread.start()

    def stop(self):
        """Stop the animation thread."""
        self.running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)

    def update_audio_level(self, level):
        """Update the current audio level for visualization."""
        with self.lock:
            self.audio_level = level
            self.audio_history.append(level)

    def set_speech_detected(self, detected):
        """Update speech detection status."""
        with self.lock:
            self.is_speech_detected = detected

    def set_transcribing(self, transcribing):
        """Update transcription status."""
        with self.lock:
            self.is_transcribing = transcribing

    def set_transcription_result(self, text, count):
        """Update latest transcription result."""
        with self.lock:
            self.last_transcription = text
            self.transcription_count = count

    def set_vad_confidence(self, confidence):
        """Update VAD confidence level."""
        with self.lock:
            self.vad_confidence = confidence

    def set_vad_method(self, method):
        """Update VAD method name."""
        with self.lock:
            self.vad_method = method

    def _get_audio_meter(self, level, width=20):
        """Generate a visual audio level meter."""
        if not COLORAMA_AVAILABLE:
            # Simple ASCII meter without colors
            filled = int(level * width)
            meter = "█" * filled + "░" * (width - filled)
            return f"[{meter}] {level:.3f}"

        # Colored meter
        filled = int(level * width)

        # Color based on level
        if level > 0.1:
            color = Fore.RED + Style.BRIGHT
        elif level > 0.05:
            color = Fore.YELLOW
        elif level > 0.01:
            color = Fore.GREEN
        else:
            color = Fore.CYAN

        meter = (
            color + "█" * filled + Style.RESET_ALL + Fore.WHITE + "░" * (width - filled)
        )
        return f"[{meter}{Style.RESET_ALL}] {level:.3f}"

    def _get_waveform(self, width=40):
        """Generate a simple waveform visualization."""
        if len(self.audio_history) < 2:
            return "─" * width

        waveform = []
        for i in range(width):
            # Sample from audio history
            if i < len(self.audio_history):
                level = self.audio_history[i]
                if level > 0.05:
                    char = (
                        "▄"
                        if not COLORAMA_AVAILABLE
                        else Fore.GREEN + "▄" + Style.RESET_ALL
                    )
                elif level > 0.02:
                    char = (
                        "▁"
                        if not COLORAMA_AVAILABLE
                        else Fore.YELLOW + "▁" + Style.RESET_ALL
                    )
                elif level > 0.005:
                    char = (
                        "."
                        if not COLORAMA_AVAILABLE
                        else Fore.CYAN + "." + Style.RESET_ALL
                    )
                else:
                    char = " "
            else:
                char = " "
            waveform.append(char)

        return "".join(waveform)

    def _get_vad_indicator(self):
        """Get Voice Activity Detection visual indicator."""
        # Create confidence bar
        confidence_bar = self._get_confidence_bar(self.vad_confidence)

        if not COLORAMA_AVAILABLE:
            if self.is_speech_detected:
                return f"🗣️  SPEECH DETECTED {confidence_bar} ({self.vad_method})"
            elif self.audio_level > 0.005:
                return f"🎤 Audio Activity {confidence_bar}"
            else:
                return f"🔇 Silence {confidence_bar}"

        # Colored indicators
        if self.is_speech_detected:
            return f"{Fore.GREEN}{Style.BRIGHT}🗣️  SPEECH DETECTED{Style.RESET_ALL} {confidence_bar} {Fore.CYAN}({self.vad_method}){Style.RESET_ALL}"
        elif self.audio_level > 0.005:
            return f"{Fore.YELLOW}🎤 Audio Activity{Style.RESET_ALL} {confidence_bar}"
        else:
            return f"{Fore.CYAN}🔇 Silence{Style.RESET_ALL} {confidence_bar}"

    def _get_confidence_bar(self, confidence, width=10):
        """Generate a confidence level bar."""
        filled = int(confidence * width)

        if not COLORAMA_AVAILABLE:
            bar = "█" * filled + "░" * (width - filled)
            return f"[{bar}] {confidence:.2f}"

        # Color based on confidence
        if confidence > 0.8:
            color = Fore.GREEN + Style.BRIGHT
        elif confidence > 0.6:
            color = Fore.GREEN
        elif confidence > 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        bar = (
            color + "█" * filled + Style.RESET_ALL + Fore.WHITE + "░" * (width - filled)
        )
        return f"[{bar}{Style.RESET_ALL}] {confidence:.2f}"

    def _get_transcription_indicator(self):
        """Get transcription status indicator."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        if self.is_transcribing:
            spinner = spinner_chars[self.animation_frame % len(spinner_chars)]
            if not COLORAMA_AVAILABLE:
                return f"{spinner} Transcribing..."
            return f"{Fore.MAGENTA}{spinner} Transcribing...{Style.RESET_ALL}"

        if self.last_transcription:
            if not COLORAMA_AVAILABLE:
                return (
                    f"✅ #{self.transcription_count}: {self.last_transcription[:60]}..."
                )
            return f"{Fore.GREEN}✅ #{self.transcription_count}: {Style.RESET_ALL}{self.last_transcription[:60]}..."

        return ""

    def _clear_lines(self, num_lines):
        """Clear the specified number of lines from console."""
        for _ in range(num_lines):
            sys.stdout.write("\033[F")  # Move cursor up one line
            sys.stdout.write("\033[K")  # Clear line

    def _animation_loop(self):
        """Main animation loop running in separate thread."""
        print("\n" * 5)  # Make space for animation

        while self.running:
            try:
                with self.lock:
                    current_level = self.audio_level

                # Clear previous animation (5 lines)
                self._clear_lines(5)

                # Line 1: Audio level meter
                meter = self._get_audio_meter(current_level)
                print(f"🎵 Audio Level: {meter}")

                # Line 2: Waveform visualization
                waveform = self._get_waveform()
                print(f"📊 Waveform:    [{waveform}]")

                # Line 3: VAD status
                vad_status = self._get_vad_indicator()
                print(f"🎯 VAD Status:  {vad_status}")

                # Line 4: Transcription status
                transcription_status = self._get_transcription_indicator()
                if transcription_status:
                    print(f"📝 Transcription: {transcription_status}")
                else:
                    print("📝 Transcription: Waiting for speech...")

                # Line 5: Separator
                separator = "─" * 80
                if COLORAMA_AVAILABLE:
                    separator = Fore.WHITE + "─" * 80 + Style.RESET_ALL
                print(separator)

                self.animation_frame += 1
                time.sleep(0.1)  # 10 FPS animation

            except (KeyboardInterrupt, RuntimeError, OSError) as e:
                print(f"Animation error: {e}")
                time.sleep(0.5)

    def display_static_message(self, message, message_type="info"):
        """Display a static message without animation."""
        if not COLORAMA_AVAILABLE:
            print(message)
            return

        colors = {
            "info": Fore.CYAN,
            "success": Fore.GREEN,
            "warning": Fore.YELLOW,
            "error": Fore.RED,
        }

        color = colors.get(message_type, Fore.WHITE)
        print(f"{color}{message}{Style.RESET_ALL}")


# Module-level animator instance
_animator = None


def get_animator():
    """Get the module animator instance."""
    # pylint: disable=global-statement
    global _animator
    if _animator is None:
        _animator = ConsoleAnimator()
    return _animator


def start_animation():
    """Start the global animation."""
    get_animator().start()


def stop_animation():
    """Stop the global animation."""
    if _animator:
        _animator.stop()
