"""Console animation utilities for real-time audio visualization and VAD feedback."""

import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import copy

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


@dataclass
class AnimationState:
    """Thread-safe container for animation state."""

    audio_level: float = 0.0
    is_speech_detected: bool = False
    is_transcribing: bool = False
    last_transcription: str = ""
    transcription_count: int = 0
    vad_confidence: float = 0.0
    vad_method: str = "Unknown"
    audio_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def snapshot(self) -> "AnimationState":
        """Create a thread-safe snapshot of current state."""
        return AnimationState(
            audio_level=self.audio_level,
            is_speech_detected=self.is_speech_detected,
            is_transcribing=self.is_transcribing,
            last_transcription=self.last_transcription,
            transcription_count=self.transcription_count,
            vad_confidence=self.vad_confidence,
            vad_method=self.vad_method,
            audio_history=copy.deepcopy(self.audio_history),
        )


class ConsoleAnimator:
    """Handles real-time console animations for audio visualization."""

    def __init__(self):
        self.running = False
        self.animation_thread: Optional[threading.Thread] = None

        # Use RLock for nested lock acquisition and better thread safety
        self._state_lock = threading.RLock()
        self._state = AnimationState()

        # Animation control
        self.animation_frame = 0
        self._last_update_time = time.time()
        self._update_interval = 0.1  # 10 FPS

        # Performance tracking
        self._frame_count = 0
        self._start_time = time.time()

    def start(self):
        """Start the animation thread."""
        with self._state_lock:
            if not self.running:
                self.running = True
                self.animation_thread = threading.Thread(
                    target=self._animation_loop, daemon=True
                )
                self.animation_thread.start()

    def stop(self):
        """Stop the animation thread."""
        with self._state_lock:
            self.running = False

        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
            self.animation_thread = None

    def update_audio_level(self, level: float):
        """Update the current audio level for visualization."""
        with self._state_lock:
            self._state.audio_level = max(0.0, min(1.0, level))  # Clamp to [0, 1]
            self._state.audio_history.append(self._state.audio_level)

    def set_speech_detected(self, detected: bool):
        """Update speech detection status."""
        with self._state_lock:
            self._state.is_speech_detected = detected

    def set_transcribing(self, transcribing: bool):
        """Update transcription status."""
        with self._state_lock:
            self._state.is_transcribing = transcribing

    def set_transcription_result(self, text: str, count: int):
        """Update latest transcription result."""
        with self._state_lock:
            self._state.last_transcription = str(text)
            self._state.transcription_count = int(count)

    def set_vad_confidence(self, confidence: float):
        """Update VAD confidence level."""
        with self._state_lock:
            self._state.vad_confidence = max(
                0.0, min(1.0, confidence)
            )  # Clamp to [0, 1]

    def set_vad_method(self, method: str):
        """Update VAD method name."""
        with self._state_lock:
            self._state.vad_method = str(method)

    def _capture_state_snapshot(self) -> AnimationState:
        """Capture an atomic snapshot of the current state."""
        with self._state_lock:
            return self._state.snapshot()

    def _get_audio_meter(self, level: float, width: int = 20) -> str:
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

    def _get_waveform(self, audio_history: deque, width: int = 40) -> str:
        """Generate a simple waveform visualization."""
        if len(audio_history) < 2:
            return "─" * width

        waveform = []
        history_list = list(audio_history)  # Convert to list for indexing

        for i in range(width):
            # Sample from audio history with proper bounds checking
            if i < len(history_list):
                level = history_list[i]
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

    def _get_vad_indicator(self, state: AnimationState) -> str:
        """Get Voice Activity Detection visual indicator."""
        # Create confidence bar
        confidence_bar = self._get_confidence_bar(state.vad_confidence)

        if not COLORAMA_AVAILABLE:
            if state.is_speech_detected:
                return f"🗣️  SPEECH DETECTED {confidence_bar} ({state.vad_method})"
            elif state.audio_level > 0.005:
                return f"🎤 Audio Activity {confidence_bar}"
            else:
                return f"🔇 Silence {confidence_bar}"

        # Colored indicators
        if state.is_speech_detected:
            speech_text = (
                f"{Fore.GREEN}{Style.BRIGHT}🗣️  SPEECH DETECTED{Style.RESET_ALL}"
            )
            method_text = f"{Fore.CYAN}({state.vad_method}){Style.RESET_ALL}"
            return f"{speech_text} {confidence_bar} {method_text}"
        elif state.audio_level > 0.005:
            return f"{Fore.YELLOW}🎤 Audio Activity{Style.RESET_ALL} {confidence_bar}"
        else:
            return f"{Fore.CYAN}🔇 Silence{Style.RESET_ALL} {confidence_bar}"

    def _get_confidence_bar(self, confidence: float, width: int = 10) -> str:
        """Generate a confidence level bar."""
        filled = int(confidence * width)

        if not COLORAMA_AVAILABLE:
            meter = "█" * filled + "░" * (width - filled)
            return f"[{meter}] {confidence:.2f}"

        # Color based on confidence
        if confidence > 0.8:
            color = Fore.GREEN + Style.BRIGHT
        elif confidence > 0.6:
            color = Fore.GREEN
        elif confidence > 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        meter = (
            color + "█" * filled + Style.RESET_ALL + Fore.WHITE + "░" * (width - filled)
        )
        return f"[{meter}{Style.RESET_ALL}] {confidence:.2f}"

    def _get_transcription_indicator(self, state: AnimationState) -> str:
        """Get transcription status indicator."""
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        if state.is_transcribing:
            spinner = spinner_chars[self.animation_frame % len(spinner_chars)]
            if not COLORAMA_AVAILABLE:
                return f"{spinner} Transcribing..."
            return f"{Fore.MAGENTA}{spinner} Transcribing...{Style.RESET_ALL}"

        if state.last_transcription:
            if not COLORAMA_AVAILABLE:
                transcription_text = f"✅ #{state.transcription_count}: "
                return transcription_text + f"{state.last_transcription[:60]}..."
            success_text = (
                f"{Fore.GREEN}✅ #{state.transcription_count}: {Style.RESET_ALL}"
            )
            return success_text + f"{state.last_transcription[:60]}..."

        return ""

    def _clear_lines(self, num_lines: int):
        """Clear the specified number of lines from console."""
        for _ in range(num_lines):
            try:
                sys.stdout.write("\033[F")  # Move cursor up one line
                sys.stdout.write("\033[K")  # Clear line
            except (OSError, UnicodeEncodeError):
                # Fallback for systems that don't support ANSI codes
                print("\r" + " " * 80 + "\r", end="")

    def _animation_loop(self):
        """Main animation loop running in separate thread."""
        print("\n" * 5)  # Make space for animation

        while self.running:
            try:
                current_time = time.time()

                # Rate limiting: only update at specified interval
                if current_time - self._last_update_time < self._update_interval:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    continue

                self._last_update_time = current_time

                # Capture atomic state snapshot
                state = self._capture_state_snapshot()

                # Clear previous animation (5 lines)
                self._clear_lines(5)

                # Line 1: Audio level meter
                meter = self._get_audio_meter(state.audio_level)
                print(f"🎵 Audio Level: {meter}")

                # Line 2: Waveform visualization
                waveform = self._get_waveform(state.audio_history)
                print(f"📊 Waveform:    [{waveform}]")

                # Line 3: VAD status
                vad_status = self._get_vad_indicator(state)
                print(f"🎯 VAD Status:  {vad_status}")

                # Line 4: Transcription status
                transcription_status = self._get_transcription_indicator(state)
                if transcription_status:
                    print(f"📝 Transcription: {transcription_status}")
                else:
                    print("📝 Transcription: Waiting for speech...")

                # Line 5: Separator with performance info
                self._frame_count += 1
                fps = (
                    self._frame_count / (current_time - self._start_time)
                    if current_time > self._start_time
                    else 0
                )
                separator = "─" * 60 + f" FPS: {fps:.1f} "
                if COLORAMA_AVAILABLE:
                    separator = Fore.WHITE + separator + Style.RESET_ALL
                print(separator)

                self.animation_frame += 1

                # Periodic performance reset to prevent overflow
                if self._frame_count > 1000:
                    self._frame_count = 0
                    self._start_time = current_time

            except (KeyboardInterrupt, RuntimeError, OSError) as e:
                print(f"Animation error: {e}")
                time.sleep(0.5)
                break
            except Exception as e:  # pylint: disable=broad-except
                # Log unexpected errors but don't crash the animation thread
                print(f"Unexpected animation error: {e}")
                time.sleep(0.1)

    def display_static_message(self, message: str, message_type: str = "info"):
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

    def get_performance_stats(self) -> dict:
        """Get animation performance statistics."""
        current_time = time.time()
        runtime = (
            current_time - self._start_time if current_time > self._start_time else 0
        )
        fps = self._frame_count / runtime if runtime > 0 else 0

        return {
            "frame_count": self._frame_count,
            "runtime_seconds": runtime,
            "average_fps": fps,
            "is_running": self.running,
            "thread_alive": (
                self.animation_thread.is_alive() if self.animation_thread else False
            ),
        }


# Module-level animator instance with thread-safe access
_ANIMATOR_LOCK = threading.Lock()
_ANIMATOR: Optional[ConsoleAnimator] = None


def get_animator() -> ConsoleAnimator:
    """Get the module animator instance (thread-safe)."""
    global _ANIMATOR  # pylint: disable=global-statement

    with _ANIMATOR_LOCK:
        if _ANIMATOR is None:
            _ANIMATOR = ConsoleAnimator()
        return _ANIMATOR


def start_animation():
    """Start the global animation."""
    get_animator().start()


def stop_animation():
    """Stop the global animation."""
    with _ANIMATOR_LOCK:
        if _ANIMATOR:
            _ANIMATOR.stop()
