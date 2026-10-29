"""Play effect sounds without blocking video frames (macOS afplay)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import call_audio

SOUND_DIR = Path(__file__).resolve().parent / "assets" / "sounds"


def sound_paths(style: str) -> dict[str, Path]:
    """Files for "out" (vanish) and "in" (reappear)."""
    return {direction: SOUND_DIR / f"{style}_{direction}.wav" for direction in ("out", "in")}


class EffectSound:
    def __init__(self, style: str):
        self.paths = sound_paths(style)
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise RuntimeError(f"Sound not found: {', '.join(missing)}")
        self.player = shutil.which("afplay") or "/usr/bin/afplay"
        self.process: subprocess.Popen | None = None

    def play(self, direction: str = "out") -> None:
        mixer = call_audio.active()
        if mixer is not None:  # "Sound into call" is on
            mixer.play(self.paths[direction])
        self.stop()
        try:
            self.process = subprocess.Popen(
                [self.player, str(self.paths[direction])], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise RuntimeError(f"Could not play the effect sound: {error}") from error

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        self.stop()
