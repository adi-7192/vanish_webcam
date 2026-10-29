"""Play the Naruto vanish sound without blocking video frames (macOS afplay)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


SOUND_PATH = Path(__file__).resolve().parent / "assets" / "vanish.MP3"


class VanishSound:
    def __init__(self, path: Path = SOUND_PATH):
        if not path.is_file():
            raise RuntimeError(f"Naruto sound not found: {path}")
        player = shutil.which("afplay") or "/usr/bin/afplay"
        self.command = [player, str(path)]
        self.process: subprocess.Popen | None = None

    def play(self) -> None:
        self.stop()
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise RuntimeError(f"Could not play the Naruto sound: {error}") from error

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        self.stop()
