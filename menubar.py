"""macOS menu-bar controller: runs vanish_cam.py headless into the virtual camera.

Pick "VanishCam" mode here, then select "OBS Virtual Camera" once in Google
Meet, WhatsApp, FaceTime or Zoom. While no effect is running the virtual camera
shows your normal webcam, so it can stay selected in every app.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import rumps

APP_DIR = Path(__file__).resolve().parent
CAM_SCRIPT = APP_DIR / "vanish_cam.py"
MODES = (("normal", "Normal Vanish (V sign)"), ("naruto", "Naruto Vanish (palms together)"))
# vanish_cam.py reports progress in Italian; show the important steps in English.
STATUS_TRANSLATIONS = (
    ("Esci dall'inquadratura", "Step out of frame – capturing background"),
    ("Acquisizione sfondo:", "Capturing background: {rest}"),
    ("Sfondo pronto", "Ready – step back in and use the gesture"),
    ("Uscita virtuale attiva", "Live on virtual camera – ready for background"),
    ("Scarico ", "Downloading model {rest}"),
    ("ERRORE:", "Error: {rest}"),
)
ICON_IDLE = "◌"
ICON_LIVE = "◉"


def translate_status(text: str) -> str:
    for prefix, english in STATUS_TRANSLATIONS:
        if text.startswith(prefix):
            return english.format(rest=text[len(prefix):].strip().replace("secondi.", "s left"))
    return text


class VanishMenuBar(rumps.App):
    def __init__(self) -> None:
        super().__init__("VanishCam", title=ICON_IDLE, quit_button=None)
        self.process: subprocess.Popen[str] | None = None
        self.mode: str | None = None
        self.status_text = "Stopped"
        self.lock = threading.Lock()

        self.status = rumps.MenuItem("Stopped")
        self.mode_items = {
            mode: rumps.MenuItem(label, callback=lambda _, m=mode: self.start(m))
            for mode, label in MODES
        }
        self.camera_items = {
            index: rumps.MenuItem(f"Camera {index}", callback=lambda _, i=index: self.set_camera(i))
            for index in range(4)
        }
        self.camera = 0
        self.camera_items[0].state = True
        camera_menu = rumps.MenuItem("Webcam")
        for item in self.camera_items.values():
            camera_menu.add(item)

        self.menu = [
            self.status,
            None,
            *self.mode_items.values(),
            rumps.MenuItem("Stop", callback=lambda _: self.stop()),
            None,
            rumps.MenuItem("Vanish now", callback=lambda _: self.send("vanish")),
            rumps.MenuItem("Reappear", callback=lambda _: self.send("reappear")),
            rumps.MenuItem("Re-capture background", callback=lambda _: self.send("background")),
            rumps.MenuItem("Show/hide preview", callback=lambda _: self.send("preview")),
            None,
            camera_menu,
            None,
            rumps.MenuItem("Quit VanishCam", callback=lambda _: self.quit()),
        ]

    def start(self, mode: str) -> None:
        self.stop()
        command = [
            sys.executable, "-u", str(CAM_SCRIPT), "--mode", mode, "--camera", str(self.camera),
            "--virtual-camera", "--no-preview", "--control-stdin",
        ]
        self.process = subprocess.Popen(
            command, cwd=APP_DIR, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.mode = mode
        self.set_status("Starting…")
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, process: subprocess.Popen[str]) -> None:
        for line in process.stdout:
            line = line.strip()
            if not line or line.startswith("objc["):
                continue
            print(line, flush=True)
            if process is self.process:
                self.set_status(translate_status(line.split("] ", 1)[-1]))
        code = process.wait()
        if process is self.process:
            self.process = None
            self.mode = None
            if code != 0 and not self.status_text.startswith("Error:"):
                self.set_status(f"Stopped (error {code}) – see Terminal")

    def send(self, command: str) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            self.set_status("Not running – pick a mode first")
            return
        try:
            process.stdin.write(command + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.set_status("VanishCam is not responding")

    def stop(self) -> None:
        process, self.process = self.process, None
        self.mode = None
        if process is not None and process.poll() is None:
            self.send_quit(process)
        self.set_status("Stopped")

    @staticmethod
    def send_quit(process: subprocess.Popen[str]) -> None:
        try:
            process.stdin.write("quit\n")
            process.stdin.close()
            process.wait(timeout=5)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            process.kill()

    def set_camera(self, index: int) -> None:
        self.camera = index
        for item_index, item in self.camera_items.items():
            item.state = item_index == index
        if self.mode is not None:
            self.start(self.mode)

    def set_status(self, text: str) -> None:
        with self.lock:
            self.status_text = text[:70]

    @rumps.timer(0.5)
    def refresh(self, _) -> None:
        with self.lock:
            self.status.title = self.status_text
        self.title = ICON_LIVE if self.mode else ICON_IDLE
        for mode, item in self.mode_items.items():
            item.state = mode == self.mode

    def quit(self) -> None:
        self.stop()
        rumps.quit_application()


if __name__ == "__main__":
    VanishMenuBar().run()
