"""Send your microphone plus the effect sounds into a call through BlackHole.

Select "BlackHole 2ch" as the microphone in Meet or Zoom. CallAudio reads the
real microphone, mixes in effect sounds when you vanish or reappear, and
writes the result to BlackHole with a few milliseconds of delay.
"""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np

RATE = 48000
BLOCK = 256                      # ~5 ms per callback
MAX_BUFFERED = int(RATE * 0.08)  # never let the voice lag more than 80 ms
KEEP_BUFFERED = int(RATE * 0.02)
VIRTUAL_MIC = "BlackHole"

_active: "CallAudio | None" = None


def active() -> "CallAudio | None":
    """The running mixer, if "Sound into call" is on."""
    return _active


def load_wav(path: Path, rate: int = RATE) -> np.ndarray:
    """Mono float32 samples at ``rate``."""
    with wave.open(str(path), "rb") as file:
        channels, width, source_rate = file.getnchannels(), file.getsampwidth(), file.getframerate()
        raw = file.readframes(file.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        samples = (samples - 128) / 128
    else:
        samples /= float(np.iinfo(dtype).max)
    samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != rate and len(samples) > 1:
        positions = np.arange(int(len(samples) * rate / source_rate)) * source_rate / rate
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


class VoiceBuffer:
    """FIFO between the microphone and output callbacks that keeps delay low."""

    def __init__(self):
        self.samples = np.zeros(0, np.float32)
        self.lock = threading.Lock()

    def push(self, block: np.ndarray) -> None:
        with self.lock:
            self.samples = np.concatenate([self.samples, block.astype(np.float32)])
            if len(self.samples) > MAX_BUFFERED:  # clock drift: drop the oldest audio
                self.samples = self.samples[-KEEP_BUFFERED:]

    def pull(self, count: int) -> np.ndarray:
        with self.lock:
            taken, self.samples = self.samples[:count], self.samples[count:]
        if len(taken) < count:  # underrun: pad with silence
            taken = np.concatenate([taken, np.zeros(count - len(taken), np.float32)])
        return taken


class Mixer:
    """Effect sounds currently playing, summed onto the voice."""

    def __init__(self, effect_gain: float = 0.8):
        self.voices: list[list] = []  # [samples, position]
        self.gain = effect_gain
        self.lock = threading.Lock()

    def add(self, samples: np.ndarray) -> None:
        with self.lock:
            self.voices.append([samples, 0])

    def mix(self, voice: np.ndarray) -> np.ndarray:
        out = voice.copy()
        count = len(out)
        with self.lock:
            for entry in self.voices:
                samples, position = entry
                part = samples[position:position + count]
                out[:len(part)] += part * self.gain
                entry[1] = position + count
            self.voices = [entry for entry in self.voices if entry[1] < len(entry[0])]
        return np.clip(out, -1.0, 1.0)


def find_device(sd, name_part: str, kind: str) -> int | None:
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    for index, device in enumerate(sd.query_devices()):
        if name_part.lower() in device["name"].lower() and device[key] > 0:
            return index
    return None


def real_microphone(sd) -> int:
    """The default input, unless that is BlackHole itself (which would loop)."""
    default_in = sd.default.device[0]
    devices = sd.query_devices()
    if default_in is not None and default_in >= 0 and VIRTUAL_MIC.lower() not in devices[default_in]["name"].lower():
        return default_in
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0 and VIRTUAL_MIC.lower() not in device["name"].lower():
            return index
    raise RuntimeError("No microphone found.")


class CallAudio:
    def __init__(self):
        import sounddevice as sd  # imported lazily: the engine works without audio

        self.sd = sd
        self.output = find_device(sd, VIRTUAL_MIC, "output")
        if self.output is None:
            raise RuntimeError("BlackHole 2ch is not installed. Run: brew install --cask blackhole-2ch")
        self.input = real_microphone(sd)
        self.buffer = VoiceBuffer()
        self.mixer = Mixer()
        self.cache: dict[Path, np.ndarray] = {}
        self.streams = []

    @property
    def microphone_name(self) -> str:
        return self.sd.query_devices(self.input)["name"]

    def start(self) -> None:
        global _active
        self.streams = [
            self.sd.InputStream(device=self.input, channels=1, samplerate=RATE, blocksize=BLOCK,
                                dtype="float32", latency="low", callback=self._capture),
            self.sd.OutputStream(device=self.output, channels=2, samplerate=RATE, blocksize=BLOCK,
                                 dtype="float32", latency="low", callback=self._send),
        ]
        for stream in self.streams:
            stream.start()
        _active = self

    def stop(self) -> None:
        global _active
        if _active is self:
            _active = None
        for stream in self.streams:
            stream.stop()
            stream.close()
        self.streams = []

    def play(self, path: Path) -> None:
        if path not in self.cache:
            self.cache[path] = load_wav(path)
        self.mixer.add(self.cache[path])

    def _capture(self, indata, frames, time_info, status) -> None:
        self.buffer.push(indata[:, 0])

    def _send(self, outdata, frames, time_info, status) -> None:
        mixed = self.mixer.mix(self.buffer.pull(frames))
        outdata[:] = mixed[:, None]
