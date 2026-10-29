"""Synthesize the effect sounds into assets/sounds (no third-party audio).

Each style gets <style>_out.wav for vanishing and <style>_in.wav, its reverse,
for reappearing. Naruto keeps assets/vanish.MP3.
"""

import wave
from pathlib import Path

import numpy as np

RATE = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds"
rng = np.random.default_rng(17)


def timeline(seconds: float) -> np.ndarray:
    return np.arange(int(RATE * seconds)) / RATE


def envelope(t: np.ndarray, attack: float, release: float) -> np.ndarray:
    total = t[-1]
    return np.clip(t / attack, 0, 1) * np.clip((total - t) / release, 0, 1)


def lowpass(signal: np.ndarray, cutoff_hz: np.ndarray) -> np.ndarray:
    """One-pole low-pass with a time-varying cutoff."""
    alpha = 1 - np.exp(-2 * np.pi * np.broadcast_to(cutoff_hz, signal.shape) / RATE)
    out = np.empty_like(signal)
    state = 0.0
    for i, (x, a) in enumerate(zip(signal, alpha)):
        state += a * (x - state)
        out[i] = state
    return out


def chirp(t: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    freq = start_hz * (end_hz / start_hz) ** (t / t[-1])
    return np.sin(2 * np.pi * np.cumsum(freq) / RATE)


def dissolve() -> np.ndarray:  # sparkling chime that rises and fades
    t = timeline(2.4)
    tones = sum(np.sin(2 * np.pi * f * t) * (0.6 ** k) for k, f in enumerate((880, 1320, 1760, 2640)))
    shimmer = 0.6 + 0.4 * np.sin(2 * np.pi * 9 * t)
    sparkle = lowpass(rng.normal(size=t.size), 6000) * (rng.random(t.size) > 0.995) * 6
    return (tones * shimmer * 0.25 + sparkle) * envelope(t, 0.05, 1.6)


def dust() -> np.ndarray:  # windy whoosh that sweeps up then fades
    t = timeline(2.4)
    noise = rng.normal(size=t.size)
    sweep = 300 + 2500 * np.sin(np.pi * t / t[-1]) ** 2
    return lowpass(noise, sweep) * envelope(t, 0.3, 1.2) * 1.8


def beam() -> np.ndarray:  # rising hum with a shimmering top
    t = timeline(1.8)
    hum = chirp(t, 180, 720) * 0.5 + chirp(t, 360, 1440) * 0.25
    trill = np.sin(2 * np.pi * 2400 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 30 * t)) * 0.12
    return (hum + trill) * envelope(t, 0.15, 0.5)


def glitch() -> np.ndarray:  # bit-crushed blips and buzzes
    t = timeline(1.3)
    square = np.sign(np.sin(2 * np.pi * 110 * t)) * 0.35
    blips = np.zeros_like(t)
    for start in rng.uniform(0, 1.2, 14):
        i, n = int(start * RATE), int(rng.uniform(0.02, 0.07) * RATE)
        blips[i:i + n] = np.sign(np.sin(2 * np.pi * rng.uniform(300, 1800) * t[:n]))[: len(blips[i:i + n])] * 0.5
    gate = (np.sin(2 * np.pi * 13 * t) > -0.3).astype(float)
    crushed = np.round((square * gate + blips) * 6) / 6
    return crushed * envelope(t, 0.01, 0.3) * 0.8


def portal() -> np.ndarray:  # deep swirl pulling inwards
    t = timeline(1.9)
    swirl = lowpass(rng.normal(size=t.size), 400 + 900 * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    sub = chirp(t, 140, 55) * 0.5
    return (swirl * 1.6 + sub) * envelope(t, 0.2, 0.6)


def write(path: Path, samples: np.ndarray) -> None:
    samples = samples / max(1e-6, np.abs(samples).max()) * 0.8
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(pcm.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("normal", dissolve), ("dust", dust), ("beam", beam),
                       ("glitch", glitch), ("portal", portal)):
        sound = make()
        write(OUT / f"{name}_out.wav", sound)
        write(OUT / f"{name}_in.wav", sound[::-1])
    print(f"Wrote sounds to {OUT}")


if __name__ == "__main__":
    main()
