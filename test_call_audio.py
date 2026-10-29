import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import call_audio
import vanish_sound
from call_audio import MAX_BUFFERED, Mixer, VoiceBuffer, load_wav, real_microphone


class VoiceBufferTests(unittest.TestCase):
    def test_passes_audio_through_in_order(self):
        buffer = VoiceBuffer()
        buffer.push(np.array([0.1, 0.2, 0.3], np.float32))
        np.testing.assert_allclose(buffer.pull(2), [0.1, 0.2])
        np.testing.assert_allclose(buffer.pull(1), [0.3])

    def test_underrun_is_silence(self):
        buffer = VoiceBuffer()
        buffer.push(np.array([0.5], np.float32))
        np.testing.assert_allclose(buffer.pull(3), [0.5, 0.0, 0.0])

    def test_delay_never_grows_past_the_cap(self):
        buffer = VoiceBuffer()
        for _ in range(100):
            buffer.push(np.ones(256, np.float32))
        self.assertLessEqual(len(buffer.samples), MAX_BUFFERED)


class MixerTests(unittest.TestCase):
    def test_effect_is_added_to_the_voice_then_removed_when_done(self):
        mixer = Mixer(effect_gain=1.0)
        mixer.add(np.full(5, 0.25, np.float32))
        first = mixer.mix(np.full(3, 0.1, np.float32))
        np.testing.assert_allclose(first, [0.35, 0.35, 0.35])
        second = mixer.mix(np.zeros(3, np.float32))
        np.testing.assert_allclose(second, [0.25, 0.25, 0.0])
        self.assertEqual(mixer.voices, [])

    def test_output_is_clipped(self):
        mixer = Mixer(effect_gain=1.0)
        mixer.add(np.full(4, 0.9, np.float32))
        self.assertLessEqual(mixer.mix(np.full(4, 0.9, np.float32)).max(), 1.0)


class LoadWavTests(unittest.TestCase):
    def test_resamples_to_48k_mono(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tone.wav"
            stereo = (np.ones((44100, 2)) * 16000).astype(np.int16)
            with wave.open(str(path), "wb") as file:
                file.setnchannels(2), file.setsampwidth(2), file.setframerate(44100)
                file.writeframes(stereo.tobytes())
            samples = load_wav(path)
        self.assertEqual(len(samples), 48000)
        self.assertAlmostEqual(float(samples.mean()), 16000 / 32767, places=3)


class DeviceTests(unittest.TestCase):
    def fake_sd(self, default_input):
        devices = [
            {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0},
        ]
        return SimpleNamespace(query_devices=lambda: devices,
                               default=SimpleNamespace(device=[default_input, 2]))

    def test_uses_the_default_microphone(self):
        self.assertEqual(real_microphone(self.fake_sd(1)), 1)

    def test_never_uses_blackhole_as_the_real_microphone(self):
        self.assertEqual(real_microphone(self.fake_sd(0)), 1)  # would feed back into itself

    def test_all_effect_sounds_load(self):
        for mode in ("normal", "naruto", "dust", "beam", "glitch", "portal"):
            for path in vanish_sound.sound_paths(mode).values():
                self.assertGreater(len(load_wav(path)), 48000 * 0.5, path)


class EffectSoundRoutingTests(unittest.TestCase):
    def test_effect_sound_goes_into_the_call_only_when_enabled(self):
        sound = vanish_sound.EffectSound("dust")
        mixer = Mock()
        with patch.object(vanish_sound.subprocess, "Popen"):
            with patch.object(call_audio, "_active", None):
                sound.play("out")
                mixer.play.assert_not_called()
            with patch.object(call_audio, "_active", mixer):
                sound.play("in")
        mixer.play.assert_called_once_with(vanish_sound.sound_paths("dust")["in"])


if __name__ == "__main__":
    unittest.main()
