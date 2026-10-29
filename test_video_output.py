import argparse
import io
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None,
        gesture_debug=False, mode="normal", no_preview=False, control_stdin=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=1920, height=1080, fps=30.0, fmt="BGR", backend="obs",
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True),):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: non scambiare rosso e blu.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


class FakeCamera:
    """Webcam whose frames also advance a fake clock (30 fps)."""

    def __init__(self, frames, clock):
        self.frames, self.clock = list(frames), clock
        self.release = Mock()

    def read(self):
        self.clock[0] += 1 / 30
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def set(self, *args):
        pass


def person_scene(count, height=24, width=40):
    room = np.full((height, width, 3), 100, np.uint8)
    person = room.copy()
    person[6:20, 14:26] = 220
    mask = np.zeros((height, width), np.float32)
    mask[6:20, 14:26] = 1.0
    calibration = int(app.CALIBRATION_SECONDS * 30) + 2
    return [room] * calibration + [person] * count, room, mask


class FitFrameTests(unittest.TestCase):
    def test_16x9_is_scaled_exactly(self):
        out = app.fit_frame(np.full((540, 960, 3), 50, np.uint8), 1920, 1080)
        self.assertEqual(out.shape, (1080, 1920, 3))
        self.assertTrue((out == 50).all())

    def test_4x3_is_letterboxed_not_stretched(self):
        out = app.fit_frame(np.full((480, 640, 3), 200, np.uint8), 1920, 1080)
        self.assertEqual(out.shape, (1080, 1920, 3))
        self.assertTrue((out[:, :200] == 0).all())  # black side bars
        self.assertTrue((out[:, 960] == 200).all())


class VideoLoopTests(unittest.TestCase):
    def run_loop(self, frames, mask, mode="normal", gestures=None, palms=None,
                 remote=None, headless=False, send_error=None):
        clock = [0.0]
        capture = FakeCamera(frames, clock)
        segmenter = Mock()
        recognizer = Mock()
        recognizer.last_hand_count = 0
        gestures = list(gestures or [])
        palms = list(palms or [])
        recognizer.detect_open_palm.side_effect = lambda image, ts: palms.pop(0) if palms else False
        recognizer.recognize_joined_hands.side_effect = lambda image, ts: gestures.pop(0) if gestures else False
        victory = lambda *args: ("Victory", 0.9) if (gestures.pop(0) if gestures else False) else ("No_Hand", 0.0)
        camera = Mock(width=frames[0].shape[1], height=frames[0].shape[0])
        camera.send.side_effect = send_error
        remote = list(remote or [])
        sent = []
        camera.send.side_effect = send_error or (lambda frame: sent.append(frame.copy()))
        args = arguments(mode=mode, no_preview=headless, control_stdin=headless)
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app, "EffectSound", return_value=Mock()))
            mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(app, "recognize_gesture", side_effect=victory))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            for name in ("namedWindow", "resizeWindow", "destroyAllWindows", "imshow"):
                mocks.enter_context(patch.object(app.cv2, name))
            mocks.enter_context(patch.object(app.cv2, "waitKey", return_value=0))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("No overlays")))
            mocks.enter_context(patch.object(app, "start_stdin_commands", return_value=Mock()))
            mocks.enter_context(patch.object(
                app, "next_command", side_effect=lambda commands: remote.pop(0) if remote else None
            ))
            mocks.enter_context(patch("builtins.print"))
            if send_error is not None:
                with self.assertRaises(type(send_error)):
                    app.run()
            else:
                with self.assertRaisesRegex(RuntimeError, "stopped sending"):
                    app.run()  # the fake webcam runs out of frames
        capture.release.assert_called_once()
        segmenter.close.assert_called_once()
        recognizer.close.assert_called_once()
        camera.close.assert_called_once()
        return sent, recognizer

    def test_gesture_held_vanishes_and_open_palm_brings_you_back(self):
        frames, room, mask = person_scene(260)
        start = int(app.CALIBRATION_SECONDS * 30) + 2
        # V sign on ~5 checks (0.7 s), then open palm on ~5 checks after vanishing.
        gestures = [True] * 5
        palms = [False] * 15 + [True] * 6
        sent, recognizer = self.run_loop(frames, mask, gestures=gestures, palms=palms)
        self.assertEqual(len(sent), len(frames))
        np.testing.assert_array_equal(sent[start], frames[start])  # visible at first
        vanished = [i for i, frame in enumerate(sent) if (frame[12, 20] == room[12, 20]).all()]
        self.assertTrue(vanished, "person never vanished")
        first_gone = vanished[0]
        np.testing.assert_array_equal(sent[first_gone][:, :5], room[:, :5])  # room stays live
        self.assertTrue(recognizer.detect_open_palm.called)
        np.testing.assert_array_equal(sent[-1], frames[-1])  # back again at the end
        self.assertGreater(len(sent) - 1, vanished[-1])

    def test_single_frame_gesture_does_not_trigger(self):
        frames, room, mask = person_scene(60)
        sent, _ = self.run_loop(frames, mask, gestures=[True, False, False, False, True])
        for frame, original in zip(sent, frames):
            np.testing.assert_array_equal(frame, original)

    def test_naruto_palms_together_then_open_palm(self):
        frames, room, mask = person_scene(200, height=48, width=80)
        mask = np.zeros((48, 80), np.float32)
        mask[10:40, 30:50] = 1.0
        for frame in frames[int(app.CALIBRATION_SECONDS * 30) + 2:]:
            frame[10:40, 30:50] = 220
        sent, recognizer = self.run_loop(frames, mask, mode="naruto",
                                         gestures=[True] * 8, palms=[False] * 25 + [True] * 8)
        self.assertTrue(any((frame[25, 40] == 100).all() for frame in sent))
        np.testing.assert_array_equal(sent[-1], frames[-1])

    def test_headless_run_is_driven_by_remote_commands(self):
        frames, room, mask = person_scene(170)
        start = int(app.CALIBRATION_SECONDS * 30) + 2
        remote = [None] * (start + 5) + ["vanish"] + [None] * 90 + ["reappear"]
        sent, _ = self.run_loop(frames, mask, remote=remote, headless=True)
        self.assertTrue(any((frame[12, 20] == room[12, 20]).all() for frame in sent))
        np.testing.assert_array_equal(sent[-1], frames[-1])

    def test_all_devices_close_on_send_error(self):
        frames, _, mask = person_scene(5)
        self.run_loop(frames, mask, send_error=RuntimeError("send failed"))

    def test_headless_without_virtual_camera_stops(self):
        capture = Mock()
        capture.read.return_value = (True, np.zeros((24, 40, 3), np.uint8))
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(no_preview=True)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(Mock(), Mock())))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=None))
            mocks.enter_context(patch("builtins.print"))
            with self.assertRaisesRegex(RuntimeError, "OBS Virtual Camera is required"):
                app.run()
        capture.release.assert_called_once()


class StdinCommandTests(unittest.TestCase):
    def test_known_commands_are_queued_and_eof_quits(self):
        with patch.object(app.sys, "stdin", io.StringIO("Vanish\nbogus\n\npreview\n")), \
                patch("builtins.print") as printed:
            commands = app.start_stdin_commands()
            received = [commands.get(timeout=2) for _ in range(3)]
        self.assertEqual(received, ["vanish", "preview", "quit"])
        printed.assert_called_once()

    def test_next_command_without_queue_or_items(self):
        self.assertIsNone(app.next_command(None))
        self.assertIsNone(app.next_command(app.queue.Queue()))


if __name__ == "__main__":
    unittest.main()
