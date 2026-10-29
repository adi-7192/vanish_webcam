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
        gesture_debug=False, button_effect="normal", no_preview=False, control_stdin=False,
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


CALIBRATION_FRAMES = int(app.CALIBRATION_SECONDS * 30) + 2


class VideoLoopTests(unittest.TestCase):
    def run_loop(self, frames, mask, timeline=(), remote=None, headless=False,
                 send_error=None, cameras=1):
        """Run one session. ``timeline`` lists (start, end, pose) in seconds of
        fake webcam time; ``remote`` lists commands, one per frame."""
        clock = [0.0]
        captures = [FakeCamera(frames, clock)]
        for _ in range(cameras - 1):  # extra webcams continue the same footage
            extra = FakeCamera([], clock)
            extra.frames = captures[0].frames
            captures.append(extra)
        opened = []

        def open_camera(index, width, height):
            opened.append(index)
            return captures[len(opened) - 1]

        def recognize_pose(image, timestamp):
            now = clock[0]
            return next((pose for start, end, pose in timeline if start <= now < end), None)

        segmenter, recognizer = Mock(), Mock()
        recognizer.last_hand_count = 0
        recognizer.recognize_pose.side_effect = recognize_pose
        camera = Mock(width=frames[0].shape[1], height=frames[0].shape[0])
        sent, messages = [], []
        camera.send.side_effect = send_error or (lambda frame: sent.append(frame.copy()))
        remote = list(remote or [])
        args = arguments(no_preview=headless, control_stdin=headless)
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=args))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", side_effect=open_camera))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            mocks.enter_context(patch.object(app, "EffectSound", return_value=Mock()))
            mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            mocks.enter_context(patch.object(app, "say", side_effect=messages.append))
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
        self.assertEqual(captures[len(opened) - 1].release.call_count, 1)
        segmenter.close.assert_called_once()
        recognizer.close.assert_called_once()
        camera.close.assert_called_once()
        opening.assert_called_once()  # the virtual camera stays open all session
        return sent, messages, opened

    @staticmethod
    def captures_of_empty_room(messages):
        return [m for m in messages if m.startswith(("Step out", "Switched webcam"))]

    def test_one_session_different_gestures_one_background(self):
        frames, room, mask = person_scene(720)
        start = CALIBRATION_FRAMES / 30
        timeline = [
            (start + 1.0, start + 2.0, "Closed_Fist"),   # ✊ Dust
            (start + 5.0, start + 6.0, "Open_Palm"),     # ✋ back
            (start + 9.0, start + 10.0, "Thumb_Up"),     # 👍 ignored
            (start + 11.0, start + 12.0, "Pointing_Up"), # ☝️ Portal
            (start + 15.0, start + 16.0, "Open_Palm"),   # ✋ back
        ]
        sent, messages, opened = self.run_loop(frames, mask, timeline)
        self.assertEqual(len(self.captures_of_empty_room(messages)), 1)
        vanished = [m for m in messages if m.startswith("Vanished with")]
        self.assertEqual([m.split(".")[0] for m in vanished], ["Vanished with Dust", "Vanished with Portal"])
        self.assertEqual(sum(m.startswith("Back!") for m in messages), 2)
        self.assertTrue(any((frame[12, 20] == room[12, 20]).all() for frame in sent))  # person hidden
        np.testing.assert_array_equal(sent[-1], frames[-1])  # visible at the end
        self.assertEqual(opened, [0])

    def test_open_palm_does_nothing_while_visible_and_vanish_poses_do_nothing_while_gone(self):
        frames, room, mask = person_scene(300)
        start = CALIBRATION_FRAMES / 30
        timeline = [(start + 0.5, start + 1.5, "Open_Palm"), (start + 2.0, start + 3.0, "Victory"),
                    (start + 6.0, start + 7.0, "ILoveYou")]
        _, messages, _ = self.run_loop(frames, mask, timeline)
        self.assertEqual([m for m in messages if m.startswith(("Vanished", "Back"))][:1],
                         ["Vanished with Dissolve. Show ✋ (or press Reappear) to come back."])
        self.assertFalse(any(m.startswith("Back") for m in messages))

    def test_quick_gesture_does_not_trigger(self):
        frames, room, mask = person_scene(60)
        start = CALIBRATION_FRAMES / 30
        sent, messages, _ = self.run_loop(frames, mask, [(start + 0.5, start + 0.75, "Thumb_Down")])
        self.assertFalse(any(m.startswith("Vanished") for m in messages))
        for frame, original in zip(sent, frames):
            np.testing.assert_array_equal(frame, original)

    def test_switching_webcam_keeps_the_session_and_virtual_camera(self):
        frames, room, mask = person_scene(200)
        remote = [None] * (CALIBRATION_FRAMES + 10) + ["camera:1"]
        _, messages, opened = self.run_loop(frames, mask, remote=remote, headless=True, cameras=2)
        self.assertEqual(opened, [0, 1])
        self.assertEqual(len(self.captures_of_empty_room(messages)), 2)  # new view, new background
        self.assertTrue(any(m.startswith("Switched webcam") for m in messages))

    def test_paused_gestures_are_ignored_until_resumed(self):
        frames, room, mask = person_scene(200)
        start = CALIBRATION_FRAMES / 30
        remote = [None] * (CALIBRATION_FRAMES + 5) + ["gestures:off"]
        _, messages, _ = self.run_loop(frames, mask, [(start + 1.0, start + 3.0, "Closed_Fist")],
                                       remote=remote, headless=True)
        self.assertIn("Gestures paused. Use the buttons to vanish and reappear.", messages)
        self.assertFalse(any(m.startswith("Vanished") for m in messages))

    def test_headless_buttons_pick_any_effect(self):
        frames, room, mask = person_scene(200)
        remote = [None] * (CALIBRATION_FRAMES + 5) + ["vanish:glitch"] + [None] * 60 + ["reappear"]
        sent, messages, _ = self.run_loop(frames, mask, remote=remote, headless=True)
        self.assertTrue(any(m.startswith("Vanished with Glitch") for m in messages))
        self.assertTrue(any(m.startswith("Back!") for m in messages))
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
            mocks.enter_context(patch.object(app, "EffectSound", return_value=Mock()))
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
