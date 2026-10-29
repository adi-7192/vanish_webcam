import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import vanish_cam as app
from effects import joined_hands_from_landmarks, open_palm_from_landmarks


def pressed_hand(center_x, upward=True):
    """Palms-together pose: fingers close together, pointing up."""
    points = [SimpleNamespace(x=center_x, y=0.53) for _ in range(21)]
    points[0] = SimpleNamespace(x=center_x, y=0.70)
    for index, offset in ((5, -0.045), (9, -0.015), (13, 0.015), (17, 0.045)):
        points[index] = SimpleNamespace(x=center_x + offset, y=0.54 if upward else 0.84)
    for index, offset in ((8, -0.045), (12, -0.015), (16, 0.015), (20, 0.045)):
        points[index] = SimpleNamespace(x=center_x + offset, y=0.34 if upward else 1.04)
    return points


def open_hand(center_x=0.5, spread=0.06, curled=()):
    """Open palm facing the camera; ``curled`` fingers (0-3) are folded down."""
    points = [SimpleNamespace(x=center_x, y=0.80) for _ in range(21)]
    for finger, base in enumerate((5, 9, 13, 17)):
        mcp_x = center_x + (finger - 1.5) * 0.03
        tip_x = center_x + (finger - 1.5) * spread
        tip_y = 0.62 if finger in curled else 0.40
        for joint in range(4):
            t = joint / 3
            y = 0.60 + (tip_y - 0.60) * t if finger not in curled else 0.60 - 0.04 * np.sin(np.pi * t)
            points[base + joint] = SimpleNamespace(x=mcp_x + (tip_x - mcp_x) * t, y=y)
    return points


def image(width=640, height=480):
    return app.mp.Image(image_format=app.mp.ImageFormat.SRGB,
                        data=np.zeros((height, width, 3), np.uint8))


class JoinedHandsTests(unittest.TestCase):
    def test_two_nearby_hands_recognized_but_one_or_separated_hands_are_not(self):
        left, right = pressed_hand(0.45), pressed_hand(0.55)
        self.assertTrue(joined_hands_from_landmarks([left, right], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, pressed_hand(0.83)], 640, 480))
        self.assertFalse(joined_hands_from_landmarks([left, pressed_hand(0.55, upward=False)], 640, 480))

    def test_rotated_joined_hands_still_recognized(self):
        hands = [pressed_hand(0.45), pressed_hand(0.55)]
        rotated = [[SimpleNamespace(x=point.y, y=1 - point.x) for point in hand]
                   for hand in hands]
        self.assertTrue(joined_hands_from_landmarks(rotated, 480, 640))

    def test_recognizer_uses_two_hands_in_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[pressed_hand(0.45), pressed_hand(0.55)])
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        self.assertTrue(recognizer.recognize_joined_hands(image(), 100))
        self.assertEqual(recognizer.last_hand_count, 2)
        alternate.detect.assert_not_called()

    def test_hand_model_recovers_joined_hands_in_a_cropped_view(self):
        primary, alternate = Mock(), Mock()
        primary.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[pressed_hand(0.45)])
        alternate.detect.side_effect = [
            SimpleNamespace(hand_landmarks=[]),
            SimpleNamespace(hand_landmarks=[pressed_hand(0.45), pressed_hand(0.55)]),
        ]
        recognizer = app.MultiViewHandLandmarker(primary, alternate)
        self.assertTrue(recognizer.recognize_joined_hands(image(), 100))
        self.assertEqual(recognizer.preferred_view, ("full", 3))
        self.assertEqual(recognizer.last_hand_count, 2)


class OpenPalmTests(unittest.TestCase):
    def test_open_spread_hand_is_a_palm(self):
        self.assertTrue(open_palm_from_landmarks([open_hand()], 640, 480))

    def test_curled_fingers_or_fingers_together_are_not(self):
        self.assertFalse(open_palm_from_landmarks([open_hand(curled=(2, 3))], 640, 480))  # V sign
        self.assertFalse(open_palm_from_landmarks([open_hand(curled=(0, 1, 2, 3))], 640, 480))  # fist
        self.assertFalse(open_palm_from_landmarks([open_hand(spread=0.025)], 640, 480))
        self.assertFalse(open_palm_from_landmarks([], 640, 480))

    def test_palms_pressed_together_never_count_as_reappear(self):
        hands = [pressed_hand(0.45), pressed_hand(0.55)]
        self.assertFalse(open_palm_from_landmarks(hands, 640, 480))

    def test_both_recognizers_detect_open_palm_with_primary_model_only(self):
        landmarker = app.MultiViewHandLandmarker(Mock(), Mock())
        landmarker.primary.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[open_hand()])
        self.assertTrue(landmarker.detect_open_palm(image(), 100))
        gestures = app.MultiViewGestureRecognizer(Mock(), Mock())
        gestures.primary.recognize_for_video.return_value = SimpleNamespace(hand_landmarks=[open_hand()])
        self.assertTrue(gestures.detect_open_palm(image(), 100))
        landmarker.alternate.detect.assert_not_called()
        gestures.alternate.recognize.assert_not_called()


class MenuTests(unittest.TestCase):
    def test_keyboard_selection(self):
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "setMouseCallback"))
            mocks.enter_context(patch.object(app.cv2, "imshow"))
            mocks.enter_context(patch.object(app.cv2, "destroyWindow"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            key = mocks.enter_context(patch.object(app.cv2, "waitKey", return_value=ord("2")))
            self.assertEqual(app.choose_mode(), "naruto")
            key.return_value = ord("1")
            self.assertEqual(app.choose_mode(), "normal")


if __name__ == "__main__":
    unittest.main()
