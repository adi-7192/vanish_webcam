import unittest
from types import SimpleNamespace
from unittest.mock import Mock

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

    def test_recognizer_reports_joined_hands_from_the_current_frame(self):
        primary, alternate = Mock(), Mock()
        primary.recognize_for_video.return_value = SimpleNamespace(
            hand_landmarks=[pressed_hand(0.45), pressed_hand(0.55)], hand_world_landmarks=[], gestures=[])
        recognizer = app.MultiViewGestureRecognizer(primary, alternate)
        self.assertEqual(recognizer.recognize_pose(image(), 100), "Joined")
        self.assertEqual(recognizer.last_hand_count, 2)
        alternate.recognize.assert_not_called()

    def test_joined_hands_recovered_in_a_zoomed_view(self):
        primary, alternate = Mock(), Mock()
        primary.recognize_for_video.return_value = SimpleNamespace(
            hand_landmarks=[pressed_hand(0.45)], hand_world_landmarks=[], gestures=[])
        alternate.recognize.side_effect = [
            SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], gestures=[]),
            SimpleNamespace(hand_landmarks=[pressed_hand(0.45), pressed_hand(0.55)],
                            hand_world_landmarks=[], gestures=[]),
        ]
        recognizer = app.MultiViewGestureRecognizer(primary, alternate)
        self.assertEqual(recognizer.recognize_pose(image(), 100), "Joined")
        self.assertEqual(recognizer.preferred_view, ("tile01", 0))
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

    def test_open_palm_found_by_the_tracker_or_a_zoomed_view(self):
        empty = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], gestures=[])
        palm = SimpleNamespace(hand_landmarks=[open_hand()], hand_world_landmarks=[], gestures=[])
        direct = app.MultiViewGestureRecognizer(Mock(), Mock())
        direct.primary.recognize_for_video.return_value = palm
        self.assertEqual(direct.recognize_pose(image(), 100), "Open_Palm")
        direct.alternate.recognize.assert_not_called()
        zoomed = app.MultiViewGestureRecognizer(Mock(), Mock())
        zoomed.primary.recognize_for_video.return_value = empty
        zoomed.alternate.recognize.return_value = palm
        self.assertEqual(zoomed.recognize_pose(image(), 100), "Open_Palm")


if __name__ == "__main__":
    unittest.main()
