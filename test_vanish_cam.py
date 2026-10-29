import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import mediapipe as mp

import vanish_cam
from vanish_cam import (
    MultiViewGestureRecognizer, recognize_gesture, victory_landmark_score,
)


def hand_points(extended=(True, True, False, False)):
    points = np.zeros((21, 3))
    for base, x, opened, spread in zip(
        (5, 9, 13, 17), (-0.45, 0.0, 0.4, 0.75), extended, (-0.35, 0.25, 0.0, 0.0)
    ):
        if opened:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x + spread * 0.45, 1.7, 0.0),
                (x + spread * 0.75, 2.15, 0.0), (x + spread, 2.5, 0.0),
            ]
        else:
            points[base : base + 4] = [
                (x, 1.0, 0.0), (x, 1.5, 0.2),
                (x, 1.3, 0.5), (x, 0.95, 0.3),
            ]
    return points


def landmarks(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


class VictoryGestureTests(unittest.TestCase):
    def test_victory_with_back_palm_rotation_mirror_and_scale(self):
        points = hand_points()
        for angle in (0.0, 0.7, 1.57, 2.6, np.pi):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            tilt = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
            for mirror in (1, -1):
                for scale in (0.02, 1.0, 100.0):
                    with self.subTest(angle=angle, mirror=mirror, scale=scale):
                        transformed = points @ rotation @ tilt
                        transformed[:, 0] *= mirror
                        transformed = transformed * scale + (0.5, 0.3, 0.1)
                        self.assertGreater(victory_landmark_score(transformed), 0.62)

    def test_other_gestures_do_not_trigger(self):
        for fingers in (
            (False, False, False, False),  # Pugno.
            (True, True, True, True),  # Palmo aperto.
            (True, False, False, False),  # Solo indice.
            (True, False, False, True),  # Corna.
            (True, True, True, False),  # Tre dita.
        ):
            with self.subTest(fingers=fingers):
                self.assertEqual(victory_landmark_score(hand_points(fingers)), 0.0)

    def test_invalid_or_degenerate_landmarks(self):
        for points in (np.zeros((21, 3)), np.zeros((20, 3)), np.full((21, 3), np.nan)):
            self.assertEqual(victory_landmark_score(points), 0.0)

    def test_landmarks_trigger_without_classifier_category(self):
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(hand_points())], hand_landmarks=[], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        name, score = recognize_gesture(recognizer, None, 1000)
        self.assertEqual(name, "Victory")
        self.assertGreater(score, 0.62)

    def test_normalized_landmark_fallback_handles_video_aspect_ratio(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        normalized = points / (image.width, image.height, image.width)
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[landmarks(normalized)], gestures=[]
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_existing_classifier_still_works(self):
        result = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, None, 1000), ("Victory", 0.9))

    def test_occluded_folded_fingers_can_appear_short_and_straight(self):
        points = hand_points()
        for base in (13, 17):
            points[base : base + 4] = [
                (0.4, 1.0, 0.0), (0.4, 1.1, 0.0),
                (0.4, 1.2, 0.0), (0.4, 1.3, 0.0),
            ]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_hidden_fingers_with_coincident_landmarks(self):
        points = hand_points()
        points[13:16] = points[13]
        points[16] = points[13]
        points[17:20] = points[17]
        points[20] = points[17]
        self.assertGreater(victory_landmark_score(points), 0.62)

    def test_visible_geometry_is_used_even_with_inaccurate_world_landmarks(self):
        image = SimpleNamespace(width=960, height=540)
        points = hand_points() * 40 + (480, 200, 0)
        world = hand_points()
        world[8] = world[6]
        self.assertEqual(victory_landmark_score(world), 0.0)
        result = SimpleNamespace(
            hand_world_landmarks=[landmarks(world)],
            hand_landmarks=[landmarks(points / (960, 540, 960))], gestures=[],
        )
        recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
        self.assertEqual(recognize_gesture(recognizer, image, 1000)[0], "Victory")

    def test_diagnostics_distinguish_missing_hand_and_rejected_gesture(self):
        for hands, expected in (([], "No_Hand"), ([landmarks(hand_points((False,) * 4))], "Hand_Detected")):
            result = SimpleNamespace(hand_world_landmarks=hands, hand_landmarks=[], gestures=[])
            recognizer = SimpleNamespace(recognize_for_video=lambda image, timestamp: result)
            self.assertEqual(recognize_gesture(recognizer, None, 1000)[0], expected)


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.empty = SimpleNamespace(hand_world_landmarks=[], hand_landmarks=[], gestures=[])
        self.victory = SimpleNamespace(
            hand_world_landmarks=[], hand_landmarks=[],
            gestures=[[SimpleNamespace(category_name="Victory", score=0.9)]],
        )
        self.image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.zeros((100, 100, 3), dtype=np.uint8)
        )

    def detector(self, recognize):
        return MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.empty),
            SimpleNamespace(recognize=recognize),
        )

    def test_primary_victory_avoids_extra_inference(self):
        calls = []
        detector = MultiViewGestureRecognizer(
            SimpleNamespace(recognize_for_video=lambda image, timestamp: self.victory),
            SimpleNamespace(recognize=lambda image: calls.append(image) or self.empty),
        )
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [])

    def test_zoomed_tile_finds_a_small_hand_and_is_preferred_next_time(self):
        calls = []

        def recognize(image):
            calls.append((image.width, image.height))
            # Only the 60x60 zoomed tiles "see" the hand.
            return self.victory if image.width == 60 else self.empty

        detector = self.detector(recognize)
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "Victory")
        self.assertEqual(calls, [(60, 60)])
        self.assertEqual(detector.preferred_view, ("tile00", 0))
        calls.clear()
        self.assertEqual(recognize_gesture(detector, self.image, 1100)[0], "Victory")
        self.assertEqual(calls[0], (60, 60))

    def test_failed_search_is_spread_over_checks_and_throttled(self):
        calls = []
        detector = self.detector(lambda image: calls.append(image) or self.empty)
        self.assertEqual(recognize_gesture(detector, self.image, 1000)[0], "No_Hand")
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1040)  # within the retry interval
        self.assertEqual(len(calls), 3)
        recognize_gesture(detector, self.image, 1100)
        self.assertEqual(len(calls), 6)

    def test_rotated_full_view_is_tried_after_the_tiles(self):
        def recognize(image):
            return self.victory if (image.width, image.height) == (100, 100) else self.empty

        detector = self.detector(recognize)
        found = [recognize_gesture(detector, self.image, 1000 + step * 100)[0] for step in range(3)]
        self.assertEqual(found[-1], "Victory")
        self.assertEqual(detector.preferred_view, ("full", 1))

    def test_hand_views_cover_the_image_with_zoomed_tiles(self):
        views = vanish_cam.hand_views(np.zeros((540, 540, 3), np.uint8))
        self.assertEqual(len(views), 7)
        self.assertEqual(views["tile00"].shape[:2], (324, 324))
        self.assertEqual(views["tile12"].shape[:2], (324, 324))

    def test_person_square_follows_the_person_smoothly(self):
        cover = np.zeros((540, 960), np.float32)
        cover[:, 700:900] = 1.0
        left, center = vanish_cam.person_square(cover, 960, 540, None)
        self.assertEqual(left, 420)  # clamped to the right edge
        left, smoothed = vanish_cam.person_square(np.zeros((540, 960), np.float32), 960, 540, 800.0)
        self.assertAlmostEqual(smoothed, 800 * 0.8 + 480 * 0.2)


class GestureModelOptionsTests(unittest.TestCase):
    def test_no_gesture_allowlist_so_all_hands_keep_their_landmarks(self):
        created = []
        fake = SimpleNamespace(create_from_options=lambda options: created.append(options) or SimpleNamespace(close=lambda: None))
        with patch.object(vanish_cam.vision, "GestureRecognizer", fake), \
                patch.object(vanish_cam.vision.ImageSegmenter, "create_from_options"):
            vanish_cam.create_tasks("normal")
        options = created[0]
        classifier = getattr(options, "canned_gesture_classifier_options", None)
        self.assertIsNone(getattr(classifier, "category_allowlist", None))


class HandModelImageTests(unittest.TestCase):
    def setUp(self):
        data = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        self.image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data)

    def test_macos_gets_rgba_with_same_pixels(self):
        with patch.object(vanish_cam.platform, "system", return_value="Darwin"):
            converted = vanish_cam.hand_model_image(self.image)
        self.assertEqual(converted.image_format, mp.ImageFormat.SRGBA)
        np.testing.assert_array_equal(converted.numpy_view()[..., :3], self.image.numpy_view())
        self.assertTrue((converted.numpy_view()[..., 3] == 255).all())

    def test_other_platforms_keep_the_original_image(self):
        with patch.object(vanish_cam.platform, "system", return_value="Windows"):
            self.assertIs(vanish_cam.hand_model_image(self.image), self.image)

    def test_macos_uses_gpu_delegate(self):
        with patch.object(vanish_cam.platform, "system", return_value="Darwin"):
            options = vanish_cam.hand_base_options(vanish_cam.HAND_MODEL)
        self.assertEqual(options.delegate, vanish_cam.python.BaseOptions.Delegate.GPU)


if __name__ == "__main__":
    unittest.main()
