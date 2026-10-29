import unittest

import numpy as np

import effects
from effects import GestureHold, LiveBackground, VanishEffect, cover_mask, replace_person

H, W = 60, 80


def scene():
    """Live frame with a 'person' block in the middle and a background plate."""
    frame = np.full((H, W, 3), 100, np.uint8)
    frame[15:50, 30:50] = 230
    background = np.full((H, W, 3), 90, np.uint8)
    mask = np.zeros((H, W), np.float32)
    mask[15:50, 30:50] = 1.0
    return frame, background, mask


class GestureHoldTests(unittest.TestCase):
    def test_fires_only_after_hold_and_survives_short_gaps(self):
        hold = GestureHold(hold=0.35, gap=0.3)
        self.assertFalse(hold.update(True, 0.0))
        self.assertFalse(hold.update(False, 0.14))  # one missed check
        self.assertFalse(hold.update(True, 0.28))
        self.assertTrue(hold.update(True, 0.42))
        self.assertFalse(hold.update(True, 0.50))  # needs a fresh hold

    def test_long_gap_resets_and_cooldown_blocks(self):
        hold = GestureHold(hold=0.35, gap=0.3)
        hold.update(True, 0.0)
        self.assertFalse(hold.update(True, 0.5))  # gap 0.5 > 0.3 restarts the hold
        self.assertFalse(hold.update(True, 0.7))
        self.assertTrue(hold.update(True, 0.9))
        hold.block(1.0, seconds=1.2)
        for now in (1.1, 1.3, 1.5, 1.7, 1.9, 2.1):
            self.assertFalse(hold.update(True, now))
        self.assertTrue(hold.update(True, 2.3))


class BackgroundTests(unittest.TestCase):
    def test_learns_only_where_the_person_is_not(self):
        frame, background, mask = scene()
        plate = LiveBackground(background)
        for _ in range(200):
            plate.update(frame, cover_mask(mask))
        learned = plate.image
        self.assertTrue(abs(int(learned[2, 2, 0]) - 100) <= 1)  # room follows the live frame
        self.assertEqual(int(learned[30, 40, 0]), 90)  # never learns the person

    def test_cover_grows_past_the_edges(self):
        _, _, mask = scene()
        cover = cover_mask(mask)
        self.assertGreater(cover[12, 40], 0.3)
        self.assertEqual(cover[0, 0], 0.0)
        self.assertTrue((cover >= mask).all())


class ReplacePersonTests(unittest.TestCase):
    def test_room_stays_live_and_person_is_replaced(self):
        frame, background, mask = scene()
        output = replace_person(frame, background, mask)
        np.testing.assert_array_equal(output[:, :20], frame[:, :20])
        np.testing.assert_array_equal(output[20:45, 33:47], background[20:45, 33:47])

    def test_nothing_to_hide_returns_frame(self):
        frame, background, _ = scene()
        self.assertIs(replace_person(frame, background, np.zeros((H, W), np.float32)), frame)


class VanishEffectTests(unittest.TestCase):
    def run_cycle(self, style, vanish_time, appear_time):
        frame, background, mask = scene()
        effect = VanishEffect(style)
        self.assertIs(effect.render(frame, background, mask, 0.0), frame)
        self.assertFalse(effect.appear(0.0, mask))  # nothing to reappear from
        self.assertTrue(effect.vanish(1.0, mask, frame))
        self.assertFalse(effect.vanish(1.1, mask))
        middle = effect.render(frame, background, mask, 1.0 + vanish_time / 2)
        self.assertTrue(np.any(middle != frame))
        gone = effect.render(frame, background, mask, 1.0 + vanish_time + 0.01)
        self.assertEqual(effect.phase, "vanished")
        np.testing.assert_array_equal(gone[20:45, 33:47], background[20:45, 33:47])
        np.testing.assert_array_equal(gone[:, :10], frame[:, :10])  # room still live
        self.assertTrue(effect.appear(10.0, mask, frame))
        effect.render(frame, background, mask, 10.0 + appear_time / 2)
        self.assertEqual(effect.phase, "appearing")
        back = effect.render(frame, background, mask, 10.0 + appear_time + 0.01)
        self.assertEqual(effect.phase, "visible")
        self.assertIs(back, frame)

    def test_normal_dissolves_out_and_back_in(self):
        self.run_cycle("normal", effects.DISSOLVE_SECONDS, effects.REFORM_SECONDS)

    def test_naruto_smoke_out_and_back_in_without_log(self):
        self.run_cycle("naruto", effects.SMOKE_SECONDS, effects.SMOKE_SECONDS)

    def test_every_new_style_goes_out_and_back_in(self):
        for name in ("dust", "beam", "glitch", "portal"):
            with self.subTest(style=name):
                style = effects.STYLES[name]
                self.run_cycle(name, style.vanish_seconds, style.appear_seconds)

    def test_styles_keep_the_room_live_mid_effect(self):
        frame, background, mask = scene()
        frame[:, :4] = 17  # a strip of room outside every effect's reach
        for name in effects.STYLES:
            with self.subTest(style=name):
                effect = VanishEffect(name)
                effect.vanish(0.0, mask, frame)
                middle = effect.render(frame, background, mask, effect.style.vanish_seconds * 0.5)
                np.testing.assert_array_equal(middle[:, :4], frame[:, :4])

    def test_dust_particles_fly_away_from_the_person(self):
        frame, background, mask = scene()
        style = effects.DustStyle()
        style.begin(frame, mask, appearing=False)
        out = style.render(frame, background, mask, 0.5, appearing=False)
        right_of_person = out[:, 52:]
        self.assertTrue((right_of_person != frame[:, 52:]).any())  # particles drifted right

    def test_naruto_person_is_hidden_inside_the_first_smoke(self):
        frame, background, mask = scene()
        effect = VanishEffect("naruto")
        effect.vanish(0.0, mask)
        early = effect.render(frame, background, mask, 0.02)
        later = effect.render(frame, background, mask, effects.SMOKE_SWAP_SECONDS + 0.01)
        self.assertGreater(early[48, 40, 0], later[48, 40, 0] - 1)  # person fades as smoke grows

    def test_reappear_can_interrupt_vanishing(self):
        frame, background, mask = scene()
        effect = VanishEffect("normal")
        effect.vanish(0.0, mask)
        effect.render(frame, background, mask, 0.5)
        before = effect.render(frame, background, mask, 0.6)
        self.assertTrue(effect.appear(0.6, mask))
        self.assertEqual(effect.phase, "appearing")
        after = effect.render(frame, background, mask, 0.6)
        np.testing.assert_array_equal(before, after)  # reverses from the same point


if __name__ == "__main__":
    unittest.main()
