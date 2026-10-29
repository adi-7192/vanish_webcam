"""Vanish/reappear rendering, hand-pose geometry, gesture timing and the live background.

The room always stays live: only the pixels covered by the person are replaced
by the learned background, so lighting, camera grain and movement behind the
person keep going while they are invisible.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

DISSOLVE_SECONDS = 2.4      # Normal: V sign -> dissolve away.
REFORM_SECONDS = 1.6        # Normal: open palm -> dissolve back in.
SMOKE_SECONDS = 1.45        # Naruto: length of the smoke puff.
SMOKE_SWAP_SECONDS = 0.13   # Naruto: person swaps while the smoke is densest.
BACKGROUND_LEARN_RATE = 0.04
GESTURE_HOLD_SECONDS = 0.35
GESTURE_GAP_SECONDS = 0.45  # small hands blink in and out of detection
GESTURE_COOLDOWN_SECONDS = 1.2
FINGER_BASES = (5, 9, 13, 17)  # MCP landmark of index, middle, ring and little finger.


# ---- Hand-pose geometry ---------------------------------------------------

def _hands_in_pixels(hands, width: int, height: int) -> list[np.ndarray]:
    points = [np.array([(p.x * width, p.y * height) for p in hand])
              for hand in hands if len(hand) == 21]
    return [hand for hand in points if np.isfinite(hand).all()]


def joined_hands_from_landmarks(hands, width: int, height: int) -> bool:
    """Recognize two nearby, similarly oriented open hands in image space."""
    if len(hands) < 2:
        return False
    points = _hands_in_pixels(hands[:2], width, height)
    if len(points) != 2:
        return False
    palms = [hand[[0, 5, 9, 13, 17]].mean(axis=0) for hand in points]
    scale = [max(float(np.linalg.norm(hand[5] - hand[17])),
                 float(np.linalg.norm(hand[0] - hand[9]))) for hand in points]
    if min(scale) < 8.0 or max(scale) > min(scale) * 2.2:
        return False
    size = sum(scale) * 0.5
    directions = [hand[9] - hand[0] for hand in points]
    alignment = float(np.dot(*directions) /
                      max(1e-6, np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])))
    fingertips = [hand[[8, 12, 16, 20]].mean(axis=0) for hand in points]
    return (
        alignment > 0.25
        and np.linalg.norm(palms[0] - palms[1]) < size * 1.55
        and np.linalg.norm(fingertips[0] - fingertips[1]) < size * 1.9
    )


def finger_extended(hand: np.ndarray, base: int) -> bool:
    finger = hand[base:base + 4]
    length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
    if length < 1e-6:
        return False
    straightness = float(np.linalg.norm(finger[3] - finger[0])) / length
    reach = float(np.linalg.norm(finger[3] - hand[0]))
    knuckle = float(np.linalg.norm(finger[1] - hand[0]))
    return straightness >= 0.85 and reach > knuckle * 1.15


def open_palm_from_landmarks(hands, width: int, height: int) -> bool:
    """One open hand with all four fingers straight (a "hi" wave).

    Palms pressed together (the Naruto vanish pose) never count.
    """
    if joined_hands_from_landmarks(hands, width, height):
        return False
    for hand in _hands_in_pixels(hands, width, height):
        knuckles = float(np.linalg.norm(hand[5] - hand[17]))
        if knuckles < 6.0:
            continue
        spread = float(np.linalg.norm(hand[8] - hand[20]))
        if all(finger_extended(hand, base) for base in FINGER_BASES) and spread > knuckles * 0.85:
            return True
    return False


# ---- Gesture timing ---------------------------------------------------------

class GestureHold:
    """Fire once a pose has been held for ``hold`` seconds, with a cooldown.

    Short detection gaps (up to ``gap`` seconds) do not reset the hold, so a
    single missed frame does not cancel the gesture.
    """

    def __init__(self, hold: float = GESTURE_HOLD_SECONDS, gap: float = GESTURE_GAP_SECONDS):
        self.hold = hold
        self.gap = gap
        self.since: float | None = None
        self.last_seen = -math.inf
        self.blocked_until = -math.inf

    def update(self, seen: bool, now: float) -> bool:
        if now - self.last_seen > self.gap:
            self.since = None
        if not seen:
            return False
        if self.since is None:
            self.since = now
        self.last_seen = now
        if now >= self.blocked_until and now - self.since >= self.hold:
            self.since = None
            return True
        return False

    def block(self, now: float, seconds: float = GESTURE_COOLDOWN_SECONDS) -> None:
        self.since = None
        self.blocked_until = now + seconds


# ---- Masks and background ---------------------------------------------------

def cover_mask(mask: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    """Smooth the person mask over time and grow it slightly past the edges,
    so hair, fingers and fast motion do not flicker through when hidden."""
    if previous is not None and previous.shape == mask.shape:
        mask = mask * 0.6 + previous * 0.4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    grown = cv2.GaussianBlur(cv2.dilate(mask, kernel), (0, 0), sigmaX=5.0)
    return np.maximum(mask, grown).astype(np.float32)


class LiveBackground:
    """Background plate that keeps learning wherever the person is not."""

    def __init__(self, image: np.ndarray):
        self.plate = image.astype(np.float32)

    def update(self, frame: np.ndarray, cover: np.ndarray) -> None:
        free = (cover < 0.02).astype(np.uint8)
        cv2.accumulateWeighted(frame, self.plate, BACKGROUND_LEARN_RATE, mask=free)

    @property
    def image(self) -> np.ndarray:
        return self.plate.astype(np.uint8)


def person_box(hide: np.ndarray, pad: int = 18):
    points = cv2.findNonZero((hide > 0.01).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return np.s_[max(0, y - pad):min(hide.shape[0], y + height + pad),
                 max(0, x - pad):min(hide.shape[1], x + width + pad)]


def replace_person(frame: np.ndarray, background: np.ndarray, hide: np.ndarray) -> np.ndarray:
    """Show the background where ``hide`` is 1 and the live frame elsewhere."""
    area = person_box(hide)
    if area is None:
        return frame
    output = frame.copy()
    amount = hide[area][:, :, None]
    output[area] = (frame[area].astype(np.float32) * (1.0 - amount)
                    + background[area].astype(np.float32) * amount).astype(np.uint8)
    return output


# ---- Normal: dissolve -------------------------------------------------------

def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def render_dissolve(frame: np.ndarray, background: np.ndarray, cover: np.ndarray,
                    progress: float, field: np.ndarray, sparkle: np.ndarray) -> np.ndarray:
    """progress 0 = fully visible, 1 = fully gone; glowing edge in between."""
    if progress <= 0.002:
        return frame
    if progress >= 0.999:
        return replace_person(frame, background, cover)
    area = person_box(cover)
    if area is None:
        return frame
    cover, field, sparkle = cover[area], field[area], sparkle[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    hide = (cover * (1.0 - survival))[:, :, None]
    mixed = frame[area].astype(np.float32) * (1.0 - hide) + background[area].astype(np.float32) * hide
    edge = ((np.abs(field - eased) < 0.018) & (cover > 0.48) & (sparkle > 0.54)).astype(np.float32)
    glow = cv2.GaussianBlur(edge, (0, 0), sigmaX=5.0, sigmaY=5.0)
    glow = (np.clip(glow * 5.0 + edge, 0.0, 1.0) * 0.8)[:, :, None]
    cyan = np.array((255, 228, 92), dtype=np.float32)
    output = frame.copy()
    output[area] = np.clip(mixed * (1.0 - glow) + cyan * glow, 0, 255).astype(np.uint8)
    return output


# ---- Naruto: smoke puff -----------------------------------------------------

def smoke_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Centre the smoke on the person's torso; size follows their height."""
    height, width = mask.shape
    binary = (mask > 0.45).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return width // 2, height // 2, int(height * 0.57)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[largest]
    size = int(np.clip(h * 0.58, height * 0.24, height * 0.70))
    center_x = int(np.clip(x + w * 0.5, size // 3, width - size // 3))
    center_y = int(np.clip(y + h * 0.52, size // 3, height - size // 3))
    return center_x, center_y, size


def draw_smoke(frame: np.ndarray, anchor: tuple[int, int, int], age: float, foreground: bool) -> None:
    if age >= SMOKE_SECONDS:
        return
    cx, cy, size = anchor
    rng = np.random.default_rng(9127 if foreground else 1934)
    reveal = min(1.0, age / 0.16)
    fade = max(0.0, 1.0 - max(0.0, age - 0.33) / 1.12)
    if fade <= 0:
        return
    margin = int(size * 1.35)
    left, top = max(0, cx - margin), max(0, cy - margin)
    right = min(frame.shape[1], cx + margin + 1)
    bottom = min(frame.shape[0], cy + margin + 1)
    if left >= right or top >= bottom:
        return
    layer = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    radius = size * (0.27 + 0.54 * min(age, 1.0))
    for _ in range(22 if foreground else 18):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * rng.uniform(0.25, 1.0)
        x = int(cx + math.cos(angle) * distance) - left
        y = int(cy + math.sin(angle) * distance * 0.63 - age * size * 0.13) - top
        puff = max(2, int(size * rng.uniform(0.09, 0.20) * (0.7 + age * 0.7)))
        color = (229, 235, 235, 190) if foreground else (152, 162, 166, 125)
        cv2.ellipse(layer, (x, y), (puff, int(puff * 0.68)),
                    rng.uniform(-25, 25), 0, 360, color, -1, cv2.LINE_AA)
    for direction in (-1, 1):
        center = (int(cx + direction * size * 0.13) - left,
                  int(cy - size * 0.13 - age * size * 0.09) - top)
        axes = (max(2, int(size * (0.30 + age * 0.27))),
                max(2, int(size * (0.09 + age * 0.12))))
        cv2.ellipse(layer, center, axes, direction * 15, 20, 230,
                    (244, 247, 245, 220), max(2, int(size * 0.025)), cv2.LINE_AA)
    occupied = cv2.findNonZero(layer[:, :, 3])
    if occupied is None:
        return
    x, y, width, height = cv2.boundingRect(occupied)
    destination = frame[top + y:top + y + height, left + x:left + x + width]
    layer = layer[y:y + height, x:x + width]
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * reveal * fade
    destination[:] = np.clip(
        layer[:, :, :3].astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def render_smoke(frame: np.ndarray, background: np.ndarray, cover: np.ndarray,
                 hidden: float, anchor: tuple[int, int, int], age: float) -> np.ndarray:
    """Hide ``hidden`` (0..1) of the person, then draw the puff on top."""
    output = replace_person(frame, background, cover * hidden)
    if output is frame:
        output = frame.copy()
    draw_smoke(output, anchor, age, foreground=False)
    draw_smoke(output, anchor, age, foreground=True)
    return output


# ---- State machine ----------------------------------------------------------

class VanishEffect:
    """visible -> vanishing -> vanished -> appearing -> visible."""

    def __init__(self, style: str):
        self.style = style
        self.phase = "visible"
        self.started = 0.0
        self.anchor: tuple[int, int, int] | None = None
        self.fields: tuple[np.ndarray, np.ndarray] | None = None

    def vanish(self, now: float, cover: np.ndarray) -> bool:
        if self.phase != "visible":
            return False
        self.phase, self.started, self.anchor = "vanishing", now, smoke_anchor(cover)
        return True

    def appear(self, now: float, cover: np.ndarray) -> bool:
        if self.phase not in ("vanishing", "vanished"):
            return False
        started = now
        if self.phase == "vanishing" and self.style == "normal":
            # Reverse from the current point of the dissolve instead of jumping.
            progress = min(1.0, (now - self.started) / DISSOLVE_SECONDS)
            started = now - (1.0 - progress) * REFORM_SECONDS
        self.phase, self.started, self.anchor = "appearing", started, smoke_anchor(cover)
        return True

    def reset(self) -> None:
        self.phase = "visible"

    def render(self, frame: np.ndarray, background: np.ndarray, cover: np.ndarray,
               now: float) -> np.ndarray:
        if self.phase == "visible":
            return frame
        age = max(0.0, now - self.started)
        if self.style == "naruto":
            return self._render_smoke(frame, background, cover, age)
        return self._render_dissolve(frame, background, cover, age)

    def _render_dissolve(self, frame, background, cover, age):
        height, width = frame.shape[:2]
        if self.fields is None or self.fields[0].shape != (height, width):
            self.fields = make_dissolve_fields(height, width)
        if self.phase == "vanishing":
            progress = age / DISSOLVE_SECONDS
            if progress >= 1.0:
                self.phase = "vanished"
        elif self.phase == "appearing":
            progress = 1.0 - age / REFORM_SECONDS
            if progress <= 0.0:
                self.phase = "visible"
                return frame
        else:
            progress = 1.0
        return render_dissolve(frame, background, cover, float(np.clip(progress, 0.0, 1.0)), *self.fields)

    def _render_smoke(self, frame, background, cover, age):
        if self.phase == "vanished":
            return replace_person(frame, background, cover)
        if self.phase == "vanishing":
            hidden = min(1.0, age / SMOKE_SWAP_SECONDS)
            if age >= SMOKE_SECONDS:
                self.phase = "vanished"
                return replace_person(frame, background, cover)
        else:  # appearing: the person returns once the smoke is dense
            hidden = 1.0 - min(1.0, max(0.0, age - 0.10) / SMOKE_SWAP_SECONDS)
            if age >= SMOKE_SECONDS:
                self.phase = "visible"
                return frame
        return render_smoke(frame, background, cover, hidden, self.anchor, age)
