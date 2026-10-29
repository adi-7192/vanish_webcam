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
GESTURE_HOLD_SECONDS = 0.45
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

# Vanish gesture -> effect style. An open palm always brings you back.
GESTURE_EFFECTS = {
    "Victory": "normal",      # ✌️ Dissolve
    "Joined": "naruto",       # 🙏 Smoke puff
    "Closed_Fist": "dust",    # ✊ Dust
    "Pointing_Up": "portal",  # ☝️ Portal
    "ILoveYou": "glitch",     # 🤟 Glitch
    "Thumb_Down": "beam",     # 👎 Teleport
}
REAPPEAR_POSE = "Open_Palm"   # ✋


class PoseHold:
    """Fire once the same pose has been held for ``hold`` seconds.

    Short detection gaps (up to ``gap`` seconds) do not reset the hold, a
    different pose restarts it, and ``block`` adds a cooldown after a switch.
    """

    def __init__(self, hold: float = GESTURE_HOLD_SECONDS, gap: float = GESTURE_GAP_SECONDS):
        self.hold = hold
        self.gap = gap
        self.pose: str | None = None
        self.since = 0.0
        self.last_seen = -math.inf
        self.blocked_until = -math.inf

    def update(self, pose: str | None, now: float) -> str | None:
        if now - self.last_seen > self.gap:
            self.pose = None
        if pose is None:
            return None
        if pose != self.pose:
            self.pose, self.since = pose, now
        self.last_seen = now
        if now >= self.blocked_until and now - self.since >= self.hold:
            self.pose = None
            return pose
        return None

    def block(self, now: float, seconds: float = GESTURE_COOLDOWN_SECONDS) -> None:
        self.pose = None
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


# ---- Styles -----------------------------------------------------------------
#
# Every style renders "how gone" the person is as p in 0..1 (0 visible, 1 gone).
# Reappearing plays the same style with p running back from 1 to 0.

def _ramp(edge0: float, edge1: float, value: float) -> float:
    return float(smoothstep(edge0, edge1, np.float32(value)))


def person_bounds(cover: np.ndarray, threshold: float = 0.3):
    """(x0, x1, y0, y1) of the person, or a centred default."""
    ys, xs = np.nonzero(cover > threshold)
    height, width = cover.shape
    if not len(xs):
        return width // 3, 2 * width // 3, height // 4, height
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def draw_points(image: np.ndarray, points: np.ndarray, colors: np.ndarray,
                alpha: np.ndarray, size: int = 2) -> None:
    """Alpha-blend small squares at ``points`` (N x 2, x/y) into ``image``."""
    height, width = image.shape[:2]
    xs, ys = points[:, 0].astype(np.int32), points[:, 1].astype(np.int32)
    colors = np.broadcast_to(np.asarray(colors, np.float32), (len(xs), 3))
    alpha = np.broadcast_to(np.asarray(alpha, np.float32), (len(xs),))
    for dx in range(size):
        for dy in range(size):
            x, y = xs + dx, ys + dy
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            a = alpha[inside][:, None]
            pixels = image[y[inside], x[inside]].astype(np.float32)
            image[y[inside], x[inside]] = (pixels * (1 - a) + colors[inside] * a).astype(np.uint8)


# Motion blur covers half a frame interval at 30 fps, like a film camera's
# 180° shutter: fast motion reads as a smooth streak instead of a jump.
SHUTTER_SECONDS = 1 / 60
BLUR_SAMPLES = 4


def ease_in_out_sine(p: float) -> float:
    """Gentle start and finish; symmetric, so reversing mid-effect stays seamless."""
    return 0.5 - 0.5 * math.cos(math.pi * p)


class Style:
    vanish_seconds = 2.0
    appear_seconds = 1.6
    eased = True        # VanishEffect applies ease_in_out_sine to p
    seconds = 2.0       # length of the current vanish or reappear
    appearing = False

    def begin(self, frame: np.ndarray | None, cover: np.ndarray, appearing: bool) -> None:
        """Called when a vanish or reappear starts; ``frame`` is the live webcam."""

    def render(self, frame, background, cover, p: float, appearing: bool) -> np.ndarray:
        raise NotImplementedError


class DissolveStyle(Style):
    """Normal: the person dissolves bottom-up with a glowing cyan edge."""

    vanish_seconds, appear_seconds = DISSOLVE_SECONDS, REFORM_SECONDS
    eased = False  # render_dissolve already eases its edge

    def __init__(self):
        self.fields: tuple[np.ndarray, np.ndarray] | None = None

    def render(self, frame, background, cover, p, appearing):
        height, width = frame.shape[:2]
        if self.fields is None or self.fields[0].shape != (height, width):
            self.fields = make_dissolve_fields(height, width)
        return render_dissolve(frame, background, cover, p, *self.fields)


class SmokeStyle(Style):
    """Naruto: a puff of smoke; the person swaps while the smoke is densest."""

    vanish_seconds = appear_seconds = SMOKE_SECONDS
    eased = False  # timed like a real puff

    def __init__(self):
        self.anchor = (0, 0, 1)

    def begin(self, frame, cover, appearing):
        self.anchor = smoke_anchor(cover)

    def render(self, frame, background, cover, p, appearing):
        if appearing:
            age = (1.0 - p) * SMOKE_SECONDS
            hidden = 1.0 - min(1.0, max(0.0, age - 0.10) / SMOKE_SWAP_SECONDS)
        else:
            age = p * SMOKE_SECONDS
            hidden = min(1.0, age / SMOKE_SWAP_SECONDS)
        return render_smoke(frame, background, cover, hidden, self.anchor, age)


class DustStyle(Style):
    """The person crumbles into particles that blow away on the wind."""

    vanish_seconds, appear_seconds = 2.6, 2.0
    LIFE = 0.9           # seconds a particle stays visible after release
    RELEASE_SPAN = 0.6   # releases happen over the first 60% of the effect
    COUNT = 6000

    def __init__(self):
        self.release_map: np.ndarray | None = None

    def begin(self, frame, cover, appearing):
        height, width = cover.shape
        rng = np.random.default_rng(4242)
        ys, xs = np.nonzero(cover > 0.5)
        self.seconds = self.appear_seconds if appearing else self.vanish_seconds
        if not len(xs):
            self.release_map = None
            return
        x0, x1 = xs.min(), xs.max()
        across = np.clip((np.arange(width, dtype=np.float32) - x0) / max(1, x1 - x0), 0, 1)
        noise = rng.random((max(2, height // 24), max(2, width // 24)), dtype=np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        # The wind blows left to right, so the left side crumbles first.
        self.release_map = (across[None, :] * 0.65 + np.clip(noise, 0, 1) * 0.35) * self.RELEASE_SPAN
        pick = rng.choice(len(xs), size=min(self.COUNT, len(xs)), replace=False)
        px, py = xs[pick], ys[pick]
        source = frame if frame is not None else np.full((height, width, 3), 200, np.uint8)
        self.colors = source[py, px].astype(np.float32)
        self.release = self.release_map[py, px]
        self.origin = np.stack([px, py], axis=1).astype(np.float32)
        self.velocity = np.stack([rng.uniform(140, 360, len(px)),
                                  rng.uniform(-160, -30, len(px))], axis=1).astype(np.float32)
        self.flutter = rng.uniform(0, 2 * np.pi, len(px)).astype(np.float32)

    def render(self, frame, background, cover, p, appearing):
        if self.release_map is None or self.release_map.shape != cover.shape:
            return replace_person(frame, background, cover * p)
        survive = smoothstep(p - 0.04, p + 0.04, self.release_map)
        output = replace_person(frame, background, cover * (1.0 - survive))
        age = (p - self.release) * self.seconds
        live = (age > 0) & (age < self.LIFE)
        if not live.any():
            return output
        if output is frame:
            output = frame.copy()
        a = age[live]
        alpha = (1 - a / self.LIFE) ** 1.5 * 0.9
        origin, velocity, flutter = self.origin[live], self.velocity[live], self.flutter[live]
        # Motion blur: draw the trail first, the particle's current spot last.
        for k in reversed(range(BLUR_SAMPLES)):
            sample = np.maximum(0.0, a - SHUTTER_SECONDS * k / max(1, BLUR_SAMPLES - 1))
            points = origin + velocity * sample[:, None]
            points[:, 1] += np.sin(sample * 7 + flutter) * 6
            weight = 1.0 - k / BLUR_SAMPLES
            draw_points(output, points, self.colors[live], alpha * weight, size=2)
        return output


class BeamStyle(Style):
    """Teleport: a shimmering column of light beams the person out."""

    vanish_seconds, appear_seconds = 1.8, 1.5
    COLOR = np.array((255, 235, 190), np.float32)  # icy blue-white (BGR)

    def __init__(self):
        self.bounds = (0, 1, 0, 1)
        self.sparks = np.zeros((0, 2), np.float32)

    def begin(self, frame, cover, appearing):
        self.bounds = person_bounds(cover)
        ys, xs = np.nonzero(cover > 0.5)
        if len(xs) > 20000:
            keep = np.random.default_rng(5).choice(len(xs), 20000, replace=False)
            xs, ys = xs[keep], ys[keep]
        self.sparks = np.stack([xs, ys], axis=1).astype(np.float32)

    def render(self, frame, background, cover, p, appearing):
        output = replace_person(frame, background, cover * _ramp(0.3, 0.7, p))
        if output is frame:
            output = frame.copy()
        beam = _ramp(0.0, 0.22, p) * (1.0 - _ramp(0.78, 1.0, p))
        if beam <= 0.01:
            return output
        height, width = output.shape[:2]
        x0, x1, y0, _ = self.bounds
        center = (x0 + x1) / 2
        half = max(8.0, (x1 - x0) / 2 * (0.55 + 0.45 * (1 - _ramp(0.7, 1.0, p))))
        left, right = int(max(0, center - half * 1.6)), int(min(width, center + half * 1.6))
        if left >= right:
            return output
        columns = np.arange(left, right, dtype=np.float32)
        across = np.exp(-((columns - center) / half) ** 2 * 2.2)
        rows = np.arange(height, dtype=np.float32)
        above = np.clip(1 - np.maximum(0, y0 - rows) / max(1, y0 + 1) * 0.6, 0.3, 1.0)
        scan = 0.85 + 0.15 * np.sin(rows * 0.35 - p * 60)
        glow = (above * scan)[:, None] * across[None, :] * beam * 0.55
        region = output[:, left:right].astype(np.float32)
        output[:, left:right] = np.clip(region + glow[:, :, None] * self.COLOR, 0, 255).astype(np.uint8)
        density = beam * max(0.0, 1 - abs(p - 0.5) * 1.4)
        count = min(len(self.sparks), int(500 * density))
        if count:
            rng = np.random.default_rng(int(p * 997))
            pick = rng.choice(len(self.sparks), count, replace=False)
            draw_points(output, self.sparks[pick], (255, 250, 240), rng.uniform(0.5, 1.0, count), size=2)
        return output


class GlitchStyle(Style):
    """The person breaks into shifted slices and split colours, then flickers out."""

    vanish_seconds, appear_seconds = 1.4, 1.2

    def render(self, frame, background, cover, p, appearing):
        area = person_box(cover, pad=24)
        if area is None:
            return frame
        rng = np.random.default_rng(int(p * 1000) + 7)
        strength = math.sin(math.pi * min(p, 1.0))
        region, mask = frame[area], cover[area]
        output = frame.copy()
        output[area] = (region.astype(np.float32) * (1 - mask[..., None])
                        + background[area].astype(np.float32) * mask[..., None]).astype(np.uint8)
        layer, visible = region.copy(), mask.copy()
        shift = int(round(strength * 10))
        if shift:  # RGB split
            layer[:, :, 0] = np.roll(layer[:, :, 0], -shift, axis=1)
            layer[:, :, 2] = np.roll(layer[:, :, 2], shift, axis=1)
        height, width = mask.shape
        if strength > 0.35:
            factor = int(2 + strength * 14)
            small = cv2.resize(layer, (max(1, width // factor), max(1, height // factor)),
                               interpolation=cv2.INTER_AREA)
            blocky = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            blocky = None
        top = 0
        while top < height:
            rows = slice(top, min(height, top + int(rng.integers(4, 28))))
            if rng.random() < p:  # this slice is gone
                visible[rows] = 0
            else:
                if blocky is not None and rng.random() < strength * 0.5:
                    layer[rows] = blocky[rows]
                dx = int(rng.normal() * strength * 45)
                if dx:
                    layer[rows] = np.roll(layer[rows], dx, axis=1)
                    visible[rows] = np.roll(visible[rows], dx, axis=1)
            top = rows.stop
        if rng.random() < p * 0.35:  # whole-person flicker
            visible[:] = 0
        a = visible[..., None]
        output[area] = (output[area].astype(np.float32) * (1 - a) + layer.astype(np.float32) * a).astype(np.uint8)
        return output


class PortalStyle(Style):
    """A glowing ring opens and the person spirals into it."""

    vanish_seconds, appear_seconds = 1.9, 1.6
    RING = np.array((60, 150, 255), np.float32)   # orange (BGR)
    CORE = np.array((190, 235, 255), np.float32)  # hot inner edge

    def __init__(self):
        self.anchor = (0, 0, 1)

    def begin(self, frame, cover, appearing):
        self.anchor = smoke_anchor(cover)

    def render(self, frame, background, cover, p, appearing):
        cx, cy, size = self.anchor
        height, width = frame.shape[:2]
        output = replace_person(frame, background, cover)
        if output is frame:
            output = frame.copy()
        opening = _ramp(0.0, 0.25, p) * (1 - _ramp(0.82, 1.0, p))
        if opening > 0.01:  # ring first, so the person spirals in front of it
            self.draw_ring(output, cx, cy, size * 0.55 * opening, p)
        # Motion blur: average the spinning, shrinking person over the shutter.
        step = SHUTTER_SECONDS / max(self.seconds, 1e-3) * (1 if self.appearing else -1)
        coverage = np.zeros((height, width, 1), np.float32)
        colour = np.zeros((height, width, 3), np.float32)
        samples = 0
        for k in range(max(1, BLUR_SAMPLES - 1)):  # three sub-steps keep the cost low
            shrink = _ramp(0.15, 0.85, float(np.clip(p + step * k / 2, 0.0, 1.0)))
            samples += 1
            if shrink >= 0.99:
                continue
            matrix = cv2.getRotationMatrix2D((float(cx), float(cy)), shrink * 300, 1 - shrink)
            weight = cv2.warpAffine(cover, matrix, (width, height))[..., None]
            coverage += weight
            colour += cv2.warpAffine(frame, matrix, (width, height)).astype(np.float32) * weight
        if coverage.any():
            coverage /= samples
            colour /= samples
            output = np.clip(output.astype(np.float32) * (1 - coverage) + colour, 0, 255).astype(np.uint8)
        return output

    def draw_ring(self, image, cx, cy, radius, p):
        radius = max(3.0, radius)
        reach = int(radius * 1.5) + 4
        left, top = max(0, cx - reach), max(0, cy - reach)
        right, bottom = min(image.shape[1], cx + reach), min(image.shape[0], cy + reach)
        if left >= right or top >= bottom:
            return
        layer = np.zeros((bottom - top, right - left, 3), np.float32)
        center = (cx - left, cy - top)
        axes = (int(radius), int(radius * 0.92))
        thickness = max(2, int(radius * 0.10))
        cv2.ellipse(layer, center, axes, 0, 0, 360, self.RING.tolist(), thickness, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(1.0, radius * 0.06))
        cv2.ellipse(layer, center, axes, 0, 0, 360, self.CORE.tolist(), max(1, thickness // 3), cv2.LINE_AA)
        rng = np.random.default_rng(int(p * 600))
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False) + p * 14
        jitter = rng.uniform(0.95, 1.18, len(angles))
        for angle, scale in zip(angles, jitter):
            x = int(center[0] + np.cos(angle) * axes[0] * scale)
            y = int(center[1] + np.sin(angle) * axes[1] * scale)
            cv2.circle(layer, (x, y), max(1, thickness // 3), self.CORE.tolist(), -1, cv2.LINE_AA)
        region = image[top:bottom, left:right].astype(np.float32)
        image[top:bottom, left:right] = np.clip(region + layer, 0, 255).astype(np.uint8)


STYLES = {
    "normal": DissolveStyle,
    "naruto": SmokeStyle,
    "dust": DustStyle,
    "beam": BeamStyle,
    "glitch": GlitchStyle,
    "portal": PortalStyle,
}


# ---- State machine ----------------------------------------------------------

class VanishEffect:
    """visible -> vanishing -> vanished -> appearing -> visible."""

    def __init__(self, style: str = "normal"):
        self.styles = {name: make() for name, make in STYLES.items()}
        self.style_name = style
        self.style = self.styles[style]
        self.phase = "visible"
        self.started = 0.0

    def vanish(self, now: float, cover: np.ndarray, frame: np.ndarray | None = None,
               style: str | None = None) -> bool:
        """Start vanishing with ``style`` (default: the last one used)."""
        if self.phase != "visible":
            return False
        if style is not None:
            self.style_name, self.style = style, self.styles[style]
        self.phase, self.started = "vanishing", now
        self.style.seconds, self.style.appearing = self.style.vanish_seconds, False
        self.style.begin(frame, cover, appearing=False)
        return True

    def appear(self, now: float, cover: np.ndarray, frame: np.ndarray | None = None) -> bool:
        if self.phase not in ("vanishing", "vanished"):
            return False
        started = now
        if self.phase == "vanishing":
            # Reverse from the current point instead of jumping to "gone".
            progress = min(1.0, (now - self.started) / self.style.vanish_seconds)
            started = now - (1.0 - progress) * self.style.appear_seconds
        self.phase, self.started = "appearing", started
        self.style.seconds, self.style.appearing = self.style.appear_seconds, True
        self.style.begin(frame, cover, appearing=True)
        return True

    def reset(self) -> None:
        self.phase = "visible"

    def render(self, frame: np.ndarray, background: np.ndarray, cover: np.ndarray,
               now: float) -> np.ndarray:
        if self.phase == "visible":
            return frame
        age = max(0.0, now - self.started)
        if self.phase == "vanishing":
            p = age / self.style.vanish_seconds
            if p >= 1.0:
                self.phase = "vanished"
        elif self.phase == "appearing":
            p = 1.0 - age / self.style.appear_seconds
            if p <= 0.0:
                self.phase = "visible"
                return frame
        if self.phase == "vanished":
            return replace_person(frame, background, cover)
        appearing = self.phase == "appearing"
        p = float(np.clip(p, 0.0, 1.0))
        if self.style.eased:
            p = ease_in_out_sine(p)
        return self.style.render(frame, background, cover, p, appearing)
