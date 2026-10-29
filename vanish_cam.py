"""VanishCam engine: gesture-triggered vanish effects sent to the OBS Virtual Camera.

One session runs from Start to Stop: the empty room is captured once, then each
vanish gesture plays its own effect and an open palm brings you back.
"""

from __future__ import annotations

import argparse
import os
import platform
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from effects import (
    GESTURE_EFFECTS, REAPPEAR_POSE, STYLES, LiveBackground, PoseHold, VanishEffect, cover_mask,
    joined_hands_from_landmarks, open_palm_from_landmarks,
)
from vanish_sound import EffectSound, sound_paths


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "VanishCam Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
}

# Callables that receive user-facing progress messages (the Mac app shows them).
STATUS_LISTENERS: list = []


def say(message: str) -> None:
    print(f"[VanishCam] {message}", flush=True)
    for listener in STATUS_LISTENERS:
        listener(message)


# Commands accepted from the keyboard (preview window) and from --control-stdin.
KEY_COMMANDS = {
    ord(" "): "background", ord("r"): "reappear", ord("v"): "vanish",
    ord("f"): "fullscreen", ord("q"): "quit", 27: "quit",
}
SIMPLE_COMMANDS = frozenset(KEY_COMMANDS.values()) | {"preview"}

# The OBS camera extension advertises only 1920x1080 to apps but passes on
# whatever size it is sent. Chrome rescales; WhatsApp and other native apps
# freeze on a mismatch, so always send exactly the advertised size.
VIRTUAL_CAMERA_SIZE = (1920, 1080)

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
GESTURE_INTERVAL_SECONDS = 0.1
VICTORY_MIN_SCORE = 0.62
POSE_MIN_SCORE = 0.6
CANNED_POSES = ("Open_Palm", "Closed_Fist", "Pointing_Up", "ILoveYou", "Thumb_Down")
EFFECT_NAMES = {
    "normal": "Dissolve", "naruto": "Smoke puff", "dust": "Dust",
    "beam": "Teleport", "glitch": "Glitch", "portal": "Portal",
}
GESTURE_HELP = "✌️ Dissolve · ✊ Dust · ☝️ Portal · 🤟 Glitch · 👎 Teleport · 🙏 Smoke puff"


def parse_command(text: str) -> str | None:
    """Normalise a control command, or None if it is not valid.

    Simple commands: vanish, reappear, background, preview, fullscreen, quit.
    With a value: vanish:<effect>, camera:<index>, gestures:on|off.
    """
    command = text.strip().lower()
    if command in SIMPLE_COMMANDS:
        return command
    name, _, value = command.partition(":")
    if name == "vanish" and value in STYLES:
        return command
    if name == "camera" and value.isdigit():
        return command
    if name == "gestures" and value in ("on", "off"):
        return command
    return None


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VanishCam: gesture-triggered vanish effects for your webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=960, help="Video width")
    parser.add_argument("--height", type=int, default=540, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start the preview in fullscreen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the webcam"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum virtual camera FPS (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require the virtual camera; by default it is used when available. --no-virtual-camera disables it",
    )
    parser.add_argument(
        "--button-effect", choices=tuple(STYLES), default="normal",
        help="Effect used by the V key / Vanish button (gestures pick their own)",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Do not open the preview window (useful during video calls)",
    )
    parser.add_argument(
        "--control-stdin", action="store_true",
        help="Read commands from stdin, one per line: vanish, vanish:<effect>, reappear, "
             "background, camera:<index>, gestures:on|off, preview, quit",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print gesture recognition details to the terminal (never on the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    """Open the OBS camera at its advertised size; ``width``/``height`` are the
    webcam's, used only for the log line."""
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=VIRTUAL_CAMERA_SIZE[0], height=VIRTUAL_CAMERA_SIZE[1], fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR, backend="obs",
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "OBS Virtual Camera is not available. Install OBS Studio 30+, click "
            "'Start Virtual Camera' once, approve the extension in System Settings, "
            "then stop the virtual camera in OBS (VanishCam needs it free). "
            f"Details: {error}"
        )
        if args.virtual_camera is True:
            raise RuntimeError(message) from error
        say(f"{message} Continuing with the local preview only.")
        return None
    say(f"Live on {camera.device} ({camera.width}x{camera.height}, webcam {width}x{height})")
    return camera


def start_stdin_commands() -> queue.Queue[str]:
    """Read control commands from stdin; EOF (parent gone) means quit."""
    commands: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        for line in sys.stdin:
            command = parse_command(line)
            if command is not None:
                commands.put(command)
            elif line.strip():
                say(f"Unknown command: {line.strip()}")
        commands.put("quit")

    threading.Thread(target=reader, name="stdin-commands", daemon=True).start()
    return commands


def next_command(commands: queue.Queue[str] | None) -> str | None:
    if commands is None:
        return None
    try:
        return commands.get_nowait()
    except queue.Empty:
        return None


def open_preview(width: int, height: int, fullscreen: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    if fullscreen:
        set_fullscreen(True)


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to exactly width x height, letterboxing instead of stretching."""
    if frame.shape[:2] == (height, width):
        return frame
    scale = min(width / frame.shape[1], height / frame.shape[0])
    size = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
    scaled = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    if size == (width, height):
        return scaled
    canvas = np.zeros((height, width, 3), np.uint8)
    left, top = (width - size[0]) // 2, (height - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = scaled
    return canvas


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    camera.send(np.ascontiguousarray(fit_frame(frame, camera.width, camera.height)))


def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        say(f"Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. "
                "Check your internet connection and try again."
            ) from error


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        raise RuntimeError(
            f"Webcam {index} is not available. Allow camera access in System Settings > "
            "Privacy & Security > Camera, close other apps using it, or pick another webcam."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def hand_base_options(model: Path) -> python.BaseOptions:
    # On macOS the hand/gesture graphs abort at startup with "Service is
    # unavailable" unless they run on the Metal (GPU) delegate.
    if platform.system() == "Darwin":
        return python.BaseOptions(model_asset_path=str(model),
                                  delegate=python.BaseOptions.Delegate.GPU)
    return python.BaseOptions(model_asset_path=str(model))


def hand_model_image(image: mp.Image) -> mp.Image:
    """The macOS GPU delegate accepts RGBA frames only (RGB aborts the process)."""
    if platform.system() != "Darwin" or image.image_format != mp.ImageFormat.SRGB:
        return image
    rgba = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2RGBA)
    return mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)


def create_tasks() -> tuple[vision.ImageSegmenter, "MultiViewGestureRecognizer"]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    # One gesture model for every pose, both hands (palms together needs two).
    # No canned-gesture allowlist: MediaPipe would drop the landmarks of every
    # hand whose gesture is filtered out, hiding open palms and joined hands.
    options = vision.GestureRecognizerOptions(
        base_options=hand_base_options(GESTURE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.5,
    )
    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = vision.GestureRecognizer.create_from_options(options)
        try:
            alternate = vision.GestureRecognizer.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, MultiViewGestureRecognizer(primary, alternate)


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def victory_landmark_score(points: np.ndarray) -> float:
    """Detect extended index and middle fingers from 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip for index, middle, ring and little finger.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # Seen from the side, folded fingers can look straight but very short:
    # do not require clearly visible joints for the ring and little finger.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be apart; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not classifier probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # Landmarks are returned even when the classifier misses "Victory"
    # (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: estimated depth can be wrong
    # when the palm and folded fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


# Retry views, tried a few per check on the IMAGE-mode model after the VIDEO
# model misses. Upper tiles come first because hands are usually raised.
RETRY_VIEWS = (
    [(f"tile0{column}", 0) for column in range(3)]
    + [(f"tile1{column}", 0) for column in range(3)]
    + [("full", 1), ("full", 3)]
)
RETRY_INTERVAL_MS = 80


def hand_views(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """The whole view plus six overlapping zoomed tiles (2 rows x 3 columns).

    MediaPipe shrinks the input to a small square before looking for hands,
    so a hand at normal laptop distance can be too small in the full view.
    """
    height, width = rgb.shape[:2]
    side = max(1, int(min(height, width) * 0.6))
    views = {"full": rgb}
    for row, top in enumerate((0, height - side)):
        for column, left in enumerate((0, (width - side) // 2, width - side)):
            views[f"tile{row}{column}"] = rgb[top:top + side, left:left + side]
    return views


def person_square(cover: np.ndarray | None, width: int, height: int,
                  previous_center: float | None) -> tuple[int, float]:
    """Left edge and smoothed centre of a full-height square around the person."""
    side = min(width, height)
    center = width / 2
    if cover is not None:
        columns = np.flatnonzero((cover > 0.3).any(axis=0))
        if columns.size:
            center = (columns[0] + columns[-1]) / 2
    if previous_center is not None:
        center = previous_center * 0.8 + center * 0.2
    left = int(np.clip(center - side / 2, 0, width - side))
    return left, center


class MultiViewHands:
    """VIDEO-mode tracking first, then zoomed/rotated IMAGE-mode retries."""

    attempts_per_check = 3

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0
        self.last_hand_count = 0

    def _search(self, image: mp.Image, timestamp_ms: int, detect, judge):
        """Try retry views; ``judge`` returns a truthy value for a match."""
        if timestamp_ms < self.next_retry_ms:
            return None
        self.next_retry_ms = timestamp_ms + RETRY_INTERVAL_MS
        views = hand_views(image.numpy_view())
        selected = [self.preferred_view] if self.preferred_view in RETRY_VIEWS else []
        while len(selected) < self.attempts_per_check:
            view = RETRY_VIEWS[self.retry_cursor % len(RETRY_VIEWS)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for name, rotation in selected:
            view = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[name], rotation)),
            )
            result = detect(hand_model_image(view))
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            verdict = judge(result, view)
            if verdict:
                self.preferred_view = (name, rotation)
                return verdict
        self.preferred_view = None
        return None

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def classify_pose(result, image: mp.Image) -> str | None:
    """The hand pose in a gesture-model result, most specific first.

    Returns "Joined" (palms together), "Open_Palm", "Victory", "Closed_Fist",
    "Pointing_Up", "ILoveYou", "Thumb_Down", or None. Thumbs up is ignored on
    purpose: people use it all the time in calls.
    """
    hands = result.hand_landmarks
    if joined_hands_from_landmarks(hands, image.width, image.height):
        return "Joined"
    if open_palm_from_landmarks(hands, image.width, image.height):
        return "Open_Palm"
    if classify_gesture_result(result, image)[0] == "Victory":
        return "Victory"
    best = None
    for categories in result.gestures:
        for category in categories:
            if (category.category_name in CANNED_POSES and category.score >= POSE_MIN_SCORE
                    and (best is None or category.score > best[1])):
                best = (category.category_name, category.score)
    return best[0] if best else None


class MultiViewGestureRecognizer(MultiViewHands):
    def recognize_pose(self, image: mp.Image, timestamp_ms: int) -> str | None:
        result = self.primary.recognize_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        pose = classify_pose(result, image)
        if pose is not None:
            return pose
        return self._search(image, timestamp_ms, self.alternate.recognize, classify_pose)


class GestureWorker:
    """Runs gesture checks on their own thread so video frames never wait for them.

    One check is in flight at a time; the video loop submits the latest frame
    whenever the worker is free and polls for results without blocking.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.jobs: queue.Queue = queue.Queue(maxsize=1)
        self.results: queue.Queue = queue.Queue()
        self.busy = False
        self.thread = threading.Thread(target=self._run, name="gestures", daemon=True)
        self.thread.start()

    def submit(self, phase: str, image: mp.Image, timestamp_ms: int, now: float) -> bool:
        if self.busy:
            return False
        self.busy = True
        self.jobs.put((phase, image, timestamp_ms, now))
        return True

    def poll(self):
        """(phase, time, pose, detail) of a finished check, or None."""
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def check(self, image: mp.Image, timestamp_ms: int) -> tuple[str | None, str]:
        pose = self.recognizer.recognize_pose(image, timestamp_ms)
        return pose, f"pose={pose} hands={self.recognizer.last_hand_count}"

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            phase, image, timestamp_ms, now = job
            try:
                pose, detail = self.check(image, timestamp_ms)
            except Exception as error:  # A failed check must not stop the video.
                pose, detail = None, f"error: {error}"
            self.results.put((phase, now, pose, detail))

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=2)


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    sounds = [path for style in STYLES for path in sound_paths(style).values()]
    required = (SEGMENTER_MODEL, GESTURE_MODEL, *sounds)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    segmenter, recognizer = create_tasks()
    try:
        recognizer.close()
    finally:
        segmenter.close()
    say("Self-test passed: models, assets and audio are present.")
    return 0


def run() -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    with ExitStack() as resources:
        return run_camera(args, resources)


def run_camera(
    args: argparse.Namespace,
    resources: ExitStack,
    commands: queue.Queue[str] | None = None,
    on_frame=None,
) -> int:
    """Run one session: capture the empty room once, then react to gestures
    until "quit". ``commands`` feeds control commands; ``on_frame`` receives
    every output frame (used by the Mac app's live thumbnail)."""
    download_models()
    camera_index = args.camera
    capture = open_camera(camera_index, args.width, args.height)
    resources.callback(lambda: capture.release())  # whichever webcam is open at the end
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks()
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    gestures = GestureWorker(recognizer)
    resources.callback(gestures.close)  # stops before the recognizer closes
    sounds = {style: EffectSound(style) for style in STYLES}
    for sound in sounds.values():
        resources.callback(sound.close)
    resources.callback(cv2.destroyAllWindows)
    if commands is None and args.control_stdin:
        commands = start_stdin_commands()
    preview = not args.no_preview
    fullscreen = args.fullscreen
    if preview:
        open_preview(args.width, args.height, fullscreen)

    effect = VanishEffect(args.button_effect)
    hold = PoseHold()
    gestures_on = True
    background: LiveBackground | None = None
    samples: list[np.ndarray] = []
    cover: np.ndarray | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    started = time.perf_counter()
    calibrating = True
    calibration_started = started
    calibration_remaining = None
    last_gesture_check = float("-inf")
    last_gesture_report = 0.0
    last_timestamp = -1
    square_center: float | None = None
    say("Step out of the frame for 4 seconds: capturing the empty room.")
    if preview:
        print("[VanishCam] Preview keys: SPACE new background, V vanish, R reappear, F fullscreen, Q quit.")

    def switch(action: str, now: float, current_cover: np.ndarray, frame: np.ndarray,
               style: str | None = None) -> None:
        if action == "vanish":
            if not effect.vanish(now, current_cover, frame, style):
                return
            sounds[effect.style_name].play("out")
            say(f"Vanished with {EFFECT_NAMES[effect.style_name]}. Show ✋ (or press Reappear) to come back.")
        else:
            if not effect.appear(now, current_cover, frame):
                return
            sounds[effect.style_name].play("in")
            say(f"Back! {GESTURE_HELP}")
        hold.block(now)

    def recalibrate(now: float, message: str) -> None:
        nonlocal calibrating, calibration_started, calibration_remaining
        calibrating, calibration_started, calibration_remaining = True, now, None
        samples.clear()
        effect.reset()
        say(message)

    try:
        while True:
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam stopped sending images.")
            if not args.no_mirror:
                camera_frame = cv2.flip(camera_frame, 1)

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
                elif not preview:
                    raise RuntimeError(
                        "Without the preview window, the OBS Virtual Camera is required."
                    )
            now = time.perf_counter()
            # MediaPipe VIDEO mode needs strictly increasing timestamps.
            timestamp = max(last_timestamp + 1, int((now - started) * 1000))
            last_timestamp = timestamp

            if calibrating:
                elapsed = now - calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining and remaining > 0:
                    say(f"Capturing the empty room: {remaining} s left. Stay out of the frame.")
                calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    stack = np.stack(samples[-18:] or [camera_frame], axis=0)
                    background = LiveBackground(np.median(stack, axis=0).astype(np.uint8))
                    samples.clear()
                    calibrating = False
                    cover = None
                    hold.block(now, 0.5)
                    say(f"Session live. {GESTURE_HELP} to vanish, ✋ to come back.")
                display = camera_frame
            else:
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                cover = cover_mask(person_mask(segmenter, mp_image, timestamp, width, height), cover)
                background.update(camera_frame, cover)
                # Look for hands in a square around the person, not the wide frame.
                square_left, square_center = person_square(cover, width, height, square_center)
                side = min(width, height)
                hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(
                    rgb[:side, square_left:square_left + side]))

                finished = gestures.poll()
                if finished is not None:
                    checked_phase, checked_at, pose, detail = finished
                    # Ignore results for a phase we have already left.
                    if gestures_on and checked_phase == effect.phase == "visible":
                        fired = hold.update(pose if pose in GESTURE_EFFECTS else None, checked_at)
                        if fired:
                            switch("vanish", now, cover, camera_frame, GESTURE_EFFECTS[fired])
                    elif gestures_on and checked_phase == effect.phase == "vanished":
                        if hold.update(pose if pose == REAPPEAR_POSE else None, checked_at):
                            switch("appear", now, cover, camera_frame)
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(f"[Gesture] {detail}", flush=True)
                        last_gesture_report = now
                if (gestures_on and effect.phase in ("visible", "vanished")
                        and now - last_gesture_check >= GESTURE_INTERVAL_SECONDS
                        and gestures.submit(effect.phase, hand_image, timestamp, now)):
                    last_gesture_check = now

                plate = background.image if effect.phase != "visible" else camera_frame
                display = effect.render(camera_frame, plate, cover, now)

            if virtual_camera is not None:
                send_virtual_frame(virtual_camera, display)
            if on_frame is not None:
                on_frame(display)
            command = None
            if preview:
                cv2.imshow(WINDOW_NAME, display)
                command = KEY_COMMANDS.get(cv2.waitKey(1) & 0xFF)
            command = command or next_command(commands)
            name, _, value = (command or "").partition(":")
            if command == "quit":
                break
            if command == "background":
                recalibrate(now, "Step out of the frame for 4 seconds: capturing a new background.")
            elif name in ("vanish", "reappear") and not calibrating:
                current = cover if cover is not None else np.zeros((height, width), np.float32)
                if name == "vanish":
                    switch("vanish", now, current, camera_frame, value or args.button_effect)
                else:
                    switch("appear", now, current, camera_frame)
            elif name == "camera" and int(value) != camera_index:
                try:
                    replacement = open_camera(int(value), args.width, args.height)
                except RuntimeError as error:
                    say(f"{error} Keeping the current webcam.")
                else:
                    capture.release()
                    capture, camera_index = replacement, int(value)
                    capture.set(cv2.CAP_PROP_FPS, args.fps)
                    recalibrate(now, "Switched webcam. Step out of the frame for 4 seconds: "
                                     "capturing the new view.")
            elif name == "gestures":
                gestures_on = value == "on"
                hold.block(now, 0.0)
                say("Gestures on." if gestures_on else "Gestures paused. Use the buttons to vanish and reappear.")
            elif command == "fullscreen" and preview:
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            elif command == "preview":
                preview = not preview
                if preview:
                    open_preview(args.width, args.height, fullscreen)
                else:
                    cv2.destroyWindow(WINDOW_NAME)
                    cv2.waitKey(1)
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[VanishCam] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
