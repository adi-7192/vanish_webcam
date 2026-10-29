"""VanishCam engine: webcam vanish effects sent to the OBS Virtual Camera on macOS."""

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
    GestureHold, LiveBackground, VanishEffect, cover_mask,
    joined_hands_from_landmarks, open_palm_from_landmarks,
)
from vanish_sound import SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "VanishCam Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"

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
CONTROL_COMMANDS = frozenset(KEY_COMMANDS.values()) | {"preview"}

# The OBS camera extension advertises only 1920x1080 to apps but passes on
# whatever size it is sent. Chrome rescales; WhatsApp and other native apps
# freeze on a mismatch, so always send exactly the advertised size.
VIRTUAL_CAMERA_SIZE = (1920, 1080)

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
GESTURE_INTERVAL_SECONDS = 0.14
NARUTO_GESTURE_INTERVAL_SECONDS = 0.08
VICTORY_MIN_SCORE = 0.62
VANISH_POSE = {"normal": "a V sign ✌️", "naruto": "palms together 🙏"}


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("FPS must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VanishCam: Normal or Naruto vanish effects for your webcam."
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
        "--mode", choices=("normal", "naruto"),
        help="Skip the start menu and use this mode",
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Do not open the preview window (useful during video calls)",
    )
    parser.add_argument(
        "--control-stdin", action="store_true",
        help="Read commands (vanish, reappear, background, preview, quit) from stdin, one per line",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print gesture recognition details to the terminal (never on the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_mode() -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    title = "Choose effect"
    selected = [None]
    window = np.zeros((420, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    buttons = (
        ("normal", "1  NORMAL VANISH", "V sign  /  dissolve", 140),
        ("naruto", "2  NARUTO VANISH", "Palms together  /  smoke puff", 260),
    )
    for mode, heading, detail, top in buttons:
        cv2.rectangle(window, (65, top), (635, top + 94), (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (65, top), (635, top + 94), (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (90, top + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.77, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (90, top + 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 65 <= x <= 635:
            for mode, _, _, top in buttons:
                if top <= y <= top + 94:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("1"):
                selected[0] = "normal"
            elif key == ord("2"):
                selected[0] = "naruto"
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


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
            command = line.strip().lower()
            if command in CONTROL_COMMANDS:
                commands.put(command)
            elif command:
                say(f"Unknown command: {command}")
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


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode == "naruto" and destination == GESTURE_MODEL:
            continue
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


def create_tasks(mode: str = "normal") -> tuple[vision.ImageSegmenter, MultiViewGestureRecognizer | MultiViewHandLandmarker]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "naruto":
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=hand_base_options(HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        # No canned-gesture allowlist: MediaPipe drops the landmarks of every
        # hand whose gesture is filtered out, which would hide open palms and
        # the landmark-based V check. classify_gesture_result picks "Victory".
        options = vision.GestureRecognizerOptions(
            base_options=hand_base_options(GESTURE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    return segmenter, wrapper_class(primary, alternate)


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


class MultiViewGestureRecognizer(MultiViewHands):
    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            return best

        def judge(view_result, view):
            candidate = classify_gesture_result(view_result, view)
            return candidate if candidate[0] == "Victory" else None

        found = self._search(image, timestamp_ms, self.alternate.recognize, judge)
        if found:
            return found
        if best[0] == "No_Hand" and self.last_hand_count:
            return "Hand_Detected", 0.0
        return best

    def detect_open_palm(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.recognize_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if open_palm_from_landmarks(result.hand_landmarks, image.width, image.height):
            return True
        judge = lambda found, view: open_palm_from_landmarks(found.hand_landmarks, view.width, view.height)
        return bool(self._search(image, timestamp_ms, self.alternate.recognize, judge))


class MultiViewHandLandmarker(MultiViewHands):
    attempts_per_check = 2

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            return True
        judge = lambda found, view: joined_hands_from_landmarks(found.hand_landmarks, view.width, view.height)
        return bool(self._search(image, timestamp_ms, self.alternate.detect, judge))

    def detect_open_palm(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if open_palm_from_landmarks(result.hand_landmarks, image.width, image.height):
            return True
        judge = lambda found, view: open_palm_from_landmarks(found.hand_landmarks, view.width, view.height)
        return bool(self._search(image, timestamp_ms, self.alternate.detect, judge))



class GestureWorker:
    """Runs gesture checks on their own thread so video frames never wait for them.

    One check is in flight at a time; the video loop submits the latest frame
    whenever the worker is free and polls for results without blocking.
    """

    def __init__(self, recognizer, mode: str):
        self.recognizer = recognizer
        self.mode = mode
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
        """(phase, time, seen, detail) of a finished check, or None."""
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        return result

    def check(self, phase: str, image: mp.Image, timestamp_ms: int) -> tuple[bool, str]:
        if phase == "vanished":
            seen = self.recognizer.detect_open_palm(image, timestamp_ms)
            return seen, f"open_palm={seen}"
        if self.mode == "normal":
            name, score = recognize_gesture(self.recognizer, image, timestamp_ms)
            return name == "Victory" and score >= VICTORY_MIN_SCORE, f"{name} score={score:.2f}"
        seen = self.recognizer.recognize_joined_hands(image, timestamp_ms)
        return seen, f"hands={self.recognizer.last_hand_count} joined={seen}"

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            phase, image, timestamp_ms, now = job
            try:
                seen, detail = self.check(phase, image, timestamp_ms)
            except Exception as error:  # A failed check must not stop the video.
                seen, detail = False, f"error: {error}"
            self.results.put((phase, now, seen, detail))

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=2)


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if isinstance(recognizer, MultiViewGestureRecognizer):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    required = (SEGMENTER_MODEL, GESTURE_MODEL, HAND_MODEL, SOUND_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing resources: {', '.join(missing)}")
    for mode in ("normal", "naruto"):
        segmenter, recognizer = create_tasks(mode)
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
    mode = args.mode or ("normal" if args.no_preview else choose_mode())
    if mode is None:
        return 0
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def run_camera(
    args: argparse.Namespace,
    resources: ExitStack,
    mode: str,
    commands: queue.Queue[str] | None = None,
    on_frame=None,
) -> int:
    """Run the effect loop. ``commands`` feeds control commands; ``on_frame``
    receives every output frame (used by the Mac app's live thumbnail)."""
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    gestures = GestureWorker(recognizer, mode)
    resources.callback(gestures.close)  # stops before the recognizer closes
    sound = VanishSound() if mode == "naruto" else None
    if sound is not None:
        resources.callback(sound.close)
    resources.callback(cv2.destroyAllWindows)
    if commands is None and args.control_stdin:
        commands = start_stdin_commands()
    preview = not args.no_preview
    fullscreen = args.fullscreen
    if preview:
        open_preview(args.width, args.height, fullscreen)

    effect = VanishEffect(mode)
    vanish_hold, appear_hold = GestureHold(), GestureHold()
    gesture_interval = (NARUTO_GESTURE_INTERVAL_SECONDS if mode == "naruto"
                        else GESTURE_INTERVAL_SECONDS)
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
    say("Step out of the frame for 4 seconds: capturing the background.")
    if preview:
        print("[VanishCam] Preview keys: SPACE new background, V vanish, R reappear, F fullscreen, Q quit.")

    def switch(action: str, now: float, current_cover: np.ndarray) -> None:
        start = effect.vanish if action == "vanish" else effect.appear
        if not start(now, current_cover):
            return
        vanish_hold.block(now)
        appear_hold.block(now)
        if sound is not None:
            sound.play()
        if action == "vanish":
            say("Vanished. Show an open palm ✋ (or press Reappear) to come back.")
        else:
            say(f"Back! Show {VANISH_POSE[mode]} to vanish again.")

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
                    say(f"Capturing background: {remaining} s left. Stay out of the frame.")
                calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    stack = np.stack(samples[-18:] or [camera_frame], axis=0)
                    background = LiveBackground(np.median(stack, axis=0).astype(np.uint8))
                    samples.clear()
                    calibrating = False
                    cover = None
                    effect.reset()
                    vanish_hold.block(now, 0.5)
                    appear_hold.block(now, 0.5)
                    say(f"Ready. Step back in and show {VANISH_POSE[mode]} to vanish.")
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
                    checked_phase, checked_at, seen, detail = finished
                    # Ignore results for a phase we have already left.
                    if checked_phase == effect.phase == "visible":
                        if vanish_hold.update(seen, checked_at):
                            switch("vanish", now, cover)
                    elif checked_phase == effect.phase == "vanished":
                        if appear_hold.update(seen, checked_at):
                            switch("appear", now, cover)
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(f"[Gesture] {detail}", flush=True)
                        last_gesture_report = now
                if (effect.phase in ("visible", "vanished")
                        and now - last_gesture_check >= gesture_interval
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
            if command == "quit":
                break
            if command == "background":
                calibrating = True
                calibration_started = now
                calibration_remaining = None
                samples.clear()
                effect.reset()
                say("Step out of the frame for 4 seconds: capturing a new background.")
            elif command in ("vanish", "reappear") and not calibrating:
                current = cover if cover is not None else np.zeros((height, width), np.float32)
                switch("vanish" if command == "vanish" else "appear", now, current)
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
