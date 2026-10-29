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
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
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
    say(f"Live on {camera.device} ({width}x{height}, up to {args.fps:g} FPS)")
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


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


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
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=hand_base_options(GESTURE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
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


class MultiViewGestureRecognizer:
    """Retry detection on rotated and cropped views without disturbing tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(hand_model_image(image), timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE mode forces a fresh detection; the VIDEO model stays on the
        # original frame, avoiding inconsistent timestamps across rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # A sideways palm is often too small in the full frame. The two lower
        # crops overlap and cover both hands, even when the image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(hand_model_image(rotated))
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def detect_open_palm(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.recognize_for_video(hand_model_image(image), timestamp_ms)
        return open_palm_from_landmarks(result.hand_landmarks, image.width, image.height)

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(hand_model_image(rotated))
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def detect_open_palm(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(hand_model_image(image), timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        return open_palm_from_landmarks(result.hand_landmarks, image.width, image.height)

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


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

                if effect.phase in ("visible", "vanished") and now - last_gesture_check >= gesture_interval:
                    last_gesture_check = now
                    if effect.phase == "visible":
                        if mode == "normal":
                            name, score = recognize_gesture(recognizer, mp_image, timestamp)
                            seen = name == "Victory" and score >= VICTORY_MIN_SCORE
                            detail = f"{name} score={score:.2f}"
                        else:
                            seen = recognizer.recognize_joined_hands(mp_image, timestamp)
                            detail = f"hands={recognizer.last_hand_count} joined={seen}"
                        if vanish_hold.update(seen, now):
                            switch("vanish", now, cover)
                    else:
                        seen = recognizer.detect_open_palm(mp_image, timestamp)
                        detail = f"open_palm={seen}"
                        if appear_hold.update(seen, now):
                            switch("appear", now, cover)
                    if args.gesture_debug and now - last_gesture_report >= 1.0:
                        print(f"[Gesture] {detail}", flush=True)
                        last_gesture_report = now

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
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
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
