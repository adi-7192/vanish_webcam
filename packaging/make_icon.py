"""Draw the VanishCam app icon and build packaging/VanishCam.icns."""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

SIZE = 1024
OUT = Path(__file__).resolve().parent / "VanishCam.icns"


def draw() -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    top, bottom = np.array([120, 60, 40]), np.array([60, 20, 110])  # BGR
    image = (top * (1 - yy[..., None]) + bottom * yy[..., None]).astype(np.uint8)
    image = np.dstack([image, np.zeros((SIZE, SIZE), np.uint8)])
    # macOS icon grid: ~824px rounded square centred in 1024.
    mask = np.zeros((SIZE, SIZE), np.uint8)
    margin, radius = 100, 185
    cv2.rectangle(mask, (margin + radius, margin), (SIZE - margin - radius, SIZE - margin), 255, -1)
    cv2.rectangle(mask, (margin, margin + radius), (SIZE - margin, SIZE - margin - radius), 255, -1)
    for cx in (margin + radius, SIZE - margin - radius):
        for cy in (margin + radius, SIZE - margin - radius):
            cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    image[..., 3] = mask
    # Person silhouette that dissolves into particles towards the right.
    person = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(person, (512, 400), 120, 255, -1, cv2.LINE_AA)
    cv2.ellipse(person, (512, 790), (250, 230), 0, 180, 360, 255, -1, cv2.LINE_AA)
    person[mask == 0] = 0
    rng = np.random.default_rng(7)
    fade = np.clip((xx - 0.42) / 0.30, 0, 1)
    keep = rng.random((SIZE, SIZE)) > fade
    solid = (person > 0) & keep
    image[solid, :3] = (245, 240, 235)
    for _ in range(260):
        x, y = rng.integers(560, 860), rng.integers(260, 820)
        if mask[y, x]:
            r = int(rng.integers(5, 16))
            cv2.circle(image, (int(x), int(y)), r, (250, 230, 210, 255), -1, cv2.LINE_AA)
    return image


def main() -> None:
    icon = draw()
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "VanishCam.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                pixels = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                cv2.imwrite(str(iconset / name), cv2.resize(icon, (pixels, pixels), interpolation=cv2.INTER_AREA))
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(OUT)], check=True)
    cv2.imwrite(str(OUT.with_suffix(".png")), cv2.resize(icon, (256, 256), interpolation=cv2.INTER_AREA))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
