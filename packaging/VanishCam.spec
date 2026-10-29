# PyInstaller spec for the macOS VanishCam.app (run via packaging/build_app.sh).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path(SPECPATH).resolve().parent
resources = [
    (root / "hand_landmarker.task", "."),
    (root / "models" / "gesture_recognizer.task", "models"),
    (root / "models" / "selfie_segmenter_landscape.tflite", "models"),
    (root / "assets" / "vanish.MP3", "assets"),
]
missing = [str(path) for path, _ in resources if not path.is_file()]
if missing:
    raise SystemExit("Missing build inputs:\n" + "\n".join(missing))

a = Analysis(
    [str(root / "app.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(path), dest) for path, dest in resources] + collect_data_files("mediapipe"),
    hiddenimports=["mediapipe.tasks.c", "pyvirtualcam"],
    excludes=["tkinter", "pygame", "matplotlib"],
    runtime_hooks=[str(root / "packaging" / "rth_no_matplotlib.py")],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VanishCam",
    console=False,
    argv_emulation=False,
    target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, name="VanishCam")
app = BUNDLE(
    coll,
    name="VanishCam.app",
    icon=str(root / "packaging" / "VanishCam.icns"),
    bundle_identifier="io.github.adi-7192.vanishcam",
    version="1.0.0",
    info_plist={
        "CFBundleDisplayName": "VanishCam",
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
        "NSCameraUsageDescription": "VanishCam reads your webcam to add vanish effects before sending the video to your calls.",
        "NSHighResolutionCapable": True,
    },
)
