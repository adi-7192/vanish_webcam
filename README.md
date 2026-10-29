# VanishCam for Mac

Vanish from your webcam in Google Meet, WhatsApp, FaceTime or Zoom. Make a gesture and you dissolve into the background, or disappear in a cloud of smoke and leave a wooden log behind.

Video is processed on your Mac. It is never recorded or sent anywhere.

<img src="packaging/VanishCam.png" width="96" alt="VanishCam icon">

## How it works

VanishCam reads your real webcam, adds the effect, and sends the result to the **OBS Virtual Camera**, a system-wide webcam that every Mac app can select. A small floating panel and a menu-bar icon let you control it without leaving your call.

## One-time setup

1. **Install OBS Studio 30 or later** from [obsproject.com](https://obsproject.com/download) or with `brew install --cask obs`.
2. **Turn on its virtual camera once.** Open OBS and click **Start Virtual Camera**. When macOS asks, open System Settings > General > Login Items & Extensions and enable **OBS Virtual Camera** under Camera Extensions. Then quit OBS. VanishCam does not need OBS to be running.
3. **Install VanishCam.** Build it (see below) and drag `dist/VanishCam.app` to your Applications folder. The first time, right-click the app and choose **Open**, because it is not signed by an Apple developer.
4. **Allow camera access** when macOS asks.

## Using it in a call

1. Open **VanishCam**. The floating panel and the menu-bar icon appear.
2. Pick **Normal** or **Naruto**, then press **Start**.
3. Step out of the frame for 4 seconds while the background is captured. The panel tells you when to step back in.
4. In Meet, WhatsApp, FaceTime or Zoom, choose **OBS Virtual Camera** as your camera. Each app remembers this. While no effect is running, it shows your normal webcam, so you can leave it selected.
5. Make the gesture, or press **Vanish**. Press **Reappear** to come back.

The panel floats above your call and never takes focus away from it. Close it any time and reopen it from the menu-bar icon, which has the same controls.

### Gestures

- **Normal:** raise your index and middle fingers in a V sign. You dissolve into the background in about 3.6 seconds.
- **Naruto:** bring your palms together in front of the webcam, fingers side by side rather than interlaced, with both hands visible. You vanish in a cloud of smoke and a wooden log falls into your place. A sound plays on your speakers.

### Tips

- Use **New background** whenever the room or lighting changes. A still background works best.
- If the thumbnail shows the wrong camera, choose another under **Webcam** in the menu-bar icon.
- WhatsApp works through the WhatsApp desktop app. Calls placed from your phone always use the phone's camera.
- The Naruto sound plays on your speakers, not into the call.
- Your self-view in Meet looks mirrored, as usual. Other people see you the right way round.

## Build the app

Requires macOS 13+ on Apple Silicon and Python 3.12 (`brew install python@3.12`).

```bash
packaging/build_app.sh
open dist/VanishCam.app
```

The script creates `.venv`, installs the dependencies, downloads the MediaPipe models, and builds `dist/VanishCam.app` with everything bundled, so the app works offline.

## Run from source

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                          # menu-bar app with the floating panel
.venv/bin/python vanish_cam.py --virtual-camera  # engine with a preview window
.venv/bin/python -m unittest test_vanish_cam test_naruto_effect test_video_output
```

`vanish_cam.py` preview keys: `SPACE` new background, `V` vanish, `R` reappear, `F` fullscreen, `Q` quit. Other options: `--mode normal|naruto`, `--camera N`, `--width`, `--height`, `--fps`, `--no-preview --control-stdin` (read commands from stdin) and `--gesture-debug`.

## Credits

Based on [VanishCam by Nicovic17](https://github.com/Nicovic17/vanish_webcam), which created the original effects, gesture recognition and assets. This fork packages it as a macOS app.

The [wooden log](assets/substitution_log.png) is an original illustration; see [assets/README.md](assets/README.md). Smoke is drawn in code by `naruto_effect.py`. No anime frames are included. Gesture and segmentation models are from [Google MediaPipe](https://ai.google.dev/edge/mediapipe).
