# VanishCam for Mac

Vanish from your webcam in Google Meet, Zoom, WhatsApp Web and other video calls. Make a gesture and you dissolve into the background or disappear in a puff of smoke. Show an open palm and you come back.

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

1. Open **VanishCam** and press **Start**. That starts a session.
2. Step out of the frame for 4 seconds while the empty room is captured. This happens once per session.
3. In Meet, Zoom or WhatsApp Web, choose **OBS Virtual Camera** as your camera. Each app remembers this. While you are visible it shows your normal webcam, so you can leave it selected.
4. From now on, just use gestures. Each vanish gesture plays its own effect, and an open palm ✋ brings you back with the same effect in reverse. No menus, no new captures.
5. Press **Stop** when you are done. Nothing else ends the session: changing the Vanish button's effect or the webcam keeps it running, and the camera stays live in every call app.

The panel floats above your call and never takes focus away from it. Close it any time and reopen it from the menu-bar icon, which has the same controls.

### Gestures

| Gesture | Effect |
|---|---|
| ✌️ V sign | **Dissolve**: you dissolve away with a glowing edge |
| ✊ Fist | **Dust**: you crumble into particles that blow away |
| ☝️ Point up | **Portal**: a glowing ring opens and you spiral into it |
| 🤟 Rock / I-love-you | **Glitch**: you break into slices and split colours, then flicker out |
| 👎 Thumbs down | **Teleport**: a shimmering column of light beams you out |
| 🙏 Palms together, fingers side by side | **Smoke puff**: you vanish in a puff of smoke |
| ✋ Open palm | **Come back**, reversing whichever effect you vanished with |

👍 is deliberately not used, because people use it all the time in calls. Hold a gesture for about half a second; quick, accidental hand shapes do nothing, and there is a short pause after each switch. Untick **Gestures on** (or choose **Pause gestures** in the menu bar) when you want to talk with your hands; the **Vanish** and **Reappear** buttons keep working, and the **Vanish button** menu chooses the effect they use.

Each effect has its own sound, played forwards when you vanish and backwards when you come back. The sounds are generated in code (`packaging/make_sounds.py`); the smoke puff uses the original `assets/vanish.MP3`. They play on your speakers, and can also go into the call (see below).

While you are invisible, the rest of the room stays live: only the area where you are is filled in with the background, which keeps updating as the light changes.

### Tips

- Capture the background with good, steady light. If you move the camera or furniture, press **New background**.
- Other people who walk into the frame while you are invisible are hidden too.
- If the thumbnail shows the wrong camera, choose another under **Webcam** in the menu-bar icon. The session keeps running; you step out once more so the new view's background can be captured.
- **WhatsApp:** use WhatsApp Web in Chrome (web.whatsapp.com), which supports video calls. The WhatsApp desktop app freezes on the OBS camera because it only accepts the 420v pixel format, which OBS does not provide ([details](https://github.com/kortexa-ai/aicamera/issues/71)). Calls from your phone always use the phone's camera.
- Your self-view in Meet looks mirrored, as usual. Other people see you the right way round.

### Sound into the call

The other person can hear the effect sounds too:

1. Install [BlackHole](https://existential.audio/blackhole/), a free virtual audio cable: `brew install --cask blackhole-2ch` (asks for your password).
2. In VanishCam, tick **Sound into call** at the bottom of the panel (or in the menu-bar icon). Allow microphone access when macOS asks.
3. In Meet or Zoom, choose **BlackHole 2ch** as your **microphone**.

VanishCam now passes your real microphone to BlackHole and mixes the effect sounds in, adding only a few milliseconds of delay. Keep VanishCam open while the call uses BlackHole: if you quit it, the call hears silence. When you are done, untick the box and switch the call's microphone back. Headphones avoid the effect sounds also reaching your microphone from your speakers.

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
.venv/bin/python -m unittest test_call_audio test_effects test_hands test_vanish_cam test_video_output
```

`vanish_cam.py` preview keys: `SPACE` new background, `V` vanish, `R` reappear, `F` fullscreen, `Q` quit. Other options: `--button-effect normal|dust|portal|glitch|beam|naruto`, `--camera N`, `--width`, `--height`, `--fps`, `--no-preview --control-stdin` (read commands such as `vanish:dust`, `reappear`, `camera:1`, `gestures:off` from stdin) and `--gesture-debug`.

## Credits

Based on [VanishCam by Nicovic17](https://github.com/Nicovic17/vanish_webcam), which created the original effects, gesture recognition and assets. This fork packages it as a macOS app.

Effects are drawn in code (`effects.py`); no anime frames or sprites are included. Gesture and segmentation models are from [Google MediaPipe](https://ai.google.dev/edge/mediapipe).
