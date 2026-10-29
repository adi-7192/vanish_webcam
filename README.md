# VanishCam — webcam disappearance effects

Video is processed on your computer. It is neither recorded nor sent to a server. The preview window and virtual camera show only the video and effects, without overlaid menus, instructions, or text.

For step-by-step instructions in English and Italian, see the [user guide](USER_GUIDE_EN_IT.md).

## macOS: use it in Google Meet, WhatsApp, FaceTime or Zoom

This fork adds macOS support. VanishCam sends its video to the **OBS Virtual Camera**, a system-wide webcam that every Mac app can select.

**One-time setup**

1. Install Python 3.12 (`brew install python@3.12`). MediaPipe does not support newer Python versions yet.
2. Install [OBS Studio](https://obsproject.com/) 30 or later (`brew install --cask obs`). Open it, click **Start Virtual Camera**, approve the system extension in System Settings (restart if asked), then quit OBS. You only need OBS for this one-time step.
3. Double-click **`Start VanishCam (Mac).command`**. The first run installs the components. If macOS blocks it, right-click it and choose **Open**.
4. Allow camera access for Terminal when macOS asks (System Settings > Privacy & Security > Camera).

**Every call**

1. Double-click `Start VanishCam (Mac).command`. A ◌ icon appears in the menu bar. Leave the Terminal window open; closing it quits VanishCam.
2. From the ◌ menu, choose **Normal Vanish** or **Naruto Vanish**. The icon turns to ◉.
3. Step out of the frame for 4 seconds while the background is captured. The first menu line shows the progress.
4. In Meet, WhatsApp, FaceTime or Zoom, choose **OBS Virtual Camera** as the camera. Each app remembers this choice. While no effect is running, the virtual camera shows your normal webcam, so you can leave it selected.
5. Make the gesture to vanish, or use **Vanish now** in the menu. Use **Reappear** to come back. **Re-capture background** starts the 4-second capture again. **Show/hide preview** opens a local window.

**Notes**

- WhatsApp works through the WhatsApp desktop app for Mac. Calls placed from your phone always use the phone's camera.
- The Naruto sound plays on your speakers, not into the call.
- If the wrong camera is used, pick another one under **Webcam** in the menu.
- To run without the menu bar: `.venv/bin/python vanish_cam.py --virtual-camera` (with a preview window and keyboard controls), or add `--no-preview --control-stdin` and type `vanish`, `reappear`, `background`, `preview` or `quit`.
- The Particles and Goku modes described below are not in the code yet.

## ☕ Buy me a coffee

If you like this project and want to support it, you can buy me a coffee through [PayPal](https://paypal.me/nicovic17) or [Revolut](https://revolut.me/nicoauriemma).

## Download for Windows

[Download VanishCam.exe]( https://nicoaauriemma.vercel.app/shop ) and double-click it to run the app. You do not need to install Python.

The published EXE may predate the Particles and Goku modes. To use them now, run the local scripts or rebuild the EXE from these sources.

## Getting started

1. Double-click `AVVIA_LOCALE.bat`.
2. In the start menu, choose **Normal Vanish**, **Naruto Vanish**, **Particles**, or **Goku** with the mouse or the `1`, `2`, `3`, and `4` keys.
3. Step out of the camera frame for 4 seconds while the background is captured. The terminal tells you when you can step back in.

**Normal Vanish:** Raise your index and middle fingers in a V sign. The dissolve starts as soon as the gesture is recognized and lasts about 3.6 seconds.

**Particles:** Make the same V sign as in Normal Vanish. Your silhouette breaks into small colored fragments that drift sideways as if carried by the wind. The effect lasts about 2.1 seconds. It captures your silhouette once when the effect starts to keep the video smooth.

**Goku:** Hold your index and middle fingers straight and together against your forehead for 1 second. Animated vertical comic-style lines run across your silhouette just before you teleport away and linger briefly afterward. To reappear, lower your hand, then repeat the gesture for another 1 second. The return uses the reverse effect. If you break the pose, the countdown starts over. `assets/teleport.MP3` plays on both disappearance and reappearance. The terminal shows the countdown for each gesture. The `V` and `R` keys are disabled in this mode: the gesture is the only way to disappear and reappear. No text is overlaid on the video.

**Naruto Vanish:** Bring your palms together in front of the webcam, with your fingers side by side rather than interlaced. Keep both hands visible. As soon as the gesture is recognized, you vanish in a cloud of smoke and a wooden log appears, falls, and remains visible until you press `R`. No microphone is needed. Naruto mode uses `hand_landmarker.task` to track both hands. `assets/vanish.MP3` plays through your computer speakers when the disappearance starts, including when triggered with `V`. You can trigger it again after pressing `R`.

To start a mode without the menu:

```powershell
AVVIA_LOCALE.bat --mode naruto
AVVIA_LOCALE.bat --mode normal
AVVIA_LOCALE.bat --mode particles
AVVIA_LOCALE.bat --mode goku
```

## Virtual camera and OBS

If [OBS Studio](https://obsproject.com/) is installed, the app automatically tries to open **OBS Virtual Camera**. The terminal shows the device name. Use `AVVIA_VIRTUAL.bat` to require a virtual camera. Select **OBS Virtual Camera** in the app that will receive the video. OBS's virtual camera must be stopped while this app is producing its output.

To use the effect in an OBS scene, run `AVVIA_OBS.bat`, add a **Window Capture** source, select **Webcam - Anteprima**, and then start the virtual camera from OBS. Alternatively, use [Unity Capture](https://github.com/schellingb/UnityCapture) as a separate virtual video source:

```powershell
AVVIA_VIRTUAL.bat --virtual-backend unitycapture
```

OBS has only one virtual camera. Do not use it as both an OBS input and output at the same time. See the [pyvirtualcam documentation](https://github.com/letmaik/pyvirtualcam#supported-virtual-cameras).

## Controls

- `SPACE`: capture a new background.
- `R`: make the person reappear.
- `V`: manually start the selected mode's effect.
- `F`: toggle fullscreen.
- `Q` or `ESC`: quit and release the cameras.

The `V` and `R` keys are disabled in Goku mode.

You can change the camera, resolution, and maximum virtual-output FPS:

```powershell
AVVIA_LOCALE.bat --camera 1 --width 1280 --height 720 --fps 30
```

To troubleshoot gesture recognition without adding text to the video, pass `--gesture-debug`. Diagnostic information appears only in the terminal.

## Visual assets

The [wooden log](assets/substitution_log.png) is an original illustration generated with ImageGen using the supplied image as a visual reference. Smoke is drawn in real time by `naruto_effect.py` so it follows the person's position and scales with the video. We also considered a [CC0 smoke sprite sheet](https://opengameart.org/content/smoke-sprite-sheet), but the procedural animation adapts better to dynamic positioning without an additional asset. No anime frames or assets are included in the project.

Goku mode uses the [MediaPipe BlazeFace short-range model](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md) to estimate the forehead position and `hand_landmarker.task` to track the fingers. The teleport's visual trace is drawn in code, without anime frames.
