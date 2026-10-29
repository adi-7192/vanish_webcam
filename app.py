"""VanishCam for macOS: menu-bar app with a floating control panel.

The effect engine (vanish_cam.run_camera) runs on a background thread and sends
its video to the OBS Virtual Camera. Pick "OBS Virtual Camera" once in Google
Meet, Zoom or WhatsApp Web; while no effect runs it shows your normal
webcam, so it can stay selected.
"""

from __future__ import annotations

import argparse
import queue
import subprocess
import sys
import threading
import time
from contextlib import ExitStack

import cv2
import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor,
    NSFloatingWindowLevel, NSFont, NSImage, NSImageScaleProportionallyUpOrDown,
    NSImageView, NSMakeRect, NSMenu, NSMenuItem, NSObject, NSPanel, NSSegmentedControl,
    NSStatusBar, NSTextField, NSButton, NSURL, NSUserDefaults, NSVariableStatusItemLength,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskClosable, NSWindowStyleMaskHUDWindow, NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled, NSWindowStyleMaskUtilityWindow, NSWorkspace,
)
from Foundation import NSData
from PyObjCTools import AppHelper

import vanish_cam

MODES = (("normal", "Normal  ✌️"), ("naruto", "Naruto  🙏"))
CAMERAS = range(4)
THUMBNAIL_FPS = 8
EXTENSION_SETTINGS = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
OBS_EXTENSION_ID = "com.obsproject.obs-studio.mac-camera-extension"


def obs_camera_state() -> str:
    """'ready', 'needs_approval' or 'missing' for the OBS camera extension."""
    try:
        listing = subprocess.run(
            ["systemextensionsctl", "list"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "ready"  # Cannot tell; let the engine report the real error.
    lines = [line for line in listing.splitlines() if OBS_EXTENSION_ID in line]
    if not lines:
        return "missing"
    if any("[activated enabled]" in line for line in lines):
        return "ready"
    return "needs_approval"


class Engine:
    """Runs vanish_cam.run_camera on a worker thread."""

    def __init__(self, on_frame, on_stopped) -> None:
        self.on_frame = on_frame
        self.on_stopped = on_stopped
        self.thread: threading.Thread | None = None
        self.commands: queue.Queue[str] | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, mode: str, camera: int) -> None:
        self.stop()
        self.commands = queue.Queue()
        args = argparse.Namespace(
            camera=camera, width=960, height=540, fps=30.0, fullscreen=False,
            # Meet/WhatsApp mirror your self-view themselves; others must see you unmirrored.
            no_mirror=True, virtual_camera=True, mode=mode, gesture_debug=False,
            no_preview=True, control_stdin=False,
        )
        self.thread = threading.Thread(
            target=self._run, args=(args, mode, self.commands), name="vanish-engine", daemon=True
        )
        self.thread.start()

    def _run(self, args, mode, commands) -> None:
        error = None
        try:
            with ExitStack() as resources:
                vanish_cam.run_camera(args, resources, mode, commands, self.on_frame)
        except Exception as exc:  # Shown to the user in the panel.
            error = str(exc)
        self.on_stopped(threading.current_thread(), error)

    def send(self, command: str) -> bool:
        if not self.running:
            return False
        self.commands.put(command)
        return True

    def stop(self) -> None:
        if self.running:
            self.commands.put("quit")
            self.thread.join(timeout=5)
        self.thread = None


class Controller(NSObject):
    def init(self):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        defaults = NSUserDefaults.standardUserDefaults()
        self.mode = defaults.stringForKey_("mode") or "normal"
        self.camera = int(defaults.integerForKey_("camera"))
        self.last_thumbnail = 0.0
        self.engine = Engine(self.frameReady, self.engineStopped)
        vanish_cam.STATUS_LISTENERS.append(
            lambda message: AppHelper.callAfter(self.setStatus_, message)
        )
        self.buildPanel()
        self.buildStatusItem()
        self.refresh()
        self.showSetupStateIfNeeded()
        return self

    # ---- UI construction -------------------------------------------------

    @objc.python_method
    def buildPanel(self) -> None:
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskUtilityWindow
                 | NSWindowStyleMaskHUDWindow | NSWindowStyleMaskNonactivatingPanel)
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 280, 404), style, NSBackingStoreBuffered, False
        )
        panel.setTitle_("VanishCam")
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)  # Clicking never steals focus from the call.
        panel.setHidesOnDeactivate_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setFrameAutosaveName_("VanishCamPanel")
        if not panel.setFrameUsingName_("VanishCamPanel"):
            panel.center()
        view = panel.contentView()

        self.preview = NSImageView.alloc().initWithFrame_(NSMakeRect(10, 248, 260, 146))
        self.preview.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.preview.setWantsLayer_(True)
        self.preview.layer().setCornerRadius_(8)
        self.preview.layer().setMasksToBounds_(True)
        view.addSubview_(self.preview)
        self.placeholder = NSTextField.labelWithString_("Camera off")
        self.placeholder.setFrame_(NSMakeRect(10, 310, 260, 20))
        self.placeholder.setAlignment_(1)  # NSTextAlignmentCenter
        self.placeholder.setTextColor_(NSColor.secondaryLabelColor())
        view.addSubview_(self.placeholder)

        self.status = NSTextField.wrappingLabelWithString_("")
        self.status.setFrame_(NSMakeRect(10, 196, 260, 46))
        self.status.setFont_(NSFont.systemFontOfSize_(11))
        view.addSubview_(self.status)

        self.segments = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_(
            [label for _, label in MODES], 0, self, "modeChanged:"
        )
        self.segments.setFrame_(NSMakeRect(10, 160, 260, 28))
        view.addSubview_(self.segments)

        self.start_button = self.button("Start", "toggleEngine:", (10, 118, 260, 34))
        self.vanish_button = self.button("Vanish", "vanish:", (10, 78, 126, 34))
        self.reappear_button = self.button("Reappear", "reappear:", (144, 78, 126, 34))
        self.background_button = self.button("New background", "background:", (10, 42, 260, 30))
        self.setup_button = self.button("Camera setup help", "openSetup:", (10, 8, 260, 28))
        for control in (self.start_button, self.vanish_button, self.reappear_button,
                        self.background_button, self.setup_button):
            view.addSubview_(control)
        self.start_button.setKeyEquivalent_("\r")
        self.panel = panel
        panel.orderFrontRegardless()

    @objc.python_method
    def button(self, title, action, frame):
        button = NSButton.buttonWithTitle_target_action_(title, self, action)
        button.setFrame_(NSMakeRect(*frame))
        return button

    @objc.python_method
    def buildStatusItem(self) -> None:
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        menu = NSMenu.alloc().init()
        self.menu_start = self.menuItem(menu, "Start", "toggleEngine:")
        menu.addItem_(NSMenuItem.separatorItem())
        self.menuItem(menu, "Vanish", "vanish:", "v")
        self.menuItem(menu, "Reappear", "reappear:", "r")
        self.menuItem(menu, "New background", "background:", "b")
        menu.addItem_(NSMenuItem.separatorItem())
        self.mode_items = {}
        for mode, label in MODES:
            item = self.menuItem(menu, label.split()[0] + " mode", "pickMode:")
            item.setRepresentedObject_(mode)
            self.mode_items[mode] = item
        webcam = NSMenu.alloc().init()
        self.camera_items = {}
        for index in CAMERAS:
            item = self.menuItem(webcam, f"Camera {index + 1}", "pickCamera:")
            item.setTag_(index)
            self.camera_items[index] = item
        webcam_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Webcam", None, "")
        webcam_item.setSubmenu_(webcam)
        menu.addItem_(webcam_item)
        menu.addItem_(NSMenuItem.separatorItem())
        self.menuItem(menu, "Show controls", "showPanel:")
        self.menuItem(menu, "Camera setup help", "openSetup:")
        menu.addItem_(NSMenuItem.separatorItem())
        self.menuItem(menu, "Quit VanishCam", "quit:", "q")
        self.status_item.setMenu_(menu)

    @objc.python_method
    def menuItem(self, menu, title, action, key=""):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key)
        item.setTarget_(self)
        menu.addItem_(item)
        return item

    # ---- State -----------------------------------------------------------

    @objc.python_method
    def refresh(self) -> None:
        running = self.engine.running
        symbol = "eye.slash.circle.fill" if running else "eye.slash.circle"
        icon = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, "VanishCam")
        icon.setTemplate_(True)
        self.status_item.button().setImage_(icon)
        self.placeholder.setHidden_(running)
        self.start_button.setTitle_("Stop" if running else "Start")
        self.menu_start.setTitle_("Stop" if running else "Start")
        for button in (self.vanish_button, self.reappear_button, self.background_button):
            button.setEnabled_(running)
        self.segments.setSelectedSegment_([m for m, _ in MODES].index(self.mode))
        for mode, item in self.mode_items.items():
            item.setState_(1 if mode == self.mode else 0)
        for index, item in self.camera_items.items():
            item.setState_(1 if index == self.camera else 0)
        if not running:
            self.preview.setImage_(None)

    def setStatus_(self, message) -> None:
        self.status.setStringValue_(message)

    @objc.python_method
    def showSetupStateIfNeeded(self) -> None:
        state = obs_camera_state()
        if state == "missing":
            self.setStatus_("One-time setup: install OBS Studio, click 'Start Virtual Camera' "
                            "once, then quit OBS. Click 'Camera setup help'.")
        elif state == "needs_approval":
            self.setStatus_("Approve 'OBS Virtual Camera' in System Settings (Camera "
                            "Extensions). Click 'Camera setup help'.")
        else:
            self.setStatus_("Press Start, then pick 'OBS Virtual Camera' as the camera in "
                            "Meet, Zoom or WhatsApp Web (in Chrome).")

    @objc.python_method
    def savePreferences(self) -> None:
        defaults = NSUserDefaults.standardUserDefaults()
        defaults.setObject_forKey_(self.mode, "mode")
        defaults.setInteger_forKey_(self.camera, "camera")

    # ---- Engine callbacks (worker thread) --------------------------------

    @objc.python_method
    def frameReady(self, frame) -> None:
        now = time.perf_counter()
        if now - self.last_thumbnail < 1 / THUMBNAIL_FPS:
            return
        self.last_thumbnail = now
        thumbnail = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            data = jpeg.tobytes()
            AppHelper.callAfter(self.showThumbnail_, data)

    def showThumbnail_(self, data) -> None:
        if self.engine.running:
            image = NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(data, len(data)))
            self.preview.setImage_(image)

    @objc.python_method
    def engineStopped(self, thread, error) -> None:
        AppHelper.callAfter(self.engineFinished_, (thread, error))

    def engineFinished_(self, payload) -> None:
        thread, error = payload
        if self.engine.thread is thread:
            self.engine.thread = None
        if error:
            self.setStatus_(f"Stopped: {error}")
        self.refresh()

    # ---- Actions ---------------------------------------------------------

    @objc.python_method
    def startEngine(self) -> None:
        self.setStatus_("Starting… loading the models.")
        self.engine.start(self.mode, self.camera)
        self.refresh()

    @objc.IBAction
    def toggleEngine_(self, sender) -> None:
        if self.engine.running:
            self.engine.stop()
            self.setStatus_("Stopped. The virtual camera shows the OBS placeholder.")
            self.refresh()
        else:
            self.startEngine()

    @objc.IBAction
    def vanish_(self, sender) -> None:
        self.engine.send("vanish")

    @objc.IBAction
    def reappear_(self, sender) -> None:
        self.engine.send("reappear")

    @objc.IBAction
    def background_(self, sender) -> None:
        self.engine.send("background")

    @objc.python_method
    def changeMode(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self.savePreferences()
        if self.engine.running:
            self.startEngine()
        self.refresh()

    @objc.IBAction
    def modeChanged_(self, sender) -> None:
        self.changeMode(MODES[sender.selectedSegment()][0])

    @objc.IBAction
    def pickMode_(self, sender) -> None:
        self.changeMode(sender.representedObject())

    @objc.IBAction
    def pickCamera_(self, sender) -> None:
        self.camera = sender.tag()
        self.savePreferences()
        if self.engine.running:
            self.startEngine()
        self.refresh()

    @objc.IBAction
    def showPanel_(self, sender) -> None:
        self.panel.orderFrontRegardless()

    @objc.IBAction
    def openSetup_(self, sender) -> None:
        state = obs_camera_state()
        if state == "missing":
            url = "https://obsproject.com/download"
        else:
            url = EXTENSION_SETTINGS
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))
        self.showSetupStateIfNeeded()

    @objc.IBAction
    def quit_(self, sender) -> None:
        self.engine.stop()
        NSApplication.sharedApplication().terminate_(None)


def main() -> None:
    if "--self-test" in sys.argv:
        raise SystemExit(vanish_cam.self_test())
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    controller = Controller.alloc().init()
    app.setDelegate_(controller)
    AppHelper.runEventLoop()


if __name__ == "__main__":
    main()
