from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
import time
import json
import os
import glob
import threading
import subprocess
from collections import deque

# Tuning is saved here so it survives a server restart. Anchored to this
# script's own folder so it's found no matter where the server is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "turret_config.json")
PRESETS_FILE = os.path.join(SCRIPT_DIR, "turret_presets.json")

app = Flask(__name__)
CORS(app)   # lets the dashboard (running on a different port) read our data

# Pose models the dashboard is allowed to switch between. Bigger = steadier
# keypoints but slower. On the Jetson you'd point these at .engine files.
ALLOWED_MODELS = ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11n.pt"]


def available_models():
    """Base pose/detection models plus any *.pt / *.engine dropped into the
    script folder — so a downloaded drone model shows up in the dashboard
    automatically. Only local files are ever loadable (no arbitrary paths)."""
    found = [os.path.basename(p)
             for p in glob.glob(os.path.join(SCRIPT_DIR, "*.pt"))
             + glob.glob(os.path.join(SCRIPT_DIR, "*.engine"))]
    return list(dict.fromkeys(ALLOWED_MODELS + sorted(found)))

# Which body part the turret aims at. Values map to COCO pose keypoint indices.
AIM_MODES = {
    "head": [0, 1, 2],   # nose, left eye, right eye
    "chest": [5, 6],     # left shoulder, right shoulder
    "hand": [9, 10],     # left wrist, right wrist
}

PREDICTORS = ["none", "ema", "kalman"]

# Everything the dashboard can tune live. POST /config to change it.
config = {
    "model": "yolo11n-pose.pt",
    "aim": "chest",          # head | chest | hand (pose models only)
    "target_class": 0,       # class index to track (pose:0=person; drone model:0)
    "auto_q": False,         # scale Kalman Q with target size (for depth changes)
    "aim_drop": 0.2,         # chest: slide aim down toward hips (0=shoulders,1=hips)
    "aim_offset_y": 0,       # px to aim below (+) / above (-) the target point
    "face_safe": False,      # never let the aim/laser rise above the shoulders
    "track_mode": "yolo",    # "yolo" (model) | "blob" (bright ball on dark bg)
    "blob_thresh": 60,       # blob mode: brightness (0-255) above which = target
    "blob_min_area": 150,    # blob mode: ignore blobs smaller than this (px area)
    "blob_circularity": 0.45,  # blob mode: min roundness (0-1). A ball is round; a
                             # light reflection is usually an irregular streak, so
                             # this rejects glare even when there's only one blob.
    "cam_manual_exposure": False,  # force a fixed short exposure (see cam_exposure).
                             # Turn ON in a dark room: kills the auto-exposure that
                             # drops FPS + blurs motion. Off = normal auto-exposure.
    "cam_exposure": 150,     # manual exposure (device units; lower = faster+sharper
                             # but darker). Slide down until FPS jumps and blur goes,
                             # adding room light to keep the ball bright.
    "predictor": "ema",      # none | ema | kalman
    "lead": 0.0,             # seconds to aim AHEAD of the target. With
                             # feedforward this is a deliberate projectile-lead
                             # offset, not lag compensation - 0 points AT it.
    "deadband": 15,          # px of slop before the PID reacts
    "kp": 0.15, "ki": 0.0, "kd": 0.005,     # PID gains. With feedforward
                             # carrying the motion, kp only trims residual error -
                             # it no longer has to be aggressive, and ki is not
                             # needed to kill steady-state lag (FF already does).
    "ff_gain": 0.9,          # velocity feedforward: fraction of the target's own
                             # angular rate to command. 1.0 cancels the motion
                             # exactly BUT is the stability boundary: against a
                             # false target that is fixed in the image, the loop
                             # gain IS ff_gain, so 1.0 self-sustains and >1.0
                             # diverges. 0.9 decays those oscillations while
                             # leaving only ~10% of the motion for the P term to
                             # trim. Do not run at 1.0 or above. 0 = off.
    "kalman_q": 8000.0,      # Kalman process noise (accel variance). Must be LARGE
                             # here: q is what lets the velocity estimate converge
                             # quickly, and feedforward is only as good as that
                             # estimate. (Sim: q=82 took ~4 s to learn a 300 px/s
                             # velocity and tracked a figure-8 at 333 px error;
                             # q=8000 converges in <1 s and halves the error.)
    "kalman_r": 1.0,         # Kalman measurement noise (blob centroid is precise)

    "move_threshold": 70,    # px/s of world speed above which the UI badge reads
                             # MOVING. Display only - the control law is the same
                             # at every speed now (feedforward), so nothing switches.
    "px_per_deg": 10.7,      # camera pixels per degree of servo rotation. Used to
                             # subtract the camera's OWN motion when judging whether
                             # the target is really moving — so panning to center a
                             # still ball doesn't read as the ball moving. Roughly
                             # frame_width / horizontal_FOV. Raise/lower it if a
                             # still target still trips into track while centering.

    # Servo output (PCA9685 over I2C). Channels are wired once at build time;
    # the min/max travel limits keep the mechanism from over-rotating; invert
    # flips an axis's direction if a servo is mounted the "wrong" way.
    "servo_enabled": True,   # master switch — off to tune the loop without moving
    "pan_channel": 0,        # PCA9685 channel for the pan servo
    "tilt_channel": 1,       # PCA9685 channel for the tilt servo
    "pan_min": 0.0, "pan_max": 180.0,     # allowed pan travel (degrees)
    "tilt_min": 0.0, "tilt_max": 180.0,   # allowed tilt travel (degrees)
    "pan_invert": False,     # reverse pan if it chases the wrong way
    "tilt_invert": False,    # reverse tilt if it chases the wrong way
    "pan_center": 90.0,      # "home" pan angle (Center button + startup)
    "tilt_center": 110.0,    # "home" tilt angle — biased up to offset droop
}

# Allowed range for each numeric setting, used to clamp incoming values.
NUMERIC_LIMITS = {
    "lead": (0.0, 1.0),
    "aim_drop": (0.0, 1.0),
    "aim_offset_y": (-300, 300),
    "blob_thresh": (0, 255),
    "blob_min_area": (10, 100000),
    "blob_circularity": (0.0, 1.0),
    "cam_exposure": (1, 2000),
    "deadband": (0, 100),
    "kp": (0.0, 0.3),
    "ki": (0.0, 0.05),
    "kd": (0.0, 0.1),
    "kalman_q": (1.0, 2000000.0),
    "kalman_r": (0.1, 50.0),
    "ff_gain": (0.0, 2.0),
    "move_threshold": (0, 2000),
    "px_per_deg": (0.0, 100.0),
    "pan_min": (0.0, 180.0), "pan_max": (0.0, 180.0),
    "tilt_min": (0.0, 180.0), "tilt_max": (0.0, 180.0),
    "pan_center": (0.0, 180.0), "tilt_center": (0.0, 180.0),
}

# Settings that are plain on/off flags and 0-15 PCA9685 channel numbers.
BOOL_SETTINGS = ("servo_enabled", "pan_invert", "tilt_invert", "face_safe", "auto_q",
                 "cam_manual_exposure")
CHANNEL_SETTINGS = ("pan_channel", "tilt_channel")

# One saved parameter set = everything that shapes the control loop.
PRESET_KEYS = ("kp", "ki", "kd", "lead", "deadband", "kalman_q", "kalman_r",
               "ff_gain", "predictor", "px_per_deg", "move_threshold")

# When the target blinks out, keep driving the aim along its last velocity for
# this many frames before giving up — long enough to ride through a brief blur or
# a clip off the frame edge, short enough not to run away on a real disappearance.
COAST_FRAMES = 6

# If the gap since we last saw this target exceeds this, treat the next sighting
# as a NEW target rather than computing velocity across the gap. After a camera
# dropout or a long occlusion prev_target_t is stale, so (pos - prev_pos)/dt and
# the Kalman step both produce a garbage velocity spike -- which feedforward then
# faithfully executes, walking the turret off-target. Observed: a camera hiccup
# yielded a phantom 1466 px/s and the aim drifted right off the screen.
MAX_TRACK_GAP = 0.5   # seconds

# Feedforward safety. Against a false target that is fixed in the IMAGE (a lens
# artifact, an auto-exposure hotspot, a motion-blurred highlight) the loop gain is
# exactly ff_gain: world_vel = px_per_deg * pan_rate, so the commanded pan_delta
# equals ff_gain * the previous pan_delta. At ff_gain 1.0 that self-sustains and
# above it diverges -- observed as the turret swinging its full travel with
# phantom speeds of 3000+ px/s. A real, world-fixed target cannot do this. So:
# ignore implausible speeds outright, and cap how far feedforward can move the
# servo in one frame, which bounds the runaway without weakening real tracking.
MAX_WORLD_SPEED = 4000.0   # px/s above which the estimate is treated as bogus
                           # (a real fast target hit 2700 px/s, so this only
                           #  catches genuinely impossible estimates)
MAX_FF_DEG = 4.0           # max degrees of feedforward per frame (~60 deg/s)

# A detection that jumps more than this many px in a single frame isn't real
# target motion (nothing physical moves that fast frame-to-frame here) — it's the
# blob snapping to a reflection or residual. The motion gate treats it as a
# relocation (fresh start point) instead of counting it as travel, so a jump can't
# false-trigger a track lock.
MAX_JUMP = 110

# Cap how far (px) the PREDICTED aim point may lead the measured target. A noisy
# Kalman velocity (high Q) times a big lead can momentarily fling the prediction
# across the frame, and the aim lurches at it for a frame before it settles.
# Bounding the lead offset keeps prediction useful without that teleport/lurch.
MAX_LEAD_PX = 150



# The live numbers the web page reads every frame.
telemetry = {
    "fps": 0.0, "target_id": None,
    "offset_x": 0, "offset_y": 0,
    "pan": 90.0, "tilt": 90.0,
    "p": 0.0, "i": 0.0, "d": 0.0,
    "speed": 0,
    "lead_frac": 0.0,   # fraction of full lead currently applied (0=off, 1=full)
    "mode": "holding",  # "moving" | "holding" - display only; one control law
    "servo": "off",   # "on" (driving hardware) | "off" (disabled) | "sim" (no board)
    "loop_latency_ms": 0.0,  # smoothed capture-read -> servo-write processing
                              # time. NOT total glass-to-servo latency - it
                              # excludes camera driver/USB buffering, which
                              # this process can't see.
}

# One-shot commands from the dashboard the vision loop picks up each frame.
control = {"recenter": False}

# Diagnostic slew-rate benchmark: steps one axis to its far travel limit and
# measures the camera's ACTUAL angular velocity from frame-to-frame
# background shift (phase correlation against px_per_deg), not just the
# commanded angle. There's no position encoder on this rig, so watching the
# scene move IS the only way to see real (not commanded) servo speed - it
# needs the camera looking at any normal textured, static scene (a shelf, a
# cluttered desk; a blank wall won't have anything to correlate against).
# Runs inside the vision thread itself (see _slew_bench_step) so it can't
# race the camera or fight the tracking loop's own servo writes. Only active
# when explicitly started via POST /bench/slew; any internal error aborts it
# without touching tracking.
slew_bench = {"active": False}

# The single vision thread publishes its latest annotated JPEG here; every
# /video viewer just streams this shared buffer. One camera, one capture — so
# extra tabs, refreshes, or reconnects can never fight over the device.
frame_lock = threading.Lock()
latest_jpeg = None

# Class names of the currently-loaded model (e.g. {0: "drone"} or the COCO
# 0-79 map). Published so the dashboard's target-class dropdown labels match
# whatever model is actually active, instead of always assuming COCO.
current_names = {}


def clamp(value, low, high):
    return max(low, min(high, value))


def apply_updates(data):
    """Validate and merge a dict of settings into config (unknown keys ignored,
    numbers clamped). Returns True if anything actually changed."""
    changed = False
    for key, value in data.items():
        if key == "model" and value in available_models():
            config["model"] = value
            changed = True
        elif key == "target_class":
            try:
                config["target_class"] = int(clamp(int(value), 0, 99))
                changed = True
            except (TypeError, ValueError):
                pass
        elif key == "aim" and value in AIM_MODES:
            config["aim"] = value
            changed = True
        elif key == "predictor" and value in PREDICTORS:
            config["predictor"] = value
            changed = True
        elif key == "track_mode" and value in ("yolo", "blob"):
            config["track_mode"] = value
            changed = True
        elif key in BOOL_SETTINGS:
            config[key] = bool(value)
            changed = True
        elif key in CHANNEL_SETTINGS:
            try:
                config[key] = int(clamp(int(value), 0, 15))
                changed = True
            except (TypeError, ValueError):
                pass
        elif key in NUMERIC_LIMITS:
            try:
                low, high = NUMERIC_LIMITS[key]
                config[key] = clamp(float(value), low, high)
                changed = True
            except (TypeError, ValueError):
                pass
    return changed


def save_config():
    """Write current settings to disk so they persist across restarts."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as exc:
        print("Could not save config:", exc)


def load_config():
    """Restore saved settings on startup, if the file exists and is valid."""
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE) as f:
            apply_updates(json.load(f))
        print("Loaded saved config from", CONFIG_FILE)
    except (OSError, ValueError) as exc:
        print("Could not load config, using defaults:", exc)


def load_presets():
    """Named parameter sets, {name: {settings...}}. Two older layouts are migrated
    on read: the acquire/track split (from the retired two-mode controller) is
    flattened, track winning on a name clash, and pre-namespace files are trimmed
    to just the keys a preset owns."""
    raw = {}
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE) as f:
                raw = json.load(f)
        except (OSError, ValueError):
            raw = {}
    if not isinstance(raw, dict):
        return {}
    if "acquire" in raw or "track" in raw:          # retired two-namespace layout
        merged = {}
        for kind in ("acquire", "track"):
            for n, c in (raw.get(kind) or {}).items():
                if isinstance(c, dict):
                    merged.setdefault(n, {}).update(c)
        raw = merged
    return {n: {k: c[k] for k in PRESET_KEYS if k in c}
            for n, c in raw.items() if isinstance(c, dict)}


def save_presets(presets):
    try:
        with open(PRESETS_FILE, "w") as f:
            json.dump(presets, f, indent=2)
    except OSError as exc:
        print("Could not save presets:", exc)


def aim_point(kpts, confs, mode, drop=0.0):
    """Pick the pixel to aim at for the chosen mode.

    head/chest average the visible keypoints; hand locks onto a SINGLE wrist
    (the more confident one) so we never aim at the empty space between two
    far-apart hands. Returns (x, y) or None when none of that part's keypoints
    are visible (YOLO reports missing keypoints as (0, 0)).
    """
    indices = AIM_MODES.get(mode, AIM_MODES["chest"])
    visible = [i for i in indices if kpts[i][0] > 0 and kpts[i][1] > 0]
    if not visible:
        return None
    if mode == "hand":
        best = max(visible, key=lambda i: confs[i] if confs is not None else 0.0)
        return int(kpts[best][0]), int(kpts[best][1])
    ax = sum(kpts[i][0] for i in visible) / len(visible)
    ay = sum(kpts[i][1] for i in visible) / len(visible)
    # For chest, slide the aim down toward the hips by `drop` (0..1).
    if mode == "chest" and drop > 0:
        hips = [i for i in (11, 12) if kpts[i][0] > 0 and kpts[i][1] > 0]
        if hips:
            hx = sum(kpts[i][0] for i in hips) / len(hips)
            hy = sum(kpts[i][1] for i in hips) / len(hips)
            ax += drop * (hx - ax)
            ay += drop * (hy - ay)
        elif len(visible) == 2:
            # Hips out of frame — estimate torso length from shoulder width.
            sw = abs(kpts[visible[0]][0] - kpts[visible[1]][0])
            ay += drop * 1.5 * sw
    return int(ax), int(ay)


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.perf_counter()
        self.p = self.i = self.d = 0.0

    def update(self, error):
        now = time.perf_counter()
        dt = now - self.prev_time
        if dt <= 0:
            dt = 1e-3
        self.p = self.kp * error
        self.integral = clamp(self.integral + error * dt, -1000, 1000)
        self.i = self.ki * self.integral
        self.d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        self.prev_time = now
        return self.p + self.i + self.d


class Kalman2D:
    """Constant-velocity Kalman filter for a 2D point in pixels.

    State is [x, y, vx, vy]. It smooths noisy detections and lets us predict
    the position some time ahead, while coasting on momentum through frames
    where the target is briefly missed.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = None    # state vector, lazily created on first measurement
        self.P = None    # covariance

    def step(self, mx, my, dt, q, r):
        # (Re)initialise on first sight or if time went backwards.
        if self.x is None or dt <= 0:
            self.x = np.array([mx, my, 0.0, 0.0], dtype=float)
            self.P = np.eye(4) * 500.0
            return

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                          [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0],
                          [0, dt3 / 2, 0, dt2]], dtype=float)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * r

        # Predict.
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Correct with the new measurement.
        z = np.array([mx, my], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def predict(self, t):
        """Where the target should be t seconds from now."""
        return float(self.x[0] + self.x[2] * t), float(self.x[1] + self.x[3] * t)

    def velocity(self):
        return float(self.x[2]), float(self.x[3])


class ServoController:
    """Drives the pan/tilt servos through a PCA9685 over I2C.

    If the adafruit-servokit library or the board itself isn't present (e.g.
    running the vision loop on a laptop with no hardware attached), it degrades
    to a silent no-op so the rest of the server keeps working. Check `.ok` to
    tell live hardware apart from simulation.
    """

    def __init__(self):
        self.kit = None
        self.ok = False
        self._warned = False
        try:
            from adafruit_servokit import ServoKit
            self.kit = ServoKit(channels=16)
            self.ok = True
            print("PCA9685 servo driver ready.")
        except Exception as exc:
            # ImportError on a dev laptop, or an I2C/board error on the Jetson.
            print("Servo driver unavailable — running in simulation "
                  "(no hardware output):", exc)

    def write(self, channel, angle):
        """Send an angle (0-180) to one channel. Safe no-op without hardware."""
        if not self.ok:
            return
        try:
            self.kit.servo[int(channel)].angle = float(clamp(angle, 0, 180))
        except Exception as exc:
            # A transient wiring/I2C hiccup shouldn't crash the vision loop.
            if not self._warned:
                print("Servo write failed (further errors suppressed):", exc)
                self._warned = True


def _v4l2_set(index, ctrls):
    """Best-effort set of V4L2 UVC controls that OpenCV's cap.set can't reach
    reliably (the exposure trio, dynamic-framerate). No-ops if v4l2-ctl or a given
    control isn't available, so it's safe on any camera/host."""
    try:
        args = ["v4l2-ctl", "-d", f"/dev/video{index}"]
        for c in ctrls:
            args += ["--set-ctrl", c]
        subprocess.run(args, timeout=3,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print("v4l2-ctl not applied:", exc)


def configure_camera(cap, index=0):
    """Force a fast, sharp feed. Two separate things pin this UVC cam near 9 FPS:

      * OpenCV opens UVC cams as YUYV, which is capped at 10 FPS at 640x480 on this
        device — MJPG does 30. So force MJPG.
      * `exposure_dynamic_framerate` lets the camera DROP its frame rate to
        lengthen exposure in a dark room (more latency + motion blur). Turn it off
        so the rate stays put; optionally pin a short manual exposure on top.

    BUFFERSIZE=1 keeps us on the newest frame, not a queued stale one. Exposure
    goes through v4l2-ctl (reliable on UVC where cap.set often isn't). Unsupported
    props/controls just no-op."""
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if config["cam_manual_exposure"]:
        _v4l2_set(index, [
            "exposure_dynamic_framerate=0",   # never trade FPS for exposure
            "auto_exposure=1",                # 1 = Manual Mode (activates the next)
            f"exposure_time_absolute={int(config['cam_exposure'])}",
        ])
    else:
        _v4l2_set(index, [
            "exposure_dynamic_framerate=0",   # keep FPS steady even on auto
            "auto_exposure=3",                # 3 = Aperture Priority (auto)
        ])
    return cap


def open_camera(preferred=0, max_index=9):
    """Open the first camera index that actually yields a frame, and return
    (cap, index). The device index can shift across reboots/re-plugs (e.g.
    /dev/video0 -> /dev/video1, or a metadata node grabbing an index), so try the
    preferred index first and then scan the rest, instead of silently failing on a
    hardcoded 0 and showing a black feed."""
    order = [preferred] + [i for i in range(max_index + 1) if i != preferred]
    for i in order:
        cap = cv2.VideoCapture(i)
        ok, _ = cap.read()
        if ok:
            print(f"Camera opened on index {i}")
            return configure_camera(cap, i), i
        cap.release()
    print(f"No working camera found (scanned 0-{max_index})")
    return cv2.VideoCapture(preferred), preferred  # loop retries/rescans


def detect_blob(frame, thresh, min_area, min_circ=0.0, last_pos=None):
    """Track a bright ball on a dark background — e.g. a colored ball on black.
    Color-agnostic: it thresholds on brightness, so any bright ball is caught,
    no neural net needed.

    Reflection rejection: a ball is round, but a glare/reflection off a light
    source is usually an irregular streak or hot-spot. Candidates below `min_circ`
    (roundness, 0-1) are dropped — unless that would drop everything (a blurred
    ball), in which case they're kept. When more than one round candidate remains
    and we had a lock last frame (`last_pos`), the nearest one wins, since the one
    real object moves continuously frame-to-frame while a reflection flares up
    elsewhere.

    Returns (annotated_frame, found, cx, cy, diameter)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    annotated = frame.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []   # (bx, by, radius, area, circularity)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perim = cv2.arcLength(c, True)
        circ = (4.0 * np.pi * area / (perim * perim)) if perim > 0 else 0.0
        (bx, by), r = cv2.minEnclosingCircle(c)
        cands.append((bx, by, r, area, circ))
    if not cands:
        return annotated, False, 0, 0, 0

    round_cands = [k for k in cands if k[4] >= min_circ]
    pool = round_cands if round_cands else cands   # don't lose a blurred ball
    if last_pos is not None and len(pool) > 1:
        pool = sorted(pool, key=lambda k: (k[0] - last_pos[0]) ** 2
                                          + (k[1] - last_pos[1]) ** 2)
        best = pool[0]
    else:
        best = max(pool, key=lambda k: k[3])        # cold start: take the biggest

    bx, by, r = best[0], best[1], best[2]
    tx, ty = int(bx), int(by)
    cv2.circle(annotated, (tx, ty), int(r), (0, 255, 0), 2)
    return annotated, True, tx, ty, int(2 * r)


def vision_loop():
    """Supervisor: keep the capture/inference loop alive. If it dies from an
    unexpected per-frame exception (a bad detection shape, a driver hiccup),
    log the traceback and restart instead of leaving the turret frozen with a
    dead thread while the web server keeps serving the last stale frame."""
    import traceback
    while True:
        try:
            _vision_run()
        except Exception:
            msg = "!! vision loop crashed — restarting in 0.5s:\n" + traceback.format_exc()
            print(msg, flush=True)
            try:
                with open(os.path.join(SCRIPT_DIR, "vision_error.log"), "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            time.sleep(0.5)


def _slew_bench_step(frame, now, dt, servos, config, bench, pan_angle, tilt_angle):
    """One iteration of the slew-rate benchmark. Returns (pan_angle, tilt_angle,
    still_running) - the caller must adopt these back into its own loop vars so
    tracking doesn't resume from a stale pre-benchmark position."""
    axis = bench["axis"]
    channel = config["pan_channel"] if axis == "pan" else config["tilt_channel"]
    lo, hi = ((config["pan_min"], config["pan_max"]) if axis == "pan"
              else (config["tilt_min"], config["tilt_max"]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")

    if bench.get("prev_gray") is None:
        # First frame: command the step to the far limit and start timing.
        # (step away from whichever limit we're currently closer to)
        current = pan_angle if axis == "pan" else tilt_angle
        target = hi if current <= (lo + hi) / 2 else lo
        bench["target_angle"] = target
        bench["t0"] = now
        bench["prev_gray"] = gray
        if config["servo_enabled"]:
            servos.write(channel, target)
        if axis == "pan":
            pan_angle = target
        else:
            tilt_angle = target
        return pan_angle, tilt_angle, True

    shift, _ = cv2.phaseCorrelate(bench["prev_gray"], gray)
    bench["prev_gray"] = gray
    dx, dy = shift
    px_shift = dx if axis == "pan" else dy
    if dt > 0:
        deg_per_s = abs(px_shift) / max(config["px_per_deg"], 0.001) / dt
        bench["samples"].append({"t": round(now - bench["t0"], 3), "deg_per_s": round(deg_per_s, 1)})

    elapsed = now - bench["t0"]
    if elapsed >= bench["duration"]:
        samples = bench["samples"]
        peak = max((s["deg_per_s"] for s in samples), default=0.0)
        # Average over samples actually in motion (>10% of peak) - excludes
        # the still-sitting-at-the-end tail from dragging the average down.
        moving = [s["deg_per_s"] for s in samples if s["deg_per_s"] > peak * 0.1]
        avg = sum(moving) / len(moving) if moving else 0.0
        bench["result"] = {
            "axis": axis, "peak_deg_per_s": round(peak, 1),
            "avg_deg_per_s": round(avg, 1), "duration_s": round(elapsed, 2),
            "sample_count": len(samples),
        }
        bench["active"] = False
        return pan_angle, tilt_angle, False

    return pan_angle, tilt_angle, True


def _vision_run():
    """Single background thread: capture, infer, drive the servos, and publish
    the latest annotated JPEG. Runs continuously regardless of how many (or how
    few) browsers are watching, so the turret tracks even with no viewer."""
    global latest_jpeg, current_names
    cap, cam_index = open_camera()
    cam_fail = 0
    servos = ServoController()
    pan_pid = PID(config["kp"], config["ki"], config["kd"])
    tilt_pid = PID(config["kp"], config["ki"], config["kd"])
    pan_angle, tilt_angle = config["pan_center"], config["tilt_center"]
    locked_id = None
    last_time = time.perf_counter()
    fps = 0.0
    loop_latency_ms = 0.0

    model = None
    model_name = None

    kf = Kalman2D()
    prev_tx = prev_ty = 0.0
    prev_target_t = None
    prev_locked_id = None
    vx = vy = 0.0            # EMA pixel velocity
    blob_last = None        # last accepted blob position (for reflection gating)
    mode = "holding"        # "moving" | "holding" - badge only, no gain switching
    lost_frames = 0         # consecutive frames with no target (for coasting)
    last_px = last_py = 0.0     # last aim point (to coast from when target lost)
    last_vx = last_vy = 0.0     # last aim velocity (px/s) to coast along
    last_pan_delta = last_tilt_delta = 0.0   # last servo step, to drift on loss
    cam_sig = None          # detects live exposure-setting changes to re-apply

    while True:
        # Hot-swap the model if the dashboard picked a different one.
        if model is None or model_name != config["model"]:
            try:
                model = YOLO(config["model"])
                model_name = config["model"]
                # e.g. {0: "person", ...} for COCO, {0: "drone"} for the custom model
                current_names = {int(k): v for k, v in model.names.items()}
                locked_id = None
                kf.reset()
            except Exception as exc:
                print("Model load failed, keeping previous:", exc)
                config["model"] = model_name or "yolo11n-pose.pt"

        ok, frame = cap.read()
        if not ok:
            # Camera down — still honor a Center request and keep the servo/fps
            # status live, so a dropped camera doesn't ALSO lock out the servos.
            if control["recenter"]:
                control["recenter"] = False
                pan_angle = clamp(config["pan_center"], config["pan_min"], config["pan_max"])
                tilt_angle = clamp(config["tilt_center"], config["tilt_min"], config["tilt_max"])
                pan_pid.integral = 0.0
                tilt_pid.integral = 0.0
                if config["servo_enabled"]:
                    servos.write(config["pan_channel"], pan_angle)
                    servos.write(config["tilt_channel"], tilt_angle)
            telemetry["fps"] = 0.0
            telemetry["servo"] = "sim" if not servos.ok else ("on" if config["servo_enabled"] else "off")
            time.sleep(0.05)   # camera hiccup — keep the thread alive, retry
            cam_fail += 1
            if cam_fail >= 40:  # ~2s of no frames: device likely re-enumerated
                print("Camera stopped delivering frames — re-scanning...")
                cap.release()
                cap, cam_index = open_camera()
                cam_sig = None   # force exposure re-apply on the fresh device
                cam_fail = 0
            continue
        cam_fail = 0

        # Re-apply camera exposure live when the dashboard changes it, so it can
        # be dialed in without restarting the server.
        sig = (config["cam_manual_exposure"], config["cam_exposure"])
        if sig != cam_sig:
            configure_camera(cap, cam_index)
            cam_sig = sig

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)   # smoothed frames per second

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        t_proc_start = now   # loop_latency_ms below times from here to servo-write

        if slew_bench["active"]:
            # Diagnostic mode: bypass tracking entirely for this frame. Any
            # failure here aborts just the benchmark, not the vision thread.
            try:
                pan_angle, tilt_angle, _ = _slew_bench_step(
                    frame, now, dt, servos, config, slew_bench, pan_angle, tilt_angle)
            except Exception as exc:
                slew_bench["active"] = False
                slew_bench["error"] = str(exc)
            cv2.putText(frame, "SLEW BENCHMARK", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            ok2, buffer = cv2.imencode(".jpg", frame)
            if ok2:
                with frame_lock:
                    latest_jpeg = buffer.tobytes()
            continue

        # Apply live-tuned PID gains and settings.
        pan_pid.kp, pan_pid.ki, pan_pid.kd = config["kp"], config["ki"], config["kd"]
        tilt_pid.kp, tilt_pid.ki, tilt_pid.kd = config["kp"], config["ki"], config["kd"]
        deadband = config["deadband"]
        lead = config["lead"]
        predictor = config["predictor"]

        # Manual "center" request from the dashboard: snap to mechanical center
        # and clear the PID so it doesn't lurch. If a target is in frame with
        # output enabled, tracking will re-aim on the next frame.
        if control["recenter"]:
            control["recenter"] = False
            pan_angle = clamp(config["pan_center"], config["pan_min"], config["pan_max"])
            tilt_angle = clamp(config["tilt_center"], config["tilt_min"], config["tilt_max"])
            pan_pid.integral = 0.0
            tilt_pid.integral = 0.0
            if config["servo_enabled"]:
                servos.write(config["pan_channel"], pan_angle)
                servos.write(config["tilt_channel"], tilt_angle)

        # --- Detect the target: the YOLO model, or a bright-blob tracker for a
        # colored ball on a dark background (no neural net needed for that).
        have_target = False
        tx = ty = 0
        target_h = 0
        has_kpts = False
        kpts = None

        if config["track_mode"] == "blob":
            annotated, have_target, tx, ty, target_h = detect_blob(
                frame, config["blob_thresh"], config["blob_min_area"],
                config["blob_circularity"], blob_last)
            if have_target:
                locked_id = "blob"
                blob_last = (tx, ty)
        else:
            results = model.track(frame, persist=True,
                                  classes=[config["target_class"]], verbose=False)
            annotated = results[0].plot()
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            if boxes is not None and boxes.id is not None:
                have_target = True
                ids = boxes.id.int().tolist()
                if locked_id is None or locked_id not in ids:
                    areas = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        areas.append((x2 - x1) * (y2 - y1))
                    locked_id = ids[areas.index(max(areas))]
                ti = ids.index(locked_id)
                bx1, by1, bx2, by2 = boxes[ti].xyxy[0].tolist()
                target_h = by2 - by1
                # Pose models expose keypoints; detection models (e.g. a drone
                # model) don't — those just aim at the bounding-box center.
                has_kpts = keypoints is not None
                aim = None
                if has_kpts:
                    kpts = keypoints.xy[ti].tolist()
                    kconf = keypoints.conf
                    confs = kconf[ti].tolist() if kconf is not None else None
                    aim = aim_point(kpts, confs, config["aim"], config["aim_drop"])
                if aim is not None:
                    tx, ty = aim
                else:
                    tx, ty = int((bx1 + bx2) / 2), int((by1 + by2) / 2)

        cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), -1)   # frame center

        if have_target:

            # Face-safety (person keypoints only): keep the aim at or below the
            # shoulder line so the always-on laser can't rise onto the face.
            if config["face_safe"] and has_kpts:
                sh = [i for i in (5, 6) if kpts[i][0] > 0 and kpts[i][1] > 0]
                if sh:
                    shoulder_y = sum(kpts[i][1] for i in sh) / len(sh)
                    if ty < shoulder_y:
                        ty = int(shoulder_y)

            # Aim a fixed number of pixels below (or above) the target point —
            # e.g. to strike a drone's body/centre of mass instead of the top
            # of its bounding box. Positive = lower on screen.
            ty += config["aim_offset_y"]
            tx, ty = int(tx), int(ty)   # cv2 draw + servo math need int pixels

            # Time since we last saw this same target (for velocity estimates).
            same_target = (prev_target_t is not None
                           and locked_id == prev_locked_id
                           and (now - prev_target_t) < MAX_TRACK_GAP)
            dt_t = (now - prev_target_t) if same_target else 0.0
            # How far the detection moved since last frame (prev_tx still valid
            # here). A huge one-frame jump = the blob snapped to a reflection/
            # residual, not real motion — the gate below uses this to relocate.
            step_px = (((tx - prev_tx) ** 2 + (ty - prev_ty) ** 2) ** 0.5) if same_target else 0.0

            # EMA velocity — used by the "ema" predictor and always for display.
            if same_target and dt_t > 0:
                vx = 0.7 * vx + 0.3 * (tx - prev_tx) / dt_t
                vy = 0.7 * vy + 0.3 * (ty - prev_ty) / dt_t
            else:
                vx = vy = 0.0

            # --- Ego-motion-free WORLD velocity ------------------------------
            # (tx,ty) shifts whenever the CAMERA pans/tilts, so an image-space
            # velocity is polluted by our own motion — that's what made a still
            # ball look like it was moving and made the lead teleport at alignment.
            # Feed the Kalman the target's WORLD position instead: image position
            # plus the camera's own angle (in px; sign flips with invert). Its
            # velocity is then the target's TRUE motion, independent of the camera.
            ppd = config["px_per_deg"]
            world_x = tx + (-pan_angle if config["pan_invert"] else pan_angle) * ppd
            world_y = ty + (-tilt_angle if config["tilt_invert"] else tilt_angle) * ppd
            if not same_target:
                kf.reset()
            q_eff = config["kalman_q"]
            if config["auto_q"] and target_h > 0:
                q_eff = clamp(q_eff * (target_h / 240.0) ** 2, 1.0, 2000000.0)
            kf.step(world_x, world_y, dt_t, q_eff, config["kalman_r"])

            prev_tx, prev_ty, prev_target_t, prev_locked_id = tx, ty, now, locked_id

            dvx, dvy = kf.velocity()            # WORLD velocity px/s (ego-free)
            speed_world = (dvx * dvx + dvy * dvy) ** 0.5

            # --- ONE control law: proportional + VELOCITY FEEDFORWARD ---------
            # `pan_angle += kp*err` is a RATE command, so for a target moving at
            # speed v the steady state needs kp*err == v*dt, i.e. a PERMANENT lag
            # of err = v*dt/kp. Shrinking that with a big kp is what the old
            # "aggressive track mode" did — but loop delay caps how big kp can be
            # before it oscillates, and a big kp x a noisy velocity is exactly what
            # made a near-stationary target twitch. Two modes were a workaround for
            # that bind.
            #
            # Feedforward removes the bind: command the target's own angular rate
            # directly (v/px_per_deg degrees per frame), so the P term only has to
            # clean up the residual and can stay small and stable. Steady-state lag
            # goes to ~0 at ANY speed, so ONE gentle gain now covers a still target
            # and a fast one — no mode switch to toggle, glitch, or lurch.
            # (Simulated: FF at kp=0.09 tracked 300 px/s to ~20 px vs ~49 px for the
            # aggressive-gain lead controller, while being HALF as jittery at rest.)
            miss = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
            mode = "moving" if speed_world >= config["move_threshold"] else "holding"
            pan_pid.kp = tilt_pid.kp = config["kp"]
            pan_pid.kd = tilt_pid.kd = config["kd"]
            pan_pid.ki = tilt_pid.ki = config["ki"]

            # Aim at the measured point. With feedforward doing the keeping-up,
            # `lead` is a deliberate offset AHEAD of the target (for a projectile's
            # flight time) — for pointing a laser AT it, 0 is correct, and the sim
            # confirmed any lead here only adds error.
            eff_lead = 0.0 if predictor == "none" else config["lead"]
            lead_frac = 1.0 if eff_lead else 0.0
            px = tx + clamp(dvx * eff_lead, -MAX_LEAD_PX, MAX_LEAD_PX)
            py = ty + clamp(dvy * eff_lead, -MAX_LEAD_PX, MAX_LEAD_PX)

            px = int(clamp(px, 0, w - 1))
            py = int(clamp(py, 0, h - 1))

            # Remember where we were aiming and how fast, so if the target blinks
            # out next frame we can coast along instead of freezing (see below).
            last_px, last_py = float(px), float(py)
            last_vx, last_vy = dvx, dvy
            lost_frames = 0

            # PID drives toward the (possibly predicted) point.
            offset_x, offset_y = px - cx, py - cy
            error_x = 0 if abs(offset_x) < deadband else offset_x
            error_y = 0 if abs(offset_y) < deadband else offset_y

            # VELOCITY FEEDFORWARD — the term that removes the steady-state lag.
            # The target sweeps dvx px/s across the world, so the camera must turn
            # at dvx/px_per_deg deg/s to stay on it: command that rate outright
            # instead of waiting for position error to build up and asking a big
            # kp to chase it. ff_gain 1.0 = exact cancellation; the P term below
            # only trims what's left, so it can stay small and stable.
            if speed_world > MAX_WORLD_SPEED:
                ff_pan = ff_tilt = 0.0        # bogus estimate - P term only
            else:
                ffk = config["ff_gain"] * dt / max(ppd, 0.001)
                ff_pan = clamp(dvx * ffk, -MAX_FF_DEG, MAX_FF_DEG)
                ff_tilt = clamp(dvy * ffk, -MAX_FF_DEG, MAX_FF_DEG)

            # invert flips a servo's direction; travel limits cap the sweep so
            # the mechanism can't drive itself into a hard stop.
            pan_delta = pan_pid.update(error_x) + ff_pan
            tilt_delta = tilt_pid.update(error_y) + ff_tilt
            if config["pan_invert"]:
                pan_delta = -pan_delta
            if config["tilt_invert"]:
                tilt_delta = -tilt_delta
            pan_angle = clamp(pan_angle + pan_delta, config["pan_min"], config["pan_max"])
            tilt_angle = clamp(tilt_angle + tilt_delta, config["tilt_min"], config["tilt_max"])
            last_pan_delta, last_tilt_delta = pan_delta, tilt_delta   # for drift-on-loss

            if config["servo_enabled"]:
                servos.write(config["pan_channel"], pan_angle)
                servos.write(config["tilt_channel"], tilt_angle)

            cv2.circle(annotated, (int(tx), int(ty)), 6, (0, 0, 255), -1)      # measured (red)
            if predictor != "none":
                cv2.circle(annotated, (int(px), int(py)), 6, (0, 165, 255), -1)  # predicted (orange)
            cv2.line(annotated, (int(cx), int(cy)), (int(px), int(py)), (0, 255, 255), 2)

            # Mode badge on the feed so you can SEE which set is driving.
            mcolor = (0, 255, 0) if mode == "moving" else (0, 165, 255)
            cv2.putText(annotated, mode.upper(), (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, mcolor, 2)

            speed = round((dvx ** 2 + dvy ** 2) ** 0.5)
            telemetry.update({
                "fps": round(fps, 1), "target_id": locked_id,
                "offset_x": offset_x, "offset_y": offset_y,
                "pan": round(pan_angle, 1), "tilt": round(tilt_angle, 1),
                "p": round(pan_pid.p, 2), "i": round(pan_pid.i, 2), "d": round(pan_pid.d, 2),
                "speed": speed, "lead_frac": round(lead_frac, 2), "mode": mode,
            })
        else:
            # Target lost. A fast ball that briefly blurs past the roundness gate
            # or clips the frame edge is usually still moving the same way, so for
            # a short window keep driving the aim along its last known velocity
            # instead of freezing (freezing guarantees we lose it). After the
            # window, give up: stop, forget the track, and reacquire fresh.
            lost_frames += 1
            # In TRACK mode, when the target drops out keep DRIFTING the servo the
            # way it was already moving, tapering to zero over the coast window — a
            # short continuation in the target's direction before giving up. It's
            # the same step we were already applying (decayed each frame), so it
            # can't wind up or run away. In acquire/pinpoint mode we don't drift
            # (the next appearance is random) — hold and re-acquire where it shows.
            coasting = (mode == "moving" and lost_frames <= COAST_FRAMES)
            if coasting:
                decay = 1.0 - lost_frames / (COAST_FRAMES + 1.0)   # 1 -> 0 taper
                pan_angle = clamp(pan_angle + last_pan_delta * decay,
                                  config["pan_min"], config["pan_max"])
                tilt_angle = clamp(tilt_angle + last_tilt_delta * decay,
                                   config["tilt_min"], config["tilt_max"])
                # project where the target would be, to bias re-acquire that way
                last_px = clamp(last_px + last_vx * dt, 0, w - 1)
                last_py = clamp(last_py + last_vy * dt, 0, h - 1)
                blob_last = (int(last_px), int(last_py))
                if config["servo_enabled"]:
                    servos.write(config["pan_channel"], pan_angle)
                    servos.write(config["tilt_channel"], tilt_angle)
                cv2.circle(annotated, (int(last_px), int(last_py)), 6, (255, 0, 255), -1)  # drifting
                cv2.putText(annotated, "DRIFT", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                telemetry.update({"fps": round(fps, 1), "target_id": None,
                                  "speed": 0, "lead_frac": 0.0, "mode": "moving"})
            else:
                prev_target_t = None   # forget velocity while there's no target
                vx = vy = 0.0
                last_vx = last_vy = 0.0
                blob_last = None       # allow reacquire anywhere in frame
                mode = "holding"       # no target -> badge reads holding
                pan_pid.integral = tilt_pid.integral = 0.0   # no leftover wind-up
                kf.reset()
                telemetry.update({"fps": round(fps, 1), "target_id": None,
                                  "speed": 0, "lead_frac": 0.0, "mode": mode})

        # "sim" = no board attached; otherwise reflect the master switch.
        telemetry["servo"] = "sim" if not servos.ok else ("on" if config["servo_enabled"] else "off")

        # Processing latency: capture-read to here (servo writes already
        # issued above). Smoothed the same way fps is. See telemetry comment
        # for what this does and doesn't include.
        proc_ms = (time.perf_counter() - t_proc_start) * 1000.0
        loop_latency_ms = 0.9 * loop_latency_ms + 0.1 * proc_ms
        telemetry["loop_latency_ms"] = round(loop_latency_ms, 1)

        ok2, buffer = cv2.imencode(".jpg", annotated)
        if ok2:
            with frame_lock:
                latest_jpeg = buffer.tobytes()


def generate_frames():
    """Per-viewer MJPEG stream of whatever frame the vision thread last made.
    Multiple viewers can run this at once — they all read the shared buffer."""
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        with frame_lock:
            buf = latest_jpeg
        if buf is not None:
            yield boundary + buf + b"\r\n"
        time.sleep(0.03)   # ~30 fps ceiling for the stream itself


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/telemetry")
def get_telemetry():
    return jsonify(telemetry)


@app.route("/center", methods=["POST"])
def center():
    """Ask the vision loop to snap pan/tilt back to mechanical center."""
    control["recenter"] = True
    return jsonify({"ok": True})


@app.route("/bench/slew", methods=["POST"])
def start_slew_bench():
    """Kick off the slew-rate benchmark (see slew_bench comment above
    _vision_run). Point the camera at a normal textured, static scene first -
    the measurement is done by watching the background move."""
    if slew_bench.get("active"):
        return jsonify({"ok": False, "error": "benchmark already running"}), 409
    data = request.get_json(silent=True) or {}
    axis = data.get("axis", "pan")
    if axis not in ("pan", "tilt"):
        return jsonify({"ok": False, "error": "axis must be 'pan' or 'tilt'"}), 400
    slew_bench.clear()
    slew_bench.update({
        "active": True, "axis": axis, "prev_gray": None, "samples": [],
        "duration": float(data.get("duration", 2.5)),
        "result": None, "error": None,
    })
    return jsonify({"ok": True})


@app.route("/bench/slew", methods=["GET"])
def get_slew_bench():
    """Poll for benchmark progress/result. `active` goes false once `result`
    (or `error`) is populated."""
    return jsonify({
        "active": slew_bench.get("active", False),
        "result": slew_bench.get("result"),
        "error": slew_bench.get("error"),
        "sample_count": len(slew_bench.get("samples", [])),
    })


@app.route("/config", methods=["GET"])
def get_config():
    """Current settings plus the option lists the dashboard renders."""
    return jsonify({
        **config,
        "_models": available_models(),
        "_aims": list(AIM_MODES.keys()),
        "_predictors": PREDICTORS,
        "_class_names": current_names,
    })


@app.route("/config", methods=["POST"])
def set_config():
    """Update one or more settings. Unknown keys are ignored; numbers clamped."""
    data = request.get_json(silent=True) or {}
    if apply_updates(data):
        save_config()
    return jsonify({"ok": True, "config": config})


@app.route("/presets", methods=["GET"])
def get_presets():
    return jsonify({"presets": sorted(load_presets().keys())})


@app.route("/presets/save", methods=["POST"])
def save_preset():
    """Snapshot the current tuning under a name."""
    name = str((request.get_json(silent=True) or {}).get("name", "")).strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    presets = load_presets()
    presets[name] = {k: config[k] for k in PRESET_KEYS}
    save_presets(presets)
    return jsonify({"ok": True, "presets": sorted(presets.keys())})


@app.route("/presets/load", methods=["POST"])
def load_preset():
    """Apply a saved preset to the live config."""
    name = str((request.get_json(silent=True) or {}).get("name", ""))
    presets = load_presets()
    if name not in presets:
        return jsonify({"ok": False, "error": "not found"}), 404
    apply_updates(presets[name])
    save_config()
    return jsonify({"ok": True, "config": config})


@app.route("/presets/delete", methods=["POST"])
def delete_preset():
    name = str((request.get_json(silent=True) or {}).get("name", ""))
    presets = load_presets()
    if presets.pop(name, None) is not None:
        save_presets(presets)
    return jsonify({"ok": True, "presets": sorted(presets.keys())})


if __name__ == "__main__":
    load_config()   # restore saved tuning before serving
    # One capture thread feeds every viewer; start it before serving.
    threading.Thread(target=vision_loop, daemon=True).start()
    # threaded=True lets /telemetry answer while /video keeps streaming.
    app.run(host="0.0.0.0", port=8000, threaded=True)
