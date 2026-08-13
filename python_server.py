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
    "face_safe": False,      # never let the aim/laser rise above the shoulders
    "predictor": "ema",      # none | ema | kalman
    "lead": 0.15,            # seconds to project the target forward
    "lead_ramp": 150.0,      # px/s at which FULL lead applies; below this the
                             # lead fades toward 0 so a stopping target doesn't
                             # get overshot (set 0 to use a constant lead)
    "deadband": 15,          # px of slop before the PID reacts
    "kp": 0.03, "ki": 0.001, "kd": 0.005,   # PID gains
    "kalman_q": 50.0,        # Kalman process noise (accel variance)
    "kalman_r": 4.0,         # Kalman measurement noise

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
    "lead_ramp": (0.0, 1000.0),
    "aim_drop": (0.0, 1.0),
    "deadband": (0, 100),
    "kp": (0.0, 0.3),
    "ki": (0.0, 0.05),
    "kd": (0.0, 0.1),
    "kalman_q": (1.0, 500.0),
    "kalman_r": (0.1, 50.0),
    "pan_min": (0.0, 180.0), "pan_max": (0.0, 180.0),
    "tilt_min": (0.0, 180.0), "tilt_max": (0.0, 180.0),
    "pan_center": (0.0, 180.0), "tilt_center": (0.0, 180.0),
}

# Settings that are plain on/off flags and 0-15 PCA9685 channel numbers.
BOOL_SETTINGS = ("servo_enabled", "pan_invert", "tilt_invert", "face_safe", "auto_q")
CHANNEL_SETTINGS = ("pan_channel", "tilt_channel")

# The live numbers the web page reads every frame.
telemetry = {
    "fps": 0.0, "target_id": None,
    "offset_x": 0, "offset_y": 0,
    "pan": 90.0, "tilt": 90.0,
    "p": 0.0, "i": 0.0, "d": 0.0,
    "speed": 0,
    "servo": "off",   # "on" (driving hardware) | "off" (disabled) | "sim" (no board)
}

# One-shot commands from the dashboard the vision loop picks up each frame.
control = {"recenter": False}

# The single vision thread publishes its latest annotated JPEG here; every
# /video viewer just streams this shared buffer. One camera, one capture — so
# extra tabs, refreshes, or reconnects can never fight over the device.
frame_lock = threading.Lock()
latest_jpeg = None


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
    """Named parameter sets, {name: {settings...}}, persisted to disk."""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE) as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            pass
    return {}


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


def vision_loop():
    """Single background thread: capture, infer, drive the servos, and publish
    the latest annotated JPEG. Runs continuously regardless of how many (or how
    few) browsers are watching, so the turret tracks even with no viewer."""
    global latest_jpeg
    cap = cv2.VideoCapture(0)
    servos = ServoController()
    pan_pid = PID(config["kp"], config["ki"], config["kd"])
    tilt_pid = PID(config["kp"], config["ki"], config["kd"])
    pan_angle, tilt_angle = config["pan_center"], config["tilt_center"]
    locked_id = None
    last_time = time.perf_counter()
    fps = 0.0

    model = None
    model_name = None

    kf = Kalman2D()
    prev_tx = prev_ty = 0.0
    prev_target_t = None
    prev_locked_id = None
    vx = vy = 0.0            # EMA pixel velocity

    while True:
        # Hot-swap the model if the dashboard picked a different one.
        if model is None or model_name != config["model"]:
            try:
                model = YOLO(config["model"])
                model_name = config["model"]
                locked_id = None
                kf.reset()
            except Exception as exc:
                print("Model load failed, keeping previous:", exc)
                config["model"] = model_name or "yolo11n-pose.pt"

        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)   # camera hiccup — keep the thread alive, retry
            continue

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)   # smoothed frames per second

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

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

        results = model.track(frame, persist=True, classes=[config["target_class"]], verbose=False)
        annotated = results[0].plot()
        cv2.circle(annotated, (cx, cy), 5, (255, 255, 255), -1)

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        if boxes is not None and boxes.id is not None:
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

            # Face-safety (person keypoints only): keep the aim at or below the
            # shoulder line so the always-on laser can't rise onto the face.
            if config["face_safe"] and has_kpts:
                sh = [i for i in (5, 6) if kpts[i][0] > 0 and kpts[i][1] > 0]
                if sh:
                    shoulder_y = sum(kpts[i][1] for i in sh) / len(sh)
                    if ty < shoulder_y:
                        ty = int(shoulder_y)

            # Time since we last saw this same target (for velocity estimates).
            same_target = prev_target_t is not None and locked_id == prev_locked_id
            dt_t = (now - prev_target_t) if same_target else 0.0

            # EMA velocity — used by the "ema" predictor and always for display.
            if same_target and dt_t > 0:
                vx = 0.7 * vx + 0.3 * (tx - prev_tx) / dt_t
                vy = 0.7 * vy + 0.3 * (ty - prev_ty) / dt_t
            else:
                vx = vy = 0.0

            # Kalman runs continuously so it's warm the instant it's selected.
            if not same_target:
                kf.reset()
            # Optionally scale Q with target size: a closer/bigger target moves
            # faster in pixels, so it needs more process noise to keep up.
            q_eff = config["kalman_q"]
            if config["auto_q"] and target_h > 0:
                q_eff = clamp(q_eff * (target_h / 240.0) ** 2, 1.0, 500.0)
            kf.step(tx, ty, dt_t, q_eff, config["kalman_r"])

            prev_tx, prev_ty, prev_target_t, prev_locked_id = tx, ty, now, locked_id

            # Velocity estimate for the chosen predictor.
            if predictor == "kalman":
                dvx, dvy = kf.velocity()
            else:
                dvx, dvy = vx, vy

            # Speed-scaled lead: apply the full lead only when the target is
            # actually moving (>= lead_ramp px/s), and fade it toward 0 as it
            # slows/stops, so the aim doesn't overshoot a stopping target.
            speed_now = (dvx ** 2 + dvy ** 2) ** 0.5
            ramp = config["lead_ramp"]
            eff_lead = lead * clamp(speed_now / ramp, 0.0, 1.0) if ramp > 0 else lead

            if predictor == "kalman":
                px, py = kf.predict(eff_lead)
            elif predictor == "ema":
                px, py = tx + dvx * eff_lead, ty + dvy * eff_lead
            else:  # "none" — aim straight at the measurement
                px, py = tx, ty

            px = int(clamp(px, 0, w - 1))
            py = int(clamp(py, 0, h - 1))

            # PID drives toward the (possibly predicted) point.
            offset_x, offset_y = px - cx, py - cy
            error_x = 0 if abs(offset_x) < deadband else offset_x
            error_y = 0 if abs(offset_y) < deadband else offset_y

            # invert flips a servo's direction; travel limits cap the sweep so
            # the mechanism can't drive itself into a hard stop.
            pan_delta = pan_pid.update(error_x)
            tilt_delta = tilt_pid.update(error_y)
            if config["pan_invert"]:
                pan_delta = -pan_delta
            if config["tilt_invert"]:
                tilt_delta = -tilt_delta
            pan_angle = clamp(pan_angle + pan_delta, config["pan_min"], config["pan_max"])
            tilt_angle = clamp(tilt_angle + tilt_delta, config["tilt_min"], config["tilt_max"])

            if config["servo_enabled"]:
                servos.write(config["pan_channel"], pan_angle)
                servos.write(config["tilt_channel"], tilt_angle)

            cv2.circle(annotated, (tx, ty), 6, (0, 0, 255), -1)      # measured (red)
            if predictor != "none":
                cv2.circle(annotated, (px, py), 6, (0, 165, 255), -1)  # predicted (orange)
            cv2.line(annotated, (cx, cy), (px, py), (0, 255, 255), 2)

            speed = round((dvx ** 2 + dvy ** 2) ** 0.5)
            telemetry.update({
                "fps": round(fps, 1), "target_id": locked_id,
                "offset_x": offset_x, "offset_y": offset_y,
                "pan": round(pan_angle, 1), "tilt": round(tilt_angle, 1),
                "p": round(pan_pid.p, 2), "i": round(pan_pid.i, 2), "d": round(pan_pid.d, 2),
                "speed": speed,
            })
        else:
            prev_target_t = None   # forget velocity while there's no target
            vx = vy = 0.0
            kf.reset()
            telemetry.update({"fps": round(fps, 1), "target_id": None, "speed": 0})

        # "sim" = no board attached; otherwise reflect the master switch.
        telemetry["servo"] = "sim" if not servos.ok else ("on" if config["servo_enabled"] else "off")

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


@app.route("/config", methods=["GET"])
def get_config():
    """Current settings plus the option lists the dashboard renders."""
    return jsonify({
        **config,
        "_models": available_models(),
        "_aims": list(AIM_MODES.keys()),
        "_predictors": PREDICTORS,
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
    """Snapshot the current settings under a name."""
    name = str((request.get_json(silent=True) or {}).get("name", "")).strip()
    if not name:
        return jsonify({"ok": False, "error": "name required"}), 400
    presets = load_presets()
    presets[name] = dict(config)   # copy of all current tunable settings
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
