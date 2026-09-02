# AutoTurret — Development Log

A record of the software work on this project: what was built, what was tried,
what broke, and *why* it broke. Written to save re-deriving the same conclusions
(and re-making the same mistakes) later.

---

## 1. System overview

| Piece | Detail |
|---|---|
| Compute | Jetson Orin Nano 8GB, JetPack 6.2.3 |
| Vision | YOLO11 (pose + custom drone detector) and an OpenCV blob tracker |
| Actuation | 2x MG996R servos via PCA9685 over I2C bus 7 |
| Payload | Always-on laser (drives the face-safety logic) |
| Backend | `python_server.py` — Flask + MJPEG stream + single vision thread, port 8000 |
| Frontend | Next.js 16 / React 19 dashboard, `app/page.tsx`, port 3000 |
| Test rig | `tools/drone_target.html` — animated on-screen target |
| Access | Tailscale + LAN; SSH alias `jetson` |

Two processes, and the distinction matters constantly:

- **Dashboard (3000)** — Next.js. Serves the UI, and spawns/kills the vision server
  via `app/api/server/route.ts`.
- **Vision server (8000)** — Python. Owns the camera, inference, PID, servos.

---

## 2. The tracking-control saga (the big one)

This consumed the most time by far. Nine distinct approaches before the right one.

### The original problem

The turret needed aggressive lead + P gain to keep up with a fast target, but
those same settings made it twitch, lurch, or oscillate when the target was slow
or stationary. The intuitive fix — two modes, gentle for acquiring and aggressive
for tracking — turned out to be treating a symptom.

### Why it was actually happening (found at the end, via simulation)

The controller is `pan_angle += kp * error`, which is a **rate** command. For a
target moving at speed `v`, the steady state requires `kp * error = v * dt`, so:

```
permanent lag = v * dt / kp
```

At 15 fps with `kp = 0.09`, a 300 px/s target sits **223 px behind, forever**. The
only way to shrink that with P alone is a big `kp` — but loop delay caps `kp`
before it oscillates, and a big `kp` multiplied by a noisy velocity estimate is
precisely what makes a near-stationary target twitch.

**That bind is the entire reason the two-mode design existed.** The premise
("aggressive gains blow up on slow targets") was correct; the conclusion
("therefore switch gains") was not.

### Attempts, in order

| # | Approach | Why it failed |
|---|---|---|
| 1 | Two modes (acquire/track), switch on distance-to-crosshair | Entering track lurched the aim, which pushed the target off-center, which ejected it back to acquire. Self-reinforcing flapping. |
| 2 | Separate saved presets for acquire and track | Useful UX, but orthogonal to the bug. Didn't address why the switch misfired. |
| 3 | Gate on "consistent velocity direction for N frames" | Instantaneous Kalman velocity is noisy; over a short window even jitter looks directionally consistent. Leaked. |
| 4 | Gate on net travel from where the target first appeared | Better (jitter oscillates in place -> ~0 net travel), but still image-space, so camera panning counted as target travel. |
| 5 | Gate on path straightness (`net displacement / path length`) | Simultaneously too leaky (a single noise jump is perfectly "straight" — one segment always is) and too strict (real curved motion scored low and never locked). |
| 6 | Jump reset — treat any >110 px single-frame jump as a relocation | Correct and kept, but only fixed reflection-snap, not the core issue. |
| 7 | Measure camera rotation *while centered* (parked vs. following) | Clever, calibration-free in principle, but still fragile and indirect. |
| 8 | **Track in world space** (`image position + camera angle`) | **Correct and kept.** Removed ego-motion from the velocity estimate. |
| 9 | Velocity-hysteresis FSM on world speed | Much better, but still a hard switch — so it still toggled at the boundary on real hardware. |
| ✅ | **Velocity feedforward, single mode** | The actual fix. See below. |

### The fix: velocity feedforward

Instead of waiting for position error to accumulate and asking a large gain to
chase it, **command the target's own angular rate directly**:

```python
ffk = ff_gain * dt / px_per_deg
pan_delta = pan_pid.update(error_x) + dvx * ffk
```

The P term then only trims residual error, so it can stay small and stable.
Steady-state lag goes to ~0 at *any* speed, which means one gentle gain covers a
still target and a fast one — **nothing left for a mode switch to do.**

Simulation results (`scratchpad/arch2.py`), constant 300 px/s target:

```
P only (kp .09)                    202 px rms
P + lead                           106 px
P_hi + lead   (the "track" mode)    49 px    jitter at rest: 3.2 px
velocity FEEDFORWARD (kp .09)       20 px    jitter at rest: 1.5 px
```

Feedforward with the *gentle* gain beat the aggressive-gain lead controller by
2.5x on moving targets **and** was half as jittery at rest — better at both ends
at once.

### Corollaries

- **With feedforward, `lead` should be 0.** Sim: lead 0 -> 19.7 px; lead 0.30 ->
  109.6 px. Feedforward already supplies the rate; lead then just parks the aim
  *ahead* of the target. Lead is only meaningful as true projectile flight-time
  lead — pointless for a laser.
- **`kalman_q` was crippled.** It was 82, with `NUMERIC_LIMITS` capping it at 500.
  At q=82 the velocity estimate takes **~4 seconds** to learn a 300 px/s target
  (figure-8 error 333 px). At q=8000 it converges in under a second (156 px).
  Feedforward is only as good as the velocity estimate — and so was every
  "is it moving?" decision, which is a large part of why mode-switching was
  so unreliable. Cap raised to 60000.

### Prior art checked

- **Velocity feedforward** — standard in telescope drives (sidereal tracking
  feedforwards Earth's rate; guiding only trims residual), camera gimbals, and
  every industrial motion controller.
- **Smith Predictor** (1957) — textbook dead-time compensator. Implemented and
  tested; at this rig's actual delay it tied plain feedforward and was *less*
  robust to delay mismatch, so it was not shipped.
- **Track initiation vs. maintenance** — real radar/fire-control does have two
  modes, but they differ in **data association** (which detections belong to the
  track; M-of-N logic, validation gates), **not in controller gains**.
- **IMM (Interacting Multiple Model)** — the principled "might be still, might be
  maneuvering" solution: parallel motion models blended by likelihood, so it
  never hard-toggles.
- **Visual servoing** — the original bug is textbook IBVS ego-motion coupling;
  world-space tracking is effectively PBVS.

### The irreducible limit

Overshoot on an *unpredicted* maneuver is approximately `v * total_latency`
(~100 px at 500 px/s with a ~200 ms loop). No algorithm beats this. Systems like
Phalanx CIWS or high-speed vision rigs win with **kHz sensor rates, not smarter
code**. At ~15 fps, latency is the binding constraint on everything here.

---

## 3. Simulation work

Rather than tuning on hardware (slow, noisy, camera kept dropping), most tuning
moved into closed-loop simulators.

| Script | Purpose | Outcome |
|---|---|---|
| `tools/tune.py` | 2D realistic sim, random search + local refine | Produced the "snappy" preset |
| `speed_sched.py` | Speed-scheduled gains (switch presets by detected speed) | **0% benefit** — system is delay-limited, not gain-limited. Lead compensates delay ~uniformly, and bounces are *also* high-speed, so speed is the wrong scheduling signal. Feature deliberately not built. |
| `tune_screen.py` | Real geometry: 30" 16:9 screen at ~8", 15.8 fps | Found the setup is frame-rate-limited; past ~600 px/s the target clips out of frame regardless of tuning. **Backing the camera off to 14–16" roughly doubles trackable speed** (angular speed = v/d). |
| `fsm_sim.py` | Acquire/track state machine | Validated hysteresis, and exposed a bug in my own sim: `r` was 100x too high, over-smoothing the velocity to ~4°/s when truth was ~30°/s |
| `arch_sim.py` / `arch2.py` / `arch3.py` | Compare control *architectures*, not just gains | Found velocity feedforward. Also caught that v1's numbers were dominated by Kalman velocity convergence time, not the control law |

**Lesson:** twice, the first run of a simulator produced confidently wrong
numbers because of a bug in the *simulator* (bad `r`; transient swamping
steady-state). Always sanity-check sim output against a hand-computed
expectation before acting on it.

---

## 4. Vision & detection

### Blob (ball) tracking

Added an OpenCV path for tracking a bright ball on a dark background — no neural
net: grayscale -> threshold -> `MORPH_OPEN` -> `findContours` ->
`minEnclosingCircle`.

**Problem:** as room light increased, reflections of the light source were picked
up as the ball.

**Fixes:**
- **Roundness gate** — `4*pi*area / perimeter^2`. A ball is round (~0.7–0.9); glare
  is an irregular streak (~0.2–0.4). Falls back to keeping all candidates if the
  gate would reject *everything* (a badly blurred ball), so it can't lose the target.
- **Nearest-to-last-position** selection when multiple round candidates survive —
  the real object moves continuously; glare flares up elsewhere.

### Aim point

- Pose models: head/chest/hand keypoints, `aim_drop` to slide down the torso.
- Detection models (e.g. drone): bounding-box center.
- `aim_offset_y` to strike below the box top (drone body vs. box edge).
- **Face-safety** — with the always-on laser, aim is clamped to at or below the
  shoulder line for people.

### Target loss behavior

Evolved several times:

1. Freeze on loss — guaranteed losing a fast target.
2. Coast along last velocity for `COAST_FRAMES` — better, but caused
   **"shoots off into nothingness."** Two causes: leftover **integral windup**
   kept driving after the target vanished, and coasting a target last seen near
   the frame edge (already on its way out) drove the servo to its travel stop.
3. Clear the integral on loss; don't coast in acquire/pinpoint (next appearance
   is random).
4. **Final:** in track mode, continue the servo's *own last step*, decayed to zero
   over the coast window. It's the step already being applied, scaled down each
   frame — so it cannot wind up or run away. Shows `TRACK (drift)` on the feed.

---

## 5. Camera & hardware

The single most recurring source of "the software is broken" reports — and it was
hardware every time.

### The recurring "no video / can't control servos"

Root cause: **the USB camera physically dropping off the bus.** Not enumerated at
all (`lsusb` showed nothing), so no `/dev/video*`. Software mitigations added:

- `open_camera()` scans indices 0–9 — the device index shifts across re-plugs
  (observed `/dev/video0` -> `/dev/video1` mid-session).
- Camera-fail branch still honors **Center** and keeps servo/FPS telemetry live,
  so a dropped camera doesn't *also* lock out the servos.
- After ~2 s of no frames, release and rescan.

### The 9 FPS discovery

The feed was running at 9 FPS in a dark room. Investigation with `v4l2-ctl` found
**two independent caps**, neither of which the code was addressing:

1. **Format.** The camera (Microdia Vitade AF, `0c45:6366`) does 30 fps in **MJPG**
   at 640x480 but only **10 fps in YUYV** — and OpenCV opens UVC cameras as YUYV
   by default. Forcing `CAP_PROP_FOURCC=MJPG` took it 9.3 -> ~15 fps, halving frame
   latency.
2. **`exposure_dynamic_framerate`** (default on) lets the camera *drop its frame
   rate* to lengthen exposure in a dark room — costing both latency and motion
   blur. Turned off.

Also learned: on this UVC device `cap.set(CAP_PROP_EXPOSURE)` is unreliable;
exposure must go through `v4l2-ctl` (`auto_exposure=1` for manual,
`exposure_time_absolute`, range 1–5000).

Measured, before/after:

```
A default          9.3 fps   fourcc=YUYV
B MJPG + dynFR off 14.9 fps  fourcc=MJPG
C + manual exp     13.7 fps  fourcc=MJPG   <- exposure was NOT the limiter
```

### USB topology

`lsusb -t` showed the camera on **Bus 01 (USB 2.0), behind an intermediate hub**,
sharing that hub with the Bluetooth radio — while **Bus 02 (USB 3.0) sat empty**.
Moving it to a direct port stabilized it. Note: the camera itself is USB 2.0
hardware (480 Mbps), so the port change was about **dedicated power and bus
arbitration**, not bandwidth.

The residual ~15 fps ceiling is MJPG-decode/USB throughput, not exposure — and
opening the camera at MJPG 30 could reset the bus hard enough to drop the SSH
session.

---

## 6. Crashes & the debugging blind spot

### The cv2 float-coordinate crash

**Symptom:** "super laggy... the moment the drone comes on screen it just kinda
pauses."

**Cause:** `cv2.error: Can't parse 'center'. Sequence item with index 1 has a
wrong type` — `aim_offset_y` is stored as a float (NUMERIC_LIMITS coerces via
`clamp(float(value))`), so `ty += aim_offset_y` made `ty` a float, and
`cv2.circle` requires ints.

**Fix:** `tx, ty = int(tx), int(ty)` plus int-casting every draw call.

### The supervisor, and why the logs were empty

Wrapped the vision loop in a supervisor that catches, logs, and restarts after
0.5 s. This turned a permanent freeze into repeating stutters — better
availability, but it also *disguised* the crash.

**Blind spot:** the spawned Python's stdout is buffered, so tracebacks never
reached `journalctl`. Fixed by also appending tracebacks to
`vision_error.log` in the script directory. That file is now the first place to
look.

---

## 7. Drone model training

Goal: a custom drone detector to replace COCO classes.

**Bump 1 — Roboflow SDK doesn't download weights.** `v.download()` fetches the
**dataset**, not trained weights (`cp: cannot stat 'drone_dl/weights.pt'`).
Weights require the UI "Download Weights" button, or training locally.

**Bump 2 — OOM on the Jetson.** Local training died with `[1]+ Killed`. The Orin
Nano's 8 GB is **unified** (shared CPU/GPU), and 6 dataloader workers exhausted
it. Fixed by stopping the turret service, adding 8 GB of swap, and retraining
with `batch=4 workers=2`. Completed: 50 epochs, 5.26 h, **mAP@50 0.976**.

**Bump 3 — weights not where expected.** Ultralytics nested them one level
deeper than assumed: `runs/detect/drone_train/run1/weights/best.pt`.

**Bump 4 — bad UI label.** The dashboard's `modelLabel` abbreviated filenames, so
`drone.pt` rendered as "D DET". Fixed to show custom model filenames verbatim, and
the backend now reports the loaded model's actual class names so the target-class
dropdown reads `0 — drone`.

---

## 8. Deployment & infrastructure bumps

These cost real time and were mostly *not* code bugs.

| Bump | Detail |
|---|---|
| **`systemctl restart` does NOT reload vision code** | The service manages the Next dashboard. The Python vision server is spawned separately — it only reloads on **Stop -> Start** in the UI. Multiple "it's still happening" reports were simply the old Python still running. |
| **The stale-server trap** | Confirmed by querying `/config` for keys that only exist in new code. At one point the running server was **two versions behind** every fix being tested. Always verify what's actually loaded before diagnosing. |
| **dev vs. prod Next.js** | `next dev` hot-reloads `page.tsx`; `next start` serves a **pre-built** bundle and ignores file changes until `npm run build`. A restart silently flipped modes, so UI edits appeared to do nothing. |
| **Browser cache** | Even after rebuild, a normal refresh reused the cached bundle. `Ctrl+Shift+R` required. |
| **`sudo` over non-interactive SSH** | `sudo systemctl restart` fails with "a password is required" — can't be automated from here. Worked around by relaunching the dashboard outside systemd with `setsid`. |
| **SIGTERM to a systemd main PID** | Reads as a *clean stop*, so `Restart=on-failure` does **not** respawn it. Left the dashboard down until manually restarted. |
| **Wrong host** | `localhost:3000` hit the dev laptop, not the Jetson. |
| **Repurposed config keys kept stale values** | `lock_min_travel` changed meaning from "50 px" to "70 px/s" but the saved `turret_config.json` still held `10` — semantically wrong under the new code, so everything tripped instantly. **Renaming a key's units silently poisons saved configs.** |
| **Encoding corruption** | Patching files via a bash heredoc + `open(p,'w')` on Windows wrote **cp1252**, mangling em-dashes/`x` into bytes that are invalid UTF-8. Broke Turbopack (`failed to convert rope into string`) and Python (`Non-UTF-8 code starting with '\x97'`). **Always pass `encoding='utf-8'` explicitly.** Repaired byte-exactly rather than by re-typing. |

---

## 9. Dashboard / UX work

- Target class: slider -> **dropdown** populated from the loaded model's real class
  names (falls back to COCO).
- Model card -> **Tracking** card: YOLO vs. Ball(blob) toggle, with blob controls
  (brightness threshold, min size, roundness gate) shown contextually.
- Pose-only controls hidden for detection models.
- **Center** button moved into the header (was buried and required scrolling).
- Live **mode badge** painted onto the video feed *and* in the sidebar.
- **Two-namespace presets** — independent `acquire` and `track` libraries that can
  be mixed and matched, with legacy flat presets auto-migrated into `track`.
  (Largely vestigial now that the controller is single-mode, but harmless.)

## 10. Test target page (`tools/drone_target.html`)

Self-contained animated target, since a real drone indoors isn't practical.

- Real drone sprite from the model's own training data (base64-embedded).
- **Bug:** changing the speed slider teleported the target — zoom/figure-8 used
  `this.t * cfg.speed`, so changing `speed` retroactively rewrote the phase.
  Fixed with a `this.a` phase accumulator (`this.a += dt * cfg.speed`).
- Modes: wander, sweep H/V, zoom, figure-8, **pinpoint** (random pop-ups with a
  deliberate blank between them, forcing a genuine re-acquire).
- **Ball** target on black for blob mode, with color picker.
- **Spawn radius** slider — pop-ups were landing outside the camera's field of
  view; this constrains them to a disk around screen center.
- Idle "bob" was briefly removed to stop it reading as motion, then **restored** —
  the correct fix was on the turret side, not by making the test easier.

---

## 11. Current state

Single control law, no mode switching. Live config:

```
ff_gain 0.9   kalman_q 8000   kalman_r 1
kp 0.15       ki 0            kd 0.005
lead 0        deadband 15     predictor ema
px_per_deg 10.7
```

Running at ~14–15 fps, no crashes. Q ceiling raised to 60000. The dashboard
exposes a **Feedforward** slider (set to 0 to A/B against the old behavior).

### Diagnostics: latency + slew rate

Added to get real numbers instead of estimates:

- **`loop_latency_ms`** in `/telemetry` — smoothed capture-read-to-servo-write
  processing time per frame. Deliberately *not* claimed as total glass-to-servo
  latency: camera driver/USB buffering happens below this process and isn't
  visible to it.
- **`POST /bench/slew`** (`{"axis": "pan"|"tilt", "duration": 2.5}`), polled
  via `GET /bench/slew` — steps the axis to its far travel limit and measures
  the camera's *actual* angular velocity from frame-to-frame background shift
  (`cv2.phaseCorrelate`, converted through `px_per_deg`), not just the
  commanded angle. There's no position encoder on this rig, so watching the
  scene move is the only way to see real servo speed. Needs the camera aimed
  at any normal textured, static scene — a blank wall gives nothing to
  correlate against. Runs inside the vision thread (gated behind
  `slew_bench["active"]`) so it can't race the camera or fight tracking's own
  servo writes; any internal error aborts just the benchmark.
- Not yet run against real hardware — thresholds (benchmark duration, the
  10%-of-peak cutoff for the "in motion" average) are first-pass defaults and
  may need adjusting once real samples come back.

### `ff_gain` = 1.0 is an instability boundary, not just "exact"

`ff_gain 1.0` was originally described as exact cancellation of the target's
motion — true against a *real* moving target, but there's a second case: a
false target that's actually fixed in the image (e.g. a lens reflection).
There, the loop gain **is** `ff_gain` itself — each frame's commanded pan
delta equals `ff_gain` times the previous one — so at 1.0 it self-sustains and
above 1.0 it diverges. Running at **0.9** damps that oscillation while still
handing the P term only ~10% of the motion to trim. Don't run at 1.0+.

### Key invariants — don't regress these

1. **Feed the Kalman world position, never raw image position.**
   `world_x = tx + (±pan_angle) * px_per_deg`. Image-space velocity cannot
   distinguish "target moved" from "camera moved."
2. **Keep velocity feedforward.** It's what makes a single gain work across all
   speeds. Without it, the two-mode bind returns.
3. **`lead` stays 0** unless firing an actual projectile.
4. **`kalman_q` must stay large** (thousands). Feedforward is only as good as the
   velocity estimate.
5. **`ff_gain` stays below 1.0** (currently 0.9) — 1.0+ self-sustains against a
   static false target. See above.
6. **Restart the vision server (Stop -> Start) to load Python changes.** A page
   refresh or `systemctl restart` is not enough.

---

## 12. Roadmap — planned improvements

Everything here traces back to the same root finding from §2: at ~15 fps,
**latency is the binding constraint**, not control-loop tuning. The control
law is done; the next real gains are hardware and pipeline, in priority order.

1. **FPS.** The current ~14–15 fps ceiling is MJPG-decode/USB throughput (§5),
   and it caps *everything downstream* — tracking, servo smoothness, all of
   it. Candidates: a camera with real USB3/higher native throughput, cutting
   pipeline stages, or moving inference off the frame-capture thread.
2. **Camera quality.** Better sensor/lens = less motion blur and a cleaner
   signal into the detector, which is now the second-order error source now
   that ego-motion and steady-state lag are solved. Related, already-known
   lever: moving the camera from ~8" to 14–16" roughly doubles trackable
   angular speed for free (`w = v/d`, found via `tune_screen.py`, §3).
3. **Vision model quality.** The custom drone detector hit mAP@50 0.976 (§7)
   but was trained on a small, single-session dataset — more/varied training
   data, and revisiting the blob tracker's glare-rejection heuristics (§4),
   are the next targets.
4. **Two-camera design for target distance (triangulation).** Previously
   scoped as "laser parallax" — flanking the laser with two cameras. Three
   options discussed: **(a)** average the two centroids to approximate a
   virtual camera at the laser (simplest, corrects parallax only), **(b)**
   true stereo triangulation for a real distance estimate (enables
   distance-aware aim correction, projectile drop compensation, etc.),
   **(c)** single-camera parallax correction using a fixed offset (cheapest,
   least capable). Distance triangulation (b) is the growth path — it's a
   superset of what (a) fixes. Bore-sight offset and servo backlash are
   separate error sources from parallax and would still need their own
   calibration.
5. **Better, smoother servos.** MG996R are hobby-grade — coarse steps and
   noticeable jitter at small deltas. Higher-resolution or digital/serial
   servos (or closed-loop servos with their own position feedback) would cut
   the mechanical noise floor the control loop is currently fighting under.

### Housekeeping (do before/while publishing)

- **Rotate the Roboflow API key** — it was pasted into a chat transcript
  (never committed) and should be regenerated in Roboflow Settings -> API
  Keys before this repo goes public.
- **Secrets scan (2026-08-19): clean.** Grepped all tracked files for
  key/token/secret/password/wifi patterns — no hits. `.claude/settings.local.json`
  (holds a WiFi credential) is gitignored and confirmed untracked. Re-run this
  scan if more config files are added before publishing.
- **USB stability** — improved by the direct-port move (§5); worth continuing
  to watch under real-world use.
