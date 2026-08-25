<div align="center">

# AutoTurret

**A camera-driven, object-tracking pan/tilt turret** — Jetson-hosted vision, a
closed-loop control law tuned in simulation before it ever touched hardware,
and a live web dashboard to run and tune it from any device on the network.

[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-149ECA?logo=react&logoColor=white)](https://react.dev)
[![Python](https://img.shields.io/badge/Python-Flask-3776AB?logo=python&logoColor=white)](https://flask.palletsprojects.com)
[![Ultralytics](https://img.shields.io/badge/Vision-YOLO11-8A2BE2)](https://docs.ultralytics.com)
[![Jetson Orin Nano](https://img.shields.io/badge/Compute-Jetson%20Orin%20Nano-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/embedded/jetson-orin)

</div>

<br>

<p align="center">
  <video src="media/drone-3rd-person.mov" controls width="100%">
    Demo video — see <code>media/drone-3rd-person.mov</code>.
  </video>
</p>

---

## Demo

**Drone tracking**

| 3rd person | Dashboard POV |
|---|---|
| <video src="media/drone-3rd-person.mov" controls width="100%"></video> | <video src="media/drone-dashboard-pov.mov" controls width="100%"></video> |

**Ball tracking** — same control law, three motion patterns

| Straight-line | Figure-8 | Pinpoint (acquire/reacquire) |
|---|---|---|
| <video src="media/ball-line.mov" controls width="100%"></video> | <video src="media/ball-figure8.mov" controls width="100%"></video> | <video src="media/ball-pinpoint.mov" controls width="100%"></video> |

## What it is

Two processes talking over HTTP: a **Python/Flask vision server** that owns
the camera, runs YOLO11 detection, and drives two servos over I2C; and a
**Next.js dashboard** that streams the annotated feed, exposes every control
gain as a live slider, and can start/stop/restart the vision server remotely.
Point it at a moving target and it tracks and aims a laser at it in real time.

The interesting part isn't the demo — it's getting from "twitches on a
stationary target, lags behind a fast one" to a single, stable control law
that handles both. That's documented in full in **[DEVLOG.md](DEVLOG.md)**.

## Highlights

- **Velocity feedforward control** — commands the target's own angular rate
  directly instead of waiting for position error to build up, collapsing what
  used to be a fragile two-mode (acquire/track) state machine into one gain
  that's stable at every speed. [Details →](DEVLOG.md#2-the-tracking-control-saga-the-big-one)
- **World-space tracking** — target velocity is estimated in world space
  (image position + camera angle), not image space, so panning to follow the
  target is never mistaken for the target moving.
- **Custom-trained YOLO11 detector** — a from-scratch drone detector
  (mAP@50 0.976) trained and fine-tuned on the Jetson itself, alongside a
  from-scratch OpenCV blob tracker as a no-neural-net fallback mode.
- **Simulated before it was tuned on hardware** — every control-law and gain
  change was validated in a closed-loop 2D simulator first; hardware time was
  spent confirming, not guessing. See `tools/tune.py`.
- **Face-safety clamp** — with an always-on laser payload, aim is hard-clamped
  to at or below the shoulder line whenever the target class is a person.
- **Fully live-tunable** — every PID/Kalman/feedforward gain, camera setting,
  and servo limit is a dashboard slider with immediate effect — no redeploy
  to tune.

## How it works

| Piece | Detail |
|---|---|
| Compute | Jetson Orin Nano 8GB, JetPack 6.2.3 |
| Vision | YOLO11 (pose + custom drone detector) and an OpenCV blob tracker |
| Control | PID + velocity feedforward on a Kalman-filtered world-space estimate |
| Actuation | 2x servos via PCA9685 over I2C |
| Payload | Always-on laser designator |
| Backend | `python_server.py` — Flask + MJPEG stream + vision/control thread, port 8000 |
| Frontend | Next.js 16 / React 19 dashboard, `app/page.tsx`, port 3000 |
| Test rig | `tools/drone_target.html` — animated on-screen target for indoor tuning |

```
┌────────────────────┐        spawns/kills, proxies stream        ┌───────────────────────┐
│  Next.js dashboard  │ ───────────────────────────────────────▶  │  Flask vision server   │
│  (port 3000)        │ ◀───────────────────────────────────────  │  (port 8000)           │
│  live config sliders│        MJPEG feed + telemetry JSON         │  camera → YOLO11/blob  │
└────────────────────┘                                            │  → Kalman → PID+FF     │
                                                                    │  → PCA9685 → servos    │
                                                                    └───────────────────────┘
```

## Hardware & CAD

Full wiring, power domains, and bring-up steps are in **[HARDWARE.md](HARDWARE.md)**.

CAD source lives in [`/cad`](cad):

| File | Format | Notes |
|---|---|---|
| [`FinalAssembly.SLDASM`](cad/FinalAssembly.SLDASM) | SolidWorks assembly | Full turret assembly, parametric |

## Tech stack

**Frontend:** Next.js 16, React 19, TypeScript
**Backend:** Python, Flask, OpenCV, NumPy
**Vision:** Ultralytics YOLO11 (pose + custom-trained detection), custom Kalman filter
**Hardware:** NVIDIA Jetson Orin Nano, PCA9685 servo driver, 2x hobby servos, USB camera, laser module

## Getting started

```bash
# one-time Jetson setup (Node, Python deps, I2C, systemd service)
sudo bash jetson_setup.sh

# dev — dashboard only (vision server is started from the UI's Start button)
npm run dev

# or reachable from other devices on the LAN
npm run dev:lan
```

The dashboard's **Start** button spawns `python_server.py` (see
`app/api/server/route.ts`); it isn't run manually. With no PCA9685 attached,
servo output degrades to a silent no-op so the vision loop still runs
standalone (e.g. on a laptop against `tools/drone_target.html`).

## Project journal

**[DEVLOG.md](DEVLOG.md)** is the full build log — nine failed approaches to
the tracking-control bug before the actual fix, the 9→15 FPS camera-driver
investigation, the drone-detector training runs, and every hardware bump
along the way. Written to capture *why*, not just *what*, so worth a read if
you want the engineering story rather than just the code.

## Roadmap

Next up, in priority order (latency is the documented bottleneck on
everything below it — see [DEVLOG §12](DEVLOG.md#12-roadmap--planned-improvements)):

1. **Camera fps + quality** — current ~15fps ceiling is USB/MJPG-decode
   throughput; a better sensor fixes both fps and image quality at once.
2. **Better, smoother servos** — hobby-grade MG996R are the current
   mechanical noise floor.
3. **Two-camera distance triangulation** — stereo rig flanking the laser for
   a real distance estimate, not just parallax correction.
4. **Vision model quality** — more/varied training data for the drone
   detector.

## License

No license file yet — all rights reserved by default until one is added.
