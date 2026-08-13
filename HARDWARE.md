# AutoTurret — Hardware wiring

Mental model: **three separate power domains that share one common ground.**
Never cross the rails — the only thing shared across all three is ground.

## Power domains
- **19V barrel-jack supply → Jetson only.**
- **5–6V brick (≥5A) → the PCA9685 V+ screw terminal only** (this powers the servos).
- **3.3V from the Jetson → the PCA9685 VCC** (logic only).

## Jetson 40-pin → PCA9685 (4-wire I2C link)
| Jetson pin | → | PCA9685 |
|---|---|---|
| 3.3V (pin 1) | → | VCC |
| GND (pin 6) | → | GND |
| SDA (pin 3) | → | SDA |
| SCL (pin 5) | → | SCL |

I2C is on **bus 7** on the Orin Nano. Verify the board with:
```bash
sudo i2cdetect -y -r 7      # expect 0x40 to appear
```

## Servo power → PCA9685
- 5–6V brick **+** and **−** → the green **V+ / GND** screw terminal.
- Add a **470–1000µF capacitor** across V+/GND to absorb current spikes.

## PCA9685 → servos
- **Pan servo → channel 0** (3-pin plug: signal / V+ / GND)
- **Tilt servo → channel 1**
- Servos draw power straight from the V+ rail through those headers — no separate wiring.

## Camera
- USB webcam straight into a Jetson USB port. Works with `cv2.VideoCapture(0)`.

## Laser (CHT1230, 3–6V module)
**DECISION: always-on.** No MOSFET, no GPIO, no code — the laser is lit whenever
the rig is powered.
- Wiring: red → 5V (pin 2), black → any GND pin.

<details><summary>Switchable option (not used — kept for reference)</summary>

MOSFET low-side switch:
- red → 5V
- black → MOSFET **drain**
- MOSFET **source** → GND
- Jetson **GPIO (pin 7)** → MOSFET **gate** through **~220Ω**, with a **10kΩ gate-to-source pulldown**.
</details>

## Common ground (do not skip)
Jetson GND, PCA9685 GND, and the 5–6V brick's minus **must all tie together**.
The GND wire in the 4-wire link plus the brick's minus wire already accomplish this.
Without it the servo signals float and jitter.

## Software notes
- Servo driver: `pip3 install adafruit-circuitpython-servokit`; enable I2C.
- **Servo control is implemented** in `python_server.py` (`ServoController`).
  `pan_angle` / `tilt_angle` (0–180°) are written to the PCA9685 each frame.
  With no board attached it degrades to a silent no-op ("sim" mode) so the
  vision loop still runs on a laptop.
- Laser: always-on, hardwired to 5V — no code.
- These live in `turret_config.json` (also settable via `POST /config`), so the
  mechanism can't over-travel and axes can be flipped without rewiring:
  - `servo_enabled` (master on/off), `pan_channel` (0), `tilt_channel` (1)
  - `pan_min` / `pan_max`, `tilt_min` / `tilt_max` (degrees)
  - `pan_invert` / `tilt_invert` (reverse an axis that chases the wrong way)
- First bring-up tip: set `servo_enabled` false, confirm the pan/tilt readouts
  on the dashboard move sanely, then enable. If an axis runs to its limit,
  flip that axis's `invert`.
