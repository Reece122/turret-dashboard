"""Closed-loop simulation of the turret to tune Kalman + PID for a walking target.

Replicates python_server.py's control law:
 - Kalman constant-velocity filter on the target's pixel position
 - prediction `lead` seconds ahead
 - incremental PID:  pan_angle += kp*err + ki*integral + kd*d(err)
 - servo modelled with a per-frame slew limit + 1-frame sensing latency

Target = a person walking at 5-10 ft: alternating steady moves and full stops,
so we can score LAG (error while moving) and OSCILLATION (motion while still).
"""
import random
import math

DT = 0.05                 # 20 fps
PPD = 640.0 / 60.0        # px per degree (640px / ~60 deg FOV)
CENTER = 320.0
NOISE_STD = 4.0           # keypoint jitter, px
SLEW = 10.0               # servo max move per frame (deg) ~ mid-grade servo
SENSOR_DELAY = 1          # frames of detect+actuate latency

# bearing(deg) over time: (t_end, bearing_at_t_end), linearly interpolated
SEG = [(1.0, 90), (3.0, 120), (4.5, 120), (7.0, 60),
       (8.5, 60), (10.0, 100), (12.0, 100)]
STILL = [(3.0, 4.5), (7.0, 8.5), (10.0, 12.0)]   # windows where target is stopped
TOTAL = 12.0


def bearing(t):
    pt, pb = 0.0, 90.0
    for te, b in SEG:
        if t <= te:
            f = (t - pt) / (te - pt) if te > pt else 1.0
            return pb + f * (b - pb)
        pt, pb = te, b
    return SEG[-1][1]


def clamp(v, a, b):
    return a if v < a else (b if v > b else v)


class Kalman1D:
    def __init__(self):
        self.pos = None
        self.vel = 0.0
        self.P = [[500.0, 0.0], [0.0, 500.0]]

    def step(self, z, dt, q, r):
        if self.pos is None:
            self.pos, self.vel = z, 0.0
            self.P = [[500.0, 0.0], [0.0, 500.0]]
            return
        self.pos += self.vel * dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = [[q * dt4 / 4, q * dt3 / 2], [q * dt3 / 2, q * dt2]]
        P = self.P
        # P = F P F^T + Q  with F = [[1,dt],[0,1]]
        a = P[0][0] + dt * P[1][0]
        b = P[0][1] + dt * P[1][1]
        c = P[1][0]
        d = P[1][1]
        P00 = a + dt * c + Q[0][0]
        P01 = b + dt * d + Q[0][1]
        P10 = c + Q[1][0]
        P11 = d + Q[1][1]
        S = P00 + r
        K0 = P00 / S
        K1 = P10 / S
        y = z - self.pos
        self.pos += K0 * y
        self.vel += K1 * y
        self.P = [[(1 - K0) * P00, (1 - K0) * P01],
                  [P10 - K1 * P00, P11 - K1 * P01]]

    def predict(self, t):
        return self.pos + self.vel * t


def simulate(kp, ki, kd, lead, lead_ramp, q, r, deadband, seed):
    kf = Kalman1D()
    pan = 90.0
    phys = 90.0
    integral = 0.0
    prev_err = 0.0
    hist = [90.0] * (SENSOR_DELAY + 1)
    rnd = random.Random(seed)
    n = int(TOTAL / DT)
    sq_err = 0.0
    still_vals = [[] for _ in STILL]
    for k in range(n):
        t = k * DT
        tb = bearing(t)
        cam = hist[-1 - SENSOR_DELAY]
        meas = CENTER + (tb - cam) * PPD + rnd.gauss(0, NOISE_STD)
        kf.step(meas, DT, q, r)
        speed = abs(kf.vel)
        eff_lead = lead * clamp(speed / lead_ramp, 0.0, 1.0) if lead_ramp > 0 else lead
        offset = kf.predict(eff_lead) - CENTER
        err = 0.0 if abs(offset) < deadband else offset
        integral = clamp(integral + err * DT, -1000, 1000)
        out = kp * err + ki * integral + kd * (err - prev_err) / DT
        prev_err = err
        pan = clamp(pan + out, 0, 180)
        phys += clamp(pan - phys, -SLEW, SLEW)
        hist.append(phys)
        sq_err += (tb - phys) ** 2
        for wi, (a0, b0) in enumerate(STILL):
            if a0 <= t < b0:
                still_vals[wi].append(phys)
    rms = math.sqrt(sq_err / n)
    jit = 0.0
    cnt = 0
    for vals in still_vals:
        if len(vals) > 2:
            m = sum(vals) / len(vals)
            jit += math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            cnt += 1
    jit = jit / cnt if cnt else 0.0
    return rms, jit


def cost_of(p, seeds=(1, 2, 3)):
    rms = jit = 0.0
    for s in seeds:
        a, b = simulate(*p, seed=s)
        rms += a
        jit += b
    rms /= len(seeds)
    jit /= len(seeds)
    return rms + 3.0 * jit, rms, jit


def show(name, p):
    c, rms, jit = cost_of(p)
    print(f"{name:10s} cost={c:6.2f}  lag(rms)={rms:5.2f}deg  jitter={jit:5.2f}deg  "
          f"| kp={p[0]:.3f} ki={p[1]:.4f} kd={p[2]:.4f} lead={p[3]*1000:.0f}ms "
          f"ramp={p[4]:.0f}px/s Q={p[5]:.0f} R={p[6]:.0f} dead={p[7]:.0f}")


# Baselines for comparison (lead_ramp=0 means constant lead, as before)
show("defaults", (0.03, 0.001, 0.005, 0.15, 0, 50, 4, 15))
show("prev tuned", (0.025, 0.0009, 0.018, 0.079, 0, 100, 1, 2))

# Random search — lead can go higher now that lead_ramp fades it near stops
rnd = random.Random(7)
best = None
for _ in range(7000):
    p = (rnd.uniform(0.01, 0.12), rnd.uniform(0.0, 0.006), rnd.uniform(0.0, 0.02),
         rnd.uniform(0.0, 0.30), rnd.uniform(0, 400), rnd.uniform(3, 100),
         rnd.uniform(2, 40), rnd.uniform(2, 35))
    c, _, _ = cost_of(p, seeds=(1, 2))
    if best is None or c < best[0]:
        best = (c, p)

# Local refine around the best (finer, 3-seed)
bp = list(best[1])
step = [0.008, 0.0006, 0.002, 0.02, 40, 8, 5, 4]
for _ in range(3000):
    cand = [clamp(bp[i] + random.Random(_ * 7 + i).uniform(-1, 1) * step[i],
                  lo, hi)
            for i, (lo, hi) in enumerate(
                [(0.005, 0.15), (0, 0.008), (0, 0.03), (0, 0.3),
                 (0, 400), (2, 100), (1, 60), (2, 40)])]
    c, _, _ = cost_of(tuple(cand))
    cc, _, _ = cost_of(tuple(bp))
    if c < cc:
        bp = cand

print("-" * 70)
show("BEST", tuple(bp))
