"""Realistic closed-loop turret sim: a 5'10" person moving side-to-side AND
toward/away from the camera (2-13.5 ft) at variable speed with jerky bursts.
High + spiky detection noise and real latency. Re-optimizes PID + Kalman + lead
(+ lead_ramp) and compares auto-Q (scale Q with apparent size) on vs off.
"""
import random
import math

DT = 1.0 / 15.0                 # ~15 fps (real on-Jetson rate)
IMG_W, IMG_H = 640, 480
HFOV = 60.0
PPD = IMG_W / HFOV              # px per degree (pan)
VFOV = 47.0
FOCAL_V = (IMG_H / 2) / math.tan(math.radians(VFOV / 2))
PERSON_H = 1.778               # 5'10" in metres
NOISE_STD = 12.0               # base keypoint noise (px) -- much higher than before
SPIKE_P = 0.06                 # fraction of frames with a big detection glitch
SPIKE_STD = 30.0
SENSOR_DELAY = 2               # frames of detect+actuate latency (~130 ms)
SLEW = 18.0                    # servo max deg/frame (~MG996R under load @15fps)
Q_REF = 240.0                  # auto-Q reference box height (matches server)
TOTAL = 14.0


def interp(kf, t):
    pt, pv = kf[0]
    for te, ve in kf[1:]:
        if t <= te:
            f = (t - pt) / (te - pt) if te > pt else 1.0
            return pv + f * (ve - pv)
        pt, pv = te, ve
    return kf[-1][1]


# lateral position (m, +right) and distance (m) keyframes
LAT = [(0, 0), (2.0, 0), (3.5, 1.0), (4.5, -1.2), (6.0, -1.2), (7.5, 1.3),
       (8.2, 1.3), (9.5, -0.4), (11.0, 0.8), (14.0, 0.0)]
DIST = [(0, 4.11), (2.0, 4.11), (4.5, 0.61), (6.0, 0.61), (7.5, 0.61),
        (9.5, 4.11), (11.0, 2.0), (14.0, 1.0)]
STILL = [(0.2, 1.9), (4.6, 5.9)]   # far-still, and near-still (body fills frame)


def bearing(t):
    return 90.0 + math.degrees(math.atan2(interp(LAT, t), interp(DIST, t)))


def box_h(t):
    return min(float(IMG_H), FOCAL_V * PERSON_H / interp(DIST, t))


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
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        Q = [[q * dt4 / 4, q * dt3 / 2], [q * dt3 / 2, q * dt2]]
        P = self.P
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


def simulate(kp, ki, kd, lead, lead_ramp, q, r, deadband, auto_q, seed):
    kf = Kalman1D()
    pan = phys = 90.0
    integral = 0.0
    prev_err = 0.0
    hist = [90.0] * (SENSOR_DELAY + 1)
    rnd = random.Random(seed)
    n = int(TOTAL / DT)
    sq = 0.0
    still_vals = [[] for _ in STILL]
    for k in range(n):
        t = k * DT
        tb = bearing(t)
        cam = hist[-1 - SENSOR_DELAY]
        noise = rnd.gauss(0, NOISE_STD)
        if rnd.random() < SPIKE_P:
            noise += rnd.gauss(0, SPIKE_STD)
        meas = IMG_W / 2 + (tb - cam) * PPD + noise
        h = box_h(t)
        q_eff = clamp(q * (h / Q_REF) ** 2, 1.0, 500.0) if auto_q else q
        kf.step(meas, DT, q_eff, r)
        speed = abs(kf.vel)
        frac = clamp(speed / lead_ramp, 0.0, 1.0) if lead_ramp > 0 else 1.0
        offset = kf.predict(lead * frac) - IMG_W / 2
        err = 0.0 if abs(offset) < deadband else offset
        integral = clamp(integral + err * DT, -1000, 1000)
        out = kp * err + ki * integral + kd * (err - prev_err) / DT
        prev_err = err
        pan = clamp(pan + out, 0, 180)
        phys += clamp(pan - phys, -SLEW, SLEW)
        hist.append(phys)
        sq += (tb - phys) ** 2
        for wi, (a0, b0) in enumerate(STILL):
            if a0 <= t < b0:
                still_vals[wi].append(phys)
    rms = math.sqrt(sq / n)
    jit = cnt = 0.0
    for vals in still_vals:
        if len(vals) > 2:
            m = sum(vals) / len(vals)
            jit += math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
            cnt += 1
    jit = jit / cnt if cnt else 0.0
    return rms, jit


def cost_of(p, auto_q, seeds=(1, 2, 3)):
    rms = jit = 0.0
    for s in seeds:
        a, b = simulate(*p, auto_q=auto_q, seed=s)
        rms += a
        jit += b
    rms /= len(seeds)
    jit /= len(seeds)
    return rms + 3.0 * jit, rms, jit


def show(name, p, auto_q):
    c, rms, jit = cost_of(p, auto_q, seeds=(1, 2, 3, 4))
    print(f"{name:22s} cost={c:6.2f}  lag(rms)={rms:5.2f}deg  jitter={jit:5.2f}deg  "
          f"| kp={p[0]:.3f} ki={p[1]:.4f} kd={p[2]:.4f} lead={p[3]*1000:.0f}ms "
          f"ramp={p[4]:.0f} Q={p[5]:.0f} R={p[6]:.0f} dead={p[7]:.0f}")


def optimize(auto_q, n_rand=3500, n_ref=1200):
    rnd = random.Random(7)
    best = None
    for _ in range(n_rand):
        p = (rnd.uniform(0.005, 0.15), rnd.uniform(0.0, 0.008), rnd.uniform(0.0, 0.04),
             rnd.uniform(0.0, 0.40), rnd.uniform(0, 500), rnd.uniform(3, 150),
             rnd.uniform(2, 80), rnd.uniform(2, 50))
        c, _, _ = cost_of(p, auto_q, seeds=(1, 2))
        if best is None or c < best[0]:
            best = (c, list(p))
    bp = best[1]
    step = [0.006, 0.0006, 0.003, 0.02, 40, 12, 8, 5]
    bounds = [(0.005, 0.15), (0, 0.008), (0, 0.04), (0, 0.4),
              (0, 500), (2, 150), (1, 80), (2, 50)]
    for i in range(n_ref):
        cand = [clamp(bp[j] + random.Random(i * 8 + j).uniform(-1, 1) * step[j], lo, hi)
                for j, (lo, hi) in enumerate(bounds)]
        if cost_of(cand, auto_q, seeds=(1, 2))[0] < cost_of(bp, auto_q, seeds=(1, 2))[0]:
            bp = cand
    return bp


DEFAULTS = (0.03, 0.001, 0.005, 0.15, 0, 50, 4, 15)
best_on = optimize(auto_q=True)
best_off = optimize(auto_q=False)

show("defaults (autoQ off)", DEFAULTS, False)
print("-" * 70)
show("BEST autoQ ON", best_on, True)
show("  ^same params, Q off", best_on, False)
show("BEST autoQ OFF", best_off, False)
