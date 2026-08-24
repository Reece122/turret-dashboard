"use client";

import { useEffect, useState } from "react";

type AimMode = "head" | "chest" | "hand";
type Predictor = "none" | "ema" | "kalman";

type Telemetry = {
  fps: number;
  target_id: number | null;
  offset_x: number;
  offset_y: number;
  pan: number;
  tilt: number;
  p: number;
  i: number;
  d: number;
  speed: number;
  lead_frac?: number; // 0..1 how much of the lead is active right now
  mode?: string; // "moving" | "holding" (display only)
  servo?: string; // "on" | "off" | "sim"
};

type Config = {
  model: string;
  aim: AimMode;
  aim_drop: number;
  aim_offset_y: number;
  face_safe: boolean;
  track_mode: "yolo" | "blob";
  blob_thresh: number;
  blob_min_area: number;
  blob_circularity: number;
  cam_manual_exposure: boolean;
  cam_exposure: number;
  target_class: number;
  auto_q: boolean;
  predictor: Predictor;
  lead: number;
  deadband: number;
  kp: number;
  ki: number;
  kd: number;
  kalman_q: number;
  kalman_r: number;
  ff_gain: number;
  move_threshold: number;
  px_per_deg: number;
  servo_enabled: boolean;
  pan_channel: number;
  tilt_channel: number;
  pan_min: number;
  pan_max: number;
  tilt_min: number;
  tilt_max: number;
  pan_invert: boolean;
  tilt_invert: boolean;
  pan_center: number;
  tilt_center: number;
  _models: string[];
  _aims: AimMode[];
  _predictors: Predictor[];
  _class_names: Record<string, string>; // {index: name} of the active model
};

const modelLabel = (m: string) => {
  const trt = m.endsWith(".engine");
  const base = m.replace(/\.(pt|engine)$/, "");
  // Custom models (not the stock yolo11 family) show their filename, e.g. "drone".
  if (!base.startsWith("yolo11")) return `${base}${trt ? " ⚡" : ""}`;
  const det = !m.includes("-pose"); // detection model (no keypoints)
  const size = (m.replace("yolo11", "")[0] || "?").toUpperCase(); // N/S/M
  return `${size}${det ? " DET" : ""}${trt ? " ⚡" : ""}`;
};
const predLabel = (p: string) => (p === "none" ? "Off" : p === "ema" ? "EMA" : "Kalman");

// COCO class indices (yolo11n/-pose). Custom single-class models (e.g. drone) use 0.
const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
  "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
  "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
  "hair drier", "toothbrush",
];

// Shown before (or without) a backend connection so the controls always render.
const DEFAULT_CONFIG: Config = {
  model: "yolo11n-pose.pt",
  aim: "chest",
  aim_drop: 0.2,
  aim_offset_y: 0,
  face_safe: false,
  track_mode: "yolo",
  blob_thresh: 60,
  blob_min_area: 150,
  blob_circularity: 0.45,
  cam_manual_exposure: false,
  cam_exposure: 150,
  target_class: 0,
  auto_q: false,
  predictor: "ema",
  lead: 0.15,
  deadband: 15,
  kp: 0.03,
  ki: 0.001,
  kd: 0.005,
  kalman_q: 8000,
  kalman_r: 1,
  ff_gain: 1.0,
  move_threshold: 70,
  px_per_deg: 10.7,
  servo_enabled: true,
  pan_channel: 0,
  tilt_channel: 1,
  pan_min: 0,
  pan_max: 180,
  tilt_min: 0,
  tilt_max: 180,
  pan_invert: false,
  tilt_invert: false,
  pan_center: 90,
  tilt_center: 110,
  _models: ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt"],
  _aims: ["head", "chest", "hand"],
  _predictors: ["none", "ema", "kalman"],
  _class_names: {},
};

export default function Dashboard() {
  const [data, setData] = useState<Telemetry | null>(null);
  const [connected, setConnected] = useState(false);
  const [cfg, setCfg] = useState<Config>(DEFAULT_CONFIG);
  const [presets, setPresets] = useState<string[]>([]);
  const [presetName, setPresetName] = useState("");

  // The Python server lives on whatever host served this page, port 8000. This
  // makes it work both locally and over WiFi from another device, with no
  // hardcoded IP. Resolved after mount to avoid an SSR/hydration mismatch.
  const [backend, setBackend] = useState("");
  useEffect(() => {
    setBackend(`http://${window.location.hostname}:8000`);
  }, []);

  // Sync controls with the server's actual settings once it's reachable.
  useEffect(() => {
    if (!backend) return;
    fetch(`${backend}/config`)
      .then((r) => r.json())
      .then((c) => setCfg({ ...DEFAULT_CONFIG, ...c }))
      .catch(() => {});
  }, [backend]);

  // Poll live telemetry for the readouts.
  useEffect(() => {
    if (!backend) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${backend}/telemetry`);
        setData(await res.json());
        setConnected(true);
      } catch {
        setConnected(false);
      }
    }, 150); // poll about 7 times a second
    return () => clearInterval(interval);
  }, [backend]);

  // The MJPEG stream is one long-lived connection; when the Python server
  // restarts it dies and the <img> won't retry on its own (hence the old
  // "hard refresh to get video back"). Bump a nonce to remount the img
  // whenever the server (re)connects, and again on any load error.
  const [videoNonce, setVideoNonce] = useState(0);
  useEffect(() => {
    if (connected) setVideoNonce((n) => n + 1);
  }, [connected]);

  // Update settings optimistically, then push to the server.
  function patch(partial: Partial<Config>) {
    setCfg((prev) => (prev ? { ...prev, ...partial } : prev));
    fetch(`${backend}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(partial),
    }).catch(() => {});
  }

  // Re-pull config from the server (used after a model swap so the target-class
  // dropdown relabels to the newly-loaded model's classes).
  function refetchConfig() {
    if (!backend) return;
    fetch(`${backend}/config`)
      .then((r) => r.json())
      .then((c) => setCfg({ ...DEFAULT_CONFIG, ...c }))
      .catch(() => {});
  }

  // Switch model, then re-pull config once it has loaded (its class names change).
  function selectModel(m: string) {
    patch({ model: m });
    setTimeout(refetchConfig, 2000); // model load takes ~1-2s
  }

  // Snap the turret back to mechanical center (90/90).
  function center() {
    if (!backend) return;
    fetch(`${backend}/center`, { method: "POST" }).catch(() => {});
  }

  // ---- Presets (named parameter sets, stored on the server) ----
  function refreshPresets() {
    if (!backend) return;
    fetch(`${backend}/presets`)
      .then((r) => r.json())
      .then((d) => setPresets(d.presets || []))
      .catch(() => {});
  }
  useEffect(refreshPresets, [backend]);

  // When the server (re)connects, re-pull config and presets. Both are otherwise
  // only fetched on mount, so a model added later (e.g. a new .engine) or presets
  // saved while the server was down wouldn't show until a manual refresh.
  useEffect(() => {
    if (!backend || !connected) return;
    fetch(`${backend}/config`)
      .then((r) => r.json())
      .then((c) => setCfg({ ...DEFAULT_CONFIG, ...c }))
      .catch(() => {});
    refreshPresets();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected]);

  async function savePreset() {
    const name = presetName.trim();
    if (!backend || !name) return;
    await fetch(`${backend}/presets/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).catch(() => {});
    setPresetName("");
    refreshPresets();
  }

  async function loadPreset(name: string) {
    if (!backend) return;
    try {
      const r = await fetch(`${backend}/presets/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const d = await r.json();
      if (d.config) setCfg({ ...DEFAULT_CONFIG, ...d.config });
    } catch {
      /* ignore */
    }
  }

  async function deletePreset(name: string) {
    if (!backend) return;
    await fetch(`${backend}/presets/delete`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).catch(() => {});
    refreshPresets();
  }

  // Start/stop the Python vision server from the dashboard.
  const [busy, setBusy] = useState(false);

  async function toggleServer() {
    setBusy(true);
    try {
      await fetch("/api/server", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: connected ? "stop" : "start" }),
      });
    } catch {
      /* ignore */
    }
    // Give Flask a moment to bind (start) or release (stop) before re-enabling.
    setTimeout(() => setBusy(false), 2000);
  }

  const hasTarget = data?.target_id != null;

  return (
    <main className="page">
      <header className="header">
        <div className="logo" aria-hidden>
          <CrosshairIcon />
        </div>
        <div className="title-wrap">
          <h1 className="title">AutoTurret</h1>
          <span className="subtitle">OBJECT-TRACKING CONTROL</span>
        </div>
        <div className="header-actions">
          <button
            type="button"
            className="server-btn"
            onClick={center}
            disabled={!connected}
            title="Snap pan/tilt back to mechanical center"
          >
            Center
          </button>
          <button
            type="button"
            className={`server-btn ${connected ? "stop" : "start"}`}
            onClick={toggleServer}
            disabled={busy}
          >
            {busy ? "Working…" : connected ? "Stop server" : "Start server"}
          </button>
          <span className="status">
            <span className={`dot ${connected ? "on" : "off"}`} />
            {connected ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </header>

      <div className="grid">
        <div className="video-col">
        <div className="video-panel">
          {/* The MJPEG stream displays in a plain img tag. */}
          {backend && (
            <img
              key={videoNonce}
              src={`${backend}/video?n=${videoNonce}`}
              alt="Turret view"
              className="video"
              onError={() => {
                // Stream dropped (server restarting) — retry shortly.
                setTimeout(() => setVideoNonce((n) => n + 1), 1500);
              }}
            />
          )}
          <div className="hud">
            <span className="bracket tl" />
            <span className="bracket tr" />
            <span className="bracket bl" />
            <span className="bracket br" />
            <div className="reticle">
              <div className="reticle-ring" />
            </div>
            <div className="feed-tag">
              CAM 01 · {connected ? `${data?.fps?.toFixed(0) ?? "--"} FPS` : "NO SIGNAL"}
            </div>
          </div>
        </div>

          <div
            className="readouts"
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: 12,
              marginTop: 12,
            }}
          >
            <Stat label="FPS" value={data ? data.fps.toFixed(1) : "--"} muted={!data} />
            <Stat
              label="Target ID"
              value={hasTarget ? data!.target_id : "none"}
              muted={!hasTarget}
            />
            <Stat label="Offset X" value={data?.offset_x ?? "--"} unit="px" muted={!data} small />
            <Stat label="Offset Y" value={data?.offset_y ?? "--"} unit="px" muted={!data} small />
            <Stat
              label="Target speed"
              value={hasTarget ? data!.speed : "--"}
              unit="px/s"
              muted={!hasTarget}
              small
            />
            <Bearing label="Pan" value={data?.pan ?? null} max={180} />
            <Bearing label="Tilt" value={data?.tilt ?? null} max={90} />
            <div className="card" style={{ gridColumn: "1 / -1" }}>
              <div className="pid-title">PID · pan</div>
              <div className="row">
                <Stat label="P" value={data?.p ?? "--"} muted={!data} small />
                <Stat label="I" value={data?.i ?? "--"} muted={!data} small />
                <Stat label="D" value={data?.d ?? "--"} muted={!data} small />
              </div>
            </div>
          </div>
        </div>

        <aside className="sidebar">
          {cfg && (
            <>
              <div className="card">
                <div className="pid-title">Presets</div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    value={presetName}
                    onChange={(e) => setPresetName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && savePreset()}
                    placeholder="preset name"
                    style={{
                      flex: 1,
                      minWidth: 0,
                      background: "#0a0e15",
                      border: "1px solid var(--border-soft)",
                      borderRadius: 8,
                      color: "var(--text)",
                      padding: "8px 10px",
                      fontFamily: "inherit",
                      fontSize: 12,
                      outline: "none",
                    }}
                  />
                  <button
                    type="button"
                    className="mode-btn"
                    style={{ flex: "0 0 auto", padding: "0 16px" }}
                    onClick={savePreset}
                  >
                    Save
                  </button>
                </div>
                {presets.length === 0 ? (
                  <div className="hint">No presets yet &mdash; tune, name it, Save.</div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                    {presets.map((n) => (
                      <div key={n} style={{ display: "flex", gap: 6 }}>
                        <button
                          type="button"
                          className="mode-btn"
                          style={{ flex: 1 }}
                          onClick={() => loadPreset(n)}
                        >
                          {n}
                        </button>
                        <button
                          type="button"
                          className="mode-btn"
                          style={{ flex: "0 0 auto", padding: "0 12px" }}
                          onClick={() => deletePreset(n)}
                          aria-label={`Delete ${n}`}
                        >
                          &times;
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="card">
                <div className="pid-title">Aim point</div>
                {cfg.model.includes("-pose") ? (
                  <>
                    <SegGroup
                      options={cfg._aims}
                      value={cfg.aim}
                      onSelect={(v) => patch({ aim: v as AimMode })}
                    />
                    {cfg.aim === "chest" && (
                      <Slider
                        label="Torso drop"
                        value={cfg.aim_drop}
                        min={0}
                        max={1}
                        step={0.01}
                        display={`${(cfg.aim_drop * 100).toFixed(0)}%`}
                        onChange={(v) => patch({ aim_drop: v })}
                      />
                    )}
                    <button
                      type="button"
                      className={`mode-btn ${cfg.face_safe ? "active" : ""}`}
                      style={{ width: "100%", marginTop: 10 }}
                      onClick={() => patch({ face_safe: !cfg.face_safe })}
                    >
                      {cfg.face_safe ? "Face-safe: ON — laser stays below face" : "Face-safe: OFF"}
                    </button>
                  </>
                ) : (
                  <div className="hint" style={{ margin: "2px 0 4px" }}>
                    Detection model — aims at the bounding-box center. Head/Chest/Hand
                    and face-safe apply to pose models only.
                  </div>
                )}
                <Slider
                  label="Aim below target"
                  value={cfg.aim_offset_y}
                  min={0}
                  max={300}
                  step={1}
                  display={`${cfg.aim_offset_y} px`}
                  onChange={(v) => patch({ aim_offset_y: v })}
                />
                <div className="slider-row" style={{ marginTop: 10 }}>
                  <label className="slider-label">Target class</label>
                  <select
                    className="class-select"
                    value={cfg.target_class}
                    onChange={(e) => patch({ target_class: Number(e.target.value) })}
                  >
                    {(Object.keys(cfg._class_names).length
                      ? Object.entries(cfg._class_names)
                          .map(([i, name]) => [Number(i), name] as [number, string])
                          .sort((a, b) => a[0] - b[0])
                      : COCO_CLASSES.map((name, i) => [i, name] as [number, string])
                    ).map(([i, name]) => (
                      <option key={i} value={i}>
                        {i} — {name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="hint">
                  Class the active model tracks (its own class names, e.g.
                  &quot;drone&quot; for the drone model). Head/Chest/Hand apply to
                  pose models only; detection models aim at box center.
                </div>
              </div>

              <div className="card">
                <div className="pid-title">Tracking</div>
                <SegGroup
                  options={["yolo", "blob"]}
                  value={cfg.track_mode}
                  label={(m) => (m === "blob" ? "Ball (blob)" : "Model (YOLO)")}
                  onSelect={(v) => patch({ track_mode: v as "yolo" | "blob" })}
                />
                {cfg.track_mode === "blob" ? (
                  <>
                    <Slider
                      label="Brightness threshold"
                      value={cfg.blob_thresh}
                      min={0}
                      max={255}
                      step={1}
                      display={`${cfg.blob_thresh}`}
                      onChange={(v) => patch({ blob_thresh: v })}
                    />
                    <Slider
                      label="Min blob size"
                      value={cfg.blob_min_area}
                      min={10}
                      max={20000}
                      step={10}
                      display={`${cfg.blob_min_area} px²`}
                      onChange={(v) => patch({ blob_min_area: v })}
                    />
                    <Slider
                      label="Roundness gate"
                      value={cfg.blob_circularity}
                      min={0}
                      max={1}
                      step={0.05}
                      display={cfg.blob_circularity.toFixed(2)}
                      onChange={(v) => patch({ blob_circularity: v })}
                    />
                    <button
                      type="button"
                      className={`mode-btn ${cfg.cam_manual_exposure ? "active" : ""}`}
                      style={{ width: "100%", marginTop: 10 }}
                      onClick={() => patch({ cam_manual_exposure: !cfg.cam_manual_exposure })}
                    >
                      {cfg.cam_manual_exposure
                        ? "Manual exposure: ON — faster + sharper"
                        : "Manual exposure: OFF (auto)"}
                    </button>
                    {cfg.cam_manual_exposure && (
                      <Slider
                        label="Exposure"
                        value={cfg.cam_exposure}
                        min={1}
                        max={2000}
                        step={1}
                        display={`${cfg.cam_exposure}`}
                        onChange={(v) => patch({ cam_exposure: v })}
                      />
                    )}
                    <div className="hint">
                      Tracks a bright ball on a dark background. Raise threshold if it
                      grabs stray light; raise min size to ignore specks. Roundness gate
                      rejects light reflections (a ball is round, glare is a streak) —
                      raise it if reflections still get picked. In a dark room turn on
                      Manual exposure and lower Exposure until FPS jumps and blur
                      disappears (add room light to keep the ball bright).
                    </div>
                  </>
                ) : (
                  <div style={{ marginTop: 10 }}>
                    <SegGroup
                      options={cfg._models}
                      value={cfg.model}
                      label={modelLabel}
                      onSelect={selectModel}
                    />
                    <div className="hint">Switching reloads the model — the feed pauses briefly.</div>
                  </div>
                )}
              </div>

              <div className="card">
                <div className="pid-title">Prediction</div>
                <SegGroup
                  options={cfg._predictors}
                  value={cfg.predictor}
                  label={predLabel}
                  onSelect={(v) => patch({ predictor: v as Predictor })}
                />
                {cfg.predictor !== "none" && (
                  <>
                    <Slider
                      label="Lead"
                      value={cfg.lead}
                      min={0}
                      max={1}
                      step={0.01}
                      display={`${(cfg.lead * 1000).toFixed(0)} ms`}
                      onChange={(v) => patch({ lead: v })}
                    />
                  </>
                )}
                {cfg.predictor === "kalman" && (
                  <>
                    <Slider
                      label="Process noise (Q)"
                      value={cfg.kalman_q}
                      min={1}
                      max={60000}
                      step={100}
                      display={cfg.kalman_q.toFixed(0)}
                      onChange={(v) => patch({ kalman_q: v })}
                    />
                    <Slider
                      label="Measurement noise (R)"
                      value={cfg.kalman_r}
                      min={0.1}
                      max={50}
                      step={0.1}
                      display={cfg.kalman_r.toFixed(1)}
                      onChange={(v) => patch({ kalman_r: v })}
                    />
                    <button
                      type="button"
                      className={`mode-btn ${cfg.auto_q ? "active" : ""}`}
                      style={{ width: "100%", marginTop: 10 }}
                      onClick={() => patch({ auto_q: !cfg.auto_q })}
                    >
                      {cfg.auto_q ? "Auto-Q: ON — scales with target size" : "Auto-Q: OFF"}
                    </button>
                  </>
                )}
              </div>

              <div className="card">
                <div className="pid-title">Motion</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <span
                    className="mode-btn active"
                    style={{
                      flex: "none",
                      padding: "4px 12px",
                      cursor: "default",
                      background: data?.mode === "moving" ? "var(--live)" : "var(--accent)",
                      borderColor: data?.mode === "moving" ? "var(--live)" : "var(--accent)",
                    }}
                  >
                    {data?.mode === "moving" ? "MOVING" : "HOLDING"}
                  </span>
                  <span className="hint" style={{ margin: 0 }}>
                    {data?.speed ?? 0} px/s world speed
                  </span>
                </div>
                <Slider
                  label="Feedforward"
                  value={cfg.ff_gain}
                  min={0}
                  max={2}
                  step={0.05}
                  display={cfg.ff_gain === 0 ? "off" : `${cfg.ff_gain.toFixed(2)}x`}
                  onChange={(v) => patch({ ff_gain: v })}
                />
                <Slider
                  label="Camera px per deg"
                  value={cfg.px_per_deg}
                  min={0}
                  max={40}
                  step={0.1}
                  display={cfg.px_per_deg.toFixed(1)}
                  onChange={(v) => patch({ px_per_deg: v })}
                />
                <Slider
                  label="Moving threshold"
                  value={cfg.move_threshold}
                  min={0}
                  max={600}
                  step={5}
                  display={`${cfg.move_threshold.toFixed(0)} px/s`}
                  onChange={(v) => patch({ move_threshold: v })}
                />
                <div className="hint">
                  Feedforward commands the target&apos;s own angular rate, so the P gain only
                  trims what&apos;s left over. That is what removes the steady-state lag
                  (<i>lag = speed x dt / P</i>) and lets one gentle gain cover a still target
                  and a fast one &mdash; no acquire/track switching. 1.00x = exact; 0 = off
                  (old lag-prone behaviour). Camera px/deg converts servo degrees to pixels
                  (&asymp; frame width &divide; camera FOV) and is what makes the velocity
                  estimate ego-motion free. Moving threshold only drives the badge above.
                </div>
              </div>

              <div className="card">
                <div className="pid-title">Tuning</div>
                <Slider
                  label="Deadband"
                  value={cfg.deadband}
                  min={0}
                  max={100}
                  step={1}
                  display={`${cfg.deadband.toFixed(0)} px`}
                  onChange={(v) => patch({ deadband: v })}
                />
                <Slider
                  label="P gain"
                  value={cfg.kp}
                  min={0}
                  max={0.3}
                  step={0.005}
                  display={cfg.kp.toFixed(3)}
                  onChange={(v) => patch({ kp: v })}
                />
                <Slider
                  label="I gain"
                  value={cfg.ki}
                  min={0}
                  max={0.05}
                  step={0.001}
                  display={cfg.ki.toFixed(3)}
                  onChange={(v) => patch({ ki: v })}
                />
                <Slider
                  label="D gain"
                  value={cfg.kd}
                  min={0}
                  max={0.1}
                  step={0.001}
                  display={cfg.kd.toFixed(3)}
                  onChange={(v) => patch({ kd: v })}
                />
              </div>

              <div className="card">
                <div className="pid-title">Servo output</div>
                <SegGroup
                  options={["on", "off"]}
                  value={cfg.servo_enabled ? "on" : "off"}
                  label={(v) => (v === "on" ? "Enabled" : "Disabled")}
                  onSelect={(v) => patch({ servo_enabled: v === "on" })}
                />
                <div className="hint">
                  {data?.servo === "sim"
                    ? "No PCA9685 detected — running in simulation (no motion)."
                    : data?.servo === "on"
                    ? `Driving servos on channels ${cfg.pan_channel} (pan) / ${cfg.tilt_channel} (tilt).`
                    : "Output off — the PID still runs but the servos won't move."}
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <div style={{ flex: 1 }}>
                    <div className="slider-label" style={{ marginBottom: 6 }}>Pan dir</div>
                    <SegGroup
                      options={["fwd", "rev"]}
                      value={cfg.pan_invert ? "rev" : "fwd"}
                      label={(v) => (v === "rev" ? "Invert" : "Normal")}
                      onSelect={(v) => patch({ pan_invert: v === "rev" })}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div className="slider-label" style={{ marginBottom: 6 }}>Tilt dir</div>
                    <SegGroup
                      options={["fwd", "rev"]}
                      value={cfg.tilt_invert ? "rev" : "fwd"}
                      label={(v) => (v === "rev" ? "Invert" : "Normal")}
                      onSelect={(v) => patch({ tilt_invert: v === "rev" })}
                    />
                  </div>
                </div>
                <Slider
                  label="Pan min"
                  value={cfg.pan_min}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.pan_min.toFixed(0)}°`}
                  onChange={(v) => patch({ pan_min: v })}
                />
                <Slider
                  label="Pan max"
                  value={cfg.pan_max}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.pan_max.toFixed(0)}°`}
                  onChange={(v) => patch({ pan_max: v })}
                />
                <Slider
                  label="Tilt min"
                  value={cfg.tilt_min}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.tilt_min.toFixed(0)}°`}
                  onChange={(v) => patch({ tilt_min: v })}
                />
                <Slider
                  label="Tilt max"
                  value={cfg.tilt_max}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.tilt_max.toFixed(0)}°`}
                  onChange={(v) => patch({ tilt_max: v })}
                />
                <Slider
                  label="Pan center"
                  value={cfg.pan_center}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.pan_center.toFixed(0)}°`}
                  onChange={(v) => patch({ pan_center: v })}
                />
                <Slider
                  label="Tilt center"
                  value={cfg.tilt_center}
                  min={0}
                  max={180}
                  step={1}
                  display={`${cfg.tilt_center.toFixed(0)}°`}
                  onChange={(v) => patch({ tilt_center: v })}
                />
              </div>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}

function SegGroup({
  options,
  value,
  onSelect,
  label,
}: {
  options: string[];
  value: string;
  onSelect: (v: string) => void;
  label?: (v: string) => string;
}) {
  return (
    <div className="mode-group">
      {options.map((o) => (
        <button
          key={o}
          type="button"
          className={`mode-btn ${value === o ? "active" : ""}`}
          onClick={() => onSelect(o)}
        >
          {label ? label(o) : o}
        </button>
      ))}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="slider-row">
      <div className="slider-head">
        <span className="slider-label">{label}</span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <input
            type="number"
            value={value}
            min={min}
            max={max}
            step={step}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
            }}
            style={{
              width: 62,
              background: "#0a0e15",
              border: "1px solid var(--border-soft)",
              borderRadius: 6,
              color: "var(--accent)",
              fontFamily: "ui-monospace, monospace",
              fontSize: 12,
              textAlign: "right",
              padding: "2px 6px",
              outline: "none",
            }}
          />
          {display && (
            <span className="slider-val" style={{ fontSize: 10, opacity: 0.6 }}>
              {display}
            </span>
          )}
        </span>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
  small,
  muted,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  small?: boolean;
  muted?: boolean;
}) {
  return (
    <div className={`card ${small ? "stat-small" : ""}`}>
      <div className="label">{label}</div>
      <div className={`value ${muted ? "muted" : ""}`}>
        {value}
        {unit ? <span className="unit"> {unit}</span> : null}
      </div>
    </div>
  );
}

// A stat with a centered bearing bar that fills left/right of center.
function Bearing({ label, value, max }: { label: string; value: number | null; max: number }) {
  const v = value ?? 0;
  const clamped = Math.max(-max, Math.min(max, v));
  const pct = (Math.abs(clamped) / max) * 50; // 0–50% from center
  const left = clamped >= 0 ? 50 : 50 - pct;

  return (
    <div className="card stat-small">
      <div className="label">{label}</div>
      <div className={`value ${value == null ? "muted" : ""}`}>
        {value == null ? "--" : `${value}°`}
      </div>
      <div className="bearing-track">
        <div className="bearing-fill" style={{ left: `${left}%`, width: `${pct}%` }} />
      </div>
    </div>
  );
}

function CrosshairIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="12" cy="12" r="7.5" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none" />
      <line x1="12" y1="1.5" x2="12" y2="5" />
      <line x1="12" y1="19" x2="12" y2="22.5" />
      <line x1="1.5" y1="12" x2="5" y2="12" />
      <line x1="19" y1="12" x2="22.5" y2="12" />
    </svg>
  );
}
