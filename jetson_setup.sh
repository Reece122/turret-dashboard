#!/usr/bin/env bash
# =====================================================================
# AutoTurret — one-time Jetson setup
# Run from inside the project folder:   sudo bash jetson_setup.sh
# Installs Node + Python deps, enables I2C, builds the dashboard, and
# registers a systemd service so the dashboard is always up on boot.
# =====================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
echo ">>> App dir: $APP_DIR   (service will run as: $RUN_USER)"

# ---- System packages -------------------------------------------------
apt-get update
# python-is-python3 makes `python` resolve to python3 (the dashboard's
# Start button spawns `python python_server.py`).
apt-get install -y python3-pip python3-venv python-is-python3 \
    libopenblas0 v4l-utils i2c-tools curl

# ---- Node.js 20 (for the Next.js dashboard) --------------------------
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

# ---- Python: control + vision deps -----------------------------------
# Everything EXCEPT torch/ultralytics, which need the Jetson-specific CUDA
# builds (see the big note at the end — do that step yourself once).
sudo -u "$RUN_USER" python3 -m pip install --upgrade pip
sudo -u "$RUN_USER" python3 -m pip install \
    flask flask-cors numpy opencv-python adafruit-circuitpython-servokit

# ---- I2C access for the PCA9685 servo driver -------------------------
usermod -aG i2c "$RUN_USER" || true

# ---- Build the dashboard --------------------------------------------
cd "$APP_DIR"
sudo -u "$RUN_USER" npm install
sudo -u "$RUN_USER" npm run build

# ---- systemd service: dashboard always on ---------------------------
NPM_BIN="$(command -v npm)"
cat > /etc/systemd/system/turret-dashboard.service <<EOF
[Unit]
Description=AutoTurret dashboard (Next.js)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=NODE_ENV=production
ExecStart=$NPM_BIN run start:lan
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable turret-dashboard.service
systemctl restart turret-dashboard.service

echo ""
echo "====================================================================="
echo " Dashboard is live on port 3000 (all interfaces)."
echo "   Over Tailscale:  http://<your-jetson-tailscale-ip>:3000"
echo ""
echo " REMAINING MANUAL STEP — the vision model (torch + ultralytics):"
echo "   These must be the Jetson/CUDA builds, NOT the PyPI CPU wheels."
echo "   Follow the official guide for your JetPack (6.2.3 / CUDA 12.6):"
echo "     https://docs.ultralytics.com/guides/nvidia-jetson/"
echo "   Typical path:"
echo "     python3 -m pip install ultralytics"
echo "     # then replace torch/torchvision with the JetPack 6 wheels"
echo "     # from the guide (jetson-ai-lab / NVIDIA index for cu126)."
echo "   Verify with:  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "   You want that to print: True"
echo "====================================================================="
