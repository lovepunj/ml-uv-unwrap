#!/usr/bin/env bash
# Oracle Cloud Always Free (Ampere A1) provisioning script for ML UV Unwrap.
# Run on a fresh Ubuntu 22.04/24.04 ARM VM as root or with sudo.
#
#   sudo bash oracle_setup.sh [app_user]
set -euo pipefail

APP_USER="${1:-ubuntu}"
APP_DIR="/home/${APP_USER}/ml-uv-unwrap"
REPO_URL="https://github.com/lovepunj/ml-uv-unwrap.git"

echo "=== [1/5] System packages ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git build-essential cmake \
    libgl1 libglib2.0-0 libegl1 \
    curl

echo "=== [2/5] Clone repo ==="
if [ ! -d "${APP_DIR}" ]; then
    git clone "${REPO_URL}" "${APP_DIR}"
else
    git -C "${APP_DIR}" pull --ff-only || echo "  (pull skipped; continuing)"
fi

echo "=== [3/5] Python venv + CPU torch ==="
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2
"${APP_DIR}/.venv/bin/pip" install --no-cache-dir -r "${APP_DIR}/requirements.txt"

echo "=== [4/5] systemd service ==="
cat > /etc/systemd/system/ml-uv-unwrap.service <<EOF
[Unit]
Description=ML UV Unwrap (FastAPI)
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_DIR}/.venv/bin/uvicorn web.server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ml-uv-unwrap
systemctl restart ml-uv-unwrap

echo "=== [5/6] Caddy HTTPS reverse proxy (optional) ==="
if command -v caddy >/dev/null 2>&1; then
    echo "  Caddy already installed."
elif ! command -v curl >/dev/null 2>&1 || [ "$(id -u)" != "0" ]; then
    echo "  Skipping Caddy (requires root + internet). Install later with:"
    echo "    apt install -y caddy && cp deploy/Caddyfile /etc/caddy/Caddyfile && systemctl restart caddy"
else
    apt-get install -y --no-install-recommends debian-keyring debian-archive-keyring apt-transport-https ca-certificates curl || true
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg || true
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null || true
    apt-get update || true
    apt-get install -y --no-install-recommends caddy || true
    if command -v caddy >/dev/null 2>&1; then
        echo "  Caddy installed. Edit /etc/caddy/Caddyfile with your domain, then: systemctl restart caddy"
    else
        echo "  Caddy install failed (fallback: expose port 8000 directly)."
    fi
fi

echo "=== [6/6] Firewall ==="
# Oracle Cloud default security lists allow 22; open 80/443 for web.
ufw allow 22/tcp || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true

echo
echo "Done. App status:"
systemctl status ml-uv-unwrap --no-pager | head -8 || true
echo
echo "Next steps:"
echo "  1. Open ports 80/443 (and optionally 8000) in your Oracle Cloud VCN"
echo "     Security List > Ingress Rules."
echo "  2. Optionally install Caddy for HTTPS + domain (see deploy/Caddyfile)."
echo "  3. App is served on port 8000: curl http://<public-ip>:8000"
