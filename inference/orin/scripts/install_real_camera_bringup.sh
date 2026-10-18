#!/usr/bin/env bash
# Install the persistent, camera-only Orin bringup. No robot command is sent.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
install -m 0644 \
  "${repo_root}/inference/orin/udev/99-iros-ramen-d405-power.rules" \
  /etc/udev/rules.d/99-iros-ramen-d405-power.rules
install -m 0644 \
  "${repo_root}/inference/orin/systemd/iros-ramen-real-camera.service" \
  /etc/systemd/system/iros-ramen-real-camera.service

udevadm control --reload-rules
udevadm trigger --action=add --subsystem-match=usb \
  --attr-match=idVendor=8086 --attr-match=idProduct=0b5b || true
systemctl daemon-reload

# The AVP image server owns the same cameras and must never race this service.
systemctl disable --now avp_teleimager.service 2>/dev/null || true
systemctl enable iros-ramen-real-camera.service

echo "Installed persistent D405 power rules and iros-ramen-real-camera.service."
echo "Start after both D405s are connected: systemctl start iros-ramen-real-camera.service"
