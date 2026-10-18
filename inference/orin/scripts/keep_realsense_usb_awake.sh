#!/usr/bin/env bash
# Keep D405 USB devices and their parent hubs out of runtime autosuspend.
# Camera-only host preparation: this script never sends a robot command.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 2
fi

shopt -s nullglob
devices=()
for device in /sys/bus/usb/devices/*; do
  [[ -f "${device}/idVendor" && -f "${device}/idProduct" ]] || continue
  [[ $(<"${device}/idVendor") == "8086" ]] || continue
  [[ $(<"${device}/idProduct") == "0b5b" ]] || continue
  devices+=("${device}")
done

if (( ${#devices[@]} != 2 )); then
  echo "Expected exactly two Intel RealSense D405 devices; found ${#devices[@]}." >&2
  exit 1
fi

for device in "${devices[@]}"; do
  speed=$(<"${device}/speed")
  if [[ ! "${speed}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
      || (( ${speed%.*} < 5000 )); then
    echo "${device} negotiated ${speed}M; D405 evaluation requires USB 3 (>=5000M)." >&2
    exit 1
  fi
done

declare -A control_paths=()
declare -A delay_paths=()
for device in "${devices[@]}"; do
  node=$(basename "${device}")
  while [[ "${node}" == *-* ]]; do
    control="/sys/bus/usb/devices/${node}/power/control"
    delay="/sys/bus/usb/devices/${node}/power/autosuspend_delay_ms"
    [[ -e "${control}" ]] && control_paths["${control}"]=1
    [[ -e "${delay}" ]] && delay_paths["${delay}"]=1
    [[ "${node}" == *.* ]] || break
    node=${node%.*}
  done
  bus=${node%%-*}
  control="/sys/bus/usb/devices/usb${bus}/power/control"
  delay="/sys/bus/usb/devices/usb${bus}/power/autosuspend_delay_ms"
  [[ -e "${control}" ]] && control_paths["${control}"]=1
  [[ -e "${delay}" ]] && delay_paths["${delay}"]=1
done

for control in "${!control_paths[@]}"; do
  echo on > "${control}"
  echo "set ${control}=on"
done
for delay in "${!delay_paths[@]}"; do
  echo -1 > "${delay}"
  echo "set ${delay}=-1"
done

echo "RealSense USB runtime autosuspend disabled for this boot."
