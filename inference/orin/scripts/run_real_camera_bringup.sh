#!/usr/bin/env bash
# Canonical read-only Orin ROS/camera bringup. This never publishes a motor command.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "${repo_root}/inference/orin"

runtime_dir=${XDG_RUNTIME_DIR:-/tmp}
exec 9>"${runtime_dir}/iros-ramen-real-camera.lock"
if ! flock -n 9; then
  echo "Another IROS RAMEN camera bringup already owns the camera lock." >&2
  exit 1
fi

if systemctl is-active --quiet avp_teleimager.service 2>/dev/null; then
  echo "avp_teleimager.service is active and owns the cameras; refusing concurrent startup." >&2
  exit 1
fi

mapfile -t head_candidates < <(
  find /dev/v4l/by-id -maxdepth 1 -type l -name '*-video-index0' \
    ! -name '*RealSense*' -print 2>/dev/null | sort
)
if (( ${#head_candidates[@]} != 1 )); then
  echo "Expected exactly one non-RealSense head capture node; found ${#head_candidates[@]}." >&2
  printf '  %s\n' "${head_candidates[@]}" >&2
  exit 1
fi
head_device=${head_candidates[0]}

mapfile -t enumerated_d405_nodes < <(
  for device in /sys/bus/usb/devices/*; do
    [[ -f "${device}/idVendor" && -f "${device}/idProduct" ]] || continue
    [[ $(<"${device}/idVendor") == 8086 && $(<"${device}/idProduct") == 0b5b ]] || continue
    printf '%s\n' "${device}"
  done
)
declare -a ready_d405_nodes=()
declare -A ready_usb_serials=()
for device in "${enumerated_d405_nodes[@]}"; do
  speed=$(cat "${device}/speed" 2>/dev/null || true)
  config=$(cat "${device}/bConfigurationValue" 2>/dev/null || true)
  serial=$(cat "${device}/serial" 2>/dev/null || true)
  if [[ ${speed} != 5000 && ${speed} != 10000 ]] || [[ ${config} != 1 ]] \
      || [[ -z ${serial} ]]; then
    echo "Ignoring unready D405 sysfs entry $(basename "${device}"): serial=${serial:-unknown} speed=${speed:-unknown}M config=${config:-unset}." >&2
    continue
  fi
  if [[ -n ${ready_usb_serials["${serial}"]+x} ]]; then
    echo "Duplicate ready D405 USB serial ${serial}; refusing ambiguous camera ownership." >&2
    exit 1
  fi
  ready_usb_serials["${serial}"]=${device}
  ready_d405_nodes+=("${device}")
done
if (( ${#ready_d405_nodes[@]} != 2 )); then
  echo "Expected exactly two configured USB3 D405 cameras; found ${#ready_d405_nodes[@]} ready (${#enumerated_d405_nodes[@]} sysfs entries)." >&2
  exit 1
fi

required_left_serial=${WRIST_LEFT_SERIAL:-128422271048}
required_right_serial=${WRIST_RIGHT_SERIAL:-128422271925}
rs_summary=$(timeout 15 rs-enumerate-devices -s 2>&1 || true)
if ! grep -Fq "${required_left_serial}" <<<"${rs_summary}" \
    || ! grep -Fq "${required_right_serial}" <<<"${rs_summary}"; then
  echo "RealSense control-plane preflight did not open both configured D405s." >&2
  echo "Required serials: left=${required_left_serial} right=${required_right_serial}" >&2
  printf '%s\n' "${rs_summary}" >&2
  exit 1
fi

# Remove only this canonical container after an unclean host shutdown.
docker rm -f iros_ramen_real_camera >/dev/null 2>&1 || true

export HEAD_CAMERA_DEVICE=${HEAD_CAMERA_DEVICE:-${head_device}}
set +e
docker compose run --rm --name iros_ramen_real_camera \
  ramen_inference \
  ros2 launch g1_bringup system.launch.py \
  mock_hardware:=false \
  enable_camera:=true \
  enable_wrist_cameras:=true \
  wrist_camera_driver:=realsense \
  enable_wrist_infra:=false \
  wrist_left_serial:="${required_left_serial}" \
  wrist_right_serial:="${required_right_serial}" \
  head_camera_device:="${HEAD_CAMERA_DEVICE}"
status=$?
set -e

# This script is the long-running systemd process.  The camera launch has no
# successful terminal state: a zero exit means a watchdog requested an orderly
# ROS shutdown after detecting stale input.  Convert that case to failure so
# Restart=on-failure actually performs recovery.  During `systemctl stop`,
# systemd owns the service state and will not restart it even if the propagated
# signal produces a non-zero status.
if (( status == 0 )); then
  echo "Camera bringup exited unexpectedly; requesting systemd recovery." >&2
  exit 1
fi
exit "${status}"
