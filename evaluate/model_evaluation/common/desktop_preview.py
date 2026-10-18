"""Process-isolated, latest-only four-camera preview for real evaluation.

This module is intentionally outside the policy/control path.  Camera polling
and JPEG encoding run in a background thread, while decoding, HUD composition,
and OpenCV GUI handling run in a separate process.  A slow or closed monitor
therefore cannot delay inference or robot shutdown.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from typing import Any, Callable

import numpy as np


CAMERA_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")
PREVIEW_SHAPE = (480, 640, 3)
ARM_JOINT_COUNT = 14
_LEFT_JOINT_LABELS = ("L SP", "L SR", "L SY", "L EL", "L WR", "L WP", "L WY")
_RIGHT_JOINT_LABELS = ("R SP", "R SR", "R SY", "R EL", "R WR", "R WP", "R WY")


def _offer_latest(updates: Any, value: Any) -> None:
    """Replace a stale paint request without blocking its producer."""

    try:
        updates.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        updates.get_nowait()
    except queue.Empty:
        pass
    try:
        updates.put_nowait(value)
    except queue.Full:
        # A concurrent GUI dequeue can race this replacement.  The next poll
        # will retry, and display loss must never affect robot control.
        pass


def _fit_preview_frame(image_bgr: np.ndarray) -> np.ndarray:
    """Fit an arbitrary BGR frame into 640x480 without changing aspect ratio."""

    import cv2

    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"camera frame must be HWC 3-channel, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.asarray(np.clip(image, 0, 255), dtype=np.uint8)
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"camera frame has invalid shape {image.shape}")
    if image.shape == PREVIEW_SHAPE:
        return np.ascontiguousarray(image)

    scale = min(PREVIEW_SHAPE[1] / width, PREVIEW_SHAPE[0] / height)
    resized_width = max(1, min(PREVIEW_SHAPE[1], round(width * scale)))
    resized_height = max(1, min(PREVIEW_SHAPE[0], round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    canvas = np.zeros(PREVIEW_SHAPE, dtype=np.uint8)
    x = (PREVIEW_SHAPE[1] - resized_width) // 2
    y = (PREVIEW_SHAPE[0] - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _status_color(status: str) -> tuple[int, int, int]:
    # OpenCV uses BGR.
    if status == "POLICY INFERENCE":
        return (90, 245, 90)
    if status in {"PRE-MOTION", "HAND INITIALIZATION", "SAFE ARM RETURN"}:
        return (80, 210, 255)
    if status == "CONTROLLED RELEASE":
        return (80, 165, 255)
    return (230, 210, 80)


def _camera_tile(
    image_bgr: np.ndarray,
    *,
    title: str,
    status: str | None = None,
    joint_labels: tuple[str, ...] = (),
    joint_angles_deg: np.ndarray | None = None,
) -> np.ndarray:
    import cv2

    tile = _fit_preview_frame(image_bgr).copy()
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (639, 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, tile, 0.35, 0.0, tile)
    cv2.putText(
        tile,
        title,
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (90, 230, 255),
        2,
        cv2.LINE_AA,
    )
    if status:
        cv2.putText(
            tile,
            status,
            (365, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            _status_color(status),
            1,
            cv2.LINE_AA,
        )
    if joint_labels:
        if joint_angles_deg is None or joint_angles_deg.shape != (7,):
            raise ValueError("wrist HUD requires seven measured joint angles")
        overlay = tile.copy()
        cv2.rectangle(overlay, (8, 52), (235, 286), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.70, tile, 0.30, 0.0, tile)
        cv2.putText(
            tile,
            "measured joints (deg)",
            (18, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        for index, (label, angle) in enumerate(
            zip(joint_labels, joint_angles_deg, strict=True)
        ):
            cv2.putText(
                tile,
                f"{label} {angle:+6.1f}",
                (18, 105 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
    return tile


def compose_four_camera_view_bgr(
    camera_bgr: dict[str, np.ndarray],
    arm_joint_position_rad: np.ndarray,
    status: str,
) -> np.ndarray:
    """Compose head L/R over wrist L/R with a measured-arm HUD."""

    missing = [role for role in CAMERA_ROLES if role not in camera_bgr]
    if missing:
        raise ValueError(f"four-camera preview missing roles: {','.join(missing)}")
    arm = np.asarray(arm_joint_position_rad, dtype=np.float64)
    if arm.shape != (ARM_JOINT_COUNT,) or not np.isfinite(arm).all():
        raise ValueError("arm_joint_position_rad must be finite 14-D")
    if not status:
        raise ValueError("status must be non-empty")
    angles = np.rad2deg(arm)
    top = np.concatenate(
        (
            _camera_tile(camera_bgr["head_left"], title="HEAD LEFT", status=status),
            _camera_tile(camera_bgr["head_right"], title="HEAD RIGHT", status=status),
        ),
        axis=1,
    )
    bottom = np.concatenate(
        (
            _camera_tile(
                camera_bgr["left_wrist"],
                title="LEFT WRIST",
                joint_labels=_LEFT_JOINT_LABELS,
                joint_angles_deg=angles[:7],
            ),
            _camera_tile(
                camera_bgr["right_wrist"],
                title="RIGHT WRIST",
                joint_labels=_RIGHT_JOINT_LABELS,
                joint_angles_deg=angles[7:],
            ),
        ),
        axis=1,
    )
    return np.concatenate((top, bottom), axis=0)


def _encode_preview_update(
    observation: dict[str, Any], status: str
) -> dict[str, Any] | None:
    """Build a compact process-safe update from one live observation."""

    import cv2

    frames = observation.get("_camera_frames")
    if not isinstance(frames, dict) or any(role not in frames for role in CAMERA_ROLES):
        return None
    joint_state = observation.get("joint_state")
    positions = getattr(joint_state, "position", None)
    joints = np.asarray(positions, dtype=np.float64)
    if joints.shape != (29,) or not np.isfinite(joints).all():
        return None

    camera_jpeg: dict[str, bytes] = {}
    for role in CAMERA_ROLES:
        frame = _fit_preview_frame(np.asarray(frames[role]))
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return None
        camera_jpeg[role] = encoded.tobytes()
    return {
        "camera_jpeg": camera_jpeg,
        "arm_joint_position_rad": joints[15:29].tolist(),
        "status": status,
    }


def _preview_worker(updates: Any, window_title: str) -> None:
    import cv2
    import tkinter as tk
    from PIL import Image, ImageTk

    last_view: np.ndarray | None = None
    root: tk.Tk | None = None
    try:
        # The runtime intentionally uses an OpenCV build without HighGUI.
        # Tk/Pillow provides a real Desktop window without changing that
        # reproducible inference environment or importing GUI work into the
        # policy process.
        root = tk.Tk()
        root.title(window_title)
        root.geometry("1200x900")
        root.minsize(640, 480)
        label = tk.Label(root, background="black")
        label.pack(fill=tk.BOTH, expand=True)
        closed = False

        def close_window(_event: Any = None) -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            root.destroy()

        def poll_updates() -> None:
            nonlocal last_view
            if closed:
                return
            update: Any = ...
            while True:
                try:
                    candidate = updates.get_nowait()
                except queue.Empty:
                    break
                update = candidate
            if update is None:
                close_window()
                return
            if update is not ...:
                camera_bgr: dict[str, np.ndarray] = {}
                for role, payload in update["camera_jpeg"].items():
                    image = cv2.imdecode(
                        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if image is None:
                        raise ValueError(f"{role} preview JPEG failed to decode")
                    camera_bgr[role] = image
                last_view = compose_four_camera_view_bgr(
                    camera_bgr,
                    np.asarray(update["arm_joint_position_rad"], dtype=np.float64),
                    str(update["status"]),
                )
            if last_view is not None:
                width = max(1, label.winfo_width())
                height = max(1, label.winfo_height())
                scale = min(width / last_view.shape[1], height / last_view.shape[0])
                display_size = (
                    max(1, round(last_view.shape[1] * scale)),
                    max(1, round(last_view.shape[0] * scale)),
                )
                rgb = np.ascontiguousarray(last_view[:, :, ::-1])
                pil_image = Image.fromarray(rgb).resize(
                    display_size, resample=Image.Resampling.BILINEAR
                )
                photo = ImageTk.PhotoImage(pil_image)
                label.configure(image=photo)
                # Tk widgets do not retain the Python image object themselves.
                label.image = photo
            root.after(15, poll_updates)

        root.bind("<Escape>", close_window)
        root.bind("q", close_window)
        root.protocol("WM_DELETE_WINDOW", close_window)
        root.after(0, poll_updates)
        root.mainloop()
    except BaseException as exc:  # noqa: BLE001
        print(
            f"[preview] monitor stopped: {type(exc).__name__}: {exc}",
            flush=True,
        )
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


class FourCameraDesktopPreview:
    """Poll live observations and publish only the latest GUI frame."""

    def __init__(
        self,
        observation_supplier: Callable[[], dict[str, Any]],
        *,
        # This is an operator diagnostic monitor, not a policy input. Encoding
        # four additional JPEGs at 15 Hz competed with the independent 30 Hz
        # recorder/decode workers and caused otherwise healthy camera streams
        # to fail capture-rate validation. Five Hz remains easy to inspect
        # while preserving the full-rate policy and recording paths.
        sample_hz: float = 5.0,
        window_title: str = "IROS 2026 RAMEN - Real Policy Evaluation",
        process_factory: Callable[[Any, str], Any] | None = None,
    ) -> None:
        if not np.isfinite(sample_hz) or sample_hz <= 0.0:
            raise ValueError("sample_hz must be finite and positive")
        self._supplier = observation_supplier
        self._sample_hz = float(sample_hz)
        self._status = "CAMERAS READY"
        self._status_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._errors: list[str] = []
        self._updates: Any | None = None
        self._process: Any | None = None
        self._process_factory = process_factory
        self._window_title = window_title

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def start(self) -> bool:
        if self._thread is not None:
            raise RuntimeError("four-camera preview was already started")
        if self._process_factory is None and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            print(
                "[preview] disabled: DISPLAY/WAYLAND_DISPLAY is not set; "
                "inference will continue",
                flush=True,
            )
            return False
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=1)
        if self._process_factory is None:
            self._process = context.Process(
                target=_preview_worker,
                args=(self._updates, self._window_title),
                name="iros-four-camera-monitor",
                daemon=True,
            )
        else:
            self._process = self._process_factory(self._updates, self._window_title)
        self._process.start()
        # This queue carries disposable latest-only paint requests.  If the
        # GUI backend exits early, Python must not wait forever for its feeder
        # thread to flush JPEGs into an unread pipe during interpreter exit.
        self._updates.cancel_join_thread()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="iros-four-camera-preview-sampler",
            daemon=True,
        )
        self._thread.start()
        return True

    def set_status(self, status: str) -> None:
        if not status:
            raise ValueError("preview status must be non-empty")
        with self._status_lock:
            self._status = str(status)

    def _sample_loop(self) -> None:
        period = 1.0 / self._sample_hz
        deadline = time.monotonic()
        while not self._stop.is_set():
            if self._process is None or not self._process.is_alive():
                return
            try:
                with self._status_lock:
                    status = self._status
                update = _encode_preview_update(self._supplier(), status)
                if update is not None and self._updates is not None:
                    _offer_latest(self._updates, update)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if not self._errors or self._errors[-1] != message:
                    self._errors.append(message)
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                deadline = time.monotonic()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self._errors.append("preview sampler did not stop within 2 seconds")
        if self._process is not None and self._process.is_alive():
            _offer_latest(self._updates, None)
            self._process.join(timeout=2.0)
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        if self._updates is not None:
            try:
                self._updates.close()
            except Exception:
                pass

    def __enter__(self) -> "FourCameraDesktopPreview":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
