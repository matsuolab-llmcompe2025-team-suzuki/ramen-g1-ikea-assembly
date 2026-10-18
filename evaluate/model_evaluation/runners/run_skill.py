"""単体 skill 実機評価 runner (Phase 2)。

# 前提

- Regular Mode (fsm_id=501, fsm_mode=0) で立ち姿勢
- ハーネス + E-stop 手元
- Orin bringup (real_hw_bridge + camera + dex1_gripper_server) 起動済
- W&B standard login済 (`~/.netrc`) または`WANDB_API_KEY`設定済

# フロー

Issue #141 (束 1-8) で、**policy を回す部分は本番と同じ `Orchestrator` になった**。
run_skill が持つのは「評価だけの手順」= 操作者の Enter・収録・preview・戻り動作・wandb。

1. skill_config.yaml + policy_config.yaml から skill / variant を resolve
2. model を読む (warmup 込み、腕の所有権を取る前に初回 forward を消化する)
3. Sensor 群 (関節角 + Dex1 + camera ×3) を init。`--actuate` 無しは Mock で dry run
4. orchestrator に本番と同じ skill の列を渡す:
   手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持 → 評価する skill
5. 物を持って始まる skill は「掴む幅へ」の手前で止めて Enter (脚を置く)
6. 開始姿勢に着いたら Enter (policy 開始)。カメラを 1 s 再検証して policy を回す
7. 評価する skill は列の最後なので、`--max-seconds` (未指定なら YAML の
   `max_seconds_hard`) で時間切れになると orchestrator が戻ってくる
8. dataset初期姿勢へ復帰 → Dex1全開 → 起動時経路を逆走して腕を下げる
9. arm_sdk controlled release、Regular Mode復帰
10. 4カメラ・state/actionをpackし、長時間実機runをwandbへupload

YOLO の重み・conf・imgsz は `policy_config.yaml` の `yolo` から取る (本番と同じ、INF-10)。
関節角は既定で Desktop が直接受ける `rt/lowstate` から (`--joint-source`、INF-18)。

# 使い方

```bash
# read-only preflight (default、real actuator に流れない):
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
    --skill move_table_base \
    --interface eth0 \
    --max-seconds 30

# real dispatch (ハーネス必須):
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
    --skill move_table_base \
    --interface eth0 \
    --max-seconds 30 \
    --actuate --use-real-waist

# 実機runはwandb uploadがdefault ON (`--no-wandb`で明示停止):
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
    --skill pick_table_leg \
    --interface eth0 \
    --actuate --no-dispatch-waist --use-real-hand \
    --wandb-run-name "pick_table_leg_20260901_trial01"
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from inference.desktop import assembly
from inference.desktop.assembly import (
    CAMERA_STALE_TIMEOUT_S,
    build_observation,
    camera_stale_roles,
    required_camera_roles,
    select_head_view,
)
from inference.desktop.recording import record_policy_tick


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_SKILL_CONFIG = REPO_ROOT / "inference/desktop/lower_policy/configs/skill_config.yaml"
DEFAULT_POLICY_CONFIG = REPO_ROOT / "inference/desktop/lower_policy/configs/policy_config.yaml"
DEFAULT_HEAD_TOPIC = "/head/camera/color/image_raw/compressed"
DEFAULT_WRIST_L_TOPIC = "/wrist_left/camera/color/image_raw/compressed"
DEFAULT_WRIST_R_TOPIC = "/wrist_right/camera/color/image_raw/compressed"
DEFAULT_JOINT_STATE_TOPIC = "/joint_states"
DEFAULT_PICK_LEG_HYBRID_CONFIG = (
    REPO_ROOT / "inference/desktop/pick_leg_hybrid/configs/pick_leg_hybrid.yaml"
)
TICK_HZ = 30.0
CAMERA_PREFLIGHT_POLL_HZ = 120.0
CAMERA_PREFLIGHT_MIN_HZ = 25.0
# The final gates run with the policy and four-camera preview resident.  A
# one-second count is too sensitive to one scheduler stall: a healthy 27--30
# Hz source can appear below 25 Hz in that short window.  Keep the same rate
# requirement, but average long enough to distinguish transient host jitter
# from sustained degradation before physical ownership or policy commands.
CAMERA_FINAL_GATE_DURATION_S = 3.0
# Startup validation also checks the observed generation rate over the whole
# window.  Give its latest-frame check a small transient allowance so a single
# DDS scheduling gap does not reject an otherwise continuous 30 Hz stream.
# The stricter limit below remains in force once policy commands are active.
CAMERA_PREFLIGHT_STALE_TIMEOUT_S = 0.75


def _resolve_hard_timeout_from_skill_config(skill_cfg: dict, skill_name: str) -> float:
    """skill_config.yaml から per-skill hard timeout (秒) を fail-closed に読む。

    YOLO transition 検知漏れ時の last-resort。値は
    compute_skill_timeout.py の p95 × 1.5 で算出したものを YAML に書いてある。
    未定義は KeyError で起動を止める (silent fallback しない、新 skill 追加時
    の書き忘れを検知させる)。
    """

    skills = skill_cfg.get("skills") if isinstance(skill_cfg, dict) else None
    if not isinstance(skills, dict):
        raise KeyError("skill_config.yaml missing top-level 'skills' mapping")
    section = skills.get(skill_name)
    if not isinstance(section, dict):
        raise KeyError(f"skill_config.yaml missing skills.{skill_name} entry")
    if "max_seconds_hard" not in section:
        raise KeyError(
            f"skill_config.yaml:skills.{skill_name} missing 'max_seconds_hard' "
            "(YOLO transition 検知漏れ時の last-resort timeout、s 単位、必須)"
        )
    value = float(section["max_seconds_hard"])
    # Issue #137: retry 有効時は skill の意味が「3 回試す」に変わるので、
    # 単発ストローク p95 x 1.5 で決めた既定より長い上限を retry block 側に置ける。
    retry_cfg = section.get("retry")
    if isinstance(retry_cfg, dict) and retry_cfg.get("max_seconds_hard") is not None:
        value = float(retry_cfg["max_seconds_hard"])
    if not value > 0:
        raise ValueError(
            f"skills.{skill_name}.max_seconds_hard must be > 0, got {value}"
        )
    return value


def resolve_evaluation_pick_mode(args: argparse.Namespace) -> None:
    """Use the production hybrid by default; legacy VLA needs an explicit opt-out."""
    if args.pick_leg_hybrid is None:
        args.pick_leg_hybrid = args.skill == "pick_table_leg"
    if args.pick_leg_phase3_executor is None:
        args.pick_leg_phase3_executor = "rule_based" if args.pick_leg_hybrid else "vla"
    if not args.pick_leg_hybrid:
        if args.pick_leg_phase3_executor != "vla":
            raise ValueError("--pick-leg-phase3-executor requires --pick-leg-hybrid")
        return
    if args.skill != "pick_table_leg":
        raise ValueError("--pick-leg-hybrid is valid only with --skill pick_table_leg")
    from inference.desktop.pick_leg_hybrid import HYBRID_PHASE1_VARIANTS

    # 区間 1 に使ってよい model だけ (先頭 = SKILL_REGISTRY の既定、Issue #188 で RAMEN-Ori を追加)
    if args.variant is not None and args.variant not in HYBRID_PHASE1_VARIANTS:
        raise ValueError(
            f"hybrid pick requires --variant in {list(HYBRID_PHASE1_VARIANTS)}; for a learned-only "
            "model comparison, explicitly pass --no-pick-leg-hybrid"
        )


def parse_args() -> argparse.Namespace:
    from evaluate.model_evaluation.common.skill_registry import SKILL_REGISTRY

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--skill", required=True, choices=sorted(SKILL_REGISTRY.keys()),
        help="evaluation 対象 skill (skill_config.yaml key と一致)",
    )
    p.add_argument(
        "--variant", default=None,
        help="policy_config.yaml variant 名 (未指定 = SKILL_REGISTRY の default)",
    )
    p.add_argument("--interface", default=None, help="cyclonedds NIC (例: eth0)")
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help=(
            "評価する skill を回す秒数。指定すると skill_config の max_seconds_hard を"
            " 上書きする (Issue #141 D6)。未指定なら YAML の値"
        ),
    )
    p.add_argument("--skill-config", type=Path, default=DEFAULT_SKILL_CONFIG)
    p.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    p.add_argument("--head-topic", default=DEFAULT_HEAD_TOPIC)
    p.add_argument("--wrist-left-topic", default=DEFAULT_WRIST_L_TOPIC)
    p.add_argument("--wrist-right-topic", default=DEFAULT_WRIST_R_TOPIC)
    p.add_argument("--joint-state-topic", default=DEFAULT_JOINT_STATE_TOPIC)
    p.add_argument(
        "--joint-source",
        choices=("lowstate", "joint_states"),
        default="lowstate",
        help=(
            "関節角の取り方 (Issue #141 INF-18)。lowstate (既定) = Desktop が"
            " rt/lowstate を直接受ける、joint_states = Orin の bridge 経由"
        ),
    )
    p.add_argument(
        "--setup-hold-sec",
        type=float,
        default=0.0,
        help=(
            "頭の手順の最後に開始姿勢で保持する秒数 (Issue #141 D2)。評価は開始 Enter が"
            " あるので既定 0、本番は 3 秒"
        ),
    )
    p.add_argument("--no-wrist-cameras", action="store_true")
    p.add_argument(
        "--pick-leg-hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "pick_table_legでは既定ON: VLM境界付きGR00T->G1 IK MP->"
            "rule-based持ち替え->insert初期姿勢。VLM不通時は起動を拒否する。"
            "--no-pick-leg-hybridは旧VLA単体の比較評価専用"
        ),
    )
    p.add_argument(
        "--pick-leg-hybrid-config",
        type=Path,
        default=DEFAULT_PICK_LEG_HYBRID_CONFIG,
    )
    p.add_argument(
        "--pick-leg-phase3-executor",
        choices=("vla", "rule_based"),
        default=None,
        help=(
            "pick-leg hybridのPhase 3実行系 (既定rule_based)。vlaは比較評価用、rule_basedは"
            "左把持→右解放→右再把持→左解放→insert初期姿勢を実測確認する有限FSM"
        ),
    )
    p.add_argument(
        "--pick-leg-vlm-endpoint",
        default=None,
        help="OpenAI-compatible chat-completions URL (default=config YAML)",
    )
    p.add_argument(
        "--pick-leg-vlm-model",
        default=None,
        help="served VLM model id (default=config YAML)",
    )
    p.add_argument(
        "--no-camera-preview",
        action="store_true",
        help=(
            "4-camera Desktop monitorを無効化。既定ではhead L/R・wrist L/Rと"
            "実測arm角を、推論・収録とは独立したlatest-only画面に表示する"
        ),
    )
    # 実 actuator gate (default OFF)
    p.add_argument("--actuate", action="store_true", help="real actuator dispatch")
    p.add_argument("--use-real-waist", action="store_true")
    p.add_argument("--use-real-hand", action="store_true")
    p.add_argument(
        "--teacher-range-observe-only",
        action="store_true",
        help=(
            "flip_table比較実験専用: 教師関節範囲の逸脱を記録するが目標は補正しない。"
            "物理関節限界・速度・加速度制限は維持する"
        ),
    )
    p.add_argument(
        "--dispatch-waist", dest="dispatch_waist", default=None,
        action="store_true",
        help="YAML の dispatch_waist を True で上書き (default=YAML 通り)",
    )
    p.add_argument(
        "--no-dispatch-waist", dest="dispatch_waist", action="store_false",
        help="YAML の dispatch_waist を False で上書き (default=YAML 通り)",
    )
    # Real actuation defaults to W&B upload.  Read-only defaults to local-only.
    p.set_defaults(wandb=None)
    p.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        help="wandb uploadを明示的に有効化 (実機runはdefault ON)",
    )
    p.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="wandb uploadを明示的に無効化",
    )
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument(
        "--log-dir", type=Path, default=None,
        help="収録 dir。default: outputs/model_eval_runs/<skill>_<ts>/",
    )
    args = p.parse_args()
    try:
        resolve_evaluation_pick_mode(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def _default_log_dir(skill_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "model_eval_runs" / f"{skill_name}_{ts}"


def _collect_obs(
    joint_state_source: Any,
    dex1_state_source: Any,
    head_source: Any,
    wrist_l_source: Any,
    wrist_r_source: Any,
    overlay_perception: Any = None,
) -> dict:
    """VlaSkill / CollisionAware step() が読む obs dict を組む。

    組み立ては本番経路と同じ `assembly.build_observation` (Issue #141 D3)。ここは
    head の取得と overlay 用の YOLO だけを行う (本番は orchestrator が同じ順序で行う)。
    """
    head = head_source.get() if head_source else None
    detections = None
    if overlay_perception is not None:
        head_rgb = None if head is None else np.asarray(getattr(head, "rgb", head))
        if head_rgb is None or head_rgb.ndim != 3 or head_rgb.shape[1] % 2 != 0:
            raise RuntimeError("overlay evaluation requires packed head stereo")
        detections = overlay_perception.predict(
            np.ascontiguousarray(select_head_view(head_rgb, "left"))
        )
    return build_observation(
        head=head,
        detections=detections,
        joint_state_source=joint_state_source,
        dex1_state_source=dex1_state_source,
        wrist_left_source=wrist_l_source,
        wrist_right_source=wrist_r_source,
    )


def _missing_observation_inputs(obs: dict, *, require_wrist: bool) -> list[str]:
    """Return missing/non-finite inputs required by a real VLA forward pass."""

    missing: list[str] = []
    js = obs.get("joint_state")
    positions = getattr(js, "position", None) if js is not None else None
    joint_array = (
        np.asarray(positions, dtype=np.float64) if positions is not None else None
    )
    if (
        joint_array is None
        or joint_array.shape != (29,)
        or not np.isfinite(joint_array).all()
    ):
        missing.append("joint_state")

    hand = obs.get("hand_state")
    hand_positions = getattr(hand, "position_rad", None) if hand is not None else None
    hand_array = (
        np.asarray(hand_positions, dtype=np.float64)
        if hand_positions is not None
        else None
    )
    if (
        hand_array is None
        or hand_array.shape != (2,)
        or not np.isfinite(hand_array).all()
    ):
        missing.append("dex1_state")

    head = obs.get("head_rgb")
    if head is None or np.asarray(head).ndim != 3:
        missing.append("head")
    if require_wrist:
        for key, label in (
            ("wrist_left_rgb", "wrist_left"),
            ("wrist_right_rgb", "wrist_right"),
        ):
            frame = obs.get(key)
            frame = getattr(frame, "rgb", frame)
            if frame is None or np.asarray(frame).ndim != 3:
                missing.append(label)
    return missing


def _camera_stale_roles(
    obs: dict,
    *,
    require_wrist: bool,
    max_age_s: float = CAMERA_STALE_TIMEOUT_S,
    now_ns: int | None = None,
) -> dict[str, float]:
    """止まっている / 来ていないカメラとその古さ (本番経路と同じ判定、Issue #141 D3)。"""
    return camera_stale_roles(
        obs, require_wrist=require_wrist, max_age_s=max_age_s, now_ns=now_ns
    )


def _validate_camera_stream_window(
    collector: Any,
    *,
    require_wrist: bool,
    duration_s: float,
    minimum_hz: float = CAMERA_PREFLIGHT_MIN_HZ,
) -> dict[str, float]:
    """Read-only proof that every required camera is still producing frames."""

    if duration_s <= 0.0 or minimum_hz <= 0.0:
        raise ValueError("camera validation duration/rate must be positive")
    roles = required_camera_roles(require_wrist=require_wrist)
    generations: dict[str, set[int]] = {role: set() for role in roles}
    start = time.monotonic()
    deadline = start + duration_s
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = collector()
        stale_now = _camera_stale_roles(
            latest,
            require_wrist=require_wrist,
            max_age_s=CAMERA_PREFLIGHT_STALE_TIMEOUT_S,
        )
        if stale_now:
            details = ", ".join(
                f"{role}=missing"
                if not np.isfinite(age_s)
                else f"{role}={age_s:.3f}s"
                for role, age_s in stale_now.items()
            )
            raise RuntimeError(
                "camera stream became stale during read-only validation: "
                + details
            )
        values = latest.get("_camera_generations")
        values = values if isinstance(values, dict) else {}
        for role in roles:
            generation = values.get(role)
            if generation is not None:
                generations[role].add(int(generation))
        time.sleep(1.0 / CAMERA_PREFLIGHT_POLL_HZ)
    elapsed = max(time.monotonic() - start, 1e-9)
    rates = {role: len(generations[role]) / elapsed for role in roles}
    stale = _camera_stale_roles(
        latest,
        require_wrist=require_wrist,
        max_age_s=CAMERA_PREFLIGHT_STALE_TIMEOUT_S,
    )
    failures = [
        f"{role}={rates[role]:.2f}Hz"
        for role in roles
        if rates[role] < minimum_hz
    ]
    failures.extend(
        f"{role}=missing/stale"
        if not np.isfinite(age_s)
        else f"{role}=stale({age_s:.3f}s)"
        for role, age_s in stale.items()
    )
    if failures:
        raise RuntimeError(
            "camera preflight failed (validation is read-only): "
            + ", ".join(failures)
        )
    print(
        "[preflight] live camera rates: "
        + " ".join(f"{role}={rates[role]:.1f}Hz" for role in roles)
    )
    return rates


def _wait_for_finite_obs(
    collector,
    timeout_s: float = 10.0,
    *,
    require_wrist: bool = True,
) -> dict:
    """Wait for the atomic set of inputs needed by the first policy tick."""
    deadline = time.monotonic() + timeout_s
    last_missing: list[str] = []
    while time.monotonic() < deadline:
        obs = collector()
        last_missing = _missing_observation_inputs(
            obs, require_wrist=require_wrist
        )
        if not last_missing:
            return obs
        time.sleep(1.0 / TICK_HZ)
    raise TimeoutError(
        f"policy observation not ready within {timeout_s:g}s; "
        f"missing={','.join(last_missing)}"
    )


def _sleep_to_next_tick(next_deadline: float) -> float:
    """Wait for one 30-Hz tick without ever emitting catch-up commands.

    Preserve the periodic phase while on time.  After an overrun, restart from
    ``now + period``: advancing to the next old-grid boundary can produce a
    sub-period command interval and invalidate the limiter's real-time
    velocity/acceleration guarantees.
    """
    now = time.monotonic()
    delta = next_deadline - now
    if delta > 0:
        time.sleep(delta)
        return next_deadline + 1.0 / TICK_HZ
    return now + 1.0 / TICK_HZ


def _measured_arm_from_observation(observation: dict[str, Any]) -> np.ndarray:
    joint_state = observation.get("joint_state")
    positions = getattr(joint_state, "position", None)
    values = np.asarray(positions, dtype=np.float64)
    if values.shape != (29,) or not np.isfinite(values).all():
        raise RuntimeError("safe return requires a finite measured 29-D joint state")
    return values[15:29].copy()


def _drive_return_waypoints(
    waypoints,
    *,
    collector,
    arm_actuator: Any,
    recorder: Any,
) -> None:
    """Run an evaluate-only reverse arm segment with measured convergence."""

    from evaluate.model_evaluation.common.safe_return import (
        BoundedArmWaypointFollower,
    )

    follower = BoundedArmWaypointFollower(
        waypoints,
        velocity_limit_rad_s=0.5,
        acceleration_limit_rad_s2=1.0,
        measured_tolerance_rad=0.10,
        # Issue 70 real logs showed a benign passive wrist-roll offset around
        # 0.19 rad after shoulders/elbows had already reached this collision-
        # clearance pose.  Keep commanding the full target, but do not strand
        # a safe reverse sweep on that wrist-only tracking residue.
        measured_tolerance_overrides={
            "return_forward_outward_clearance": 0.20
        }
        if any(
            waypoint.name == "return_forward_outward_clearance"
            for waypoint in waypoints
        )
        else None,
        stage_timeout_s=15.0,
    )
    observation = collector()
    measured = _measured_arm_from_observation(observation)
    now = time.monotonic()
    arm_actuator.send_action(follower.start(measured, now))
    active_index = follower.stage_index
    active = follower.waypoints[active_index]
    print(f"[return {active_index + 1}/{len(follower.waypoints)}] {active.name} started")
    recorder.write_event({
        "kind": "return_motion_stage_started",
        "stage": active.name,
        "target_arm_rad": active.as_array().tolist(),
    })
    next_deadline = now + 1.0 / TICK_HZ
    while not follower.complete:
        observation = collector()
        measured = _measured_arm_from_observation(observation)
        now = time.monotonic()
        previous_index = follower.stage_index
        previous = follower.current_waypoint
        command = follower.step(measured, now)
        arm_actuator.send_action(command)
        if follower.stage_index != previous_index:
            error = float(np.max(np.abs(previous.as_array() - measured)))
            print(
                f"[return {previous_index + 1}/{len(follower.waypoints)}] "
                f"{previous.name} reached (error={error:.4f}rad)"
            )
            recorder.write_event({
                "kind": "return_motion_stage_complete",
                "stage": previous.name,
                "max_measured_error_rad": error,
            })
            if not follower.complete:
                active = follower.current_waypoint
                print(
                    f"[return {follower.stage_index + 1}/{len(follower.waypoints)}] "
                    f"{active.name} started"
                )
                recorder.write_event({
                    "kind": "return_motion_stage_started",
                    "stage": active.name,
                    "target_arm_rad": active.as_array().tolist(),
                })
        next_deadline = _sleep_to_next_tick(next_deadline)


def _move_hands_to_target(
    *,
    hand_actuator: Any,
    dex1_state_source: Any,
    recorder: Any,
    target_rad: tuple[float, float],
    event_prefix: str,
    description: str,
    velocity_limit_rad_s: float = 1.5,
    tolerance_rad: float = 0.05,
    allow_closing_contact: bool = False,
) -> None:
    """Slew both Dex1 motors and confirm target or a bounded physical grasp.

    ``allow_closing_contact`` is reserved for the operator-supervised object
    placement gate.  It accepts a hand that travelled at least 80% of its
    commanded closing stroke and then remained stable near the target.  The
    permitted contact residual scales with the requested stroke, capped at
    0.75 rad, because a real table leg can stop the gripper farther open than
    the unobstructed dataset target.  Opening moves and ordinary frame-zero
    moves still require the strict positional tolerance.
    """

    from inference.desktop.lower_policy.skills.hand_ramp import (
        build_hand_target_ramp,
        hand_target_completion_mode,
    )

    state = dex1_state_source.get()
    if state is None:
        raise RuntimeError(f"{description} requires bilateral Dex1 state")
    now_ns = time.monotonic_ns()
    state_age_s = max(
        now_ns - int(state.left_received_monotonic_ns),
        now_ns - int(state.right_received_monotonic_ns),
    ) / 1e9
    if state_age_s > 0.5:
        raise RuntimeError(
            f"{description} Dex1 state is stale (age={state_age_s:.3f}s)"
        )
    start = np.asarray(state.position_rad, dtype=np.float64)
    target = np.asarray(target_rad, dtype=np.float64)
    if start.shape != (2,) or not np.isfinite(start).all():
        raise RuntimeError(f"{description} measured Dex1 pose is invalid: {start}")
    ramp = build_hand_target_ramp(
        start,
        target,
        command_hz=TICK_HZ,
        velocity_limit_rad_s=velocity_limit_rad_s,
    )
    print(f"[hand] {description}: {start.tolist()} -> {target.tolist()}")
    recorder.write_event({
        "kind": f"{event_prefix}_started",
        "measured_start_rad": start.tolist(),
        "target_rad": target.tolist(),
    })
    for command in ramp:
        hand_actuator.send_action(command)
        time.sleep(1.0 / TICK_HZ)

    stable_samples = 0
    measured_history: list[np.ndarray] = []
    expected_motion_s = float(np.max(np.abs(target - start))) / velocity_limit_rad_s
    deadline = time.monotonic() + max(5.0, expected_motion_s + 3.0)
    measured = start
    while time.monotonic() < deadline:
        state = dex1_state_source.get()
        if state is not None:
            now_ns = time.monotonic_ns()
            state_age_s = max(
                now_ns - int(state.left_received_monotonic_ns),
                now_ns - int(state.right_received_monotonic_ns),
            ) / 1e9
            measured = np.asarray(state.position_rad, dtype=np.float64)
            valid = (
                state_age_s <= 0.5
                and measured.shape == (2,)
                and np.isfinite(measured).all()
            )
            completion_mode = None
            if valid:
                measured_history.append(measured.copy())
                measured_history = measured_history[-10:]
                completion_mode = hand_target_completion_mode(
                    start=start,
                    target=target,
                    measured_history=measured_history,
                    tolerance_rad=tolerance_rad,
                    allow_closing_contact=allow_closing_contact,
                )
            else:
                measured_history.clear()
            stable_samples = stable_samples + 1 if completion_mode == "target" else 0
            if stable_samples >= 5 or completion_mode == "closing_contact":
                residual = np.abs(measured - target)
                recorder.write_event({
                    "kind": f"{event_prefix}_complete",
                    "measured_rad": measured.tolist(),
                    "target_rad": target.tolist(),
                    "completion_mode": completion_mode,
                    "residual_rad": residual.tolist(),
                })
                print(
                    f"[hand] {completion_mode} confirmed "
                    f"(measured={measured.tolist()}, residual={residual.tolist()})"
                )
                return
        time.sleep(1.0 / TICK_HZ)
    raise TimeoutError(
        f"{description} did not reach target within timeout "
        f"(target={target.tolist()}, measured={measured.tolist()})"
    )


def _open_hands_fully(
    *,
    hand_actuator: Any,
    dex1_state_source: Any,
    recorder: Any,
    event_prefix: str = "return_hands_open",
    description: str = "opening both Dex1 hands to the 5.4rad maximum",
) -> None:
    """Slew both Dex1 motors to the physical open endpoint and confirm it."""

    from inference.desktop.lower_policy.actuators.hand import HAND_GRIP_MAX

    _move_hands_to_target(
        hand_actuator=hand_actuator,
        dex1_state_source=dex1_state_source,
        recorder=recorder,
        target_rad=(HAND_GRIP_MAX, HAND_GRIP_MAX),
        event_prefix=event_prefix,
        description=description,
    )


def _load_evaluation_initial_pose(
    skill_config_path: Path, skill_name: str, *, skill_config: dict | None = None
):
    """Load a skill pose, including the documented move-table fallback.

    ``move_table_base`` intentionally has no explicit ``initial_pose`` in the
    current configuration.  Production enters it from the final ``setup``
    stage, so the standalone evaluator must use that same pose instead of
    rejecting the skill before a read-only preflight can start.
    """

    from inference.desktop.lower_policy.initial_pose import (
        SkillInitialPose,
        initial_pose_from_config,
        load_initial_pose,
    )

    try:
        return (
            initial_pose_from_config(
                skill_config, skill_name, source=str(skill_config_path)
            )
            if skill_config is not None
            else load_initial_pose(skill_config_path, skill_name)
        )
    except KeyError:
        if skill_name != "move_table_base":
            raise

    import yaml
    from inference.desktop.lower_policy.skills.setup_skill import SetupSkill

    if skill_config is None:
        with open(skill_config_path, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    else:
        config = skill_config
    try:
        setup_config = config["skills"]["setup"]
    except (KeyError, TypeError) as exc:
        raise KeyError(
            "move_table_base requires either an explicit initial_pose or "
            "skills.setup.stages"
        ) from exc
    setup = SetupSkill.from_config(setup_config)
    return SkillInitialPose(
        arm_position_rad=setup.stages[-1].pose_rad.copy(),
        # No hand target is sent for this fallback.  These values exist only
        # to satisfy the shared metadata contract.
        dex1_opening_fraction=(1.0, 1.0),
        requires_separate_hand_initialization=False,
        hand_initialization_instruction="",
        dataset_repo_id="",
        dataset_revision="",
        training_episode_count=0,
        statistic="skill_config.setup.final_stage",
    )


def _require_strict_regular(loco_actuator: Any) -> Any:
    """Fail closed unless the physical G1 is in strict Regular Mode."""

    status = loco_actuator.get_loco_status()
    actual = (status.fsm_id, status.fsm_mode)
    if actual != (501, 0):
        raise RuntimeError(
            "G1 must be in strict Regular Mode immediately before actuation: "
            f"actual={actual}, expected=(501, 0)"
        )
    print("[preflight] strict Regular=(501,0); actuation gate passed")
    return status


def _wait_for_regular_handoff(loco_actuator: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        status = loco_actuator.get_loco_status()
        last = (status.fsm_id, status.fsm_mode)
        if last == (501, 0):
            print("[shutdown] strict Regular handoff verified (501,0)")
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"controlled release did not restore strict Regular Mode: last={last}"
    )


def main() -> int:
    args = parse_args()
    wandb_enabled = args.actuate if args.wandb is None else bool(args.wandb)

    from evaluate.model_evaluation.common.skill_registry import (
        SKILL_REGISTRY,
        resolve_vla_skill_class,
    )
    from inference.desktop.recording import RunRecorder
    from evaluate.model_evaluation.common.desktop_preview import (
        FourCameraDesktopPreview,
    )
    from evaluate.model_evaluation.common.wandb_exporter import (
        evaluate_eligibility,
        upload_run,
        write_upload_status,
    )

    entry = SKILL_REGISTRY[args.skill]
    if args.teacher_range_observe_only and args.skill != "flip_table":
        raise ValueError(
            "--teacher-range-observe-only is valid only with --skill flip_table"
        )
    variant_name = args.variant or entry.default_variant
    log_dir = args.log_dir or _default_log_dir(args.skill)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] skill={args.skill}, variant={variant_name}, log_dir={log_dir}")

    hybrid_runtime = None
    hybrid_cfg = None
    if args.pick_leg_hybrid:
        if args.skill != "pick_table_leg":
            raise ValueError("--pick-leg-hybrid is valid only with --skill pick_table_leg")
        if args.actuate and not args.use_real_hand:
            raise ValueError("physical --pick-leg-hybrid requires --use-real-hand")
        if args.no_wrist_cameras:
            raise ValueError("pick-leg GR00T hybrid requires both wrist cameras")
        if args.use_real_waist or args.dispatch_waist is True:
            raise ValueError(
                "pick-leg hybrid leaves waist/legs to Regular Mode; do not request real waist"
            )
        from inference.desktop.pick_leg_hybrid.real_skill import (
            load_reference_images,
            probe_vlm_endpoint,
        )

        hybrid_cfg, references = load_reference_images(
            args.pick_leg_hybrid_config,
            endpoint_override=args.pick_leg_vlm_endpoint,
            model_override=args.pick_leg_vlm_model,
        )
        endpoint = probe_vlm_endpoint(hybrid_cfg, references)
        hybrid_runtime = {
            "config": str(args.pick_leg_hybrid_config.resolve()),
            "phase3_executor": args.pick_leg_phase3_executor,
            "reference_count": len(references),
            **endpoint,
        }
        print(
            "[hybrid] VLM/VLA/MP preflight passed: "
            f"model={endpoint['model']} references={len(references)} "
            f"phase3={args.pick_leg_phase3_executor}; waist/legs=Regular"
        )
    elif args.pick_leg_phase3_executor != "vla":
        raise ValueError(
            "--pick-leg-phase3-executor requires --pick-leg-hybrid"
        )

    import yaml
    with open(args.skill_config) as f:
        skill_cfg = yaml.safe_load(f)
    from inference.desktop.lower_policy.initial_pose import (
        apply_policy_variant_profile,
    )

    skill_cfg = apply_policy_variant_profile(skill_cfg, args.skill, variant_name)

    # 1) initial_pose 読取り
    initial_pose = _load_evaluation_initial_pose(
        args.skill_config, args.skill, skill_config=skill_cfg
    )
    print(
        f"[init] initial_pose: ep={initial_pose.training_episode_count}, "
        f"hand_init_required={initial_pose.requires_separate_hand_initialization}"
    )
    if (
        args.actuate
        and initial_pose.requires_separate_hand_initialization
        and not args.use_real_hand
    ):
        raise ValueError(
            f"{args.skill} requires --use-real-hand: its physical sequence is "
            "Dex1 full-open -> dataset grasp -> policy"
        )

    # 2) skill_config.yaml から dispatch_waist を読む (CLI override 適用)
    hybrid_next_initial_pose = None
    if args.pick_leg_hybrid and args.pick_leg_phase3_executor == "rule_based":
        from inference.desktop.lower_policy.initial_pose import (
            initial_pose_from_config,
        )

        hybrid_next_initial_pose = initial_pose_from_config(
            skill_cfg,
            "insert_table_leg",
            source=str(args.skill_config),
        )
        assert hybrid_runtime is not None
        hybrid_runtime["next_skill_initial_pose"] = {
            "skill": "insert_table_leg",
            "arm_position_rad": hybrid_next_initial_pose.arm_position_rad.tolist(),
            "dex1_target_rad": list(hybrid_next_initial_pose.dex1_target_rad),
            "dataset_repo_id": hybrid_next_initial_pose.dataset_repo_id,
            "dataset_revision": hybrid_next_initial_pose.dataset_revision,
            "statistic": hybrid_next_initial_pose.statistic,
        }
    yaml_dispatch = bool(
        skill_cfg["skills"][args.skill].get("dispatch_waist", True)
    )
    dispatch_waist = (
        yaml_dispatch if args.dispatch_waist is None else bool(args.dispatch_waist)
    )
    if args.pick_leg_hybrid:
        dispatch_waist = False
    print(f"[init] dispatch_waist={dispatch_waist} (yaml={yaml_dispatch})")

    # 評価する skill を回す秒数。`--max-seconds` は skill_config の max_seconds_hard を
    # 上書きする (Issue #141 D6-4。秒数と動きは D4 で 1 本にまとめた)。
    yaml_hard_timeout_s = _resolve_hard_timeout_from_skill_config(skill_cfg, args.skill)
    skill_seconds = (
        yaml_hard_timeout_s if args.max_seconds is None else float(args.max_seconds)
    )
    if args.max_seconds is None and hybrid_cfg is not None:
        # The old learned-only pick budget can be shorter than the full hybrid FSM.
        skill_seconds = max(skill_seconds, hybrid_cfg.runtime.hard_timeout_sec)
    if skill_seconds <= 0:
        raise ValueError(f"--max-seconds must be > 0, got {args.max_seconds}")
    print(
        f"[timeout] skill={args.skill} seconds={skill_seconds:g} "
        f"(yaml max_seconds_hard={yaml_hard_timeout_s:g})"
    )

    gravity_compensator = None
    if args.actuate:
        # Load Pinocchio/Boost before the policy stack (Torch/OpenCV) and the
        # Unitree SDK can bind the system libstdc++ into this process.
        from inference.desktop.lower_policy.actuators.gravity_compensation import (
            OfficialG1ArmGravityCompensator,
        )

        gravity_compensator = OfficialG1ArmGravityCompensator()
        print(
            "[init] official G1 gravity feed-forward ready "
            f"cache={gravity_compensator.cache_sha256[:12]}"
        )

    # 3) policy variant 読取り + Model load (warmup 込み、本番と同じ関数)
    #    腕の所有権を取る前にここで読む: 初回 forward の lazy 初期化 (GR00T の Qwen/flow で
    #    2〜3 s、ACT / DP の CUDA 初期化と cuDNN autotune) を、30 Hz の loop に乗せないため。
    from inference.desktop.lower_policy.policies.config_loader import (
        load_policy_variant,
        load_yolo_settings,
    )

    variant = load_policy_variant(args.policy_config, variant_name)
    yolo_settings = load_yolo_settings(args.policy_config)
    print(
        f"[init] policy type={variant.policy_type}, mode={variant.policy_config.mode}, "
        f"ckpt_ref={variant.policy_config.ckpt_ref}"
    )
    policy = assembly.load_policy(variant, label=f"{args.skill}:{variant.name}")

    # lazy: env-isolated dependencies (cleaner は runtime env の numpy/yaml に依存)
    from inference.desktop.perception.cleaner import (
        load_cleanup_config,
        resolve_median_match_params,
    )

    exit_code = 0
    policy_start_time = 0.0
    policy_end_time = 0.0
    tick_count = 0
    joint_state_source = None
    dex1_state_source = None
    head_source = None
    wrist_l_source = None
    wrist_r_source = None
    arm_actuator = None
    hand_actuator = None
    return_hand_actuator = None
    loco_actuator = None
    skill = None
    arm_started = False
    hand_started = False
    return_hand_started = False
    skill_started = False
    measured_initial_arm: np.ndarray | None = None
    desktop_preview: FourCameraDesktopPreview | None = None
    preview_started = False

    with RunRecorder(log_dir) as recorder:
        # metadata dump
        recorder.write_metadata({
            "skill": args.skill,
            "variant": variant_name,
            "policy_type": variant.policy_type,
            "policy_mode": variant.policy_config.mode,
            "ckpt_ref": variant.policy_config.ckpt_ref,
            # Issue #137: 実行層の切替軸。これが残っていないと wandb から
            # 「どの設定で走った run か」を後追いできない (09-07 の実機 run が
            # まさにそうなっていた)。RTC / async / TE を実機で振るので必須。
            "action_space": variant.policy_config.action_space,
            "replan_family": variant.policy_config.replan_family,
            "execution_steps": variant.policy_config.execution_steps,
            "temporal_lambda": variant.policy_config.temporal_lambda,
            "rtc": {
                "enabled": variant.policy_config.rtc.enabled,
                "frozen_steps": variant.policy_config.rtc.frozen_steps,
                "overlap_steps": variant.policy_config.rtc.overlap_steps,
                "ramp_rate": variant.policy_config.rtc.ramp_rate,
            },
            "initial_pose": {
                "arm_position_rad": initial_pose.arm_position_rad.tolist(),
                "dex1_opening_fraction": list(initial_pose.dex1_opening_fraction),
                "requires_separate_hand_initialization": (
                    initial_pose.requires_separate_hand_initialization
                ),
                "dataset_repo_id": initial_pose.dataset_repo_id,
                "dataset_revision": initial_pose.dataset_revision,
                "training_episode_count": initial_pose.training_episode_count,
                "statistic": initial_pose.statistic,
            },
            "wrist_tool_offset": skill_cfg["skills"][args.skill].get(
                "wrist_tool_offset"
            ),
            "dispatch_waist": dispatch_waist,
            "yolo_ckpt_ref": yolo_settings.ckpt_ref,
            "yolo_conf": yolo_settings.conf,
            "yolo_imgsz": yolo_settings.imgsz,
            "joint_source": args.joint_source,
            "median_match": dict(
                zip(
                    ("max_centroid_dist", "ambiguity_ratio"),
                    resolve_median_match_params(
                        load_cleanup_config().get("median_filter", {})
                    ),
                )
            ),
            "setup_hold_sec": args.setup_hold_sec,
            "actuate": args.actuate,
            "use_real_waist": args.use_real_waist,
            "use_real_hand": args.use_real_hand,
            "max_seconds": skill_seconds,
            "four_camera_preview_requested": not args.no_camera_preview,
            "wandb_enabled": wandb_enabled,
            "teacher_range_observe_only": args.teacher_range_observe_only,
            "pick_leg_hybrid": hybrid_runtime,
            "started_at": datetime.now().isoformat(),
        })

        try:
            # 4) SDK init + sensor init (lazy import で default env 保護)
            from inference.desktop.lower_policy.actuators.g1_sdk import (
                G1SDKWalkActuator,
            )
            from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1ArmActuator
            from inference.desktop.lower_policy.actuators.hand import (
                G1HandActuator, MockHandActuator,
            )
            from inference.desktop.lower_policy.actuators.waist import (
                G1WaistActuator, MockWaistActuator,
            )
            from inference.desktop.lower_policy.skills.collision_aware_pre_motion import (
                CollisionAwareArmPreMotionSkill,
            )
            from inference.desktop.lower_policy.dispatcher import (
                SkillDispatchLowerPolicy,
            )
            from inference.desktop.orchestrator import (
                LiveSourceSafetyError,
                Orchestrator,
                enter_never,
            )
            from inference.desktop.perception.g1_urdf_fk import DEFAULT_URDF_PATH
            from inference.desktop.perception.policy_filter import (
                PolicyDetectionFilter,
                load_policy_filter_config,
            )
            from inference.desktop.perception.stream import DetectionStream

            if args.actuate:
                loco_actuator = G1SDKWalkActuator(
                    interface=args.interface or "eth0"
                )
            elif args.interface:
                from unitree_sdk2py.core.channel import ChannelFactoryInitialize

                ChannelFactoryInitialize(0, args.interface)

            # 手首 FK は skill ごとの tool offset を使う (本番と同じ、INF-9)
            fk_factory = assembly.FkFactory()

            # Sensors (本番と同じ関数。関節角は既定で rt/lowstate から、INF-18)
            sensors = assembly.build_sensor_sources(
                head_topic=args.head_topic,
                wrist_left_topic=None if args.no_wrist_cameras else args.wrist_left_topic,
                wrist_right_topic=None if args.no_wrist_cameras else args.wrist_right_topic,
                joint_source=args.joint_source,
                joint_state_topic=args.joint_state_topic,
            )
            joint_state_source = sensors.joint
            dex1_state_source = sensors.dex1
            head_source = sensors.head
            wrist_l_source = sensors.wrist_left
            wrist_r_source = sensors.wrist_right
            print(f"[init] joint_source={args.joint_source}")

            # --actuate 依存の flag consistency check (arm 無しで waist/hand real は
            # arm_sdk publisher が居ないので DDS 経路が破綻する)
            if not args.actuate and (args.use_real_waist or args.use_real_hand):
                raise ValueError(
                    "--use-real-waist / --use-real-hand require --actuate "
                    "(no arm_sdk publisher without --actuate)"
                )

            # Arm actuator (real / mock 経路 = --actuate で分岐)
            if args.actuate:
                arm_actuator = G1ArmActuator(
                    gravity_compensator=gravity_compensator
                )
                print("[init] arm_actuator initialized; start deferred to safety gate")
            else:
                arm_actuator = None
                print("[init] dry-run mode (no arm dispatch)")

            waist_actuator = (
                G1WaistActuator(arm_actuator)
                if args.use_real_waist and arm_actuator is not None
                else MockWaistActuator()
            )
            hand_actuator = (
                G1HandActuator() if args.use_real_hand else MockHandActuator()
            )
            # 検出は mode に依らず毎 tick 回す (本番と同じ、Issue #141 D3)。
            # 重み・conf・imgsz は policy_config.yaml の `yolo` から (INF-10)。
            perception = assembly.build_yolo_perception(
                args.policy_config, device=variant.policy_config.device
            )
            # preflight (sensor の待ちとカメラの検査) は検出を使わない
            def collector():
                return _collect_obs(
                    joint_state_source, dex1_state_source,
                    head_source, wrist_l_source, wrist_r_source,
                )
            print("[warmup] waiting for joint + Dex1 + camera observations...")
            warmup_observation = _wait_for_finite_obs(
                collector,
                require_wrist=not args.no_wrist_cameras,
            )
            recorder.write_event({"kind": "sensors_warmup_done"})
            camera_rates = _validate_camera_stream_window(
                collector,
                require_wrist=not args.no_wrist_cameras,
                duration_s=2.0,
            )
            recorder.write_event({
                "kind": "camera_rate_preflight_passed",
                "duration_s": 2.0,
                "rates_hz": camera_rates,
            })
            head_rgb = np.asarray(warmup_observation["head_rgb"])
            warmup_detections = perception.predict(
                np.ascontiguousarray(select_head_view(head_rgb, "left"))
            )
            recorder.write_event(
                {
                    "kind": "overlay_yolo_warmup_done",
                    "detections": len(warmup_detections),
                    "weight": yolo_settings.ckpt_ref,
                }
            )
            print(f"[warmup] YOLO path ready (detections={len(warmup_detections)})")

            if args.no_wrist_cameras and not args.no_camera_preview:
                print(
                    "[preview] disabled because --no-wrist-cameras was requested; "
                    "four physical views are required"
                )
            elif not args.no_camera_preview:
                desktop_preview = FourCameraDesktopPreview(
                    collector,
                    window_title=(
                        f"IROS 2026 RAMEN - {args.skill} / {variant_name}"
                    ),
                )
                preview_started = desktop_preview.start()
                if preview_started:
                    print(
                        "[preview] four-camera Desktop monitor enabled "
                        "(head L/R, wrist L/R, measured arm angles); "
                        "q/Esc closes only the monitor"
                    )
                    recorder.write_event({"kind": "desktop_preview_started"})

            if args.actuate:
                if desktop_preview is not None:
                    desktop_preview.set_status("WAITING FOR PRE-MOTION")
                setup_description = (
                    "Dex1 will open fully, then the arms will move to the "
                    "dataset initial pose"
                    if initial_pose.requires_separate_hand_initialization
                    else "the arms will move to the dataset initial pose"
                )
                input(
                    "Harness / E-stop / arm, waist, and hand clearance confirmed. "
                    f"Press Enter and {setup_description}, or Ctrl+C to cancel: "
                )
                # The operator may spend an arbitrary amount of time at the
                # prompt.  Prove the streams are live again immediately before
                # acquiring any physical arm ownership.
                _validate_camera_stream_window(
                    collector,
                    require_wrist=not args.no_wrist_cameras,
                    duration_s=CAMERA_FINAL_GATE_DURATION_S,
                )
                _require_strict_regular(loco_actuator)
                if desktop_preview is not None:
                    desktop_preview.set_status("PRE-MOTION")
                arm_actuator.start()
                arm_started = True
                if args.use_real_hand and hand_actuator is not None:
                    # The operator can remain at the Enter gate indefinitely.
                    # Prime from the current bilateral sample, not the earlier
                    # warmup snapshot, so the first write can never jump back
                    # to a stale hand pose.
                    current_hand_state = dex1_state_source.get()
                    if current_hand_state is None:
                        raise RuntimeError(
                            "fresh bilateral Dex1 state disappeared at the final gate"
                        )
                    hand_age_s = max(
                        time.monotonic_ns()
                        - int(current_hand_state.left_received_monotonic_ns),
                        time.monotonic_ns()
                        - int(current_hand_state.right_received_monotonic_ns),
                    ) / 1e9
                    if hand_age_s > 0.5:
                        raise RuntimeError(
                            "bilateral Dex1 state is stale at the final gate "
                            f"(age={hand_age_s:.3f}s)"
                        )
                    measured_hand = current_hand_state.position_rad
                    hand_actuator.prime_hold(measured_hand)
                    hand_actuator.start()
                    hand_started = True
                print("[init] real actuator publishers started after final gate")

                # Object-in-hand tasks must reach the arm frame-zero pose with
                # both grippers physically open.  This leaves an unambiguous
                # operator window to place the table leg before the dataset
                # grasp is commanded.
                # 頭の手順に手の skill が入るとき (--use-real-hand) は、その 1 番目が
                # 同じ全開まで開くので二度やらない。
                if (
                    initial_pose.requires_separate_hand_initialization
                    and not args.use_real_hand
                ):
                    _open_hands_fully(
                        hand_actuator=hand_actuator,
                        dex1_state_source=dex1_state_source,
                        recorder=recorder,
                        event_prefix="startup_hands_open",
                        description=(
                            "opening both Dex1 hands before arm pre-motion"
                        ),
                    )

            # 5) 頭の手順 + policy を orchestrator で回す (Issue #141 束 1-8 / D6)。
            #    本番と同じ skill の列
            #      手を開く → 腕の pre-motion → 手を開始の開度へ → N 秒保持 → 評価する skill
            #    を、評価だけの区切り (脚を置く Enter / 開始 Enter) で分けて run_live する。
            #    read-only (--actuate 無し) は腕が動かないので頭の手順を飛ばす (従来どおり)。
            skill_cfg["skills"][args.skill]["dispatch_waist"] = dispatch_waist
            VlaSkillCls = resolve_vla_skill_class(entry)
            extra_skill_kwargs = None
            if args.teacher_range_observe_only:
                extra_skill_kwargs = {"teacher_range_observe_only": True}
            if args.pick_leg_hybrid:
                from inference.desktop.pick_leg_hybrid.real_skill import (
                    RealPickLegHybridVlaSkill,
                )
                VlaSkillCls = RealPickLegHybridVlaSkill
                extra_skill_kwargs = {
                    "hybrid_config_path": args.pick_leg_hybrid_config,
                    "hybrid_vlm_endpoint": args.pick_leg_vlm_endpoint,
                    "hybrid_vlm_model": args.pick_leg_vlm_model,
                    "phase3_executor": args.pick_leg_phase3_executor,
                }
                if hybrid_next_initial_pose is not None:
                    extra_skill_kwargs.update({
                        "next_initial_arm_target": (
                            hybrid_next_initial_pose.arm_position_rad
                        ),
                        "next_initial_hand_target": np.asarray(
                            hybrid_next_initial_pose.dex1_target_rad,
                            dtype=np.float64,
                        ),
                    })
            built = assembly.build_vla_skill(
                skill_name=args.skill,
                vla_skill_cls=VlaSkillCls,
                variant=variant,
                skill_config=skill_cfg,
                waist_actuator=waist_actuator,
                hand_actuator=hand_actuator,
                fk_factory=fk_factory,
                policy=policy,
                extra_skill_kwargs=extra_skill_kwargs,
            )
            skill = built.skill
            probe_state: dict = {}

            measured_initial_arm = _measured_arm_from_observation(collector())
            head_skills: list = []
            if arm_actuator is None:
                recorder.write_event({"kind": "pre_motion_skipped_read_only"})
                print("[pre-motion] skipped in read-only mode; no robot command sent")
            else:
                head_skills = assembly.build_head_procedure(
                    skill_config=skill_cfg,
                    skill_name=args.skill,
                    initial_pose=initial_pose,
                    hand_actuator=hand_actuator,
                    hold_sec=args.setup_hold_sec,
                    include_hand=args.use_real_hand,
                )
            sequence = [s.name for s in head_skills] + [args.skill]
            registry = {s.name: s for s in head_skills}
            registry[args.skill] = skill
            dispatcher = SkillDispatchLowerPolicy(registry)
            orch = Orchestrator(
                perception,
                DetectionStream(load_cleanup_config()),
                dispatcher,
                initial_skill=sequence[0],
                transitions={
                    name: ([sequence[i + 1]] if i + 1 < len(sequence) else [])
                    for i, name in enumerate(sequence)
                },
                enter_check={name: enter_never for name in sequence},
                actuator_send_fn=(
                    arm_actuator.send_action if arm_actuator is not None else None
                ),
                joint_state_source=joint_state_source,
                dex1_state_source=dex1_state_source,
                wrist_left_source=wrist_l_source,
                wrist_right_source=wrist_r_source,
                policy_filter=PolicyDetectionFilter(load_policy_filter_config()),
                hard_timeout_by_skill={args.skill: skill_seconds},
                on_tick=lambda result, obs, active: record_policy_tick(
                    recorder,
                    obs,
                    active,
                    learned_skills=frozenset({args.skill}),
                    arm_actuator=arm_actuator,
                    probe_state=probe_state,
                ),
            )
            print(f"[orch] skill sequence: {' -> '.join(sequence)}")

            def _run_until(stop_before: str | None) -> None:
                """次の区切りまで回す。`stop_before` に入った時点で戻る。"""
                try:
                    orch.run_live(
                        head_source,
                        hz=TICK_HZ,
                        stop_on_skills=frozenset() if stop_before is None else frozenset({stop_before}),
                    )
                except LiveSourceSafetyError as exc:
                    recorder.write_event({
                        "kind": (
                            "camera_stream_stale_hold"
                            if "camera" in str(exc)
                            else "live_source_safety_error"
                        ),
                        "detail": str(exc),
                    })
                    # The arm publisher still owns arm_sdk and continuously
                    # holds the last safe target here.  Releasing immediately
                    # can drop a table leg (and makes a benign convergence
                    # timeout look like sudden arm power loss).  During an
                    # actuated policy interval, require the operator to secure
                    # the object/arms before the existing controlled release.
                    # E-stop remains the immediate escape for unsafe motion;
                    # Ctrl+C also proceeds to controlled release.
                    if args.actuate and arm_started and policy_start_time > 0:
                        if desktop_preview is not None:
                            desktop_preview.set_status(
                                "SAFETY HOLD - SECURE OBJECT, THEN ENTER"
                            )
                        recorder.write_event({
                            "kind": "operator_safety_hold_started",
                            "detail": str(exc),
                        })
                        print(
                            "[safety-hold] Last safe arm/Dex1 targets remain "
                            "actively held. Secure the object and support the "
                            "arms; use E-stop for immediate danger.",
                            file=sys.stderr,
                        )
                        try:
                            input(
                                "[safety-hold] Press Enter only after it is safe "
                                "to perform the controlled arm_sdk release "
                                "(Ctrl+C to release now): "
                            )
                            recorder.write_event({
                                "kind": "operator_safety_hold_released"
                            })
                        except EOFError:
                            # Non-interactive invocations cannot acknowledge a
                            # hold.  Record that fact and retain the established
                            # finally-path controlled release rather than hang.
                            recorder.write_event({
                                "kind": "operator_safety_hold_unavailable",
                                "reason": "stdin_eof",
                            })
                    raise

            # 6) 頭の手順。物を持って始まる skill は「掴む幅へ」の手前で操作者を待つ。
            grasp_skill_name = next(
                (s.name for s in head_skills if s.name.startswith("hand_grasp_")), None
            )
            if head_skills:
                if desktop_preview is not None:
                    desktop_preview.set_status("PRE-MOTION")
                recorder.write_event({"kind": "pre_motion_start"})
                _run_until(grasp_skill_name or args.skill)
                if grasp_skill_name is not None:
                    if desktop_preview is not None:
                        desktop_preview.set_status("HAND INITIALIZATION")
                    print()
                    print(f"[hand-init] {initial_pose.hand_initialization_instruction}")
                    input(
                        "[hand-init] Hands are fully open and arms are at the dataset "
                        "initial pose. Place the table leg between the hands, then "
                        "press Enter to move Dex1 to the dataset initial width: "
                    )
                    _run_until(args.skill)
                recorder.write_event({"kind": "pre_motion_complete"})
                print(
                    "[pre-motion] dataset initial arm/hand pose reached and held; "
                    "policy has not started"
                )

            # 7) Policy start gate
            if args.actuate:
                if desktop_preview is not None:
                    desktop_preview.set_status("READY FOR POLICY")
                input(
                    "[gate] Dataset initial arm/hand pose is held. Press Enter "
                    "to start policy (Ctrl+C to abort): "
                )
            else:
                print("[gate] read-only policy preflight starting automatically")
            # Pre-motion and object placement can take tens of seconds.  A
            # device that disappeared during that interval must prevent the
            # very first policy command while the measured pose is still held.
            policy_camera_rates = _validate_camera_stream_window(
                collector,
                require_wrist=not args.no_wrist_cameras,
                duration_s=CAMERA_FINAL_GATE_DURATION_S,
            )
            recorder.write_event({
                "kind": "camera_rate_policy_gate_passed",
                "duration_s": CAMERA_FINAL_GATE_DURATION_S,
                "rates_hz": policy_camera_rates,
            })

            # 8) Policy (orchestrator が回す。列の最後なので時間切れで戻ってくる)
            skill_started = True
            if desktop_preview is not None:
                desktop_preview.set_status("POLICY INFERENCE")
            recorder.write_event({"kind": "policy_start", "variant": variant_name})
            policy_start_time = time.monotonic()
            recorder.start_policy_capture(
                time.monotonic_ns(),
                observation_supplier=lambda: _collect_obs(
                    None,
                    None,
                    head_source,
                    wrist_l_source,
                    wrist_r_source,
                ),
                # Poll at 2x the 30 Hz source rate so latest-only DDS frames
                # are not phase-aliased with slower model inference.
                camera_sample_hz=60.0,
            )
            _run_until(None)
            skill_started = False
            policy_end_time = time.monotonic()
            elapsed_s = policy_end_time - policy_start_time
            stop_reason = (
                "requested_max_seconds" if elapsed_s >= skill_seconds else "skill_complete"
            )
            recorder.stop_policy_capture(time.monotonic_ns())
            recorder.write_event({
                "kind": "policy_stop",
                "elapsed_s": elapsed_s,
                "stop_reason": stop_reason,
            })
            recorder.policy_seconds = elapsed_s
            tick_count = recorder.tick_count
            print(f"[policy] loop end, ticks={tick_count}, "
                  f"elapsed={recorder.policy_seconds:.2f}s")

            # A normal evaluation completion follows the same proven reverse
            # path used by issue 70.  Fault/Ctrl+C paths do not start a new
            # autonomous sweep; they go directly to controlled release below.
            if arm_actuator is not None:
                if desktop_preview is not None:
                    desktop_preview.set_status("SAFE ARM RETURN")
                if measured_initial_arm is None:
                    raise RuntimeError(
                        "safe return lacks the measured arm pose from before pre-motion"
                    )
                from evaluate.model_evaluation.common.safe_return import (
                    build_return_arm_waypoints,
                )

                return_waypoints = build_return_arm_waypoints(
                    measured_initial_arm,
                    initial_pose.arm_position_rad,
                )
                recorder.write_event({"kind": "return_motion_started"})
                try:
                    # First restore the model's dataset/frame-zero arm pose.
                    _drive_return_waypoints(
                        return_waypoints[:1],
                        collector=collector,
                        arm_actuator=arm_actuator,
                        recorder=recorder,
                    )

                    # Then release any object before the large reverse sweep.
                    physical_hand = hand_actuator if args.use_real_hand else None
                    if physical_hand is None:
                        current_hand = dex1_state_source.get()
                        if current_hand is None:
                            raise RuntimeError(
                                "safe return cannot initialize Dex1 without state"
                            )
                        return_hand_actuator = G1HandActuator()
                        return_hand_actuator.prime_hold(current_hand.position_rad)
                        return_hand_actuator.start()
                        return_hand_started = True
                        physical_hand = return_hand_actuator
                        recorder.write_event({
                            "kind": "return_hand_actuator_started_at_measured_pose"
                        })
                    _open_hands_fully(
                        hand_actuator=physical_hand,
                        dex1_state_source=dex1_state_source,
                        recorder=recorder,
                    )

                    # Reverse the remaining startup waypoints and finish at
                    # the measured lowered pose captured before takeover.
                    _drive_return_waypoints(
                        return_waypoints[1:],
                        collector=collector,
                        arm_actuator=arm_actuator,
                        recorder=recorder,
                    )
                except Exception as exc:
                    recorder.write_event({
                        "kind": "return_motion_error", "message": repr(exc)
                    })
                    raise
                recorder.write_event({"kind": "return_motion_complete"})
                print(
                    "[shutdown] reverse arm path complete; arms are back at "
                    "the measured lowered pose"
                )

        except KeyboardInterrupt:
            print("[abort] KeyboardInterrupt received", file=sys.stderr)
            recorder.write_event({"kind": "abort_keyboard_interrupt"})
            if policy_start_time > 0 and policy_end_time == 0:
                policy_end_time = time.monotonic()
                recorder.policy_seconds = policy_end_time - policy_start_time
            exit_code = 130
        except Exception as exc:
            print(f"[error] {exc!r}", file=sys.stderr)
            recorder.write_event({"kind": "exception", "message": repr(exc)})
            exit_code = 1
        finally:
            # 9) Controlled release
            recorder.stop_policy_capture(time.monotonic_ns())
            tick_count = recorder.tick_count
            if policy_start_time > 0 and policy_end_time == 0:
                policy_end_time = time.monotonic()
                recorder.policy_seconds = policy_end_time - policy_start_time
            shutdown_failures: list[str] = []
            if skill_started and skill is not None:
                try:
                    skill.stop()
                    skill_started = False
                except Exception as exc:
                    shutdown_failures.append(f"skill stop: {exc!r}")
            if arm_started and arm_actuator is not None:
                try:
                    if desktop_preview is not None:
                        desktop_preview.set_status("CONTROLLED RELEASE")
                    print("[shutdown] arm_sdk controlled release started")
                    arm_actuator.controlled_release(duration_s=2.0)
                    recorder.write_event({"kind": "arm_actuator_controlled_release"})
                    print("[shutdown] arm_sdk controlled release complete (weight=0)")
                except Exception as exc:
                    shutdown_failures.append(f"arm controlled release: {exc!r}")
                try:
                    arm_actuator.stop()
                    arm_started = False
                except Exception as exc:
                    shutdown_failures.append(f"arm publisher stop: {exc!r}")
            if hand_started and hand_actuator is not None:
                try:
                    hand_actuator.stop()
                    hand_started = False
                    recorder.write_event({"kind": "hand_actuator_stopped"})
                except Exception as exc:
                    shutdown_failures.append(f"hand publisher stop: {exc!r}")
            if return_hand_started and return_hand_actuator is not None:
                try:
                    return_hand_actuator.stop()
                    return_hand_started = False
                    recorder.write_event({"kind": "return_hand_actuator_stopped"})
                except Exception as exc:
                    shutdown_failures.append(
                        f"return-hand publisher stop: {exc!r}"
                    )
            if args.actuate and loco_actuator is not None:
                try:
                    _wait_for_regular_handoff(loco_actuator)
                except Exception as exc:
                    shutdown_failures.append(f"Regular handoff: {exc!r}")
            if desktop_preview is not None:
                try:
                    desktop_preview.close()
                    if preview_started:
                        recorder.write_event({
                            "kind": "desktop_preview_stopped",
                            "errors": list(desktop_preview.errors),
                        })
                except Exception as exc:
                    # A diagnostic monitor is never allowed to turn a safe
                    # robot run into an actuation/shutdown failure.
                    recorder.write_event({
                        "kind": "desktop_preview_close_error",
                        "message": repr(exc),
                    })
            for source in (
                wrist_r_source,
                wrist_l_source,
                head_source,
                dex1_state_source,
                joint_state_source,
            ):
                if source is not None:
                    try:
                        source.close()
                    except Exception as exc:
                        shutdown_failures.append(
                            f"{type(source).__name__} close: {exc!r}"
                        )
                        recorder.write_event({
                            "kind": "source_close_error", "message": repr(exc)
                        })
            try:
                policy.close()
            except Exception as exc:
                shutdown_failures.append(f"policy close: {exc!r}")
            if shutdown_failures:
                exit_code = 1
                for failure in shutdown_failures:
                    recorder.write_event({
                        "kind": "controlled_shutdown_error", "message": failure
                    })
                    print(f"[shutdown] {failure}", file=sys.stderr)

    # Capture packaging finishes on RunRecorder context exit.  A run with a
    # truncated camera stream is diagnostic, never a successful evaluation,
    # even if arm control and the safe return completed normally.
    capture_report_path = log_dir / "capture" / "capture_report.json"
    if policy_start_time > 0.0 and capture_report_path.is_file():
        capture_report = json.loads(capture_report_path.read_text(encoding="utf-8"))
        if not bool(capture_report.get("complete", False)):
            if exit_code == 0:
                exit_code = 1
            print(
                "[capture] evaluation marked failed: full four-camera policy "
                "interval was not captured",
                file=sys.stderr,
            )

    policy_seconds = (
        policy_end_time - policy_start_time if policy_start_time else 0.0
    )
    run_status = "completed" if exit_code == 0 else (
        "interrupted" if exit_code == 130 else "failed"
    )
    (log_dir / "run_result.json").write_text(
        json.dumps(
            {
                "runner_status": run_status,
                "runner_exit_code": exit_code,
                "policy_seconds": policy_seconds,
                "tick_count": tick_count,
                "requested_max_seconds": args.max_seconds,
                "abnormal_runs_are_uploadable": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # 10) W&B upload.  Long physical failures are diagnostics, not discarded
    # runs; the duration/tick gate still excludes 1-second smoke tests.
    if wandb_enabled:
        decision = evaluate_eligibility(
            actuated=args.actuate,
            policy_seconds=policy_seconds,
            tick_count=tick_count,
            exit_code=exit_code,
            requested_seconds=args.max_seconds,
        )
        url: str | None = None
        if decision.eligible:
            run_name = args.wandb_run_name or (
                f"{args.skill}_{variant_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            url = upload_run(log_dir, metadata={
                "skill": args.skill, "variant": variant_name,
                "policy_seconds": policy_seconds,
                "tick_count": tick_count,
                "requested_max_seconds": args.max_seconds,
                "runner_status": run_status,
                "runner_exit_code": exit_code,
            }, run_name=run_name, tags=(args.skill, variant_name))
        else:
            print(f"[wandb] upload skipped: {', '.join(decision.reasons)}")
        write_upload_status(log_dir, decision=decision, url=url)

    print(f"[done] log_dir={log_dir}, exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    # CycloneDDS owns native recvMC threads which can outlive closed Python
    # reader wrappers and crash during interpreter finalization.  All robot,
    # sensor, preview, worker, recording, and W&B resources are explicitly
    # finalized by main(); terminate the CLI without running native module
    # destructors after flushing operator output.
    result = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)
