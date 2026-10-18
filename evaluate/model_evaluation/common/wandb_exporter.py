"""W&B export for long physical evaluations, including failed runs.

# 認証

標準の`wandb login`で`~/.netrc`へ認証を保存するか、runner起動前に
`WANDB_API_KEY`を環境変数へ設定する:

    pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill --wandb ...

code側では認証情報を読み書きせず、wandb SDKの標準認証を使う。

# Upload eligibility

正常runは以下を全て満たした場合にuploadする:

- `--actuate` 有り (real dispatch した run のみ upload)
- policy 実 loop 時間 >= `min_seconds` (default 10 秒)
- action.jsonl の tick 数 >= `min_ticks` (default 30 = 1 秒 @ 30Hz、緩め)

実行中例外、Ctrl+C、安全復帰失敗は、要求時間が10秒以上の本番試行でpolicy
commandが1件でも出ていれば、実時間が10秒未満でもuploadする。意図的な1秒
smokeとread-onlyは従来通り除外する。

# Entity / Project

CLAUDE.md 記載の team 固定:
    entity:  ken05-matuo-llm-88_llm_2025_suzuki
    project: iros-2026-ramen-real-policy-evaluation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_WANDB_ENTITY = "ken05-matuo-llm-88_llm_2025_suzuki"
DEFAULT_WANDB_PROJECT = "iros-2026-ramen-real-policy-evaluation"
DEFAULT_MIN_SECONDS = 10.0
DEFAULT_MIN_TICKS = 30


@dataclass(frozen=True)
class UploadDecision:
    eligible: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_eligibility(
    *,
    actuated: bool,
    policy_seconds: float,
    tick_count: int,
    exit_code: int,
    requested_seconds: float | None = None,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    min_ticks: int = DEFAULT_MIN_TICKS,
) -> UploadDecision:
    """Upload 条件全て満たしたら eligible=True、reasons は違反 list。"""
    reasons: list[str] = []
    if not actuated:
        reasons.append("run was not actuated (--actuate missing)")
    abnormal = exit_code != 0
    if abnormal:
        if requested_seconds is not None and requested_seconds < min_seconds:
            reasons.append(
                f"requested duration {requested_seconds:.2f}s is a short smoke "
                f"(< {min_seconds:g}s)"
            )
        if policy_seconds <= 0.0 or tick_count < 1:
            reasons.append("abnormal run ended before any policy command was recorded")
    else:
        if policy_seconds < min_seconds:
            reasons.append(
                f"policy ran only {policy_seconds:.2f}s (< {min_seconds:g}s)"
            )
        if tick_count < min_ticks:
            reasons.append(
                f"only {tick_count} tick(s) recorded (< {min_ticks})"
            )
    return UploadDecision(eligible=not reasons, reasons=tuple(reasons))


def upload_run(
    log_dir: Path,
    *,
    metadata: dict[str, Any],
    run_name: str,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
    tags: tuple[str, ...] = (),
) -> str | None:
    """Upload telemetry, available camera videos, and diagnostic artifacts.

    Returns:
        wandb run URL (str) on success, None on failure (auth/network 系)。
    """
    try:
        import wandb  # lazy: eval env のみ install 想定
    except ImportError:
        print("[wandb] wandb SDK not installed; skipping upload")
        return None
    if not callable(getattr(wandb, "init", None)):
        location = getattr(wandb, "__file__", None)
        print(
            "[wandb] imported module is not the W&B SDK "
            f"(module_file={location!r}, init missing); skipping upload"
        )
        return None

    try:
        stored_metadata_path = Path(log_dir) / "metadata.json"
        stored_metadata = (
            json.loads(stored_metadata_path.read_text(encoding="utf-8"))
            if stored_metadata_path.is_file()
            else {}
        )
        run_config = {**stored_metadata, **metadata}
        run = wandb.init(
            entity=entity,
            project=project,
            name=run_name,
            config=run_config,
            tags=[
                *list(tags),
                (
                    "completed"
                    if int(metadata.get("runner_exit_code", 0)) == 0
                    else "runner-failed"
                ),
            ],
        )
    except Exception as exc:
        print(f"[wandb] init failed: {exc!r}; upload skipped")
        return None

    try:
        for filename in (
            "metadata.json",
            "run_result.json",
            "events.jsonl",
            "states.jsonl",
            "actions.jsonl",
            "capture/capture_report.json",
            "capture/camera_frames.jsonl",
            "capture/video_manifest.json",
        ):
            path = Path(log_dir) / filename
            if path.exists():
                wandb.save(str(path), base_path=str(log_dir), policy="now")
        manifest_path = Path(log_dir) / "capture" / "video_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            media = {}
            for role, video in manifest.get("videos", {}).items():
                path = Path(log_dir) / video["relative_path"]
                if path.is_file():
                    media[f"camera/{role}"] = wandb.Video(str(path), format="mp4")
            if media:
                run.log(media)

        def read_jsonl(path: Path) -> list[dict[str, Any]]:
            if not path.is_file():
                return []
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        states = read_jsonl(Path(log_dir) / "states.jsonl")
        actions = read_jsonl(Path(log_dir) / "actions.jsonl")
        count = min(len(states), len(actions))
        if count:
            maximum = 1000
            indices = (
                list(range(count))
                if count <= maximum
                else sorted(
                    {
                        round(i * (count - 1) / (maximum - 1))
                        for i in range(maximum)
                    }
                )
            )
            columns = (
                ["policy_time_s", "latency_ms"]
                + [f"action_{i:02d}" for i in range(19)]
                + [f"state_joint_{i:02d}" for i in range(29)]
                + [f"state_velocity_{i:02d}" for i in range(29)]
                + [f"state_effort_{i:02d}" for i in range(29)]
                + ["state_hand_left", "state_hand_right"]
            )
            first_t = int(actions[indices[0]].get("t", 0))
            rows = []
            for index in indices:
                action = actions[index]
                state = states[index]
                action_values = list(action.get("chunk_0") or [])
                joint_values = list(state.get("joint_position") or [])
                velocity_values = list(
                    state.get("joint_velocity") or [None] * 29
                )
                # Issue #137: effort は #137 以前の run には無い。列を落とすと
                # 新旧で table schema が変わるため、欠損は None 埋めで列を保つ。
                effort_values = list(
                    state.get("joint_effort") or [None] * 29
                )
                hand_values = list(state.get("hand_position_rad") or [])
                if (
                    len(action_values) != 19
                    or len(joint_values) != 29
                    or len(velocity_values) != 29
                    or len(effort_values) != 29
                    or len(hand_values) != 2
                ):
                    continue
                rows.append(
                    [
                        (int(action.get("t", first_t)) - first_t) / 1e9,
                        action.get("latency_ms"),
                        *action_values,
                        *joint_values,
                        *velocity_values,
                        *effort_values,
                        *hand_values,
                    ]
                )
            if rows:
                run.log({
                    "telemetry/state_action_trace": wandb.Table(
                        columns=columns, data=rows
                    )
                })
        run.summary.update({
            "runner/exit_code": int(metadata.get("runner_exit_code", 0)),
            "runner/status": metadata.get("runner_status", "unknown"),
            "policy/duration_s": metadata.get("policy_seconds", 0.0),
            "policy/tick_count": metadata.get("tick_count", 0),
        })
        capture_report_path = Path(log_dir) / "capture" / "capture_report.json"
        if capture_report_path.is_file():
            capture_report = json.loads(
                capture_report_path.read_text(encoding="utf-8")
            )
            run.summary.update({
                "capture/complete": bool(capture_report.get("complete", False)),
                "capture/queue_drop_count": int(
                    capture_report.get("queue_drop_count", 0)
                ),
                "capture/errors": list(capture_report.get("errors", [])),
                "capture/validation_errors": list(
                    capture_report.get("validation_errors", [])
                ),
            })
            for role, count in capture_report.get(
                "camera_written_frame_count", {}
            ).items():
                run.summary[f"capture/{role}/frames"] = int(count)
            for role, hz in capture_report.get("camera_effective_hz", {}).items():
                run.summary[f"capture/{role}/hz"] = float(hz)
            for role, value in capture_report.get(
                "camera_coverage_fraction", {}
            ).items():
                run.summary[f"capture/{role}/coverage"] = float(value)
            for role, value in capture_report.get(
                "camera_max_interframe_gap_s", {}
            ).items():
                run.summary[f"capture/{role}/max_gap_s"] = float(value)
        # W&B <=0.18 exposed ``Run.get_url()`` while current SDKs expose the
        # same value as the ``Run.url`` property.  Support both so a completed
        # upload is not incorrectly recorded as pending after SDK upgrades.
        get_url = getattr(run, "get_url", None)
        url = (
            get_url()
            if callable(get_url)
            else getattr(run, "url", None)
        ) or ""
        wandb.finish(exit_code=int(metadata.get("runner_exit_code", 0)))
        return url
    except Exception as exc:
        print(f"[wandb] upload failed: {exc!r}")
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        return None


def write_upload_status(
    log_dir: Path,
    *,
    decision: UploadDecision,
    url: str | None,
) -> None:
    """upload 判定 + 結果を wandb_upload.json に残す (再送 / 監査用)。"""
    payload = {
        "eligible": decision.eligible,
        "reasons": list(decision.reasons),
        "uploaded": bool(url),
        "status": (
            "uploaded"
            if url
            else "upload_failed_or_pending"
            if decision.eligible
            else "ineligible"
        ),
        "wandb_url": url,
    }
    (Path(log_dir) / "wandb_upload.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
