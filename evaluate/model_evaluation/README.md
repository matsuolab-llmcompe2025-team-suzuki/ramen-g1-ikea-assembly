# Real skill evaluation

`evaluate.model_evaluation.runners.run_skill` evaluates one registered VLA skill
without changing the production orchestrator. It reuses the production policy,
sensor, actuator, and collision-aware startup components, while evaluation-only
safety return and recording remain in this directory.

## Safety flow

With `--actuate`, the runner requires an operator confirmation and strict G1
Regular Mode `(501, 0)` before starting any publisher. Arms follow the shared
startup clearance path and finish at the skill's dataset/frame-zero pose. The
policy-start Enter is accepted only after that measured pose has converged; a
transient clearance pose is never treated as the policy start pose.

For `insert_table_leg`, `rotate_leg_to_tighten`, and `insert_and_tighten`,
`--use-real-hand` is mandatory. Both Dex1 hands first move to the physical 5.4 rad open endpoint and
stay open while the arms reach frame zero. The operator then places the table
leg between the hands and presses Enter once to move to the dataset initial
grasp width. A second, separate Enter starts policy inference. Both hand moves
are ramped and confirmed from fresh bilateral state before the next gate.

After a normal policy deadline:

1. return the arms to dataset/frame-zero;
2. open both Dex1 hands to the physical 5.4 rad maximum and confirm live state;
3. traverse the resolved startup waypoints in exact reverse order;
4. restore the measured lowered arm pose captured before takeover;
5. ramp `arm_sdk` weight to zero and verify Regular Mode `(501, 0)`.

An exception or `Ctrl+C` does not start a new arm sweep. It stops camera capture,
holds the last command through controlled release, stops all publishers
independently, and records any shutdown failure.

## Recording and W&B

After model and sensor warmup, the runner opens a latest-only Desktop monitor by
default. It shows head left/right above wrist left/right, with measured arm
angles and the current phase (`PRE-MOTION`, `READY FOR POLICY`, `POLICY
INFERENCE`, `SAFE ARM RETURN`, or `CONTROLLED RELEASE`). Camera polling, JPEG
encoding, HUD composition, and GUI handling are isolated from the policy/control
thread. Closing the window with `q` or `Esc` closes only the monitor. Use
`--no-camera-preview` for a deliberately headless run.

The monitor remains live through prompts, pre-motion, policy inference, and the
safe return, but file recording still covers only the policy interval. The
recording camera sampler polls independently at 60 Hz so the three physical 30
Hz camera streams are not aliased by slower model inference. Outputs are written
under the selected run directory:

```text
metadata.json
run_result.json
events.jsonl
states.jsonl
actions.jsonl
capture/capture_report.json
capture/camera_frames.jsonl
capture/video_manifest.json
capture/videos/{head_left,head_right,left_wrist,right_wrist}.mp4
wandb_upload.json  # W&B enabled run only
```

Physical runs default to W&B upload; read-only runs default to local-only.
`--no-wandb` disables upload explicitly. A normally completed run requires at
least 10 policy seconds and 30 action ticks. A requested 10-second-or-longer
physical run that fails or is interrupted is uploaded after its first recorded
policy command, even if it stopped early. Deliberate one-second tests and
read-only preflights are not uploaded.

Authentication uses the standard `WANDB_API_KEY` or existing W&B login in
`~/.netrc`. Upload failures do not change robot shutdown; the local run remains
retryable and `wandb_upload.json` records the result.

## Typical use

```bash
# Read-only: model and live sensors, no command, no W&B upload.
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
  --skill rotate_table_base \
  --variant ramen_ori_default \
  --interface enx58278cbf8be0 \
  --max-seconds 30

# Physical evaluation: W&B is enabled by default.
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
  --skill rotate_table_base \
  --variant ramen_ori_default \
  --interface enx58278cbf8be0 \
  --max-seconds 30 \
  --actuate --use-real-waist --use-real-hand
```

per-skill runbook / smoke procedure は `docs/handoff/` 配下の skill 別 handoff MD
(現行は `realmachine_rotate_table_base_eval.md`) を参照。
