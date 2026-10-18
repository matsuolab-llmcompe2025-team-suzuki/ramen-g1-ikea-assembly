# RAMEN G1 IKEA Assembly

Team RAMEN's code for the **IROS 2026 Humanoid IKEA Assembly Challenge**: a Unitree G1 humanoid with Dex1-1 grippers
assembles an IKEA UTTER children's table (pick a leg, insert it, tighten it, turn the table top, repeat, then flip the
table over).

This repository contains the whole stack we used:

- **Inference** for the real robot: perception (YOLO-OBB on the head camera), a stage / skill state machine, and the
  learned policies that drive the arms and grippers through the Unitree SDK.
- **Training** of the policies: RAMEN-Ori (our own vision-language-action-style policy), GR00T / ACT / Diffusion Policy
  subtask policies, and the YOLO-OBB detector.
- **Data tools** that turn the official BitRobot dataset into per-skill training sets, and the annotation and curation
  tools behind them.

## Scope

- **Supported: running on your own G1 setup**, with our desktop PC talking to the robot through the Unitree SDK
  (`--action-sink sdk`, the default) and the robot's Orin publishing the cameras.
- **Not supported: the competition-venue path** (`--action-sink boundary`, through the organizer's whole-body
  controller). The organizer's interface package is not included (it has no license); see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The venue-specific code is still in the tree, and selecting it
  stops with a message that says where to get the package.
- Code comments and most documents are in Japanese. Numbers such as `#141` or `Issue #188` refer to our internal
  issue tracker.

## Models and datasets

Everything we trained and the data we built are on Hugging Face:

- [**RAMEN G1 IKEA Assembly — Models**](https://huggingface.co/collections/Team-RAMEN/ramen-g1-ikea-assembly-models-6ac8dcb33fde8c28691c127e)
- [**RAMEN G1 IKEA Assembly — Datasets**](https://huggingface.co/collections/Team-RAMEN/ramen-g1-ikea-assembly-datasets-6ac8dcfc0ad04368b87c6bf1)

The inference config (`inference/desktop/lower_policy/configs/policy_config.yaml`, `default_variant_by_skill`) uses
these public models by default, pinned to a commit. They are downloaded from Hugging Face on first use.

| skill | default policy | model |
|---|---|---|
| `rotate_table_base` | RAMEN-Ori | [`ramen-g1-ikea-assembly-ramen-ori-rotate-table-base`](https://huggingface.co/Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base) |
| `pick_table_leg` | RAMEN-Ori | [`ramen-g1-ikea-assembly-ramen-ori-pick-table-leg`](https://huggingface.co/Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-pick-table-leg) |
| `insert_table_leg` | RAMEN-Ori | [`ramen-g1-ikea-assembly-ramen-ori-insert-table-leg`](https://huggingface.co/Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-insert-table-leg) |
| `rotate_leg_to_tighten` | GR00T N1.7 fine-tune | [`IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_200k`](https://huggingface.co/Team-RAMEN/IROS2026_RAMEN_takada_rotate_leg_to_tighten_optimal_gr00t_200k) |
| `flip_table` | RAMEN-Ori (not yet validated on the robot) | [`ramen-g1-ikea-assembly-ramen-ori-flip-table`](https://huggingface.co/Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-flip-table) |
| (perception) | YOLO-OBB | [`ramen-g1-ikea-assembly-yolo-obb`](https://huggingface.co/Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb) |

The GR00T policy also loads NVIDIA's base model [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B).
Its vision-language backbone [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is gated,
so you may need to request access and set `HF_TOKEN` to a token that has it. Some alternative policies listed in the
config are not public.

## Repository layout

| path | what |
|---|---|
| `inference/desktop/` | The robot-side runtime on the desktop PC: `entrypoint.py` (start here), `orchestrator.py`, `perception/`, `skill_planner/`, `lower_policy/` (skills, policies, actuators, configs), `pick_leg_hybrid/` |
| `inference/orin/` | ROS 2 Humble packages for the G1's onboard Orin: cameras, hand and SDK bridge, robot description |
| `model/ramen_ori/` | RAMEN-Ori: model, training (Hydra configs under `configs/`) |
| `model/subtask_policy_training/` | GR00T / ACT / Diffusion Policy training on the subtask datasets (LeRobot) |
| `model/yolo_obb/` | YOLO-OBB training and prediction |
| `data/bitrobot_lerobot_subtask_datasets/` | Cuts the BitRobot dataset into skills, builds the curated datasets and the pre-rendered image caches |
| `data/curation_tool/` | The per-segment curation UI and its loader |
| `data/annotation_tool/` | Label Studio setup and scripts for the YOLO-OBB annotations |
| `data/yolo_obb/` | Builds the YOLO-OBB training set (including the synthetic images) |
| `evaluate/model_evaluation/` | Single-skill runner for the real robot (`runners/run_skill.py`) and offline analysis tools |
| `evaluate/pick_leg_hybrid/` | Evaluation of the hybrid pick (VLM + policy + IK) |
| `docs/` | Dataset notes, the RAMEN-Ori design, G1 hardware notes and the full-run guide (Japanese) |

## Setup

Requirements: Ubuntu 22.04 or 24.04 (the environments are defined for `linux-64` only), an NVIDIA GPU for the policies,
[pixi](https://pixi.sh/), and git.

GPU memory: 9.1 GiB with `--gpu-models 2` and the default models (measured on an RTX 5090 after loading and one
warm-up step: YOLO-OBB 0.9 GiB, each RAMEN-Ori 1.0 GiB, GR00T 7.2 GiB), so 12 GB or more is recommended. The unit
tests need no GPU.

```bash
# 1. The Unitree SDK (installed editable into the runtime environment), at the commit we used on the robot
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git third_party/unitree_sdk2_python
git -C third_party/unitree_sdk2_python checkout 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5

# 2. Environments
pixi install                 # default: light, for unit tests (Python 3.12)
pixi install -e runtime      # runtime: Unitree SDK, CycloneDDS, torch, ultralytics, for the robot (Python 3.10)
(cd inference/desktop && pixi install)   # policy workers: GR00T / ACT / Diffusion Policy (Python 3.12 + LeRobot)
```

Training uses separate pixi workspaces: `model/ramen_ori/`, `model/yolo_obb/` and `model/subtask_policy_training/`
(see their READMEs).

Optional `.env` at the repository root (git-ignored): `HF_TOKEN` (for gated or private models) and `WANDB_API_KEY`
(training logs). The data tools and the YOLO-OBB scripts read it themselves; for everything else (running on the
robot, RAMEN-Ori and GR00T / ACT / Diffusion Policy training), load it into the shell first with
`set -a && . ./.env && set +a`.

## Running on your robot

Prepare a harness, an E-stop operator and clear space. The robot must be standing in its walking mode, and the Orin
must be publishing the cameras (`inference/orin/README.md`). Run each model once without `--actuate` first.

```bash
# The whole task, Stage 0 (walk to the table) to Stage 5 (flip the table)
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --phase3-full --actuate --use-real-waist --use-real-hand \
    --setup-hold-sec 3 --gpu-models 2 --log /tmp/run.jsonl

# One stage at a time (N = 0..5)
pixi run -e runtime python -m inference.desktop.entrypoint \
    --interface eth0 --stage N --actuate --use-real-waist --use-real-hand \
    --setup-hold-sec 3 --gpu-models 1

# A single skill with a chosen policy
pixi run -e runtime python -m evaluate.model_evaluation.runners.run_skill \
    --interface eth0 --skill insert_table_leg --variant insert_table_leg_ramen_ori_venue_all6_20k
```

`--interface` is the network interface that reaches the G1. The full procedure (resuming from a stage, the safety
stop and return, what to check) is in [`docs/handoff/realmachine_full_run_eval.md`](docs/handoff/realmachine_full_run_eval.md).
Other options: `python -m inference.desktop.entrypoint --help`. Policy choices go in `policy_config.yaml`
(`--policy-variant-<skill>` overrides one skill), and all skill parameters in
`inference/desktop/lower_policy/configs/skill_config.yaml`.

## Tests

```bash
pixi run -e runtime test-desktop-runtime   # all inference tests (what CI runs; no GPU or robot needed)
pixi run test-lower-policy                 # lower_policy only, in the light default environment
```

## Training

- RAMEN-Ori: [`model/ramen_ori/README.md`](model/ramen_ori/README.md). The training configs assume our training
  server's `/nvme` layout; see that README for how to point them elsewhere.
- YOLO-OBB: [`model/yolo_obb/README.md`](model/yolo_obb/README.md) and [`data/yolo_obb/README.md`](data/yolo_obb/README.md)
- GR00T / ACT / Diffusion Policy: [`model/subtask_policy_training/README.md`](model/subtask_policy_training/README.md)
- Datasets: [`data/bitrobot_lerobot_subtask_datasets/README.md`](data/bitrobot_lerobot_subtask_datasets/README.md),
  [`data/curation_tool/README.md`](data/curation_tool/README.md), [`data/annotation_tool/README.md`](data/annotation_tool/README.md)

## License

Apache License 2.0 ([`LICENSE`](LICENSE)). Third-party files, dependencies and the models and data used by the code
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). In particular, the YOLO-OBB parts use
[Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, and the published YOLO-OBB model is
AGPL-3.0. Our datasets are CC BY 4.0, adapted from BitRobot's datasets.

## Acknowledgements

The training data comes from BitRobot's
[`G1_WBT_Dex1_Building-Children-Table`](https://huggingface.co/datasets/BitRobot/G1_WBT_Dex1_Building-Children-Table)
and [`2026-humanoid-ikea-assembly-challenge`](https://huggingface.co/datasets/BitRobot/2026-humanoid-ikea-assembly-challenge)
datasets (CC BY 4.0). We thank the organizers of the IROS 2026 Humanoid IKEA Assembly Challenge.
