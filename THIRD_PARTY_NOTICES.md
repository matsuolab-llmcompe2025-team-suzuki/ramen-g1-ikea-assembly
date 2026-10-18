# Third-party notices

This repository is licensed under the Apache License 2.0 (see `LICENSE`). It also contains, depends on, or is meant to
be used with the third-party works below, each under its own license.

## Included in this repository

| path | what | license |
|---|---|---|
| `inference/orin/ros2_ws/src/g1_description/meshes/`, `inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1/g1_29dof_mode_15_with_dex1_1.urdf` | Unitree G1 URDF and meshes from [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros) | BSD-3-Clause, Unitree Robotics. Source and license text in `inference/orin/ros2_ws/src/g1_description/licenses/` |
| `model/ramen_ori/assets/urdf/g1_29dof_mode_15_with_dex1_1.urdf` | The same Unitree G1 URDF (identical copy) | BSD-3-Clause, Unitree Robotics. Source and license text in the same directory |
| `inference/desktop/perception/assets/organizer_ik/g1_29dof_with_hand.urdf` | Unitree G1 URDF with hands, the kinematics used by the competition organizer's IK | BSD-3-Clause, Unitree Robotics. License text in the same directory |
| `model/subtask_policy_training/gr00t/assets/dex1_g1_synergy.json` | Open and closed joint values of the G1 7-joint hand, estimated from [`nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1) (revision `0d7bdd0`), used to map the Dex1-1 gripper onto GR00T's G1 hand | The source dataset is CC BY 4.0, NVIDIA |

## Not included

- **The competition organizer's interface package** (`inference/desktop/boundary/`). It has no license, so it is not
  distributed here. It is needed only for the competition-venue path (`--action-sink boundary`), which this repository
  does not support. To use that path anyway, copy `boundary/` from the organizer's public repository
  [`iacevaltest/iros_g1_orin_package`](https://github.com/iacevaltest/iros_g1_orin_package) at commit `497f3ab` to
  `inference/desktop/boundary/`.

## Packages installed by the environments

The pixi environments install these packages; they are not part of this repository. Their exact versions are in the
`pixi.lock` files.

| package | license | used for |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** | YOLO-OBB training (`model/yolo_obb/`) and YOLO-OBB inference in the `runtime` environment |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | GR00T and other policies (`model/subtask_policy_training/`, `inference/desktop/`) |
| [LeRobot fork](https://github.com/matsuolab-llmcompe2025-team-suzuki/lerobot) (`ramen` branch) | Apache-2.0 | RAMEN-Ori training (`model/ramen_ori/`) |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | BSD-3-Clause | Talking to the robot (`runtime` environment) |
| [lingbot-vision](https://github.com/robbyant/lingbot-vision) | Apache-2.0 | RAMEN-Ori's image encoder |

All other packages are under their own licenses.

## Models and data you download yourself

The code loads these from their publishers; we do not redistribute them. Follow each publisher's license.

| item | used for | license |
|---|---|---|
| [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) | Base model of the GR00T policies | See the publisher's terms |
| [`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | The vision-language backbone of GR00T N1.7 | NVIDIA Open Model License (access must be requested) |
| [`Qwen/Qwen3-VL-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) (and the 8B / 2B variants) | The VLM of the hybrid `pick_table_leg` | Apache-2.0 |
| [`robbyant/lingbot-vision-vit-base`](https://huggingface.co/robbyant/lingbot-vision-vit-base) | RAMEN-Ori's image encoder | Apache-2.0 |
| [`BitRobot/G1_WBT_Dex1_Building-Children-Table`](https://huggingface.co/datasets/BitRobot/G1_WBT_Dex1_Building-Children-Table), [`BitRobot/2026-humanoid-ikea-assembly-challenge`](https://huggingface.co/datasets/BitRobot/2026-humanoid-ikea-assembly-challenge) | Training data | CC BY 4.0, BitRobot |
| [Places365](http://places2.csail.mit.edu/) | Backgrounds for synthetic YOLO-OBB training images (`data/yolo_obb/augmentation/scripts/`) | Non-commercial research and educational use only; see its [terms](http://places2.csail.mit.edu/download.html) |

## Models and data we published

Our trained models and datasets are on Hugging Face, each with its license in its card:

- [RAMEN G1 IKEA Assembly — Models](https://huggingface.co/collections/Team-RAMEN/ramen-g1-ikea-assembly-models-6ac8dcb33fde8c28691c127e):
  the RAMEN-Ori policies (Apache-2.0; they contain the LingBot ViT-B/16 weights, Apache-2.0) and the YOLO-OBB detector
  (AGPL-3.0, because it was fine-tuned from Ultralytics' pretrained weights).
- [RAMEN G1 IKEA Assembly — Datasets](https://huggingface.co/collections/Team-RAMEN/ramen-g1-ikea-assembly-datasets-6ac8dcfc0ad04368b87c6bf1):
  CC BY 4.0, adapted from the BitRobot datasets above.
