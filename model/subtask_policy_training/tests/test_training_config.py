from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_default_configs_are_independent_of_working_directory() -> None:
    expected = (ROOT / "configs" / "subtask_training.json").resolve()
    script_names = (
        "download_source_mcaps.py",
        "materialize_lerobot_training_view.py",
        "resolve_training_config.py",
        "upload_lerobot_dataset.py",
        "write_groot_modality_json.py",
    )
    for index, script_name in enumerate(script_names):
        module = load_module(
            f"default_config_path_{index}",
            ROOT / "scripts" / script_name,
        )
        assert module.DEFAULT_CONFIG == expected
        assert module.DEFAULT_CONFIG.is_file()


def test_vast_flow_matching_config_pins_a_dataset_revision_not_a_cache_path() -> None:
    config = (ROOT / "deployment" / "vast" / "flow_matching_bc.conf").read_text(
        encoding="utf-8"
    )
    assert 'DATASET_REVISION="10a6ec05f9993b8d59faad2957e47153b0f15f37"' in config
    assert "SOURCE_DATASET_ROOT=" not in config
    assert ".cache/huggingface" not in config


def test_act_diagnostic_uses_strict_processor_contract(tmp_path: Path) -> None:
    module = load_module(
        "diagnose_act_processor_contract",
        ROOT / "scripts" / "diagnose_act_policy_outputs.py",
    )
    state_path = tmp_path / "stats.safetensors"
    state_path.write_bytes(b"state")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": state_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved, eps = module.resolve_processor_state(
        tmp_path,
        "policy_preprocessor.json",
        "normalizer_processor",
    )
    assert resolved == state_path.resolve()
    assert eps == pytest.approx(1e-8)
    with pytest.raises(KeyError, match="statistics are missing"):
        module.normalize_feature({}, "observation.state", module.torch.zeros(19), eps)

    source = (ROOT / "scripts" / "diagnose_act_policy_outputs.py").read_text(encoding="utf-8")
    assert 'load_kwargs: dict[str, Any] = {"strict": True}' in source
    assert "/home/" not in source


def test_policy_uploader_validates_processor_state_files(tmp_path: Path) -> None:
    module = load_module(
        "upload_policy_processor_contract",
        ROOT / "scripts" / "upload_policy.py",
    )
    for name in ("config.json", "train_config.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": "normalizer.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "device_processor", "config": {}}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="normalizer.safetensors"):
        module.validate_model_dir(tmp_path)
    (tmp_path / "normalizer.safetensors").write_bytes(b"state")
    module.validate_model_dir(tmp_path)

    processor = json.loads((tmp_path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    processor["steps"][0]["state_file"] = "../outside.safetensors"
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(processor), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes model directory"):
        module.validate_model_dir(tmp_path)


def test_resolve_default_training_config(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in [
        "SUBTASK",
        "POLICY_TYPE",
        "CONTROL_SCOPE",
        "DATASET_REPO_ID",
        "DATASET_REVISION",
        "GROOT_DATASET_REPO_ID",
        "POLICY_REPO_ID",
        "OUTPUT_DIR",
        "JOB_NAME",
        "DEVICE",
        "WANDB_ENABLE",
        "WANDB_PROJECT",
        "PUSH_TO_HUB",
        "UPLOAD_AFTER_TRAIN",
        "PRIVATE",
        "MATERIALIZE_TRAINING_VIEW",
        "TRAINING_VIEW_ROOT",
        "TRAINING_VIEW_FORCE",
        "POLICY_INPUT_FEATURES",
        "POLICY_OUTPUT_FEATURES",
        "GROOT_BASE_MODEL_PATH",
        "GROOT_BASE_MODEL_REVISION",
        "GROOT_EMBODIMENT_TAG",
        "GROOT_CHUNK_SIZE",
        "GROOT_N_ACTION_STEPS",
        "GROOT_USE_RELATIVE_ACTIONS",
        "GROOT_RELATIVE_EXCLUDE_JOINTS",
        "GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR",
        "GROOT_PROCESSOR_OVERLAY_ROOT",
        "GROOT_USE_BF16",
    ]:
        monkeypatch.delenv(name, raising=False)

    values = module.resolve(config)

    assert values["SUBTASK"] == "pick_leg"
    assert values["POLICY_TYPE"] == "act"
    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
    assert values["DATASET_REVISION"] == ""
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
    assert values["POLICY_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_act_1"
    assert values["OUTPUT_DIR"] == "outputs/train/act_pick_leg"
    assert values["WANDB_PROJECT"] == "iros2026-ramen-pick-leg"
    assert values["PUSH_TO_HUB"] == "false"
    assert values["UPLOAD_AFTER_TRAIN"] == "true"
    assert values["CONTROL_SCOPE"] == "upper_body"
    assert values["STATE_DIM"] == "19"
    assert values["ACTION_DIM"] == "19"
    assert values["CAMERAS"] == "head_left,left_wrist,right_wrist"
    assert values["MATERIALIZE_TRAINING_VIEW"] == "true"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/act_pick_leg"
    assert values["TRAINING_VIEW_FORCE"] == "false"
    assert values["TRAIN_IMAGE_TRANSFORMS_ENABLE"] == "true"
    assert values["TRAIN_BATCH_SIZE"] == "8"
    assert values["TRAIN_STEPS"] == "300000"
    assert values["TRAIN_EVAL_STEPS"] == "10000"
    assert values["TRAIN_MAX_EVAL_SAMPLES"] == "512"
    assert values["ACT_CHUNK_SIZE"] == "100"
    assert values["ACT_N_ACTION_STEPS"] == "10"
    assert json.loads(values["SOURCE_CAMERA_MAP"]) == {
        "observation.images.head_left": "observation.images.cam_0",
        "observation.images.left_wrist": "observation.images.cam_2",
        "observation.images.right_wrist": "observation.images.cam_3",
    }
    assert values["POLICY_VIEW_LAYOUT"] == "robot_q_upper_body_19d"
    input_features = json.loads(values["POLICY_INPUT_FEATURES"])
    output_features = json.loads(values["POLICY_OUTPUT_FEATURES"])
    assert list(input_features) == [
        "observation.state",
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert "observation.images.head_right" not in input_features
    assert input_features["observation.images.head_left"] == {
        "type": "VISUAL",
        "shape": [3, 480, 640],
    }
    assert input_features["observation.state"] == {"type": "STATE", "shape": [19]}
    assert output_features == {"action": {"type": "ACTION", "shape": [19]}}
    assert "GROOT_BASE_MODEL_PATH" not in values


def test_resolve_allows_subtask_override(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "tighten")
    monkeypatch.setenv("POLICY_TYPE", "diffusion")

    values = module.resolve(config)

    assert values["TASK"] == "rotate leg to tighten"
    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_1"
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_1"
    assert values["POLICY_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_diffusion_1"
    assert values["WANDB_PROJECT"] == "iros2026-ramen-tighten"


def test_resolve_flow_matching_defaults(monkeypatch) -> None:
    module = load_module("resolve_flow_matching", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("POLICY_TYPE", "flow_matching")

    values = module.resolve(config)

    assert values["CONTROL_SCOPE"] == "upper_body"
    assert values["STATE_DIM"] == "19"
    assert values["ACTION_DIM"] == "19"
    assert values["FLOW_ACTION_HORIZON"] == "24"
    assert values["FLOW_N_ACTION_STEPS"] == "6"
    assert values["FLOW_INFERENCE_STEPS"] == "10"
    assert values["FLOW_MODEL_DIM"] == "384"
    assert values["FLOW_TRANSFORMER_LAYERS"] == "6"
    assert values["FLOW_TRANSFORMER_HEADS"] == "8"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/flow_matching_flip_table"
    assert values["POLICY_REPO_ID"].endswith("_flip_table_flow_matching_1")

    monkeypatch.setenv("FLOW_ACTION_HORIZON", "4")
    monkeypatch.setenv("FLOW_N_ACTION_STEPS", "5")
    with pytest.raises(ValueError, match="cannot exceed"):
        module.resolve(config)


def test_resolve_diffusion_defaults(monkeypatch) -> None:
    module = load_module("resolve_diffusion", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("SUBTASK", "rotate_table_base")
    monkeypatch.setenv("POLICY_TYPE", "diffusion")

    values = module.resolve(config)

    assert values["CONTROL_SCOPE"] == "upper_body"
    assert values["STATE_DIM"] == "19"
    assert values["ACTION_DIM"] == "19"
    assert values["DIFFUSION_HORIZON"] == "32"
    assert values["DIFFUSION_N_ACTION_STEPS"] == "8"
    assert values["DIFFUSION_N_OBS_STEPS"] == "2"
    assert values["DIFFUSION_NUM_INFERENCE_STEPS"] == "10"
    # 末尾 drop は horizon から導出 (32 - 8 - 2 + 1)。
    assert values["DIFFUSION_DROP_N_LAST_FRAMES"] == "23"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/diffusion_rotate_table_base"


def test_resolve_diffusion_rejects_invalid_temporal_contract(monkeypatch) -> None:
    module = load_module("resolve_diffusion_invalid", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("POLICY_TYPE", "diffusion")

    # U-Net の downsampling factor (8) の倍数でない horizon は LeRobot 側で落ちるので先に弾く。
    monkeypatch.setenv("DIFFUSION_HORIZON", "20")
    with pytest.raises(ValueError, match="multiple of 8"):
        module.resolve(config)

    # n_action_steps + n_obs_steps - 1 が horizon を超える組合せも弾く。
    monkeypatch.setenv("DIFFUSION_HORIZON", "8")
    monkeypatch.setenv("DIFFUSION_N_ACTION_STEPS", "8")
    monkeypatch.setenv("DIFFUSION_N_OBS_STEPS", "2")
    with pytest.raises(ValueError, match="cannot exceed DIFFUSION_HORIZON"):
        module.resolve(config)


def test_resolve_diffusion_drop_n_last_frames_follows_overrides(monkeypatch) -> None:
    module = load_module("resolve_diffusion_drop", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("POLICY_TYPE", "diffusion")
    monkeypatch.setenv("DIFFUSION_HORIZON", "16")

    assert module.resolve(config)["DIFFUSION_DROP_N_LAST_FRAMES"] == "7"

    monkeypatch.setenv("DIFFUSION_DROP_N_LAST_FRAMES", "0")
    assert module.resolve(config)["DIFFUSION_DROP_N_LAST_FRAMES"] == "0"


def test_resolve_allows_pinned_dataset_revision(monkeypatch) -> None:
    module = load_module("resolve_training_config_revision", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("DATASET_REVISION", "10a6ec05f9993b8d59faad2957e47153b0f15f37")

    values = module.resolve(config)

    assert values["DATASET_REVISION"] == "10a6ec05f9993b8d59faad2957e47153b0f15f37"


def test_resolve_rejects_incomplete_camera_and_temporal_contracts(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    config["cameras"] = ["head_left", "left_wrist"]
    with pytest.raises(ValueError, match="exactly"):
        module.resolve(config)

    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("ACT_CHUNK_SIZE", "4")
    monkeypatch.setenv("ACT_N_ACTION_STEPS", "5")
    with pytest.raises(ValueError, match="cannot exceed"):
        module.resolve(config)


def test_resolve_groot_n17_defaults(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("POLICY_TYPE", "groot")

    values = module.resolve(config)

    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1"
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1"
    assert values["POLICY_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_1"
    assert values["WANDB_PROJECT"] == "iros2026-ramen-flip-table"
    assert values["CONTROL_SCOPE"] == "upper_body_relative_eef"
    assert values["STATE_DIM"] == "49"
    assert values["ACTION_DIM"] == "53"
    assert values["ACTION_SEMANTICS"] == "real_g1_relative_eef_relative_arm_absolute_hand_waist"
    assert values["POLICY_VIEW_LAYOUT"] == "real_g1_relative_eef_relative_joints"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/groot_flip_table"
    assert json.loads(values["POLICY_INPUT_FEATURES"])["observation.state"] == {
        "type": "STATE",
        "shape": [49],
    }
    assert json.loads(values["POLICY_OUTPUT_FEATURES"]) == {
        "action": {"type": "ACTION", "shape": [53]}
    }
    assert values["GROOT_BASE_MODEL_PATH"] == "nvidia/GR00T-N1.7-3B"
    assert values["GROOT_BASE_MODEL_REVISION"] == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert values["GROOT_EMBODIMENT_TAG"] == "real_g1_relative_eef_relative_joints"
    assert values["GROOT_CHUNK_SIZE"] == "16"
    assert values["GROOT_N_ACTION_STEPS"] == "16"
    assert values["GROOT_USE_RELATIVE_ACTIONS"] == "true"
    assert values["GROOT_RELATIVE_EXCLUDE_JOINTS"] == '["hand","waist","base_height","navigate"]'
    assert values["GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR"] == "true"
    assert values["GROOT_PROCESSOR_OVERLAY_ROOT"] == (
        "outputs/groot_base_overlays/real_g1_relative_eef_3cam_dex1_v2"
    )
    assert values["GROOT_USE_BF16"] == "true"
    assert values["GROOT_IMAGE_TRANSFORMS_ENABLE"] == "true"
    assert values["GROOT_BATCH_SIZE"] == "64"
    assert values["GROOT_STEPS"] == "20000"


def test_resolve_groot_n17_allows_temporal_overrides_but_not_slot_semantics(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_CHUNK_SIZE", "40")
    monkeypatch.setenv("GROOT_N_ACTION_STEPS", "25")

    values = module.resolve(config)

    assert values["GROOT_CHUNK_SIZE"] == "40"
    assert values["GROOT_N_ACTION_STEPS"] == "25"
    assert values["GROOT_USE_RELATIVE_ACTIONS"] == "true"
    assert values["GROOT_RELATIVE_EXCLUDE_JOINTS"] == '["hand","waist","base_height","navigate"]'

    monkeypatch.setenv("GROOT_RELATIVE_EXCLUDE_JOINTS", '["left_gripper","right_gripper"]')
    with pytest.raises(ValueError, match="must be exactly"):
        module.resolve(config)


def test_resolve_rejects_unknown_policy_type(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("POLICY_TYPE", "unknown")

    with pytest.raises(ValueError, match="POLICY_TYPE must be one of"):
        module.resolve(config)


def test_resolve_groot_rejects_disabling_relative_eef(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_USE_RELATIVE_ACTIONS", "false")

    try:
        module.resolve(config)
    except ValueError as exc:
        assert "chunk's current observation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("disabling native relative EEF must fail")


def test_resolve_groot_rejects_disabling_native_eef_processor(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR", "false")

    with pytest.raises(ValueError, match="must stay true"):
        module.resolve(config)


def test_resolve_allows_wandb_project_override(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("WANDB_PROJECT", "custom-project")

    values = module.resolve(config)

    assert values["WANDB_PROJECT"] == "custom-project"


# ============================================================================
# Issue #122: subtask_dataset_repos (list) 対応、DATASET_REPO_IDS export
# ============================================================================


def test_resolve_single_repo_backward_compat_populates_repo_ids(monkeypatch) -> None:
    """既存 single-repo subtask (pick_leg 等) は DATASET_REPO_IDS = [DATASET_REPO_ID] を保証。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in ["SUBTASK", "POLICY_TYPE", "DATASET_REPO_ID", "DATASET_REPO_IDS"]:
        monkeypatch.delenv(name, raising=False)

    values = module.resolve(config)

    single_repo = "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
    assert values["DATASET_REPO_ID"] == single_repo
    assert json.loads(values["DATASET_REPO_IDS"]) == [single_repo]


def test_resolve_multi_repo_subtask_task5_rotate_table_base(monkeypatch) -> None:
    """rotate_table_base (task 5) は Issue #122 で 8 chunks を merged 1 repo に統合。
    HF pull 高速化 (~1h → ~2min) + deterministic pin の revision 数減で運用単純化。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in ["DATASET_REPO_ID", "DATASET_REPO_IDS"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUBTASK", "rotate_table_base")

    values = module.resolve(config)

    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    assert repo_ids == ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base"]
    # DATASET_REPO_ID は list の 1 つ目 (downstream single-repo 互換)
    assert values["DATASET_REPO_ID"] == repo_ids[0]


def test_resolve_multi_repo_subtask_task7_move_table_base(monkeypatch) -> None:
    """move_table_base (task 7) は 1 chunk の list (new subtask, Issue #122 追加)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in ["DATASET_REPO_ID", "DATASET_REPO_IDS"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUBTASK", "move_table_base")

    values = module.resolve(config)

    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    assert repo_ids == ["Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-move-table-base"]
    assert values["DATASET_REPO_ID"] == repo_ids[0]
    assert values["TASK"] == "move table base"


def test_resolve_combined_task5_7_yields_task7_plus_merged(monkeypatch) -> None:
    """combined_task5_7 は Issue #122 で 9 chunks → 2 repos (task 7 + task 5 merged) に統合。
    prepare_multi_repo_source.py の入力サイズが 9→2 に減り merge も短縮。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in ["DATASET_REPO_ID", "DATASET_REPO_IDS"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")

    values = module.resolve(config)

    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    assert repo_ids == [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-move-table-base",
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base",
    ]


def test_resolve_env_override_dataset_repo_ids_wins(monkeypatch) -> None:
    """DATASET_REPO_IDS env で明示 override 可能 (test/pilot での差替用)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "rotate_table_base")
    custom = json.dumps(["custom/repo1", "custom/repo2"])
    monkeypatch.setenv("DATASET_REPO_IDS", custom)

    values = module.resolve(config)

    assert values["DATASET_REPO_IDS"] == custom


def test_resolve_rejects_invalid_dataset_repos_type() -> None:
    """dataset_repos が list of str でなければ error (schema 保護)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    # Inject invalid type into a subtask
    config["subtasks"]["rotate_table_base"]["dataset_repos"] = "not_a_list"
    os.environ.pop("DATASET_REPO_ID", None)
    os.environ.pop("DATASET_REPO_IDS", None)
    os.environ["SUBTASK"] = "rotate_table_base"
    try:
        with pytest.raises(ValueError, match="must be a list of strings"):
            module.resolve(config)
    finally:
        os.environ.pop("SUBTASK", None)


# ============================================================================
# Issue #122 Phase C-D: materialize --allow-multi-task / --split-json wiring
# ============================================================================


def test_materialize_load_external_split_json(tmp_path: Path) -> None:
    """外部 split JSON を load できる (RAMEN-Ori export と compat)。"""
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    payload = {
        "schema_version": "team_ramen_grouped_episode_split_v1",
        "seed": 42,
        "splits": {
            "train": {"episode_indices": [0, 1, 2, 5, 6, 7]},
            "validation": {"episode_indices": [3]},
            "test": {"episode_indices": [4]},
        },
    }
    p = tmp_path / "split.json"
    p.write_text(json.dumps(payload))
    loaded = module.load_external_split_json(p, total_episodes=8)
    assert loaded == payload


def test_materialize_load_external_split_json_rejects_missing_episodes(tmp_path: Path) -> None:
    """全 episode が含まれていない split JSON は error。"""
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    payload = {
        "splits": {
            "train": {"episode_indices": [0, 1]},
            "validation": {"episode_indices": [2]},
            "test": {"episode_indices": [3]},
        },
    }
    p = tmp_path / "split.json"
    p.write_text(json.dumps(payload))
    # total_episodes=8 なのに 4 しかない
    with pytest.raises(ValueError, match="cover all"):
        module.load_external_split_json(p, total_episodes=8)


def test_materialize_load_external_split_json_rejects_missing_split_names(tmp_path: Path) -> None:
    """train/validation/test の全 3 split が必要。"""
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    payload = {"splits": {"train": {"episode_indices": [0, 1, 2]}}}
    p = tmp_path / "split.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="train.*validation.*test"):
        module.load_external_split_json(p, total_episodes=3)


def test_train_sh_wires_multi_repo_and_split_json() -> None:
    """train_lerobot.sh が Issue #122 の Phase D 拡張を含むこと (multi-repo pre-merge + split-json 経路)。"""
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()
    # multi-repo pre-merge branch
    assert "prepare_multi_repo_source.py" in script
    assert "DATASET_REPO_IDS" in script
    # materialize side wiring
    assert "--allow-multi-task" in script
    assert "--split-json" in script
    # dry-run 出力にも含まれる
    assert "prepare_multi_repo:" in script


def test_train_sh_anchors_paths_to_repo_root() -> None:
    """Issue #122: MERGED_SOURCE_DIR / PYTHONPATH が repo-root anchored である事。

    - MERGED_SOURCE_DIR は RAMEN-Ori config (`outputs/merged_sources/...` repo-root
      relative) と同じ物理 dir を指す必要がある (cache 共有・重複排除)。
    - PYTHONPATH は `python -m model.subtask_policy_training....` を cwd=ROOT_DIR
      から起動する時に必要 (sys.path[0]=ROOT_DIR には model/ package が無いため)。
    """
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()
    # REPO_ROOT 導出行 (ROOT_DIR の 2 階層上)
    assert 'REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"' in script
    # MERGED_SOURCE_DIR default が ${REPO_ROOT}/outputs/... 形式
    assert '${REPO_ROOT}/outputs/merged_sources/${SUBTASK}_from_hf' in script
    assert '${REPO_ROOT}/outputs/merged_sources/${SUBTASK}' in script
    # 旧 relative default (`outputs/merged_sources/...` 単体) は無くなっている
    assert 'MERGED_SOURCE_DIR:-outputs/merged_sources' not in script
    # PYTHONPATH に REPO_ROOT が prefix される (既存 PYTHONPATH は保持)
    assert 'export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"' in script


def test_train_sh_wires_frame_cache_wrapper_and_precompute() -> None:
    """Issue #122: train_lerobot.sh が JPG frame cache pipeline を組み込むこと。

    - lerobot-train の代わりに lerobot_train_with_frame_cache wrapper を invoke
    - LEROBOT_FRAME_CACHE_ENABLE を env に export (default true)
    - materialize 後、resolve 前に precompute_frame_cache.py を呼ぶ (settings hash で reuse 判定)
    - Cache target = merged_source (multi-repo)、policy_type 越しに共有可能
    - mp4 auto drop は常時 ON (lerobot_frame_cache_patch の _check_cached_episodes_sufficient
      差替で video 不在でも train 成立、Issue #122)
    - dry-run 出力にも precompute command を含む
    """
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()
    # wrapper 呼び出し (lerobot-train CLI 直接ではなく wrapper module)
    assert "lerobot_train_with_frame_cache" in script
    # env var の export (default true、外部で false override 可能)
    assert 'LEROBOT_FRAME_CACHE_ENABLE="${LEROBOT_FRAME_CACHE_ENABLE:-true}"' in script
    # precompute step
    assert "precompute_frame_cache.py" in script
    # Cache target 選択: multi-repo なら merged_source、single-repo なら training view
    assert "frame_cache_target" in script
    assert "merged_source_root" in script
    # FRAME_CACHE_PRECOMPUTE で precompute step 自体を skip 可能
    assert "FRAME_CACHE_PRECOMPUTE" in script
    # KEEP_VIDEOS 系 env / flag は削除済 (Issue #122): mp4 drop 常時 ON
    assert "FRAME_CACHE_KEEP_VIDEOS" not in script
    assert "--keep-source-videos" not in script
    # dry-run 出力
    assert "precompute_frame_cache:" in script


def test_frame_cache_patch_and_wrapper_scripts_exist() -> None:
    """Issue #122 で追加した 2 script が存在し、意図した entry を持つこと。

    Issue #122 D-4: wrapper に obb_overlay_setup.setup_from_env() の呼び出しも
    追加された (GR00T 等 lerobot-train 経由 policy に対する overlay hook register)。
    """
    patch = ROOT / "scripts" / "lerobot_frame_cache_patch.py"
    wrapper = ROOT / "scripts" / "lerobot_train_with_frame_cache.py"
    assert patch.is_file()
    assert wrapper.is_file()
    patch_src = patch.read_text()
    wrapper_src = wrapper.read_text()
    # patch は decode_video_frames を monkey-patch する
    assert "decode_video_frames" in patch_src
    assert "def apply_patch" in patch_src
    # wrapper は apply_patch → obb_overlay_setup.setup_from_env → lerobot-train main を呼ぶ
    assert "apply_patch" in wrapper_src
    assert "obb_overlay_setup" in wrapper_src
    assert "setup_from_env" in wrapper_src
    assert "lerobot.scripts.lerobot_train" in wrapper_src


def test_prepare_multi_repo_source_script_exists_and_calls_merge() -> None:
    """prepare_multi_repo_source.py が Phase A の merge script を呼び出す前提を検証。"""
    src = (ROOT / "scripts" / "prepare_multi_repo_source.py").read_text()
    assert "merge_curated_lerobot_datasets.py" in src
    assert "--allow-duplicate-task-indices" in src
    assert "snapshot_download" in src


def test_train_wrapper_passes_policy_feature_overrides() -> None:
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()

    assert '--policy.input_features="$POLICY_INPUT_FEATURES"' in script
    assert '--policy.output_features="$POLICY_OUTPUT_FEATURES"' in script
    assert "materialize_lerobot_training_view.py" in script
    assert "prepare_groot_n17_real_g1_overlay.py" in script
    assert "--overlay-root \"$GROOT_LEROBOT_SOURCE_OVERLAY_ROOT\"" in script
    assert 'export PYTHONPATH="$GROOT_LEROBOT_SOURCE_OVERLAY_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert "patch_lerobot_groot_relative_eef.py --check" in script
    assert "restore_groot_base_model_path.py" in script
    assert '--dataset.root="$TRAINING_VIEW_ROOT"' in script
    assert "accelerate launch" in script
    assert '--num_processes="$NUM_GPUS"' in script
    assert '--mixed_precision="$ACCELERATE_MIXED_PRECISION"' in script
    assert "--fsdp_sharding_strategy=FULL_SHARD" in script
    assert "--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP" in script
    assert (
        "--fsdp_transformer_layer_cls_to_wrap="
        "Qwen3VLTextDecoderLayer,Qwen3VLVisionBlock,BasicTransformerBlock"
    ) in script
    assert "--fsdp_use_orig_params=true" in script
    assert '--policy.max_steps="$GROOT_STEPS"' in script
    assert '--wandb.mode="$WANDB_MODE"' in script


def test_two_gpu_slurm_batch_pins_dataset_and_validates_sdk_mapping() -> None:
    script = (
        ROOT / "deployment" / "slurm" / "train_groot_n17_coarse_insert_2gpu.sbatch"
    ).read_text()

    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --cpus-per-task=16" in script
    assert "#SBATCH --partition=debug" in script
    assert '"${SLURM_SUBMIT_DIR:-}"' in script
    assert "find_repository_root" in script
    assert 'container_image="${CONTAINER_IMAGE:-$host_repository_root/gr00t-lerobot.sif}"' in script
    assert 'exec "$singularity_bin" exec' in script
    assert "--nv" in script
    assert '--bind "$host_repository_root:$container_repository_root"' in script
    assert "--env RAMEN_IN_SINGULARITY=true" in script
    assert 'export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"' in script
    assert 'export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"' in script
    assert "wandb_api_key_in_container: set" in script
    assert 'export VENV_DIR="/opt/lerobot/.venv"' in script
    assert 'export NUM_GPUS="${NUM_GPUS:-2}"' in script
    assert 'ACCELERATE_DISTRIBUTED_STRATEGY="${ACCELERATE_DISTRIBUTED_STRATEGY:-fsdp}"' in script
    assert 'export SUBTASK="coarse_insert"' in script
    assert "338c3db7f106e5108f29df5bb564a983897c0eca" in script
    assert "verify_g1_sdk_joint_mapping.py" in script
    assert "memory_gb = props.total_memory / 10**9" in script
    assert "below NVIDIA's recommended 40 GB VRAM class" in script
    assert "requires 40 GiB+" not in script
    assert 'export GROOT_BATCH_SIZE="${GROOT_BATCH_SIZE:-4}"' in script
    assert 'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in script
    assert 'if [[ -n "${WANDB_PROJECT:-}" ]]; then' in script
    assert 'export WANDB_ENABLE="${WANDB_ENABLE:-$wandb_enable_default}"' in script
    assert 'export WANDB_MODE="offline"' in script
    assert "W&B credentials are not available in the job" in script
    assert 'export UPLOAD_AFTER_TRAIN="${UPLOAD_AFTER_TRAIN:-false}"' in script
    assert './scripts/train_groot_n17.sh "$@"' in script


def test_lerobot_training_view_materializer_maps_official_schema() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    table = pa.table(
        {
            "observation.state.ee_state": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2],
            "observation.state.robot_q_current": [[float(i) for i in range(36)]],
            "observation.state.hand_state": [[36.0, 37.0]],
            "action.ee_action": [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0] * 2],
            "action.robot_q_desired": [[float(i) for i in range(100, 136)]],
            "action.hand_cmd": [[136.0, 137.0]],
            "timestamp": [0.0],
            "frame_index": [0],
            "episode_index": [0],
            "index": [0],
            "task_index": [0],
        }
    )

    state_rows, action_rows = module.build_policy_vectors_from_table(
        table,
        config=config,
        policy_type="act",
    )
    assert state_rows == [[float(i) for i in range(19, 38)]]
    assert action_rows == [[float(i) for i in range(119, 138)]]

    state_rows, action_rows = module.build_policy_vectors_from_table(
        table,
        config=config,
        policy_type="groot",
    )
    assert len(state_rows[0]) == 49
    assert len(action_rows[0]) == 53
    mapping = load_module(
        "g1_full_body_mapping_schema",
        ROOT / "gr00t" / "g1_full_body_mapping.py",
    )
    assert state_rows[0][0:9] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert state_rows[0][18:25] == pytest.approx(
        mapping.dex1_to_hand(36.0, side="left", kind="state")
    )
    assert state_rows[0][25:32] == pytest.approx(
        mapping.dex1_to_hand(37.0, side="right", kind="state")
    )
    assert state_rows[0][32:39] == [22, 23, 24, 25, 26, 27, 28]
    assert state_rows[0][39:46] == [29, 30, 31, 32, 33, 34, 35]
    assert state_rows[0][46:49] == [19, 20, 21]
    assert action_rows[0][0:9] == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert action_rows[0][18:25] == pytest.approx(
        mapping.dex1_to_hand(136.0, side="left", kind="action")
    )
    assert action_rows[0][25:32] == pytest.approx(
        mapping.dex1_to_hand(137.0, side="right", kind="action")
    )
    assert action_rows[0][32:39] == [122, 123, 124, 125, 126, 127, 128]
    assert action_rows[0][39:46] == [129, 130, 131, 132, 133, 134, 135]
    assert action_rows[0][46:49] == [119, 120, 121]
    assert action_rows[0][49:53] == [0.0, 0.0, 0.0, 0.0]


def test_materializer_rejects_nonfinite_source_values() -> None:
    module = load_module(
        "materialize_lerobot_training_view_finite",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        module.as_float_list([0.0, float("nan")])


def test_grouped_split_is_disjoint_and_resolves_for_lerobot(tmp_path) -> None:
    materializer = load_module(
        "materialize_lerobot_training_view_split",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    resolver = load_module(
        "resolve_training_split",
        ROOT / "scripts" / "resolve_training_split.py",
    )
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    episode_path = tmp_path / "source" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    (tmp_path / "source" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 10}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(10)),
                "tasks": [["flip table"] for _ in range(10)],
                "source_episode_name": [f"recording_{index:02d}" for index in range(10)],
            }
        ),
        episode_path,
    )

    split = materializer.build_grouped_episode_split(tmp_path / "source", config=config, pq=pq)
    dataset_root = tmp_path / "view"
    split_path = dataset_root / resolver.SPLIT_PATH
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    resolved = resolver.resolve_split(dataset_root)

    episode_sets = {
        name: set(value["episode_indices"]) for name, value in split["splits"].items()
    }
    assert episode_sets["train"].isdisjoint(episode_sets["validation"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["validation"].isdisjoint(episode_sets["test"])
    selected = json.loads(resolved["DATASET_EPISODES_JSON"])
    assert selected[-resolved["VALIDATION_EPISODE_COUNT"] :] == sorted(episode_sets["validation"])
    assert math.ceil(len(selected) * float(resolved["DATASET_EVAL_SPLIT"])) == len(
        episode_sets["validation"]
    )


def test_grouped_split_accepts_curated_source_episode_indices(tmp_path) -> None:
    materializer = load_module(
        "materialize_lerobot_training_view_curated_split",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source = tmp_path / "source"
    episode_path = source / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    (source / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 10}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(10)),
                "tasks": [["insert table leg to table base"] for _ in range(10)],
                "source_episode_index": [index // 2 for index in range(10)],
            }
        ),
        episode_path,
    )

    split = materializer.build_grouped_episode_split(source, config=config, pq=pq)

    assert split["group_key"] == "source_episode_index"
    source_sets = {
        name: set(value["source_episode_names"]) for name, value in split["splits"].items()
    }
    assert source_sets["train"].isdisjoint(source_sets["validation"])
    assert source_sets["train"].isdisjoint(source_sets["test"])
    assert source_sets["validation"].isdisjoint(source_sets["test"])
    assert all(
        value.startswith("source_episode_index:")
        for values in source_sets.values()
        for value in values
    )


def test_training_view_fingerprint_changes_with_source_data(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view_fingerprint",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    source = tmp_path / "source"
    for relative, content in (
        ("meta/info.json", b"{}"),
        ("meta/stats.json", b"{}"),
        ("meta/tasks.parquet", b"tasks"),
        ("meta/episodes/chunk-000/file-000.parquet", b"episodes"),
        ("data/chunk-000/file-000.parquet", b"data-v1"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    before = module.source_dataset_fingerprint(source)
    (source / "data/chunk-000/file-000.parquet").write_bytes(b"data-v2")
    after = module.source_dataset_fingerprint(source)

    assert before != after


def test_groot_materializer_rejects_ambiguous_source_eef_format() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    config["source_dataset"]["eef_pose_format"]["angle_unit"] = "degree"

    with pytest.raises(ValueError, match="Euler-XYZ radians"):
        module.validate_source_eef_pose_format(config["source_dataset"])


def test_groot_data_rewrite_emits_49d_53d_parquet_and_stats(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_file = tmp_path / "source" / "data" / "chunk-000" / "file-000.parquet"
    source_file.parent.mkdir(parents=True)
    rows = 2
    table = pa.table(
        {
            "observation.state.ee_state": [[0.0] * 12 for _ in range(rows)],
            "observation.state.robot_q_current": [
                [float(i + row) for i in range(36)] for row in range(rows)
            ],
            "observation.state.hand_state": [[4.0, 4.5], [3.5, 4.0]],
            "action.ee_action": [
                [0.01 * (i + row) for i in range(12)] for row in range(rows)
            ],
            "action.robot_q_desired": [
                [float(i + row) + 0.1 for i in range(36)] for row in range(rows)
            ],
            "action.hand_cmd": [[3.9, 4.4], [3.4, 3.9]],
            "timestamp": [0.0, 1.0 / 30.0],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
        }
    )
    pq.write_table(table, source_file)

    stats = module.rewrite_data_parquets(
        tmp_path / "source",
        tmp_path / "output",
        config=config,
        policy_type="groot",
        pa=pa,
        pq=pq,
    )

    rewritten = pq.read_table(tmp_path / "output" / "data" / "chunk-000" / "file-000.parquet")
    assert rewritten.schema.field("observation.state").type.list_size == 49
    assert rewritten.schema.field("action").type.list_size == 53
    assert len(stats["observation.state"]["mean"]) == 49
    assert len(stats["action"]["q99"]) == 53


def test_training_view_stats_exclude_validation_and_test_episodes(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view_train_stats",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_file = tmp_path / "source" / "data" / "chunk-000" / "file-000.parquet"
    source_file.parent.mkdir(parents=True)
    table = pa.table(
        {
            "observation.state.robot_q_current": [[0.0] * 36, [0.0] * 36, [100.0] * 36],
            "observation.state.hand_state": [[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]],
            "action.robot_q_desired": [[1.0] * 36, [1.0] * 36, [101.0] * 36],
            "action.hand_cmd": [[1.0, 1.0], [1.0, 1.0], [101.0, 101.0]],
            "timestamp": [0.0, 1.0 / 30.0, 0.0],
            "frame_index": [0, 1, 0],
            "episode_index": [0, 0, 1],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
        }
    )
    pq.write_table(table, source_file)

    stats = module.rewrite_data_parquets(
        tmp_path / "source",
        tmp_path / "output",
        config=config,
        policy_type="act",
        stats_episode_indices={0},
        pa=pa,
        pq=pq,
    )

    assert stats["observation.state"]["mean"] == pytest.approx([0.0] * 19)
    assert stats["action"]["mean"] == pytest.approx([1.0] * 19)


def test_lerobot_training_view_features_and_stats_use_cam0_as_head_left() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_features = {
        "observation.images.cam_0": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_1": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_2": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_3": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state.ee_state": {"dtype": "float32", "shape": [12]},
        "observation.state.robot_q_current": {"dtype": "float32", "shape": [36]},
        "observation.state.hand_state": {"dtype": "float32", "shape": [2]},
        "action.ee_action": {"dtype": "float32", "shape": [12]},
        "action.robot_q_desired": {"dtype": "float32", "shape": [36]},
        "action.hand_cmd": {"dtype": "float32", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }

    features = module.build_training_features(
        config=config,
        policy_type="act",
        source_features=source_features,
    )
    assert list(key for key in features if key.startswith("observation.images.")) == [
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert features["observation.images.head_left"] == {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    }
    assert features["observation.state"]["shape"] == [19]
    assert features["action"]["shape"] == [19]

    source_stats = {
        "observation.state.robot_q_current": {"min": list(range(36)), "max": list(range(36)), "count": [10]},
        "observation.state.hand_state": {"min": [36, 37], "max": [36, 37], "count": [10]},
        "action.robot_q_desired": {"min": list(range(100, 136)), "max": list(range(100, 136)), "count": [10]},
        "action.hand_cmd": {"min": [136, 137], "max": [136, 137], "count": [10]},
        "observation.images.cam_0": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
        "observation.images.cam_2": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
        "observation.images.cam_3": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
    }
    mapped_stats = {
        "observation.state": {"min": list(range(49)), "max": list(range(49)), "count": [10]},
        "action": {"min": list(range(100, 153)), "max": list(range(100, 153)), "count": [10]},
    }
    stats = module.build_training_stats(
        config=config,
        policy_type="groot",
        source_stats=source_stats,
        source_features=source_features,
        mapped_vector_stats=mapped_stats,
    )
    assert stats["observation.images.head_left"] == source_stats["observation.images.cam_0"]
    assert stats["observation.state"] == mapped_stats["observation.state"]
    assert stats["action"] == mapped_stats["action"]


def test_groot_real_g1_relative_eef_slot_mapping() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    state, action = module.map_source_row_to_real_g1_relative_eef(
        ee_state=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, -0.2, 0.1, 0.0, 0.0, math.pi / 2],
        ee_action=[0.2, 0.4, 0.6, 0.0, 0.0, math.pi / 2, 0.3, -0.1, 0.2, 0.0, 0.0, 0.0],
        robot_q_current=[float(i) for i in range(36)],
        robot_q_desired=[float(i + 100) for i in range(36)],
        hand_state=[4.0, 3.0],
        hand_cmd=[2.0, 1.0],
    )

    assert len(state) == 49
    assert len(action) == 53
    assert state[:9] == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert state[9:18] == pytest.approx([0.4, -0.2, 0.1, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert state[18:25] == pytest.approx(
        module.dex1_to_hand(4.0, side="left", kind="state")
    )
    assert state[25:32] == pytest.approx(
        module.dex1_to_hand(3.0, side="right", kind="state")
    )
    assert state[32:39] == [22, 23, 24, 25, 26, 27, 28]
    assert state[39:46] == [29, 30, 31, 32, 33, 34, 35]
    assert state[46:49] == [19, 20, 21]
    assert action[:9] == pytest.approx([0.2, 0.4, 0.6, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert action[18:25] == pytest.approx(
        module.dex1_to_hand(2.0, side="left", kind="action")
    )
    assert action[25:32] == pytest.approx(
        module.dex1_to_hand(1.0, side="right", kind="action")
    )
    assert action[32:39] == [122, 123, 124, 125, 126, 127, 128]
    assert action[39:46] == [129, 130, 131, 132, 133, 134, 135]
    assert action[46:49] == [119, 120, 121]
    assert action[49:53] == [0.0, 0.0, 0.0, 0.0]


def test_dex1_groot_hand_synergy_round_trips_and_clips() -> None:
    mapping = load_module("g1_dex1_range_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")

    for side in ("left", "right"):
        for value in (0.0, 0.75, 2.25, 4.5):
            canonical = mapping.dex1_to_hand(value, side=side, kind="action")
            assert mapping.hand_to_dex1(canonical, side=side, kind="action") == pytest.approx(value)
    assert mapping.hand_to_dex1(
        mapping.dex1_to_hand(4.8, side="right", kind="action"),
        side="right",
        kind="action",
    ) == pytest.approx(4.5)


def test_dataset_robot_q_maps_to_runtime_sdk_motor_ids() -> None:
    module = load_module("g1_sdk_dataset_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    runtime = load_module(
        "g1_runtime_joint_mapping",
        ROOT.parents[1]
        / "inference"
        / "orin"
        / "ros2_ws"
        / "src"
        / "g1_hw_bridge"
        / "g1_hw_bridge"
        / "joint_mapping.py",
    )

    assert tuple(module.G1_SDK_JOINT_NAMES) == tuple(runtime.G1_JOINT_NAMES)
    assert module.DATASET_ROBOT_Q_INDEX_BY_SDK_ID == tuple(range(7, 36))
    assert module.map_dataset_robot_q_to_sdk_order(list(range(36))) == list(range(7, 36))
    assert module.dataset_robot_q_index_from_sdk_id(0) == 7
    assert module.dataset_robot_q_index_from_sdk_id(28) == 35
    with pytest.raises(ValueError, match="sdk_id must be in"):
        module.dataset_robot_q_index_from_sdk_id(29)
    with pytest.raises(TypeError, match="sdk_id must be an integer"):
        module.dataset_robot_q_index_from_sdk_id(True)


def test_g1_sdk_mapping_verifier_reports_all_29_motors() -> None:
    module = load_module(
        "verify_g1_sdk_joint_mapping",
        ROOT / "scripts" / "verify_g1_sdk_joint_mapping.py",
    )

    rows = module.verified_mapping_rows()
    assert len(rows) == 29
    assert rows[0] == {
        "sdk_id": 0,
        "joint_name": "left_hip_pitch_joint",
        "dataset_robot_q_index": 7,
    }
    assert rows[-1] == {
        "sdk_id": 28,
        "joint_name": "right_wrist_yaw_joint",
        "dataset_robot_q_index": 35,
    }


def test_training_view_marker_records_sdk_motor_mapping(tmp_path: Path) -> None:
    module = load_module(
        "materialize_sdk_mapping_marker",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(
        json.dumps({"features": {}}), encoding="utf-8"
    )

    marker = module.build_marker(
        repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1",
        source_root=tmp_path,
        config=config,
        policy_type="groot",
        source_fingerprint="fixture-sha256",
    )

    assert marker["schema_version"] == "team_ramen_training_view_v7"
    assert marker["groot_dex1_hand_encoding"] == {
        "source_range": [0.0, 4.5],
        "encoding": "fixed official G1 seven-joint synergy",
        "inverse": "least-squares projection",
    }
    assert marker["groot_action_loss_excluded_indices"] == list(range(49, 53))
    assert len(marker["source_robot_q_sdk_mapping"]) == 29
    assert marker["source_robot_q_sdk_mapping"][15] == {
        "sdk_id": 15,
        "dataset_index": 22,
        "joint_name": "left_shoulder_pitch_joint",
    }


def test_training_view_normalizes_sparse_source_task_ids(tmp_path: Path) -> None:
    module = load_module(
        "materialize_sparse_task_ids",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source = tmp_path / "source"
    data_path = source / "data/chunk-000/file-000.parquet"
    tasks_path = source / "meta/tasks.parquet"
    data_path.parent.mkdir(parents=True)
    tasks_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_index": 3, "__index_level_0__": "rotate leg to tighten"}]
        ),
        tasks_path,
    )
    pq.write_table(
        pa.table(
            {
                "observation.state.robot_q_current": [[0.0] * 36] * 2,
                "observation.state.hand_state": [[0.0, 0.0]] * 2,
                "action.robot_q_desired": [[0.0] * 36] * 2,
                "action.hand_cmd": [[0.0, 0.0]] * 2,
                "timestamp": [0.0, 0.1],
                "frame_index": [0, 1],
                "episode_index": [0, 0],
                "index": [0, 1],
                "task_index": [3, 3],
            }
        ),
        data_path,
    )

    task_index_map, task_names = module.load_dense_task_index_mapping(source)
    assert task_index_map == {3: 0}
    assert task_names == {3: "rotate leg to tighten"}
    output = tmp_path / "output"
    module.write_dense_tasks_metadata(
        output / "meta/tasks.parquet",
        task_index_map=task_index_map,
        task_names=task_names,
    )
    module.rewrite_data_parquets(
        source,
        output,
        config=config,
        policy_type="act",
        task_index_map=task_index_map,
        pa=pa,
        pq=pq,
    )

    tasks = __import__("pandas").read_parquet(output / "meta/tasks.parquet")
    assert tasks.index.tolist() == ["rotate leg to tighten"]
    assert tasks["task_index"].tolist() == [0]
    rewritten = pq.read_table(output / "data/chunk-000/file-000.parquet")
    assert rewritten["task_index"].to_pylist() == [0, 0]


def test_groot_relative_eef_uses_se3_not_component_subtraction() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    current = module.source_euler_xyz_pose_to_xyz_rot6d([1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2])
    target = module.source_euler_xyz_pose_to_xyz_rot6d([1.0, 3.0, 3.0, 0.0, 0.0, math.pi])

    relative = module.absolute_eef_xyz_rot6d_to_relative(current, target)

    assert relative == pytest.approx([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], abs=1e-7)


def test_groot_real_g1_relative_eef_modality_contract() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    modality = module.build_real_g1_relative_eef_modality_json(
        [
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ]
    )

    assert list(modality["state"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
    ]
    assert list(modality["action"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert modality["meta"]["action_configs"]["left_wrist_eef_9d"] == {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "left_wrist_eef_9d",
    }
    sdk_mapping = modality["meta"]["source_robot_q_sdk_mapping"]
    assert len(sdk_mapping) == 29
    assert sdk_mapping[0] == {
        "sdk_id": 0,
        "dataset_index": 7,
        "joint_name": "left_hip_pitch_joint",
    }
    assert sdk_mapping[-1] == {
        "sdk_id": 28,
        "dataset_index": 35,
        "joint_name": "right_wrist_yaw_joint",
    }
    assert list(modality["video"]) == ["head_left", "left_wrist", "right_wrist"]


def test_lerobot_groot_processor_converts_complete_eef_chunk_from_current_state() -> None:
    patcher = load_module(
        "patch_lerobot_groot_relative_eef",
        ROOT / "scripts" / "patch_lerobot_groot_relative_eef.py",
    )
    installed_source = patcher.processor_path().read_text(encoding="utf-8")
    assert any(
        marker in installed_source
        for marker in (patcher.PATCH_MARKER, patcher.PATCH_MARKER_V2, patcher.PATCH_MARKER_V1)
    )
    assert "Qwen/Qwen3-VL-2B-Instruct" in patcher.PROCESSOR_ASSET_REPLACEMENT

    import torch
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep

    mapping = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    state = torch.zeros(1, mapping.REAL_G1_RELATIVE_EEF_STATE_DIM)
    action = torch.zeros(1, 2, mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    for span in (
        mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["left_wrist_eef_9d"],
        mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["right_wrist_eef_9d"],
    ):
        state[:, span[0] + 3 : span[1]] = identity
    for span in (
        mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_wrist_eef_9d"],
        mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_wrist_eef_9d"],
    ):
        action[:, :, span[0] + 3 : span[1]] = identity

    left_eef = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_wrist_eef_9d"]
    action[0, 0, left_eef[0] : left_eef[0] + 3] = torch.tensor([0.1, 0.0, 0.0])
    action[0, 1, left_eef[0] : left_eef[0] + 3] = torch.tensor([0.2, 0.0, 0.0])
    left_arm_state = mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["left_arm"]
    left_arm_action = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_arm"]
    state[:, slice(*left_arm_state)] = 1.0
    action[:, :, slice(*left_arm_action)] = torch.tensor([[[1.5] * 7, [2.0] * 7]])
    left_hand = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_hand"]
    waist = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["waist"]
    action[:, :, slice(*left_hand)] = 0.75
    action[:, :, slice(*waist)] = 0.25

    state_keys = list(mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES)
    action_keys = list(mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)
    raw_stats = {
        "state": {
            key: {"mean": [0.0] * (end - start)}
            for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES.items()
        },
        "action": {
            key: {"mean": [0.0] * (end - start)}
            for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES.items()
        },
    }
    modality = {
        "state": {"modality_keys": state_keys},
        "action": {
            "modality_keys": action_keys,
            "action_configs": [mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in action_keys],
        },
    }
    processor = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality)

    converted = processor._convert_relative_action_groups_for_training(action, state)

    assert converted[0, :, left_eef[0] : left_eef[0] + 3] == pytest.approx(
        torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    )
    assert converted[0, :, slice(*left_arm_action)] == pytest.approx(
        torch.tensor([[0.5] * 7, [1.0] * 7])
    )
    assert converted[0, :, slice(*left_hand)] == pytest.approx(torch.full((2, 7), 0.75))
    assert converted[0, :, slice(*waist)] == pytest.approx(torch.full((2, 3), 0.25))


def test_lerobot_groot_processor_masks_only_base_action_slots() -> None:
    import torch
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
    from lerobot.types import TransitionKey

    mapping = load_module("g1_loss_mask_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    processor = GrootN17PackInputsStep(
        action_horizon=2,
        valid_action_horizon=2,
        max_action_dim=mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM,
        normalize_min_max=False,
        modality_config={
            "action": {
                "loss_excluded_indices": list(
                    mapping.REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES
                )
            }
        },
    )
    result = processor(
        {
            TransitionKey.ACTION: torch.zeros(
                1, 2, mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM
            )
        }
    )
    mask = result[TransitionKey.COMPLEMENTARY_DATA]["action_mask"]

    assert tuple(mask.shape) == (1, 2, mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM)
    assert mask.sum().item() == 2 * len(mapping.REAL_G1_DEX1_ACTION_LOSS_VALID_INDICES)
    assert torch.all(mask[:, :, list(mapping.REAL_G1_DEX1_ACTION_LOSS_VALID_INDICES)] == 1)
    assert torch.all(mask[:, :, list(mapping.REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES)] == 0)


def test_prepare_groot_overlay_preserves_action_contract_and_enables_three_cameras(tmp_path) -> None:
    overlay_module = load_module(
        "prepare_groot_n17_real_g1_overlay",
        ROOT / "scripts" / "prepare_groot_n17_real_g1_overlay.py",
    )
    mapping = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    source = tmp_path / "source"
    source.mkdir()
    tag = mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
    state_keys = list(mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES)
    action_keys = list(mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)
    processor_config = {
        "processor_kwargs": {
            "use_relative_action": True,
            "max_action_horizon": mapping.GROOT_N17_NATIVE_ACTION_HORIZON,
            "modality_configs": {
                tag: {
                    "video": {"delta_indices": [-20, 0], "modality_keys": ["ego_view"]},
                    "state": {"delta_indices": [0], "modality_keys": state_keys},
                    "action": {
                        "delta_indices": list(range(mapping.GROOT_N17_NATIVE_ACTION_HORIZON)),
                        "modality_keys": action_keys,
                        "action_configs": [
                            mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in action_keys
                        ],
                    },
                }
            },
        }
    }
    statistics = {
        tag: {
            "state": {
                key: {"mean": [0.0] * (end - start)}
                for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES.items()
            },
            "action": {
                key: {"mean": [0.0] * (end - start)}
                for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES.items()
            },
            "relative_action": {
                key: {
                    "mean": [
                        [0.0] * (mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES[key][1]
                                 - mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES[key][0])
                        for _ in range(mapping.GROOT_N17_NATIVE_ACTION_HORIZON)
                    ]
                }
                for key in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_arm", "right_arm")
            },
        }
    }
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "action_horizon": mapping.GROOT_N17_NATIVE_ACTION_HORIZON,
            }
        )
    )
    (source / "embodiment_id.json").write_text(
        json.dumps({tag: mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID})
    )
    (source / "processor_config.json").write_text(json.dumps(processor_config))
    (source / "statistics.json").write_text(json.dumps(statistics))
    (source / "model.safetensors").write_bytes(b"fixture")

    output = overlay_module.prepare_overlay(
        source_root=source,
        output_root=tmp_path / "overlay",
        model_path="nvidia/GR00T-N1.7-3B",
        revision="fixture-revision",
        force=False,
    )

    overlay_config = json.loads((output / "processor_config.json").read_text())
    overlay_modality = overlay_config["processor_kwargs"]["modality_configs"][tag]
    assert overlay_modality["video"] == {
        "delta_indices": [0],
        "modality_keys": ["head_left", "left_wrist", "right_wrist"],
    }
    assert overlay_modality["action"]["action_configs"] == (
        processor_config["processor_kwargs"]["modality_configs"][tag]["action"]["action_configs"]
    )
    assert overlay_modality["action"]["loss_excluded_indices"] == list(
        mapping.REAL_G1_DEX1_ACTION_LOSS_EXCLUDED_INDICES
    )
    assert (output / "model.safetensors").is_symlink()
    assert overlay_module.prepare_overlay(
        source_root=source,
        output_root=output,
        model_path="nvidia/GR00T-N1.7-3B",
        revision="fixture-revision",
        force=False,
    ) == output


def test_restore_groot_base_model_path_makes_checkpoints_portable(tmp_path) -> None:
    module = load_module(
        "restore_groot_base_model_path",
        ROOT / "scripts" / "restore_groot_base_model_path.py",
    )
    checkpoint = tmp_path / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps({"type": "groot", "base_model_path": "/tmp/local-overlay"}) + "\n"
    )

    changed = module.restore_base_model_paths(
        tmp_path,
        runtime_path="/tmp/local-overlay",
        canonical_path="nvidia/GR00T-N1.7-3B",
    )

    assert changed == 1
    assert json.loads(config_path.read_text())["base_model_path"] == "nvidia/GR00T-N1.7-3B"


# ============================================================================
# Issue #122 D-1: skill-merged HF repo + USE_MERGED_HF_REPO opt-in
# ============================================================================


def _clear_merged_env(monkeypatch) -> None:
    for name in [
        "SUBTASK",
        "DATASET_REPO_ID",
        "DATASET_REPO_IDS",
        "MERGED_DATASET_REPO_ID",
        "MERGED_COMPONENT_REPO_IDS",
        "USE_MERGED_HF_REPO",
    ]:
        monkeypatch.delenv(name, raising=False)


def test_resolve_merged_dataset_repo_id_from_template(monkeypatch) -> None:
    """merged_repo_template から skill-level merged repo id が format される。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "rotate_table_base")

    values = module.resolve(config)

    assert (
        values["MERGED_DATASET_REPO_ID"]
        == "Team-RAMEN/IROS2026_RAMEN_HARA_rotate_table_base_merged_v1"
    )


def test_resolve_merged_dataset_repo_id_empty_for_union_subtask(monkeypatch) -> None:
    """union subtask (component_subtasks あり) は MERGED_DATASET_REPO_ID を持たない。

    orchestrator (prepare_merged_skill_dataset.sh) が union に対して誤って
    push しようとした場合に fail する guard。
    """
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")

    values = module.resolve(config)

    assert values["MERGED_DATASET_REPO_ID"] == ""
    # component list は引き続き export される
    components = json.loads(values["MERGED_COMPONENT_REPO_IDS"])
    assert len(components) == 2


def test_resolve_merged_component_repo_ids_from_component_subtasks(monkeypatch) -> None:
    """combined_task5_7 の component_subtasks から 2 skill-merged repo id を得る。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")

    values = module.resolve(config)

    components = json.loads(values["MERGED_COMPONENT_REPO_IDS"])
    assert components == [
        "Team-RAMEN/IROS2026_RAMEN_HARA_rotate_table_base_merged_v1",
        "Team-RAMEN/IROS2026_RAMEN_HARA_move_table_base_merged_v1",
    ]


def test_resolve_merged_component_repo_ids_single_when_no_components(monkeypatch) -> None:
    """component_subtasks 未定義の subtask では MERGED_COMPONENT_REPO_IDS = [self merged]。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "rotate_table_base")

    values = module.resolve(config)

    components = json.loads(values["MERGED_COMPONENT_REPO_IDS"])
    assert components == [values["MERGED_DATASET_REPO_ID"]]


def test_use_merged_hf_repo_default_false_preserves_config_dataset_repos(monkeypatch) -> None:
    """USE_MERGED_HF_REPO 未 set (default false) 時は config `dataset_repos` を素直に返す。

    Issue #122 (2026-08-19) で task 5 の 8 chunks を 1 merged repo に統合 (元 chunks は削除)、
    combined_task5_7 の default は [task 7 + task 5 merged] = 2 repos に短縮。過去に
    「9 chunks を default に維持」してた assertion は premise が invalid (chunks 削除済) で
    ここに置き換え。opt-in path (USE_MERGED_HF_REPO=true、下 test 参照) は skill-name
    based の `merged_repo_template` で解決するため export 名が異なる (rotate_table_base_merged /
    move_table_base_merged)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")

    values = module.resolve(config)

    assert values["USE_MERGED_HF_REPO"] == "false"
    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    # combined_task5_7 の dataset_repos は [task 7 (単体) + task 5 merged] = 2
    assert repo_ids == [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-move-table-base",
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-table-base",
    ]


def test_use_merged_hf_repo_opt_in_overrides_dataset_repo_ids(monkeypatch) -> None:
    """USE_MERGED_HF_REPO=true 時、DATASET_REPO_IDS が skill-merged component list に差替わる。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")
    monkeypatch.setenv("USE_MERGED_HF_REPO", "true")

    values = module.resolve(config)

    assert values["USE_MERGED_HF_REPO"] == "true"
    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    assert repo_ids == [
        "Team-RAMEN/IROS2026_RAMEN_HARA_rotate_table_base_merged_v1",
        "Team-RAMEN/IROS2026_RAMEN_HARA_move_table_base_merged_v1",
    ]
    assert values["DATASET_REPO_ID"] == repo_ids[0]


def test_use_merged_hf_repo_opt_in_single_skill(monkeypatch) -> None:
    """USE_MERGED_HF_REPO=true + 単一 skill 時、DATASET_REPO_IDS = [1 merged repo]。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "rotate_table_base")
    monkeypatch.setenv("USE_MERGED_HF_REPO", "true")

    values = module.resolve(config)

    repo_ids = json.loads(values["DATASET_REPO_IDS"])
    assert repo_ids == ["Team-RAMEN/IROS2026_RAMEN_HARA_rotate_table_base_merged_v1"]


def test_use_merged_hf_repo_requires_template(monkeypatch) -> None:
    """merged_repo_template 未定義で USE_MERGED_HF_REPO=true は error。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "rotate_table_base")
    monkeypatch.setenv("USE_MERGED_HF_REPO", "true")
    # simulate missing template
    del config["merged_repo_template"]

    with pytest.raises(ValueError, match="merged_repo_template"):
        module.resolve(config)


def test_env_override_merged_component_repo_ids_wins(monkeypatch) -> None:
    """MERGED_COMPONENT_REPO_IDS env で明示 override 可能 (pilot/test 用)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    monkeypatch.setenv("SUBTASK", "combined_task5_7")
    monkeypatch.setenv("USE_MERGED_HF_REPO", "true")
    custom = json.dumps(["custom/skill_a_merged", "custom/skill_b_merged"])
    monkeypatch.setenv("MERGED_COMPONENT_REPO_IDS", custom)

    values = module.resolve(config)

    assert values["MERGED_COMPONENT_REPO_IDS"] == custom
    # DATASET_REPO_IDS も opt-in で custom list に追従する
    assert json.loads(values["DATASET_REPO_IDS"]) == json.loads(custom)


def test_rejects_component_subtasks_referencing_unknown_subtask(monkeypatch) -> None:
    """component_subtasks に未定義 subtask 名があれば error (schema 保護)。"""
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    _clear_merged_env(monkeypatch)
    config["subtasks"]["combined_task5_7"]["component_subtasks"] = ["not_a_real_subtask"]
    monkeypatch.setenv("SUBTASK", "combined_task5_7")

    with pytest.raises(ValueError, match="unknown component_subtask"):
        module.resolve(config)


def test_train_sh_echoes_merged_hf_state() -> None:
    """train_lerobot.sh が USE_MERGED_HF_REPO 状態を echo し、opt-in 時に namespace 分離する。"""
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()
    assert 'echo "use_merged_hf_repo: $USE_MERGED_HF_REPO"' in script
    # opt-in 時の merged dir namespace (chunk-merged と衝突しない)
    assert "_from_hf" in script


def test_train_sh_echoes_and_validates_obb_overlay_env() -> None:
    """Issue #122 D-4-S3: train_lerobot.sh が OBB_OVERLAY_ENABLE の状態を echo し、
    enable + OBB_PRECOMPUTED_ROOT 未指定は fail-fast (exit 2) にする。usage 例も含む。"""
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()
    # env state echo
    assert 'echo "obb_overlay_enable: $OBB_OVERLAY_ENABLE_VAL"' in script
    # fail-fast validation
    assert "OBB_OVERLAY_ENABLE=true requires OBB_PRECOMPUTED_ROOT" in script
    assert "exit 2" in script
    # dry-run 出力にも overlay 状態を含む
    assert 'OBB_OVERLAY_ENABLE=%s' in script
    # 使用例 (GR00T + overlay) がコメントで説明されている
    assert "OBB_OVERLAY_ENABLE=true" in script
    # optional env の default 表示
    assert "OBB_OVERLAY_CONF" in script
    assert "OBB_OVERLAY_THICKNESS" in script
    assert "OBB_OVERLAY_CLASS_FILTER" in script


def test_prepare_merged_skill_dataset_script_exists() -> None:
    """D-1 orchestrator script が存在し、merge + upload の両方を wire している。"""
    script = (
        ROOT.parents[1]
        / "data"
        / "bitrobot_lerobot_subtask_datasets"
        / "scripts"
        / "prepare_merged_skill_dataset.sh"
    )
    assert script.is_file()
    src = script.read_text()
    # 3 stage が揃っている: resolve config + merge + upload
    assert "resolve_training_config.py" in src
    assert "prepare_multi_repo_source.py" in src
    assert "upload_lerobot_dataset.py" in src
    # skill-merged 側 name を使う (chunk template ではなく merged template)
    assert "MERGED_DATASET_REPO_ID" in src
    # skill-agnostic (SUBTASK env で切替) を明示
    assert 'SUBTASK=rotate_table_base' in src or "SUBTASK env is required" in src


def test_prepare_merged_skill_dataset_script_is_executable() -> None:
    """orchestrator script は executable bit を持つ (bash 直起動対応)。"""
    import stat

    script = (
        ROOT.parents[1]
        / "data"
        / "bitrobot_lerobot_subtask_datasets"
        / "scripts"
        / "prepare_merged_skill_dataset.sh"
    )
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR, f"script not executable: mode={oct(mode)}"
