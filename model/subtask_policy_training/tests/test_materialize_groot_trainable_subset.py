from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from model.subtask_policy_training.scripts.materialize_groot_trainable_subset import (
    materialize,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materialize_replaces_only_approved_subset(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    save_file(
        {"action.weight": torch.zeros(2), "frozen.weight": torch.ones(3)},
        base / "model.safetensors",
    )
    subset = tmp_path / "000500.safetensors"
    save_file({"action.weight": torch.full((2,), 4.0)}, subset)
    subset.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": _sha256(subset),
                "parameter_count": 2,
                "approved_prefixes": ["action."],
            }
        )
    )

    output = tmp_path / "candidate"
    manifest = materialize(base, subset, output)

    with safe_open(output / "model.safetensors", framework="pt") as weights:
        assert torch.equal(weights.get_tensor("action.weight"), torch.full((2,), 4.0))
        assert torch.equal(weights.get_tensor("frozen.weight"), torch.ones(3))
    assert manifest["replaced_tensor_count"] == 1
    assert manifest["optimizer_state_transferred"] is False


def test_materialize_rejects_unknown_tensor(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    save_file({"action.weight": torch.zeros(2)}, base / "model.safetensors")
    subset = tmp_path / "000500.safetensors"
    save_file({"action.unknown": torch.zeros(2)}, subset)
    subset.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sha256": _sha256(subset),
                "parameter_count": 2,
                "approved_prefixes": ["action."],
            }
        )
    )

    with pytest.raises(ValueError, match="unknown tensors"):
        materialize(base, subset, tmp_path / "candidate")
