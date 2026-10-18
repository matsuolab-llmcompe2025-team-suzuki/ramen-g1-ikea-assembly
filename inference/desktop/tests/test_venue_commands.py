"""Verify the documented CLI and default pick mode without running main()."""

from pathlib import Path
import re
import shlex
import sys

import pytest

from inference.desktop import entrypoint
from inference.desktop.lower_policy.policies.config_loader import load_pick_leg_hybrid_default

# 会場の手順書。公開用 repo には入れない (Issue #190)
pytestmark = pytest.mark.skipif(
    not Path("docs/handoff/venue_commands.md").exists(),
    reason="docs/handoff/venue_commands.md が無い (公開用 repo には入れない)",
)


def _documented_args():
    doc = Path("docs/handoff/venue_commands.md").read_text()
    command = next(b for b in re.findall(r"```bash\n(.*?)```", doc, re.S) if "python -m inference.desktop.entrypoint" in b)
    command = command.split("python -m inference.desktop.entrypoint", 1)[1]
    command = command.replace("\\\n", " ").replace("$STAGE", "2")
    command = command.replace("${ORIN_IP}", "127.0.0.1").replace("$ORIN_IP", "127.0.0.1")
    command = command.replace("$PICK_VLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
    return shlex.split(command)


@pytest.mark.parametrize("stage", range(6))
@pytest.mark.parametrize("actuate", [False, True])
def test_documented_boundary_command_validates_all_stages(monkeypatch, stage, actuate):
    argv = _documented_args()
    argv[argv.index("--stage") + 1] = str(stage)
    if actuate:
        argv.append("--actuate")
    # Explicit endpoints must also win over contradictory inherited lab settings.
    monkeypatch.setenv("IROS_ORIN_HOST", "192.0.2.1")
    monkeypatch.setenv("G1_IMAGE_SERVER_IP", "192.0.2.2")
    monkeypatch.setattr(sys, "argv", ["entrypoint", *argv])
    args = entrypoint.parse_args()
    entrypoint.normalize_phase3_range_args(args)
    entrypoint.configure_official_endpoints(args)
    entrypoint.resolve_production_pick_mode(args)
    entrypoint.apply_default_policy_variants(args, config_path=args.policy_config)
    entrypoint._validate_phase3_config(args)
    # Stage 1〜4 の pick のやり方は policy_config.yaml の pick_leg_hybrid (Issue #188)
    assert args.pick_leg_hybrid is (
        stage in (1, 2, 3, 4) and load_pick_leg_hybrid_default(args.policy_config)
    )
    # VLM はこの run の中で立てる (外の VLM の宛先は渡さない)
    assert args.spawn_vlm_server is True
    assert args.pick_leg_vlm_endpoint is None
    assert args.pick_leg_phase3_executor == "rule_based"
    assert args.actuate is actuate
    assert args.boundary_state_host == "127.0.0.1"


def test_documented_full_run_includes_hybrid_pick_after_stage_zero(monkeypatch):
    argv = _documented_args()
    index = argv.index("--stage")
    argv[index:index + 2] = ["--phase3-full", "--phase3-start-stage", "0", "--phase3-end-stage", "5"]
    monkeypatch.setattr(sys, "argv", ["entrypoint", *argv])
    args = entrypoint.parse_args()
    entrypoint.normalize_phase3_range_args(args)
    entrypoint.configure_official_endpoints(args)
    entrypoint.resolve_production_pick_mode(args)
    entrypoint.apply_default_policy_variants(args, config_path=args.policy_config)
    entrypoint._validate_phase3_config(args)
    assert args.pick_leg_hybrid is load_pick_leg_hybrid_default(args.policy_config)
    assert entrypoint.resolve_stage_selection(args) == (0, (0, 1, 2, 3, 4, 5))
    assert entrypoint.resolve_include_hand_in_head(args, range(6)) is True
