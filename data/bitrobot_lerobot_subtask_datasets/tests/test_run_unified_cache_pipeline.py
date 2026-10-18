"""run_unified_cache_pipeline.py + configs/unified_frame_cache.yaml の unit test (Issue #139)。"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
CONFIG = ROOT / "configs" / "unified_frame_cache.yaml"
BASE_YAML = REPO / "model" / "ramen_ori" / "configs" / "base.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_unified_cache_pipeline", ROOT / "scripts" / "run_unified_cache_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipe = _load()
CFG = pipe.load_config(CONFIG)


def _sp(tmp_path: Path):
    return pipe.SkillPaths(tmp_path / "rotate_table_base")


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_yolo_hash_matches_existing_hf_cache_dir() -> None:
    # HF 上の既存 obb_yolo/ce877e8237be と同じ ckpt / revision → 同じ dir に上書きされる
    assert pipe.yolo_cache_hash(CFG) == "ce877e8237be"


def test_augmentation_equals_base_yaml_phase_g_values() -> None:
    """焼き込みの aug は base.yaml と同じ値 + enabled / geometric を on にしたもの。"""
    base = yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))["augmentation"]
    want = copy.deepcopy(base)
    want["enabled"] = True
    want["geometric"]["enabled"] = True
    assert CFG["augmentation"] == want
    assert set(CFG["augmentation"]["wrist_cam_keys"]) == set(CFG["cameras"]["wrist"])


ISSUE_139_SKILLS = ["rotate_table_base", "move_table_base", "insert_table_leg"]
ISSUE_146_SKILLS = ["pick_table_leg", "flip_table", "rotate_leg_to_tighten"]


def _skill(name: str) -> dict:
    return next(s for s in CFG["skills"] if s["name"] == name)


def test_skills_cover_six_datasets() -> None:
    assert [s["name"] for s in CFG["skills"]] == ISSUE_139_SKILLS + ISSUE_146_SKILLS
    # v0 展開は一緒に流す skill が同時に走るので、流す組ごとに 24 vCPU に収める
    for group in (ISSUE_139_SKILLS, ISSUE_146_SKILLS):
        assert sum(_skill(n)["decode_jobs"] for n in group) <= 24


def test_issue_146_skills_skip_token_and_use_merged_repos() -> None:
    assert [_skill(n)["repo_id"] for n in ISSUE_146_SKILLS] == [
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-pick-table-leg",
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-flip-table",
        "Team-RAMEN/ramen-g1-ikea-assembly-ramen-ori-rotate-leg-to-tighten",
    ]
    for n in ISSUE_146_SKILLS:
        assert pipe.skill_kinds(_skill(n)) == ("overlay", "common")
    # Issue #139 の 3 skill は HF にある 3 種のまま
    for n in ISSUE_139_SKILLS:
        assert pipe.skill_kinds(_skill(n)) == pipe.V2_KINDS


def test_skill_kinds_rejects_missing_common_or_head_or_unknown() -> None:
    import pytest

    for kinds in (["overlay"], ["common"], ["overlay", "common", "depth"]):
        with pytest.raises(pipe.PipelineError):
            pipe.skill_kinds({"name": "x", "kinds": kinds})
    assert pipe.skill_kinds({"name": "x", "kinds": ["common", "token"]}) == ("token", "common")


def test_main_rejects_bad_kinds_before_starting(tmp_path: Path) -> None:
    import pytest

    cfg = copy.deepcopy(CFG)
    cfg["skills"][0]["kinds"] = ["overlay"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(SystemExit):
        pipe.main(["--config", str(cfg_path), "--work-root", str(tmp_path / "w"), "--dry-run"])


# ---------------------------------------------------------------------------
# command
# ---------------------------------------------------------------------------


def test_decode_cmd_limits_cams_and_keeps_mp4(tmp_path: Path) -> None:
    cmd = pipe.decode_cmd(CFG, _sp(tmp_path), CFG["skills"][0])
    assert _flag(cmd, "--video-keys") == (
        "observation.images.cam_0,observation.images.cam_2,observation.images.cam_3"
    )
    assert "--keep-source-mp4" in cmd
    assert _flag(cmd, "--jobs") == "8"


def test_yolo_cmd_is_full_detection_and_sharded(tmp_path: Path) -> None:
    cmd = pipe.yolo_cmd(CFG, _sp(tmp_path), 2)
    assert cmd[:4] == ["pixi", "run", "--manifest-path", str(REPO / "model/yolo_obb/pixi.toml")]
    assert _flag(cmd, "--cameras") == "observation.images.cam_0"
    assert _flag(cmd, "--top-k") == "32"
    assert "--fail-on-truncation" in cmd
    assert (_flag(cmd, "--num-shards"), _flag(cmd, "--shard-index")) == ("4", "2")
    assert _flag(cmd, "--yolo-revision") == CFG["yolo"]["revision"]


def test_bake_cmd_writes_three_roots_with_unified_config(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    cmd = pipe.bake_cmd(CFG, sp, _skill("rotate_table_base"), CONFIG, "ce877e8237be")
    for kind in ("overlay", "token", "common"):
        assert _flag(cmd, f"--output-{kind}-root") == str(sp.v2 / kind)
    assert _flag(cmd, "--aug-config-yaml") == str(CONFIG)
    assert _flag(cmd, "--obb-top-k") == "32"
    assert _flag(cmd, "--obb-cameras") == "observation.images.cam_0"
    assert _flag(cmd, "--num-variants") == "10"


def test_bake_cmd_omits_token_root_when_not_in_kinds(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    cmd = pipe.bake_cmd(CFG, sp, _skill("flip_table"), CONFIG, "ce877e8237be")
    assert "--output-token-root" not in cmd
    assert _flag(cmd, "--output-overlay-root") == str(sp.v2 / "overlay")
    assert _flag(cmd, "--output-common-root") == str(sp.v2 / "common")


def test_dry_run_packs_only_kinds_of_skill(tmp_path: Path, capsys) -> None:
    """token を焼かない skill は token の tar 化も作らない (overlay / common / v0 だけ)。"""
    assert pipe.main(
        ["--config", str(CONFIG), "--work-root", str(tmp_path), "--skills", "flip_table", "--dry-run"]
    ) == 0
    out = capsys.readouterr().out
    assert "--output-token-root" not in out
    assert "frame_cache_v2_tars/token" not in out
    for kind in ("overlay", "common"):
        assert f"frame_cache_v2_tars/{kind}" in out
    assert "ramen-g1-ikea-assembly-ramen-ori-flip-table" in out


def test_pack_v2_cmd_uses_wrist_for_common_and_head_otherwise(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    common = pipe.pack_v2_cmd(CFG, sp, "common")
    overlay = pipe.pack_v2_cmd(CFG, sp, "overlay")
    assert _flag(common, "--cameras") == "observation.images.cam_2,observation.images.cam_3"
    assert _flag(overlay, "--cameras") == "observation.images.cam_0"
    assert _flag(overlay, "--done-flag") == str(sp.marker("bake"))
    assert _flag(overlay, "--dest-dir") == str(sp.staging / "frame_cache_v2_tars" / "overlay")


def test_upload_cmd_waits_for_pack_marker(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    cmd = pipe.sync_cmd(CFG, sp, "Team-RAMEN/x", "upload")
    assert _flag(cmd, "--until-flag") == str(sp.marker("pack"))
    assert "--until-flag" not in pipe.sync_cmd(CFG, sp, "Team-RAMEN/x", "verify")


def test_download_patterns_skip_ir_and_head_right() -> None:
    full = pipe.download_patterns(CFG, None)
    assert "videos/observation.images.cam_1/*" not in full
    assert "videos/observation.images.cam_0/*" in full
    limited = pipe.download_patterns(CFG, 1)
    assert "videos/observation.images.cam_2/chunk-000/file-000.mp4" in limited
    assert not any(p.endswith("/*") and p.startswith("videos/") for p in limited)


def test_cam_decoded_requires_frame_count_for_every_mp4(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    cam = "observation.images.cam_0"
    assert not pipe.cam_decoded(sp, cam)
    for i in range(2):
        mp4 = sp.src / "videos" / cam / "chunk-000" / f"file-{i:03d}.mp4"
        mp4.parent.mkdir(parents=True, exist_ok=True)
        mp4.write_bytes(b"x")
    (sp.v0 / cam / "chunk-000" / "file-000").mkdir(parents=True)
    (sp.v0 / cam / "chunk-000" / "file-000" / "_frame_count.txt").write_text("3")
    assert not pipe.cam_decoded(sp, cam)
    (sp.v0 / cam / "chunk-000" / "file-001").mkdir(parents=True)
    (sp.v0 / cam / "chunk-000" / "file-001" / "_frame_count.txt").write_text("3")
    assert pipe.cam_decoded(sp, cam)


def test_load_dotenv_does_not_override_existing(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('HF_TOKEN="from-file"\nexport OTHER=1\n# c\n', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    monkeypatch.delenv("OTHER", raising=False)
    pipe.load_dotenv(env)
    import os

    assert os.environ["HF_TOKEN"] == "from-env"
    assert os.environ["OTHER"] == "1"


def test_dry_run_prints_every_stage_without_executing(tmp_path: Path, capsys) -> None:
    rc = pipe.main(
        ["--config", str(CONFIG), "--work-root", str(tmp_path), "--skills", "move_table_base", "--dry-run"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    for needle in (
        "snapshot_download", "precompute_frame_cache.py", "precompute_yolo_obb.py",
        "precompute_augmented_frame_cache_ramen_ori.py", "pack_frame_cache_tars.py",
        "sync_frame_cache_to_hf.py upload", "sync_frame_cache_to_hf.py verify",
        "sync_frame_cache_to_hf.py cleanup", "ALL DONE",
    ):
        assert needle in out, needle
    assert "--execute" not in out  # 削除は手動
    assert not (tmp_path / "move_table_base").exists()


def test_bake_cmd_adds_gpu_jpeg_encode_only_when_enabled(tmp_path: Path) -> None:
    sp = _sp(tmp_path)
    cfg = copy.deepcopy(CFG)
    cfg["bake"]["gpu_jpeg_encode"] = False
    assert "--gpu-jpeg-encode" not in pipe.bake_cmd(cfg, sp, _skill("rotate_table_base"), CONFIG, "h")
    cfg["bake"]["gpu_jpeg_encode"] = True
    assert pipe.bake_cmd(cfg, sp, _skill("rotate_table_base"), CONFIG, "h")[-1] == "--gpu-jpeg-encode"


def test_smoke_frames_passes_prefix_limit_to_decode(tmp_path: Path) -> None:
    cmd = pipe.decode_cmd(CFG, _sp(tmp_path), CFG["skills"][0], smoke_frames=300)
    assert _flag(cmd, "--max-frames-per-mp4") == "300"
    assert "--max-frames-per-mp4" not in pipe.decode_cmd(CFG, _sp(tmp_path), CFG["skills"][0])


def test_smoke_frames_requires_no_upload(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(SystemExit):
        pipe.main(["--config", str(CONFIG), "--work-root", str(tmp_path), "--smoke-frames", "10", "--dry-run"])
    assert pipe.main(
        ["--config", str(CONFIG), "--work-root", str(tmp_path), "--skills", "move_table_base",
         "--smoke-frames", "10", "--no-upload", "--dry-run"]
    ) == 0


def test_bake_cmd_passes_cuda_worker_split(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["bake"]["jobs"], cfg["bake"]["cuda_workers"] = 24, 10
    cmd = pipe.bake_cmd(cfg, _sp(tmp_path), _skill("rotate_table_base"), CONFIG, "h")
    assert (_flag(cmd, "--jobs"), _flag(cmd, "--cuda-workers")) == ("24", "10")


def test_mps_env_lives_under_work_root(tmp_path: Path) -> None:
    env = pipe.mps_env(tmp_path)
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "mps" / "pipe")
    assert env["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "mps" / "log")


def test_dry_run_with_mps_does_not_start_daemon(tmp_path: Path, capsys) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["gpu"]["mps"] = True
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert pipe.main(
        ["--config", str(cfg_path), "--work-root", str(tmp_path / "w"), "--skills", "move_table_base", "--dry-run"]
    ) == 0
    assert "(dry-run) start CUDA MPS" in capsys.readouterr().out
    assert not (tmp_path / "w").exists()
