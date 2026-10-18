"""OBB → root_link 把持ポーズ変換の unit test (Issue #123)。

中心は **往復検証**: 平面上の既知の長方形をカメラへ順投影 → 得た OBB を
provider に通す → 元の姿勢が復元できるか。
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.lower_policy.obb_grasp import (
    CameraExtrinsic,
    ObbGraspPoseProvider,
    PinholeCamera,
    SupportPlane,
    intersect_ray_plane,
    make_obb,
)

# dataset 実測の把持ポーズ近傍に脚を置く
LEG_CENTER = np.array([0.366, -0.163, 0.101])
LEG_LEN = 0.40
LEG_WIDTH = 0.045


def leg_corners(center=LEG_CENTER, yaw=0.30, length=LEG_LEN, width=LEG_WIDTH) -> np.ndarray:
    """平面上に置いた脚の 4 隅 (root_link 座標)。"""
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])       # 長軸
    v = np.array([-np.sin(yaw), np.cos(yaw), 0.0])      # 短軸
    hl, hw = length / 2, width / 2
    return np.array([
        center - u * hl - v * hw,
        center + u * hl - v * hw,
        center + u * hl + v * hw,
        center - u * hl + v * hw,
    ])


def project(provider: ObbGraspPoseProvider, pts_root: np.ndarray,
            waist=None) -> np.ndarray:
    """root_link の 3D 点 → 正規化画素 (順投影、往復検証用)。"""
    origin, R = provider.camera_pose_in_root(waist)
    cam = (np.asarray(pts_root) - origin) @ R          # R^T @ (p - o) を行方向で
    assert np.all(cam[:, 2] > 0), "カメラ後方に落ちている"
    u = provider.camera.fx * cam[:, 0] / cam[:, 2] + provider.camera.cx
    v = provider.camera.fy * cam[:, 1] / cam[:, 2] + provider.camera.cy
    return np.stack([u / provider.camera.width, v / provider.camera.height], axis=1)


def leg_corners_len(length: float, center=LEG_CENTER, yaw=0.30) -> np.ndarray:
    """長さを変えた脚 (把持比率の逆算を試すため)。"""
    return leg_corners(center=center, yaw=yaw, length=length)


@pytest.fixture
def provider() -> ObbGraspPoseProvider:
    return ObbGraspPoseProvider()


# ---------------------------------------------------------------- 幾何の素

def test_ray_plane_intersection_basic():
    plane = SupportPlane(normal=(0, 0, 1), offset=0.1)
    hit = intersect_ray_plane(np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, -1.0]), plane)
    np.testing.assert_allclose(hit, [0.0, 0.0, 0.1])


def test_ray_parallel_or_backwards_returns_none():
    plane = SupportPlane(normal=(0, 0, 1), offset=0.1)
    assert intersect_ray_plane(np.array([0., 0., .5]), np.array([1., 0., 0.]), plane) is None
    assert intersect_ray_plane(np.array([0., 0., .5]), np.array([0., 0., 1.]), plane) is None


def test_plane_normal_is_normalized_and_validated():
    assert np.isclose(np.linalg.norm(SupportPlane(normal=(0, 0, 5)).normal), 1.0)
    with pytest.raises(ValueError, match="non-zero"):
        SupportPlane(normal=(0, 0, 0))


def test_camera_rays_point_forward_in_optical_frame():
    cam = PinholeCamera()
    r = cam.ray(np.array([[320.0, 240.0], [0.0, 0.0], [639.0, 479.0]]))
    assert np.all(r[:, 2] > 0)
    np.testing.assert_allclose(r[0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1.0)


def test_camera_rejects_bad_intrinsics_and_pixels():
    with pytest.raises(ValueError, match="fx/fy"):
        PinholeCamera(fx=0.0)
    with pytest.raises(ValueError, match="width/height"):
        PinholeCamera(width=0)
    with pytest.raises(ValueError, match=r"\(\.\.\.,2\)"):
        PinholeCamera().ray(np.zeros((3, 3)))


def test_camera_looks_forward_and_downward_at_the_support_plane(provider):
    """外部パラメータの向きの不変条件。

    どんな校正値でも head カメラは「前方かつ下向き」で、視線中心は
    腕の作業域内で支持平面に当たるはず。ここが崩れたら符号か
    光学フレーム規約の取り違え。
    """
    origin, R = provider.camera_pose_in_root()
    axis = R @ np.array([0.0, 0.0, 1.0])
    assert axis[0] > 0.3, "前方を向いていない"
    assert axis[2] < -0.3, "下方を向いていない"
    assert origin[2] > 0.3, "カメラが胴体上部にない"
    hit = intersect_ray_plane(origin, axis, provider.plane)
    assert hit is not None
    assert 0.2 < hit[0] < 0.6, f"視線中心が作業域外: {hit}"
    assert abs(hit[1]) < 0.4


def test_default_extrinsic_is_a_placeholder_not_a_calibration(provider):
    """default 外部パラメータでは把持領域が画角中心から外れることを記録する。

    default は URDF の `d435_joint` だが、docs/dataset/depth.md のとおり実機 head は
    D435 系ではない別 stereo モジュールで、外部パラメータも未校正
    (docs/flip_table/2026-07-10_act_policy_root_cause_report.md)。
    実測では視線中心が平面に当たる点は [0.394, +0.018] で、dataset の把持点中央値
    [0.366, -0.163] とは約 0.18 m ずれる。**実機投入前に校正値で上書きすること。**
    この test は default をそのまま本番に持ち込む事故を防ぐための覚書。
    """
    origin, R = provider.camera_pose_in_root()
    hit = intersect_ray_plane(origin, R @ np.array([0.0, 0.0, 1.0]), provider.plane)
    offset = float(np.linalg.norm(hit[:2] - LEG_CENTER[:2]))
    assert offset > 0.1, (
        "default 外部パラメータが把持領域に合ってしまった。校正値を入れたなら "
        "本 test は削除し、上の不変条件 test だけ残すこと"
    )


# ---------------------------------------------------------------- 往復検証

@pytest.mark.parametrize("yaw", [-0.4, -0.1, 0.0, 0.3, 0.9])
def test_roundtrip_recovers_position_and_yaw(provider, yaw):
    corners = leg_corners(yaw=yaw)
    px = project(provider, corners)
    obs = {"cleaned": [make_obb(px)]}
    pose = provider.grasp_pose(obs)
    assert pose is not None
    np.testing.assert_allclose(pose.position, LEG_CENTER, atol=1e-6)
    # yaw は長軸方向。反対向き (pi ずれ) は端の選び方で決まるので絶対値で比較
    got = pose.rpy[2]
    assert min(abs(got - yaw), abs(abs(got - yaw) - np.pi)) < 1e-6


def test_roundtrip_is_invariant_to_vertex_order(provider):
    """頂点の並び順が変わっても同じ把持ポーズになること。"""
    corners = leg_corners(yaw=0.3)
    px = project(provider, corners)
    base = provider.grasp_pose({"cleaned": [make_obb(px)]})
    for shift in (1, 2, 3):
        rolled = np.roll(px, shift, axis=0)
        got = provider.grasp_pose({"cleaned": [make_obb(rolled)]})
        np.testing.assert_allclose(got.position, base.position, atol=1e-9)
    reversed_ = provider.grasp_pose({"cleaned": [make_obb(px[::-1])]})
    np.testing.assert_allclose(reversed_.position, base.position, atol=1e-9)


def test_roundtrip_with_waist_rotation(provider):
    """腰が回っていてもカメラ変換に反映され、同じ 3D 点を復元すること。"""
    waist = np.array([0.12, 0.0, 0.0])
    corners = leg_corners()
    px = project(provider, corners, waist)

    class JS:
        position = np.concatenate([np.zeros(12), waist, np.zeros(14)])
        name = ()

    pose = provider.grasp_pose({"cleaned": [make_obb(px)], "joint_state": JS()})
    np.testing.assert_allclose(pose.position, LEG_CENTER, atol=1e-6)


def test_ignoring_waist_when_it_is_nonzero_causes_error(provider):
    """腰を無視すると誤差が出ることを固定 (waist 引数が飾りでないことの確認)。"""
    waist = np.array([0.15, 0.0, 0.0])
    px = project(provider, leg_corners(), waist)
    without = provider.grasp_pose({"cleaned": [make_obb(px)]})  # joint_state 無し
    assert np.linalg.norm(without.position - LEG_CENTER) > 0.01


def test_grasp_fraction_moves_along_the_long_axis():
    corners = leg_corners(yaw=0.0)
    near = ObbGraspPoseProvider(grasp_fraction=0.0)
    far = ObbGraspPoseProvider(grasp_fraction=1.0)
    px = project(near, corners)
    p0 = near.grasp_pose({"cleaned": [make_obb(px)]})
    p1 = far.grasp_pose({"cleaned": [make_obb(px)]})
    assert np.isclose(p0.position_error(p1), LEG_LEN, atol=1e-6)
    # 0.0 側のほうがロボットに近い
    assert np.linalg.norm(p0.position[:2]) < np.linalg.norm(p1.position[:2])


def test_grasp_fraction_end_is_deterministic_under_yaw_flip():
    """脚の向きが 180 度反転しても、基準端は「ロボットに近い側」で一定。"""
    prov = ObbGraspPoseProvider(grasp_fraction=0.0)
    a = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners(yaw=0.3)))]})
    b = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners(yaw=0.3 + np.pi)))]})
    np.testing.assert_allclose(a.position, b.position, atol=1e-6)


def test_roll_and_pitch_are_the_configured_constants():
    prov = ObbGraspPoseProvider(approach_roll=0.187, approach_pitch=0.872)
    pose = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners()))]})
    assert np.isclose(pose.rpy[0], 0.187)
    assert np.isclose(pose.rpy[1], 0.872)


def test_position_z_is_the_plane_height():
    prov = ObbGraspPoseProvider(plane=SupportPlane(offset=0.135))
    pose = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners(
        center=np.array([0.366, -0.163, 0.135]))))]})
    assert np.isclose(pose.position[2], 0.135, atol=1e-9)


def test_grasp_height_offset_does_not_change_projected_metric_scale():
    """Tool-frame height is applied after ray/plane intersection."""
    surface_z = 0.0
    tool_offset_z = 0.101
    center = np.array([0.366, -0.163, surface_z])
    prov = ObbGraspPoseProvider(
        plane=SupportPlane(offset=surface_z),
        grasp_height_offset_m=tool_offset_z,
    )
    corners = leg_corners(center=center)
    pose = prov.grasp_pose({"cleaned": [make_obb(project(prov, corners))]})
    assert pose is not None
    assert np.isclose(prov.last_leg_length, LEG_LEN, atol=1e-6)
    assert np.isclose(pose.position[2], surface_z + tool_offset_z, atol=1e-9)
    np.testing.assert_allclose(prov.last_corners_root[:, 2], surface_z, atol=1e-9)


# ---------------------------------------------------------------- 選別

def test_returns_none_without_detections(provider):
    assert provider.grasp_pose({}) is None
    assert provider.grasp_pose({"cleaned": []}) is None


def test_filters_by_class_and_confidence(provider):
    px = project(provider, leg_corners())
    assert provider.grasp_pose({"cleaned": [make_obb(px, class_name="hole")]}) is None
    assert provider.grasp_pose({"cleaned": [make_obb(px, confidence=0.2)]}) is None
    assert provider.grasp_pose({"cleaned": [make_obb(px, confidence=0.9)]}) is not None


def test_search_mask_rejects_legs_outside_the_expected_area():
    prov = ObbGraspPoseProvider(search_center=(0.366, -0.163), search_radius=0.15)
    inside = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners()))]})
    assert inside is not None
    far = leg_corners(center=np.array([0.366, 0.35, 0.101]))
    assert prov.grasp_pose({"cleaned": [make_obb(project(prov, far))]}) is None


def test_picks_the_highest_confidence_candidate():
    prov = ObbGraspPoseProvider()
    a = leg_corners(center=np.array([0.33, -0.20, 0.101]))
    b = leg_corners(center=np.array([0.40, -0.10, 0.101]))
    dets = [make_obb(project(prov, a), confidence=0.6),
            make_obb(project(prov, b), confidence=0.95)]
    pose = prov.grasp_pose({"cleaned": dets})
    np.testing.assert_allclose(pose.position, [0.40, -0.10, 0.101], atol=1e-6)


def test_diagnostics_belong_to_returned_candidate_not_last_rejected_obb():
    """A later bad detection must not make a valid returned pose look rejected."""
    prov = ObbGraspPoseProvider(
        handover_clearance_m=0.187,
        handover_margin_m=0.03,
    )
    accepted = leg_corners_len(0.40)
    rejected = leg_corners_len(0.15, center=np.array([0.45, 0.10, 0.101]))
    pose = prov.grasp_pose({
        "cleaned": [
            make_obb(project(prov, accepted), confidence=0.95),
            make_obb(project(prov, rejected), confidence=0.80),
        ]
    })

    assert pose is not None
    assert prov.last_reject_reason is None
    assert prov.last_reachable
    assert prov.last_leg_length == pytest.approx(0.40, abs=1e-6)
    assert prov.last_grasp_fraction == pytest.approx(1.0 - 0.217 / 0.40)


def test_exposes_last_corners_for_diagnostics(provider):
    px = project(provider, leg_corners())
    provider.grasp_pose({"cleaned": [make_obb(px)]})
    assert provider.last_corners_root.shape == (4, 3)
    np.testing.assert_allclose(provider.last_corners_root[:, 2], 0.101, atol=1e-9)


def test_rejects_bad_vertex_shape(provider):
    with pytest.raises(ValueError, match=r"\(4, 2\)"):
        provider.project_to_plane(np.zeros((3, 2)))
    with pytest.raises(ValueError, match=r"\(4, 3\)"):
        provider.pose_from_corners(np.zeros((4, 2)))


def test_rejects_bad_waist_shape(provider):
    with pytest.raises(ValueError, match="shape"):
        provider.camera_pose_in_root(np.zeros(2))


# ---------------------------------------------------------------- config

def test_from_config_defaults_and_overrides():
    prov = ObbGraspPoseProvider.from_config(None)
    assert prov.leg_class == "leg"
    prov = ObbGraspPoseProvider.from_config({
        "camera": {"fx": 500.0, "cx": 310.0},
        "extrinsic": {"xyz": [0.05, 0.0, 0.43], "rpy": [0.0, 0.85, 0.0]},
        "support_plane": {"offset": 0.12},
        "grasp_height_offset_m": 0.08,
        "leg_class": "table_leg",
        "grasp_fraction": 0.25,
        "search_center": [0.36, -0.16],
        "search_radius": 0.2,
    })
    assert prov.camera.fx == 500.0 and prov.camera.cx == 310.0
    assert prov.camera.fy == PinholeCamera().fy      # 未指定は default
    assert prov.plane.offset == 0.12
    assert prov.grasp_height_offset_m == 0.08
    assert prov.leg_class == "table_leg"
    assert prov.grasp_fraction == 0.25
    np.testing.assert_allclose(prov.search_center, [0.36, -0.16])


@pytest.mark.parametrize("cfg,match", [
    ({"bogus": 1}, "unknown key"),
    ({"camera": {"nope": 1}}, "unknown key"),
    ({"extrinsic": {"nope": 1}}, "unknown key"),
    ({"support_plane": {"nope": 1}}, "unknown key"),
    ({"extrinsic": {"xyz": [1, 2]}}, "3 values"),
    ({"support_plane": {"normal": [1, 2]}}, "3 values"),
    ({"grasp_height_offset_m": float("nan")}, "finite"),
    ({"grasp_fraction": 1.5}, r"\[0, 1\]"),
    ({"search_center": [1, 2, 3]}, "2 values"),
])
def test_from_config_rejects_invalid(cfg, match):
    with pytest.raises(ValueError, match=match):
        ObbGraspPoseProvider.from_config(cfg)


def test_satisfies_grasp_pose_provider_protocol():
    from inference.desktop.lower_policy.grasp_pose import GraspPoseProvider

    assert isinstance(ObbGraspPoseProvider(), GraspPoseProvider)


# ---------------------------------------------------------------- 把持比率と持ち替えの整合

def _shipped_cfg() -> dict:
    import yaml
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "skill_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["skills"]["pick_table_leg"]


def test_handover_clearance_matches_the_stage_offset():
    """出荷 YAML の 2 つの設定値が噛み合っていること。

    左手は右グリッパから `|offset|` 離れた点を掴む。把持点より先にその距離ぶん
    脚が出ていなければ左手は空を掴む。`handover_clearance_m` はこの距離と
    一致していなければならないが、両者は別 section にあり黙ってずれうる。
    この test がその結合を固定する。
    """
    cfg = _shipped_cfg()
    stage = next(s for s in cfg["stages"] if s["name"] == "approach_handover")
    offset = np.linalg.norm(np.array(stage["goal"]["offset"], dtype=float))
    clearance = float(cfg["obb_grasp"]["handover_clearance_m"])
    assert clearance == pytest.approx(offset, abs=0.005), (
        f"handover_clearance_m={clearance} が持ち替え offset の大きさ {offset:.4f} と"
        " ずれている。左手が脚の先の空を掴む"
    )


def test_grasp_fraction_is_derived_so_protrusion_is_constant():
    """脚長が変わっても、把持点より先に出る長さは一定になること。"""
    prov = ObbGraspPoseProvider(handover_clearance_m=0.187, handover_margin_m=0.03)
    for length in (0.30, 0.36, 0.45, 0.60):
        prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners_len(length)))]})
        protrusion = (1.0 - prov.last_grasp_fraction) * prov.last_leg_length
        assert protrusion == pytest.approx(0.217, abs=1e-6), f"脚長 {length}"


def test_short_leg_is_rejected_before_grasping():
    """持ち替え距離を確保できない脚は掴む前に捨てる。"""
    prov = ObbGraspPoseProvider(handover_clearance_m=0.187, handover_margin_m=0.03)
    ok = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners_len(0.36)))]})
    assert ok is not None
    short = prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners_len(0.15)))]})
    assert short is None
    assert prov.last_reject_reason == "leg_too_short"


def test_near_end_margin_rejects_a_barely_long_enough_projection():
    """Do not accept a projection that would put the right hand at the leg end."""
    prov = ObbGraspPoseProvider(
        handover_clearance_m=0.187,
        handover_margin_m=0.03,
        grasp_near_end_margin_m=0.04,
    )
    too_short = prov.grasp_pose({
        "cleaned": [make_obb(project(prov, leg_corners_len(0.24)))]
    })
    assert too_short is None
    assert prov.last_reject_reason == "leg_too_short"

    adequate = prov.grasp_pose({
        "cleaned": [make_obb(project(prov, leg_corners_len(0.30)))]
    })
    assert adequate is not None
    assert prov.last_grasp_fraction == pytest.approx(1.0 - 0.217 / 0.30)


def test_fixed_fraction_is_used_when_clearance_is_not_configured():
    prov = ObbGraspPoseProvider(grasp_fraction=0.25)
    prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners_len(0.36)))]})
    assert prov.last_grasp_fraction == pytest.approx(0.25)
    assert prov.last_reject_reason is None


def test_leg_length_is_measured_from_the_detection():
    prov = ObbGraspPoseProvider()
    for length in (0.28, 0.36, 0.50):
        prov.grasp_pose({"cleaned": [make_obb(project(prov, leg_corners_len(length)))]})
        assert prov.last_leg_length == pytest.approx(length, abs=1e-6)


def test_clearance_and_margin_validation():
    with pytest.raises(ValueError, match="handover_clearance_m"):
        ObbGraspPoseProvider(handover_clearance_m=0.0)
    with pytest.raises(ValueError, match="handover_margin_m"):
        ObbGraspPoseProvider(handover_margin_m=-0.01)
    with pytest.raises(ValueError, match="grasp_near_end_margin_m"):
        ObbGraspPoseProvider(grasp_near_end_margin_m=-0.01)


def test_from_config_reads_clearance():
    prov = ObbGraspPoseProvider.from_config(
        {"handover_clearance_m": 0.20, "handover_margin_m": 0.05}
    )
    assert prov.handover_clearance_m == 0.20
    assert prov.handover_margin_m == 0.05
    assert ObbGraspPoseProvider.from_config({}).handover_clearance_m is None
