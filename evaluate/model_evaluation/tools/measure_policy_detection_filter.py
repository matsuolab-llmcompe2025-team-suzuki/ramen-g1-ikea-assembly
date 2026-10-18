"""policy 用の検出の filter (`inference/desktop/perception/policy_filter.py`) の値を決めるための計測 (Issue #141 束 1-9)。

policy (overlay・memory・progress monitor) は学習時に「画像と同じ frame の生の検出 (conf ≥ 0.30)」で描いた
画像を見ている。filter の案ごとに、policy が見る枠が生の検出とどれだけ違うかを、教師の区間と実機の場面で比べる。
上位 Planner 用の cleaner (今の本番で policy にも渡している、tick t に frame t−2 の枠を出す) も参考に並べる。

env が違うので 2 段に分かれる:

1. detect (runtime env、ultralytics): 実機の head_left の動画に YOLO (`policy_config.yaml` の `yolo.ckpt_ref`、conf 0.25 =
   本番と同じ) を掛け、frame ごとの検出を `<out>/<動画の親の run>.npz` に保存する

       pixi run -e runtime python -m evaluate.model_evaluation.tools.measure_policy_detection_filter \\
           detect --videos <head_left.mp4> ... --out <dir>

2. measure (model/ramen_ori env、pyarrow): 教師の区間の生の検出 (統合 cache の OBB、`obb_yolo/<hash>`) と 1 の npz に
   案を掛けて比べ、`report.md` / `metrics.json` / 並べた画像 (`preview_<案>.png`) を書く

       PYTHONPATH=. pixi run --manifest-path model/ramen_ori/pixi.toml python -m \\
           evaluate.model_evaluation.tools.measure_policy_detection_filter measure \\
           --teacher rotate=<dataset snapshot> ... --real <1 の out> --out <dir>

比べる量 (区間の先頭 3 frame は filter が溜まるまでなので除く):

| 量 | 意味 |
|---|---|
| 個数が変わる frame | どれかの class で、描く枠の数が生の検出 (conf ≥ 0.30) と違う frame の割合 |
| 消える枠 / 増える枠 | 生の枠の数に対する、描かれない枠 / 生に無いのに描かれる枠の割合 (class ごと) |
| 見えなくなる frame | 1 個しか無い class で、生では見えているのに描く枠が 0 になる frame の割合 |
| ずれ | 1 個しか無い class の、同じ frame の生の枠との中心の距離 [px、640×480] |
| ブレ | 1 個しか無い class の中心の 2 階差分 `|c_t − 2c_{t−1} + c_{t−2}|` の median [px] |
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml

from inference.desktop.perception.cleaner import load_cleanup_config
from inference.desktop.perception.policy_filter import (
    PolicyDetectionFilter,
    load_policy_filter_config,
)
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_CONFIG = REPO_ROOT / "inference/desktop/lower_policy/configs/policy_config.yaml"
CAMERA = "observation.images.cam_0"   # head_left (overlay を描くカメラ)
OBB_HASH = "ce877e8237be"             # 統合 cache の OBB (YOLO v11b @8221d0a、上位 32 個)
IMAGE_WH = np.array([640.0, 480.0])
CONF_DRAW = 0.30                      # overlay の閾値 (学習と同じ)
CONF_YOLO = 0.25                      # 本番の YoloObbPerception の conf (統合 cache の OBB は 0.15 以上を持つ)
SKIP_HEAD_FRAMES = 3
# YOLO v11b の class 番号 (data/yolo_obb/configs/dataset_v11b.yaml、memory_features と同じ)
CLASS_NAMES = {0: "workspace", 1: "leg", 2: "leg_tip", 3: "hole", 4: "table_top", 5: "hand_right", 6: "hand_left"}
SINGLE_CLASSES = ("workspace", "table_top", "hand_left", "hand_right")
ALL_CLASSES = SINGLE_CLASSES + ("leg", "leg_tip", "hole")

# 案: None は特別扱い (raw = 生の検出、planner_cleaner = 今の cleaner)。dict は cleanup.yaml の policy 節からの差分
VARIANTS: dict[str, dict | None] = {
    "raw": None,                       # 生の検出 (学習の画像と同じ)
    "planner_cleaner": None,           # 今の Planner 用 cleaner (tick t に frame t−2 の枠)
    "planner_fill": None,              # 同上 + 上限の扱いを fill_by_conf に (束 1-9 F-3)
    "policy_default": {},              # policy 節の既定 (persist 1 / continue 1 / keep_continuing / ならしなし)
    "continue2": {"continue_frames": 2},
    "persist0": {"persist_frames": 0},
    "fill": {"over_cap": "fill_by_conf"},
    "fill_continue2": {"over_cap": "fill_by_conf", "continue_frames": 2},
    "fill_persist0": {"over_cap": "fill_by_conf", "persist_frames": 0},
    "fill_continue2_median3": {"over_cap": "fill_by_conf", "continue_frames": 2, "smoothing": "median3_past"},
}
PREVIEW_VARIANTS = ("fill_continue2", "policy_default", "planner_fill")

Episode = list[list[OBBDetection]]   # frame ごとの検出


# ---------------------------------------------------------------------------
# 1. detect (runtime env)
# ---------------------------------------------------------------------------


def detect(videos: list[Path], out_dir: Path, device: str | None) -> None:
    import cv2  # lazy: env-isolated (opencv は runtime env)

    from inference.desktop.perception.yolo_obb import YoloObbPerception, resolve_yolo_ckpt_ref

    yolo = yaml.safe_load(POLICY_CONFIG.read_text())["yolo"]
    ckpt_ref = yolo["ckpt_ref"]
    perception = YoloObbPerception(
        resolve_yolo_ckpt_ref(ckpt_ref, yolo["ckpt_file"]), device=device
    )
    # 焼き込み (統合 cache の manifest) は imgsz 800。本番と同じく imgsz を渡さないときに使われる値を残す
    print(f"[detect] {ckpt_ref} imgsz={perception._model.overrides.get('imgsz')} conf={perception._conf}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for video in videos:
        run_id = video.parts[video.parts.index("media") - 1] if "media" in video.parts else video.stem
        cap = cv2.VideoCapture(str(video))
        rows: list[tuple[int, OBBDetection]] = []
        n_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rows.extend((n_frames, d) for d in perception.predict(frame))
            n_frames += 1
        cap.release()
        np.savez_compressed(
            out_dir / f"{run_id}.npz",
            frame=np.array([f for f, _ in rows], dtype=np.int32),
            class_id=np.array([d.class_id for _, d in rows], dtype=np.int16),
            conf=np.array([d.confidence for _, d in rows], dtype=np.float32),
            verts=np.array([d.verts for _, d in rows], dtype=np.float32).reshape(-1, 4, 2),
            n_frames=n_frames,
            video=str(video),
            ckpt_ref=ckpt_ref,
            class_names=json.dumps(perception.class_names),
        )
        print(f"[detect] {run_id}: {n_frames} frames, {len(rows)} detections -> {out_dir / f'{run_id}.npz'}")


# ---------------------------------------------------------------------------
# 2. measure (model/ramen_ori env)
# ---------------------------------------------------------------------------


def _det(class_id: int, conf: float, verts: np.ndarray) -> OBBDetection:
    return OBBDetection(class_id=int(class_id), class_name=CLASS_NAMES[int(class_id)], confidence=float(conf), verts=verts)


def iter_teacher(snapshot: Path, max_episodes: int | None, seed: int) -> Iterator[tuple[str, Episode]]:
    """教師の区間を 1 本ずつ返す (全部を同時に持たない = メモリの山を作らない)。"""
    import pandas as pd  # lazy: env-isolated (model/ramen_ori env)

    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # lazy: env-isolated (pyarrow / torch)

    episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((snapshot / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    cache = ObbPrecomputedCache(
        root=snapshot / "obb_yolo", ckpt_hash=OBB_HASH, camera_keys=[CAMERA], top_K=None, num_classes=7, fps=30.0
    )
    cache.build_episode_mapping(episodes)
    rows = episodes[["episode_index", "length"]].to_numpy()
    if max_episodes is not None and len(rows) > max_episodes:
        rows = rows[np.sort(np.random.default_rng(seed).choice(len(rows), max_episodes, replace=False))]
    for ep, length in rows:
        arr = cache.episode_arrays(int(ep), int(length), CAMERA)
        verts = arr["verts"].reshape(int(length), -1, 4, 2)
        yield str(int(ep)), [
            [_det(c, s, v) for c, s, v, ok in zip(arr["class_id"][t], arr["conf"][t], verts[t], arr["valid"][t]) if ok]
            for t in range(int(length))
        ]


def load_run(path: Path) -> Episode:
    """detect の npz 1 本 → frame ごとの検出。"""
    z = np.load(path)
    names = {int(k): v for k, v in json.loads(str(z["class_names"])).items()}
    if any(names.get(k) != v for k, v in CLASS_NAMES.items()):
        raise ValueError(f"{path}: YOLO class names {names} differ from {CLASS_NAMES}")
    frames: Episode = [[] for _ in range(int(z["n_frames"]))]
    for f, c, s, v in zip(z["frame"], z["class_id"], z["conf"], z["verts"]):
        frames[int(f)].append(_det(c, s, v))
    return frames


def iter_real(npz_dir: Path) -> Iterator[tuple[str, Episode]]:
    for path in sorted(npz_dir.glob("*.npz")):
        yield path.stem, load_run(path)


def run_variant(name: str, episode: Episode, planner_cfg: dict, policy_cfg: dict) -> Episode:
    """区間 1 本に案を掛け、各 tick に policy が見る枠 (conf ≥ 0.30) を返す。"""
    overrides = VARIANTS[name]
    if name == "raw":
        return [[d for d in frame if d.confidence >= CONF_DRAW] for frame in episode]
    if name in ("planner_cleaner", "planner_fill"):
        cfg = {**planner_cfg, "over_max_mode": "fill_by_conf" if name == "planner_fill" else "keep_continuing"}
        stream = DetectionStream(cfg)
        return [
            [d for d in (stream.push([d for d in frame if d.confidence >= CONF_YOLO]) or []) if d.confidence >= CONF_DRAW]
            for frame in episode
        ]
    f = PolicyDetectionFilter({**policy_cfg, **overrides})
    return [f.push(frame) for frame in episode]


def _counts(frame: list[OBBDetection]) -> dict[str, int]:
    counts = dict.fromkeys(ALL_CLASSES, 0)
    for d in frame:
        counts[d.class_name] += 1
    return counts


def _center_px(frame: list[OBBDetection], name: str) -> np.ndarray | None:
    dets = [d for d in frame if d.class_name == name]
    if not dets:
        return None
    top = max(dets, key=lambda d: d.confidence)
    return top.verts.mean(axis=0) * IMAGE_WH


class Comparison:
    """案 1 つ分の集計。区間ごとに `update` を呼ぶ (区間をまたいで状態は持たない)。"""

    def __init__(self) -> None:
        self.n_frames = 0
        self.changed = 0
        self.raw_boxes = dict.fromkeys(ALL_CLASSES, 0)
        self.removed = dict.fromkeys(ALL_CLASSES, 0)
        self.added = dict.fromkeys(ALL_CLASSES, 0)
        self.visible = dict.fromkeys(SINGLE_CLASSES, 0)
        self.lost = dict.fromkeys(SINGLE_CLASSES, 0)
        self.shift: dict[str, list[float]] = {c: [] for c in SINGLE_CLASSES}
        self.jitter_raw: dict[str, list[float]] = {c: [] for c in SINGLE_CLASSES}
        self.jitter_out: dict[str, list[float]] = {c: [] for c in SINGLE_CLASSES}

    def update(self, raw_ep: Episode, out_ep: Episode) -> None:
        history: dict[str, list] = {c: [] for c in SINGLE_CLASSES}
        for t in range(SKIP_HEAD_FRAMES, len(raw_ep)):
            raw, out = raw_ep[t], out_ep[t]
            rc, oc = _counts(raw), _counts(out)
            self.n_frames += 1
            self.changed += any(rc[c] != oc[c] for c in ALL_CLASSES)
            for c in ALL_CLASSES:
                self.raw_boxes[c] += rc[c]
                self.removed[c] += max(0, rc[c] - oc[c])
                self.added[c] += max(0, oc[c] - rc[c])
            for c in SINGLE_CLASSES:
                r, o = _center_px(raw, c), _center_px(out, c)
                if r is not None:
                    self.visible[c] += 1
                    self.lost[c] += o is None
                if r is not None and o is not None:
                    self.shift[c].append(float(np.linalg.norm(r - o)))
                history[c] = (history[c] + [(r, o)])[-3:]
                if len(history[c]) == 3:
                    for idx, acc in ((0, self.jitter_raw[c]), (1, self.jitter_out[c])):
                        pts = [h[idx] for h in history[c]]
                        if all(p is not None for p in pts):
                            acc.append(float(np.linalg.norm(pts[2] - 2 * pts[1] + pts[0])))

    def result(self) -> dict:
        def pct(a: int, b: int) -> float | None:
            return None if b == 0 else 100.0 * a / b

        def q(values: list[float], p: float) -> float | None:
            return None if not values else float(np.percentile(values, p))

        total_raw = sum(self.raw_boxes.values())
        return {
            "frames": self.n_frames,
            "count_changed_frames_pct": pct(self.changed, self.n_frames),
            "removed_boxes_pct": pct(sum(self.removed.values()), total_raw),
            "added_boxes_pct": pct(sum(self.added.values()), total_raw),
            "removed_boxes_pct_by_class": {c: pct(self.removed[c], self.raw_boxes[c]) for c in ALL_CLASSES},
            "added_boxes_pct_by_class": {c: pct(self.added[c], self.raw_boxes[c]) for c in ALL_CLASSES},
            "lost_frames_pct": {c: pct(self.lost[c], self.visible[c]) for c in SINGLE_CLASSES},
            "shift_px_median": {c: q(self.shift[c], 50) for c in SINGLE_CLASSES},
            "shift_px_p90": {c: q(self.shift[c], 90) for c in SINGLE_CLASSES},
            "jitter_px_median_raw": {c: q(self.jitter_raw[c], 50) for c in SINGLE_CLASSES},
            "jitter_px_median": {c: q(self.jitter_out[c], 50) for c in SINGLE_CLASSES},
            "raw_boxes_by_class": self.raw_boxes,
        }


def measure_source(
    episodes: Iterator[tuple[str, Episode]],
    planner_cfg: dict,
    policy_cfg: dict,
    picks: dict[str, list] | None = None,
) -> dict[str, dict]:
    """区間を 1 本ずつ流して集計する。`picks` を渡すと、枠の数が変わった frame の (区間, t, 差) を集める。"""
    accs = {n: Comparison() for n in VARIANTS}
    for key, ep in episodes:
        outs = {n: run_variant(n, ep, planner_cfg, policy_cfg) for n in VARIANTS}
        for n in VARIANTS:
            accs[n].update(outs["raw"], outs[n])
        if picks is not None:
            for name in PREVIEW_VARIANTS:
                for t in range(SKIP_HEAD_FRAMES, len(ep)):
                    rc, oc = _counts(outs["raw"][t]), _counts(outs[name][t])
                    diff = {c: oc[c] - rc[c] for c in ALL_CLASSES if oc[c] != rc[c]}
                    if diff:
                        picks[name].append((key, t, diff))
        del outs
    return {n: a.result() for n, a in accs.items()}


def _fmt(v: float | None, digits: int = 2) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def write_report(metrics: dict[str, dict[str, dict]], policy_cfg: dict, out_dir: Path) -> None:
    lines = [
        "# policy 用の検出の filter の計測",
        "",
        f"policy 用の既定 (cleanup.yaml の policy 節): `{json.dumps(policy_cfg, ensure_ascii=False)}`",
        "",
        "| 案 | 中身 |",
        "|---|---|",
    ]
    for name, v in VARIANTS.items():
        if name == "raw":
            what = "生の検出 (conf ≥ 0.30、学習の画像と同じ)"
        elif name == "planner_cleaner":
            what = "今の Planner 用 cleaner (tick t に frame t−2 の枠)"
        elif name == "planner_fill":
            what = "Planner 用 cleaner + 上限を conf の高い順で埋める"
        else:
            what = json.dumps(v, ensure_ascii=False) if v else "policy 節の既定のまま"
        lines.append(f"| {name} | {what} |")
    lines.append("")
    for source, by_variant in metrics.items():
        frames = next(iter(by_variant.values()))["frames"]
        lines += [f"## {source} ({frames} frame)", "", "| 案 | 個数が変わる frame % | 消える枠 % | 増える枠 % |", "|---|---:|---:|---:|"]
        lines += [
            f"| {n} | {_fmt(m['count_changed_frames_pct'])} | {_fmt(m['removed_boxes_pct'])} | {_fmt(m['added_boxes_pct'])} |"
            for n, m in by_variant.items()
        ]
        lines += ["", "消える枠 % (class ごと):", "", "| 案 | " + " | ".join(ALL_CLASSES) + " |", "|---|" + "---:|" * len(ALL_CLASSES)]
        lines += [
            f"| {n} | " + " | ".join(_fmt(m["removed_boxes_pct_by_class"][c]) for c in ALL_CLASSES) + " |"
            for n, m in by_variant.items()
        ]
        lines += [
            "",
            "1 個しか無い class: 見えなくなる frame % / ずれ px (median・p90) / ブレ px (median、生 → 案):",
            "",
            "| 案 | " + " | ".join(SINGLE_CLASSES) + " |",
            "|---|" + "---|" * len(SINGLE_CLASSES),
        ]
        lines += [
            f"| {n} | "
            + " | ".join(
                f"{_fmt(m['lost_frames_pct'][c])}% / {_fmt(m['shift_px_median'][c], 1)}・{_fmt(m['shift_px_p90'][c], 1)} / "
                f"{_fmt(m['jitter_px_median_raw'][c], 1)} → {_fmt(m['jitter_px_median'][c], 1)}"
                for c in SINGLE_CLASSES
            )
            + " |"
            for n, m in by_variant.items()
        ]
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _read_frame(video: str, index: int) -> np.ndarray:
    import cv2  # lazy: env-isolated (opencv)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} of {video}")
    return frame


def write_previews(
    picks: dict[str, list], real_dir: Path, planner_cfg: dict, policy_cfg: dict, out_dir: Path, per_variant: int
) -> None:
    """枠の数が変わった frame を並べる (左 = 生、右 = 案)。選んだ frame の run だけ読み直す。"""
    import cv2  # lazy: env-isolated (opencv)

    from inference.desktop.lower_policy.policies.ramen_ori import overlay_obb_on_frame

    for name, found in picks.items():
        if not found:
            continue
        chosen = [found[i] for i in np.linspace(0, len(found) - 1, min(per_variant, len(found))).astype(int)]
        tiles = []
        for run in sorted({run for run, _, _ in chosen}):
            ep = load_run(real_dir / f"{run}.npz")
            video = str(np.load(real_dir / f"{run}.npz")["video"])
            outs = {n: run_variant(n, ep, planner_cfg, policy_cfg) for n in ("raw", name)}
            for pick_run, t, diff in chosen:
                if pick_run != run:
                    continue
                frame = _read_frame(video, t)
                left = overlay_obb_on_frame(frame.copy(), outs["raw"][t])
                right = overlay_obb_on_frame(frame.copy(), outs[name][t])
                label = f"{run} t={t} " + " ".join(f"{c}{d:+d}" for c, d in diff.items())
                for img, tag in ((left, "raw"), (right, name)):
                    cv2.putText(img, f"{tag} | {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                tiles.append(np.hstack([left, right]))
        grid = np.vstack(tiles)
        grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"preview_{name}.png"), grid)
        print(f"[measure] preview_{name}.png: {len(chosen)} of {len(found)} changed frames")


def measure(teacher: dict[str, Path], real_dir: Path | None, out_dir: Path, max_episodes: int | None, per_variant: int) -> None:
    planner_cfg = load_cleanup_config()
    policy_cfg = load_policy_filter_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, dict]] = {}
    for skill, snapshot in teacher.items():
        metrics[f"教師 {skill}"] = measure_source(iter_teacher(snapshot, max_episodes, seed=0), planner_cfg, policy_cfg)
        print(f"[measure] 教師 {skill} done")
    if real_dir is not None:
        picks: dict[str, list] = {n: [] for n in PREVIEW_VARIANTS}
        metrics["実機 09-09"] = measure_source(iter_real(real_dir), planner_cfg, policy_cfg, picks)
        print("[measure] 実機 done")
        write_previews(picks, real_dir, planner_cfg, policy_cfg, out_dir, per_variant)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(metrics, policy_cfg, out_dir)
    print(f"[measure] wrote {out_dir / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--videos", type=Path, nargs="+", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--device", default=None)
    m = sub.add_parser("measure")
    m.add_argument("--teacher", nargs="*", default=[], help="<skill>=<dataset snapshot (meta/ と obb_yolo/ を含む)>")
    m.add_argument("--real", type=Path, default=None, help="detect の out")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--max-episodes", type=int, default=None, help="skill ごとに無作為に選ぶ区間の数 (既定: 全部)")
    m.add_argument("--preview-frames", type=int, default=12)
    args = p.parse_args()
    if args.cmd == "detect":
        detect(args.videos, args.out, args.device)
    else:
        teacher = dict(item.split("=", 1) for item in args.teacher)
        measure({k: Path(v) for k, v in teacher.items()}, args.real, args.out, args.max_episodes, args.preview_frames)


if __name__ == "__main__":
    main()
