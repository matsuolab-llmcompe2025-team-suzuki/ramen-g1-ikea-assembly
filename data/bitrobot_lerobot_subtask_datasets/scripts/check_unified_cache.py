#!/usr/bin/env python3
"""統合 overlay cache の本番前の確認 (Issue #146、Sakura の関門 1 = smoke の出力にかける)。

pipeline が自分で止まる条件 (YOLO の上限超え、枠が 1 つも当たらない、jpg の欠け、HF との size 不一致など)
では拾えない「中身が正しいか」を、本番の焼き込みの前に確かめる。本番の後には使わない (作り直さないため)。

subcommand:
- merge: passthrough で merge した dataset を、元の chunk と LeRobot で読み比べる (各 chunk の先頭・末尾と無作為の
  episode の先頭・中央・末尾の frame で、全カメラの画素と数値が一致するか)。merge の直後に使う
- yolo: skill・class ごとの検出率・1 frame の個数・conf の分布と、1 frame の検出数。生の枠を v0 に描いた一覧画像
  (数字と画像を出すだけで合否は付けない = 常に exit 0。目で見て判断する)
- aug: 焼いた jpg を、焼き込みと同じ手順 (色の変化 → 枠 → 幾何変換、seed も同じ) を同じ device で作り直した基準と
  比べる。head は枠ごとに描かれているかと、枠の外に余計な差が無いか。wrist は幾何変換が掛かっていないか。一覧画像。
  ノイズの乱数は画像のある device で引かれ、CPU ではその後の JPEG / RandomErasing の抽選もずれるので、
  `--device` は焼き込みの worker と同じにする (本番の設定は全 worker が GPU = `--device cuda`)
- roundtrip: smoke の tar を restore.sh で展開したものを、RAMEN-Ori の学習と同じ組み立て
  (`hydra.utils.instantiate(data config)`、jpg が欠けたら止まる) で読み、skill_id を確かめる

merge / yolo / aug は main env (model/subtask_policy_training/pixi.toml)、roundtrip は RAMEN-Ori env (model/ramen_ori/pixi.toml)。
確認の script 自体の誤りを目で切り分けられるよう、合否だけでなく数字と一覧画像を出す。

Usage (repo root から。<work> は pipeline の skill の作業 dir = `<work_root>/<skill>`):

    C=data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml
    S=data/bitrobot_lerobot_subtask_datasets/scripts/check_unified_cache.py
    pixi run --manifest-path model/subtask_policy_training/pixi.toml python $S merge \\
        --merged /nvme/merged/flip_table --chunk /nvme/chunks/task_2_chunk0 --chunk /nvme/chunks/task_2_chunk1 ...
    pixi run --manifest-path model/subtask_policy_training/pixi.toml python $S yolo --config $C \\
        --skill flip_table=/nvme/uc_smoke/flip_table --skill rotate_table_base=/nvme/obb_ref/rotate_table_base \\
        --out /nvme/uc_check
    pixi run --manifest-path model/subtask_policy_training/pixi.toml python $S aug --config $C \\
        --skill flip_table=/nvme/uc_smoke/flip_table --device cuda --out /nvme/uc_check
    pixi run --manifest-path model/ramen_ori/pixi.toml python $S roundtrip \\
        --data-config model/ramen_ori/configs/data/unified_beta.yaml \\
        --sub flip_table=/nvme/uc_smoke/flip_table/src,/nvme/uc_restore/flip_table/overlay --max-frame 300

yolo の `--skill` の dir は `obb_yolo/<hash>/` を持つこと (pipeline の作業 dir、または HF から `obb_yolo/` だけ
落とした dir)。`v0/` もあれば枠を描いた一覧画像を出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image, JpegImagePlugin

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for p in (SCRIPT_DIR, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_unified_cache_pipeline as pipe  # noqa: E402

BAKE_SCRIPT = REPO_ROOT / "model" / "ramen_ori" / "scripts" / "precompute_augmented_frame_cache_ramen_ori.py"
OBB_FPS = 30  # pipeline は焼き込みに --obb-fps を渡さない (既定 30)
THUMB = (320, 240)


def _parse_named(specs: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"{flag} は NAME=VALUE の形で渡す: {spec!r}")
        out[name] = value
    return out


def _v0_frames(cam_dir: Path) -> list[Path]:
    """v0 の jpg (`<cam>/<chunk>/<file>/frame_XXXXXX.jpg`、variant を除く) を path 順に。"""
    return sorted(p for p in cam_dir.glob("*/*/frame_*.jpg") if "_v" not in p.stem)


def _spread(items: list, n: int) -> list:
    """items から n 個を等間隔に (全体から散らして選ぶ)。"""
    if len(items) <= n:
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n)]


def _grid(rows: list[list[np.ndarray]], out: Path) -> None:
    """RGB uint8 の画像の行列を 1 枚の jpg に並べる。"""
    w, h = THUMB
    cols = max(len(r) for r in rows)
    sheet = Image.new("RGB", (w * cols, h * len(rows)), (255, 255, 255))
    for y, row in enumerate(rows):
        for x, img in enumerate(row):
            sheet.paste(Image.fromarray(img).resize(THUMB), (x * w, y * h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def merge_picks(chunk_episode_counts: list[int], n_random: int, seed: int = 0) -> list[tuple[int, int, int]]:
    """比べる episode の (merge 後の番号, chunk の順番, chunk の中の番号)。各 chunk の先頭・末尾と無作為 n_random 本。

    merge は chunk を渡した順に episode を並べ直すので、merge 後の番号は chunk の episode 数の累積で決まる。
    """
    rng = np.random.default_rng(seed)
    picks: list[tuple[int, int, int]] = []
    start = 0
    for ci, n in enumerate(chunk_episode_counts):
        chosen = {0, n - 1} | {int(e) for e in rng.choice(n, size=min(n_random, n), replace=False)}
        picks += [(start + e, ci, e) for e in sorted(chosen)]
        start += n
    return picks


def _pad_tasks(ds) -> None:
    """元の chunk は frame の task_index が元の番号 (例 3) のまま、tasks は 1 行なので、LeRobot の
    `tasks.iloc[task_index]` が範囲外になる。番号の位置まで埋めて読めるようにする (merge 後は 0 に振り直し済み)。"""
    t = ds.meta.tasks
    n = int(t["task_index"].max()) + 1
    names = [f"__unused_task_{i}__" for i in range(n)]
    for name, row in t.iterrows():
        names[int(row["task_index"])] = name
    ds.meta.tasks = pd.DataFrame({"task_index": range(n)}, index=pd.Index(names, name="task"))


def cmd_merge(args: argparse.Namespace) -> int:
    # lazy: main env の lerobot 0.6.0 で読む (RAMEN-Ori env は fork 0.5.1)
    import os  # noqa: PLC0415

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415

    merged = LeRobotDataset("local/merged", root=args.merged, video_backend="pyav")
    chunks = [LeRobotDataset(f"local/{p.name}", root=p, video_backend="pyav") for p in args.chunk]
    for c in chunks:
        _pad_tasks(c)
    cams = merged.meta.video_keys
    numeric = [k for k, v in merged.meta.features.items()
               if v["dtype"] in ("float32", "float64", "int64") and k not in ("index", "episode_index", "task_index")]
    results: dict[str, tuple[bool, str]] = {}
    totals = (merged.meta.total_episodes, merged.meta.total_frames)
    want = (sum(c.meta.total_episodes for c in chunks), sum(c.meta.total_frames for c in chunks))
    results["episode・frame の数"] = (totals == want, f"merge 後 {totals} / chunk の合計 {want}")

    mismatches: list[str] = []
    n_frames = 0
    picks = merge_picks([c.meta.total_episodes for c in chunks], args.random)
    for e, ci, ce in picks:
        m_ep, c_ep = merged.meta.episodes[e], chunks[ci].meta.episodes[ce]
        length = int(m_ep["dataset_to_index"]) - int(m_ep["dataset_from_index"])
        if length != int(c_ep["dataset_to_index"]) - int(c_ep["dataset_from_index"]):
            mismatches.append(f"ep {e}: length")
            continue
        for f in sorted({0, length // 2, length - 1}):
            a = merged[int(m_ep["dataset_from_index"]) + f]
            b = chunks[ci][int(c_ep["dataset_from_index"]) + f]
            bad = [k for k in cams if (a[k] - b[k]).abs().max().item() != 0]
            bad += [k for k in numeric if not torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k]))]
            if bad:
                mismatches.append(f"ep {e} (chunk{ci} ep {ce}) frame {f}: {bad}")
            n_frames += 1
    results["画素・数値が元と一致"] = (
        not mismatches,
        f"episode {len(picks)}、frame {n_frames}、カメラ {len(cams)} + 数値 {len(numeric)}、不一致 {len(mismatches)}",
    )
    # 並び順まで見る: 同じ task の chunk を束ねる使い方では集合が必ず一致してしまい、
    # episode の割り当ての入れ替わりや一部 episode の取りこぼしを見逃す
    got_src = [int(ep["source_task_index"]) for ep in merged.meta.episodes]
    want_src = [int(ep["source_task_index"]) for c in chunks for ep in c.meta.episodes]
    n_diff = sum(g != w for g, w in zip(got_src, want_src)) + abs(len(got_src) - len(want_src))
    results["source_task_index"] = (
        got_src == want_src,
        f"merge 後 {len(got_src)} 件 {sorted(set(got_src))} / chunk を渡した順 {len(want_src)} 件 "
        f"{sorted(set(want_src))}、並びの不一致 {n_diff}",
    )
    prov = json.loads((args.merged / "meta" / "team_ramen_curation.json").read_text(encoding="utf-8"))
    results["video_mode"] = (prov.get("video_mode") == "passthrough", str(prov.get("video_mode")))
    videos = list((args.merged / "videos").rglob("*.mp4"))
    linked = sum(os.stat(p).st_nlink > 1 for p in videos)
    print(f"== {args.merged} (mp4 {len(videos)}、うち hardlink {linked})")
    for k, (ok, s) in results.items():
        print(f"  {'OK' if ok else 'NG'}  {k}: {s}")
    for m in mismatches[:20]:
        print(f"    {m}")
    return 0 if all(ok for ok, _ in results.values()) else 1


# ---------------------------------------------------------------------------
# yolo
# ---------------------------------------------------------------------------


def load_detections(obb_dir: Path, cam: str) -> pd.DataFrame:
    """`<obb_dir>/<cam>/<chunk>/<file>.parquet` を全部読み、1 行 = (frame, 検出枠) の表にする。"""
    files = sorted((obb_dir / cam).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"YOLO の parquet が無い: {obb_dir / cam}")
    tables = []
    for f in files:
        t = pq.read_table(f, columns=["frame_index", "conf", "class_id", "valid"]).to_pandas()
        t["mp4"] = f"{f.parent.name}/{f.stem}"
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def yolo_stats(df: pd.DataFrame, class_names: dict[int, str], overlay_conf: float, top_k: int) -> dict[str, Any]:
    """skill 1 つ分の数字。検出率は overlay に描かれる conf 以上の枠が 1 つ以上ある frame の割合。"""
    key = ["mp4", "frame_index"]
    frames = df.groupby(key).size().index
    det = df[df["valid"]]
    counts = det.groupby(key).size().reindex(frames, fill_value=0)
    drawn = det[det["conf"] >= overlay_conf]
    per_class = {}
    for cid, name in sorted(class_names.items()):
        d = det[det["class_id"] == cid]
        dd = drawn[drawn["class_id"] == cid]
        q = d["conf"].quantile([0.1, 0.5, 0.9]).tolist() if len(d) else [None] * 3
        per_class[name] = {
            "frame_rate": dd[key].drop_duplicates().shape[0] / len(frames),
            "per_frame": len(dd) / len(frames),
            "conf_p10": q[0], "conf_p50": q[1], "conf_p90": q[2],
        }
    return {
        "frames": int(len(frames)),
        "dets_per_frame_mean": float(counts.mean()),
        "dets_per_frame_max": int(counts.max()),
        "frames_without_det": int((counts == 0).sum()),
        "top_k": top_k,
        "per_class": per_class,
    }


def yolo_markdown(stats: dict[str, dict[str, Any]]) -> str:
    names = list(stats)
    lines = ["| skill | frame | 検出数 / frame (平均 / 最大、上限) | 検出 0 の frame |", "|---|---:|---:|---:|"]
    for n in names:
        s = stats[n]
        lines.append(
            f"| {n} | {s['frames']:,} | {s['dets_per_frame_mean']:.1f} / {s['dets_per_frame_max']} ({s['top_k']}) "
            f"| {s['frames_without_det']:,} |"
        )
    classes = list(next(iter(stats.values()))["per_class"])
    lines += ["", "class ごと: 検出率 (overlay の閾値以上が 1 つ以上ある frame) / 1 frame の個数 / conf の p10・p50・p90", ""]
    lines.append("| class | " + " | ".join(names) + " |")
    lines.append("|---|" + "---:|" * len(names))
    for c in classes:
        cells = []
        for n in names:
            pc = stats[n]["per_class"][c]
            conf = "—" if pc["conf_p50"] is None else f"{pc['conf_p10']:.2f}・{pc['conf_p50']:.2f}・{pc['conf_p90']:.2f}"
            cells.append(f"{pc['frame_rate']:.0%} / {pc['per_frame']:.2f} / {conf}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def draw_raw_boxes(v0_path: Path, obb_dir: Path, cam: str, class_names: dict[int, str]) -> np.ndarray:
    """v0 に YOLO の生の枠 (保存されている全件) を class の色と conf の値つきで描く。RGB uint8。"""
    import cv2  # lazy: main env にしか無い

    from model.ramen_ori.overlay import CLASS_COLORS_BGR

    img = cv2.imread(str(v0_path))
    h, w = img.shape[:2]
    mp4 = obb_dir / cam / v0_path.parent.parent.name / f"{v0_path.parent.name}.parquet"
    frame = int(v0_path.stem.split("_")[1])
    rows = pq.read_table(mp4, filters=[("frame_index", "=", frame), ("valid", "=", True)]).to_pylist()
    for r in rows:
        pts = (np.asarray(r["verts"], dtype=np.float32).reshape(4, 2) * [w, h]).astype(np.int32)
        color = CLASS_COLORS_BGR.get(int(r["class_id"]), (255, 255, 255))
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=2)
        label = f"{class_names.get(int(r['class_id']), r['class_id'])} {r['conf']:.2f}"
        cv2.putText(img, label, tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(img, f"{v0_path.parent.parent.name}/{v0_path.parent.name} #{frame}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img[..., ::-1].copy()


def cmd_yolo(args: argparse.Namespace) -> int:
    cfg = pipe.load_config(args.config)
    obb_hash = pipe.yolo_cache_hash(cfg)
    cam = cfg["cameras"]["head"]
    stats = {}
    for name, d in _parse_named(args.skill, "--skill").items():
        obb_dir = Path(d) / "obb_yolo" / obb_hash
        manifest = json.loads((obb_dir / "manifest.json").read_text(encoding="utf-8"))
        class_names = {int(k): v for k, v in manifest["yolo_class_names"].items()}
        stats[name] = yolo_stats(load_detections(obb_dir, cam), class_names, cfg["bake"]["overlay_conf"], cfg["yolo"]["top_k"])
        v0_cam = Path(d) / "v0" / cam
        if v0_cam.is_dir():
            picks = _spread(_v0_frames(v0_cam), args.samples)
            imgs = [draw_raw_boxes(p, obb_dir, cam, class_names) for p in picks]
            _grid([imgs[i : i + 5] for i in range(0, len(imgs), 5)], args.out / f"yolo_{name}.jpg")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "yolo.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    md = yolo_markdown(stats)
    (args.out / "yolo.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    return 0


# ---------------------------------------------------------------------------
# aug
# ---------------------------------------------------------------------------


def _load_bake_module():
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("precompute_augmented_frame_cache_ramen_ori", BAKE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_jpg(path: Path) -> tuple[torch.Tensor, int]:
    """(3, H, W) float [0,1] と JPEG の色差 (PIL の get_sampling: 0 = 4:4:4、2 = 4:2:0)。"""
    with Image.open(path) as im:
        sampling = JpegImagePlugin.get_sampling(im)
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0, sampling


def _to_u8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().sum(0)


def _chmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().amax(0)


def head_overlay_metrics(
    got: torch.Tensor,
    base: torch.Tensor,
    params,
    aug_pipe,
    per_frame: dict[str, Any],
    overlay_conf: float,
    thickness: int,
    line_diff: float,
    spurious_diff: float,
) -> dict[str, Any]:
    """焼いた head の overlay 画像 1 枚を、同じ手順で作り直した基準と比べる。

    ref_on = 幾何変換(枠(色の変化後))、ref_off = 幾何変換(色の変化後)。枠 k だけを描いた画像と ref_off の差が
    line_diff を超える画素を「枠 k の線」とし、そこで got が ref_off より ref_on に近い画素の割合 (presence) を測る。
    線の外 (2 px 広げた外側) で got と ref_off の差が spurious_diff を超える画素の割合が spurious。
    """
    from model.ramen_ori.overlay import draw_overlay_on_tensor  # noqa: PLC0415

    def draw(valid: np.ndarray) -> torch.Tensor:
        return draw_overlay_on_tensor(
            base, verts=per_frame["verts"], conf=per_frame["conf"], class_id=per_frame["class_id"],
            valid=valid, conf_threshold=overlay_conf, line_thickness=thickness,
        )

    valid = np.asarray(per_frame["valid"], dtype=bool)
    ref_off = aug_pipe.apply_geometric(base, params)
    ref_on = aug_pipe.apply_geometric(draw(valid), params)
    closer_on = _l1(got, ref_on) < _l1(got, ref_off)
    drawable = np.flatnonzero(valid & (np.asarray(per_frame["conf"], dtype=np.float32) >= overlay_conf))
    presence = []
    for k in drawable:
        only_k = np.zeros_like(valid)
        only_k[k] = True
        mask = _chmax(aug_pipe.apply_geometric(draw(only_k), params), ref_off) > line_diff
        if mask.any():  # 幾何変換で画像の外に出た枠は数えない
            presence.append(float(closer_on[mask].float().mean()))
    lines = (_chmax(ref_on, ref_off) > line_diff).float()[None, None]
    near = torch.nn.functional.max_pool2d(lines, 5, stride=1, padding=2)[0, 0].bool()
    outside = ~near
    spurious = float((_chmax(got, ref_off)[outside] > spurious_diff).float().mean()) if outside.any() else 0.0
    return {
        "drawable": int(len(drawable)),
        "presence": presence,
        "spurious": spurious,
        "mean_diff": float((got - ref_on).abs().mean()),
    }


def check_all_mp4_dirs(
    sp, *, kinds: tuple[str, ...], head: str, wrists: list[str], n_variants: int,
    want_sampling: int, bake, r: dict[str, Any]
) -> None:
    """全 mp4 dir で jpg の枚数と完了マーカーを見る (画素の比較は標本なので、取りこぼしはここで全件拾う)。"""
    targets = [(k, head) for k in kinds if k != "common"] + [("common", c) for c in wrists if "common" in kinds]
    for kind, cam in targets:
        for v0_dir in sorted(p for p in (sp.v0 / cam).glob("*/*") if p.is_dir()):
            n_v0 = len([p for p in v0_dir.glob("frame_*.jpg") if "_v" not in p.stem])
            out_dir = sp.v2 / kind / cam / v0_dir.parent.name / v0_dir.name
            n_jpg = len(list(out_dir.glob("frame_*_v*.jpg")))
            r["dirs"] += 1
            if n_jpg != n_v0 * (n_variants - 1):
                r["bad_dirs"].append(f"{out_dir}: jpg {n_jpg} / v0 {n_v0} × variant {n_variants - 1}")
            done = out_dir / bake.DONE_MARKER_NAME
            want_chroma = "4:4:4" if want_sampling == 0 else "4:2:0"
            if not done.is_file() or json.loads(done.read_text()).get("jpg_subsampling") != want_chroma:
                r["bad_done_marker"].append(str(out_dir))


def check_aug_skill(
    name: str, work: Path, cfg: dict[str, Any], args: argparse.Namespace, bake, aug_pipe, obb_hash: str
) -> tuple[dict[str, Any], list[list[np.ndarray]], list[list[np.ndarray]]]:
    from model.ramen_ori.obb_cache import ObbPrecomputedCache  # noqa: PLC0415

    skill = next((s for s in cfg["skills"] if s["name"] == name), None)
    if skill is None:
        raise SystemExit(f"config に skill {name!r} が無い")
    kinds = pipe.skill_kinds(skill)
    sp = pipe.SkillPaths(work)
    b = cfg["bake"]
    dev = torch.device(args.device)
    head, wrists = cfg["cameras"]["head"], cfg["cameras"]["wrist"]
    want_sampling = 0 if b.get("gpu_jpeg_encode") else 2
    cache = ObbPrecomputedCache(
        root=sp.obb, ckpt_hash=obb_hash, camera_keys=[head], top_K=cfg["yolo"]["top_k"],
        num_classes=cfg["yolo"]["num_classes"], fps=OBB_FPS,
    )
    variants = range(1, int(b["num_variants"]))
    r: dict[str, Any] = {"kinds": list(kinds), "missing": [], "bad_sampling": 0, "bad_size": 0,
                         "boxes_drawable": 0, "boxes_visible": 0, "presence": [], "spurious": [],
                         "head_mean_diff": [], "token_mean_diff": [], "wrist_mean_diff": [], "bad_done_marker": [],
                         "dirs": 0, "bad_dirs": []}
    head_rows: list[list[np.ndarray]] = []
    wrist_rows: list[list[np.ndarray]] = []
    check_all_mp4_dirs(sp, kinds=kinds, head=head, wrists=wrists, n_variants=int(b["num_variants"]),
                       want_sampling=want_sampling, bake=bake, r=r)

    def read(path: Path, size: tuple[int, int]) -> torch.Tensor | None:
        if not path.is_file():
            r["missing"].append(str(path))
            return None
        img, sampling = _load_jpg(path)
        r["bad_sampling"] += int(sampling != want_sampling)
        r["bad_size"] += int(tuple(img.shape[-2:]) != size)
        return img.to(dev)

    # head: YOLO の結果がある frame を全体から散らして選ぶ
    head_frames = [p for p in _v0_frames(sp.v0 / head)
                   if cache.lookup_mp4_frame(head, f"{p.parent.parent.name}/{p.parent.name}", int(p.stem.split("_")[1])) is not None]
    head_picks = _spread(head_frames, args.frames)
    sheet_head = set(_spread(list(range(len(head_picks))), args.sheet_frames))  # 一覧画像も全体から散らす
    for i, v0_path in enumerate(head_picks):
        _, mp4_rel, frame_idx, frame_uid = bake._derive_frame_info(v0_path, sp.v0)
        per_frame = cache.lookup_mp4_frame(head, mp4_rel, frame_idx)
        v0 = bake._load_v0_as_tensor(v0_path)[0].to(dev)
        rel = v0_path.relative_to(sp.v0).parent
        row = [_to_u8(v0)]
        for v in variants:
            idx = f"{frame_uid}:v{v}"
            base = aug_pipe.photometric(v0, head, idx)
            name_v = f"{v0_path.stem}_v{v}.jpg"
            if "token" in kinds:
                got = read(sp.v2 / "token" / rel / name_v, tuple(v0.shape[-2:]))
                if got is not None:
                    r["token_mean_diff"].append(float((got - base).abs().mean()))
            if "overlay" in kinds:
                got = read(sp.v2 / "overlay" / rel / name_v, tuple(v0.shape[-2:]))
                if got is None:
                    continue
                m = head_overlay_metrics(
                    got, base, aug_pipe.geometric_params(head, idx), aug_pipe, per_frame,
                    float(b["overlay_conf"]), int(b["overlay_thickness"]), args.line_diff / 255, args.spurious_diff / 255,
                )
                r["boxes_drawable"] += m["drawable"]
                r["boxes_visible"] += len(m["presence"])
                r["presence"] += m["presence"]
                r["spurious"].append(m["spurious"])
                r["head_mean_diff"].append(m["mean_diff"])
                if i in sheet_head:
                    row.append(_to_u8(got))
        if i in sheet_head:
            head_rows.append(row)

    # wrist: 色の変化だけの基準と比べる (幾何変換が掛かっていれば全体がずれて差が大きくなる)
    for cam in wrists:
        if aug_pipe.geometric_params(cam, "probe").use:
            raise SystemExit(f"{cam} に幾何変換が掛かる設定になっている (augmentation.wrist_cam_keys を確認)")
        wrist_picks = _spread(_v0_frames(sp.v0 / cam), args.wrist_frames)
        for j, v0_path in enumerate(wrist_picks):
            _, _, _, frame_uid = bake._derive_frame_info(v0_path, sp.v0)
            v0 = bake._load_v0_as_tensor(v0_path)[0].to(dev)
            rel = v0_path.relative_to(sp.v0).parent
            row = [_to_u8(v0)]
            for v in variants:
                got = read(sp.v2 / "common" / rel / f"{v0_path.stem}_v{v}.jpg", tuple(v0.shape[-2:]))
                if got is None:
                    continue
                ref = aug_pipe.photometric(v0, cam, f"{frame_uid}:v{v}")
                r["wrist_mean_diff"].append(float((got - ref).abs().mean()))
                row.append(_to_u8(got))
            if j == len(wrist_picks) // 2:  # cam ごと 1 行、先頭に偏らないよう中央の frame
                wrist_rows.append(row)
    return r, head_rows, wrist_rows


def aug_verdict(r: dict[str, Any], args: argparse.Namespace) -> dict[str, tuple[bool, str]]:
    """項目ごとの (一致したか, 数字)。"""
    def mx(xs: list[float]) -> float:
        return max(xs) if xs else 0.0

    pres = r["presence"]
    drawn = sum(p >= args.presence for p in pres)
    n_head, n_wrist = len(r["head_mean_diff"]) + len(r["token_mean_diff"]), len(r["wrist_mean_diff"])
    # 空の dir を渡すと以降の項目が「0 件で不一致 0」= OK になるので、まず見た量を確かめる。
    # 枠の数もここで見て、検出が 0 のときに枠の項目が二重に NG になるのを避ける
    seen_ok = n_head > 0 and n_wrist > 0 and r["dirs"] > 0
    if "overlay" in r["kinds"]:
        seen_ok = seen_ok and r["boxes_drawable"] > 0
    out = {
        "確認できた枚数": (seen_ok,
                     f"head {n_head} 枚、wrist {n_wrist} 枚、mp4 dir {r['dirs']}、"
                     f"閾値以上の検出 {r['boxes_drawable']}"),
        "jpg がそろう": (not r["missing"] and not r["bad_dirs"],
                     f"標本の欠け {len(r['missing'])}、枚数の合わない mp4 dir {len(r['bad_dirs'])} / {r['dirs']}"),
        "色差・大きさ": (r["bad_sampling"] == 0 and r["bad_size"] == 0,
                     f"色差の不一致 {r['bad_sampling']}、大きさの不一致 {r['bad_size']}"),
        "完了マーカー": (not r["bad_done_marker"], f"不一致 {len(r['bad_done_marker'])} / {r['dirs']} dir"),
    }
    if "overlay" in r["kinds"]:
        # presence は「線を測れた枠」しか見ないので、測れた枠の割合も別に確かめる
        # (幾何変換で画像の外に出た枠と、下地と色が近くて線を拾えない枠がここに落ちる)。
        # 検出が 0 件のときは「確認できた枚数」が NG になるので、ここでは判定しない
        visible = r["boxes_visible"] / r["boxes_drawable"] if r["boxes_drawable"] else 1.0
        out["枠が画像内にある"] = (visible >= args.visible_ratio,
                            f"測れた枠 {r['boxes_visible']} / 閾値以上の検出 {r['boxes_drawable']} = {visible:.3f}")
        out["枠が検出どおり"] = (drawn == len(pres),
                            f"描かれた枠 {drawn} / 画像内の枠 {len(pres)}、"
                            f"presence 最小 {min(pres) if pres else float('nan'):.3f}")
        out["枠の外に余計な差なし"] = (mx(r["spurious"]) <= args.spurious_rate, f"最大 {mx(r['spurious']):.5f}")
        out["head と基準の差"] = (mx(r["head_mean_diff"]) <= args.max_mean_diff, f"平均の最大 {mx(r['head_mean_diff']):.4f}")
    if "token" in r["kinds"]:
        out["token と基準の差"] = (mx(r["token_mean_diff"]) <= args.max_mean_diff, f"平均の最大 {mx(r['token_mean_diff']):.4f}")
    out["wrist は幾何変換なし"] = (mx(r["wrist_mean_diff"]) <= args.max_mean_diff, f"平均の最大 {mx(r['wrist_mean_diff']):.4f}")
    return out


def cmd_aug(args: argparse.Namespace) -> int:
    from model.ramen_ori.image_aug import ImageAugPipeline  # noqa: PLC0415

    cfg = pipe.load_config(args.config)
    b = cfg["bake"]
    if 0 < int(b["cuda_workers"]) < int(b["jobs"]):
        # 混成だと frame ごとにどちらの worker が焼いたか分からず、基準を 1 つの device で作れない
        # (cuda_workers が jobs 以上なら全 worker が GPU = 均質なので通す)
        raise SystemExit(
            f"bake.cuda_workers {b['cuda_workers']} が 0 と jobs {b['jobs']} の間 (CPU / GPU 混成)。"
            "混成で焼いた cache はこの確認では判定できない (乱数が device で変わる)"
        )
    obb_hash = pipe.yolo_cache_hash(cfg)
    bake = _load_bake_module()
    aug_pipe = ImageAugPipeline(cfg["augmentation"])
    all_ok = True
    report = {}
    for name, d in _parse_named(args.skill, "--skill").items():
        r, head_rows, wrist_rows = check_aug_skill(name, Path(d), cfg, args, bake, aug_pipe, obb_hash)
        verdict = aug_verdict(r, args)
        report[name] = {"verdict": {k: {"ok": ok, "detail": s} for k, (ok, s) in verdict.items()},
                        "missing": r["missing"][:20], "bad_done_marker": r["bad_done_marker"][:20],
                        "bad_dirs": r["bad_dirs"][:20]}
        print(f"== {name} (kinds {r['kinds']})")
        for k, (ok, s) in verdict.items():
            print(f"  {'OK' if ok else 'NG'}  {k}: {s}")
            all_ok &= ok
        if head_rows:
            _grid(head_rows, args.out / f"aug_{name}_head.jpg")
        if wrist_rows:
            _grid(wrist_rows, args.out / f"aug_{name}_wrist.jpg")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "aug.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# roundtrip
# ---------------------------------------------------------------------------


def eligible_indices(src: Path, cams: list[str], fps: float, max_frame: int | None, max_files: int | None) -> list[int]:
    """smoke の cache にある frame だけを指す dataset の index (全 cam で file 番号 < max_files、
    file の中の位置 < max_frame)。smoke は mp4 ごとに先頭 max_frame だけ展開・焼き込みするため。"""
    rows = []
    for f in sorted((src / "meta" / "episodes").glob("*/*.parquet")):
        rows += pq.read_table(f).to_pylist()
    out: list[int] = []
    for ep in rows:
        if max_files is not None and any(
            int(ep[f"videos/{c}/chunk_index"]) != 0 or int(ep[f"videos/{c}/file_index"]) >= max_files for c in cams
        ):
            continue
        starts = [round(float(ep[f"videos/{c}/from_timestamp"]) * fps) for c in cams]
        n = int(ep["length"]) if max_frame is None else min(int(ep["length"]), max_frame - max(starts))
        out += range(int(ep["dataset_from_index"]), int(ep["dataset_from_index"]) + max(n, 0))
    return out


def cmd_roundtrip(args: argparse.Namespace) -> int:
    # lazy: hydra / lerobot fork は RAMEN-Ori env にしか無い
    import hydra  # noqa: PLC0415
    from omegaconf import OmegaConf  # noqa: PLC0415

    from model.ramen_ori.skill_mapping import SKILL_ID_TO_NAME  # noqa: PLC0415

    name_to_id = {v: k for k, v in SKILL_ID_TO_NAME.items()}
    subs = {}
    for name, spec in _parse_named(args.sub, "--sub").items():
        src, _, overlay = spec.partition(",")
        if name not in name_to_id or not overlay:
            raise SystemExit(f"--sub は skill 名 ({sorted(name_to_id)})=SRC,OVERLAY: {name}={spec}")
        subs[name] = (Path(src), Path(overlay))
    data = OmegaConf.load(args.data_config)
    data.chunk_len = args.chunk_len  # 学習の config では ${model.chunk_len}
    data.memory = False
    data.sub_datasets = [
        {"merged_source_root": str(src), "repo_id": f"local/{name}", "frame_cache_root_baked": str(overlay)}
        for name, (src, overlay) in subs.items()
    ]
    ds = hydra.utils.instantiate(data, augmentation_cfg=None)
    cams = list(data.camera_keys)
    all_ok = True
    for k, (name, (src, _)) in enumerate(subs.items()):
        fps = json.loads((src / "meta" / "info.json").read_text())["fps"]
        idx = eligible_indices(src, cams, fps, args.max_frame, args.max_files)
        picks = _spread(idx, args.samples)
        got_ids = set()
        for i in picks:
            item = ds.subs[k][i]
            got_ids.add(int(item["skill_id"]))
            if not torch.isfinite(item["images"]).all():
                raise SystemExit(f"{name} index {i}: 画像に非有限値")
        ok = bool(picks) and got_ids == {name_to_id[name]}
        all_ok &= ok
        print(f"  {'OK' if ok else 'NG'}  {name}: 読めた sample {len(picks)} (smoke の範囲の index {len(idx):,})、"
              f"skill_id {sorted(got_ids)} (期待 {name_to_id[name]})、images {tuple(item['images'].shape) if picks else '-'}")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("merge", help="passthrough の merge 後と元の chunk を LeRobot で読み比べる")
    m.add_argument("--merged", type=Path, required=True)
    m.add_argument("--chunk", type=Path, action="append", required=True, help="merge に渡した順に (repeatable)")
    m.add_argument("--random", type=int, default=3, help="chunk ごとに無作為に足す episode 数")
    m.set_defaults(func=cmd_merge)

    y = sub.add_parser("yolo", help="YOLO の検出の数字 + 枠を描いた一覧画像")
    y.add_argument("--config", type=Path, required=True)
    y.add_argument("--skill", action="append", required=True, help="NAME=DIR (DIR/obb_yolo/<hash>、あれば DIR/v0)")
    y.add_argument("--samples", type=int, default=20, help="一覧画像の frame 数 (skill ごと)")
    y.add_argument("--out", type=Path, required=True)
    y.set_defaults(func=cmd_yolo)

    a = sub.add_parser("aug", help="焼いた jpg を同じ手順で作り直した基準と比べる")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--skill", action="append", required=True, help="NAME=pipeline の作業 dir (v0 / v2 / obb_yolo)")
    a.add_argument("--frames", type=int, default=8, help="head の frame 数 (skill ごと、全 variant を見る)")
    a.add_argument("--wrist-frames", type=int, default=4, help="wrist の frame 数 (cam ごと)")
    a.add_argument("--sheet-frames", type=int, default=3, help="一覧画像に並べる head の frame 数")
    a.add_argument("--line-diff", type=float, default=30, help="枠の線とみなす差 (0-255)")
    a.add_argument("--presence", type=float, default=0.8, help="枠が描かれているとみなす、線の画素の割合")
    a.add_argument("--visible-ratio", type=float, default=0.8,
                   help="閾値以上の検出のうち、線を測れた枠の割合の下限 (幾何変換で画像外に出た枠などが落ちる)")
    a.add_argument("--spurious-diff", type=float, default=80, help="枠の外で余計な差とみなす差 (0-255)")
    a.add_argument("--spurious-rate", type=float, default=0.002, help="枠の外の余計な差の画素の割合の上限")
    a.add_argument("--max-mean-diff", type=float, default=0.03, help="基準との差の平均の上限 (0-1)")
    a.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="基準を作る device。焼き込みの worker と同じにする (ノイズ以降の乱数が device で変わる)")
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_aug)

    r = sub.add_parser("roundtrip", help="展開した cache を RAMEN-Ori の学習と同じ組み立てで読む")
    r.add_argument("--data-config", type=Path, required=True, help="共通の設定を取る学習の data config")
    r.add_argument("--sub", action="append", required=True, help="SKILL=SRC,OVERLAY (restore.sh の出力の overlay)")
    r.add_argument("--chunk-len", type=int, default=32)
    r.add_argument("--max-frame", type=int, default=None, help="smoke の --smoke-frames")
    r.add_argument("--max-files", type=int, default=None, help="smoke の --limit-files")
    r.add_argument("--samples", type=int, default=4)
    r.set_defaults(func=cmd_roundtrip)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
