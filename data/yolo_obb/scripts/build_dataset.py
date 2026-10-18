"""HF から annotations + frames を pull して YOLO-OBB 学習 dataset を組む。

Epic #43 / Issue #45 (v1、100 枚固定) → Issue #53 (v2、400 枚、multi-ep 対応)。

処理:
1. HF `Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb` から
   `yolo_obb/labels/**/*.txt` を全部 pull
2. 対応 image を同じ repo から
   `frames/**/frame_YY.png` で pull
3. skill stratified split で train/val に分割 (default 90/10、seed 固定)
   - `--train-only-eps` で指定した ep は split をバイパスして全部 train へ
     (v2: ep 62 の 300 枚を全部 train、ep 60 の 100 枚は従来通り 90/10 で
      seed=42 → v1 val と bit-perfect 一致、v1 vs v2 直接 A/B 可能)
4. ultralytics 標準 layout に copy:
     dataset/images/{train,val}/{skill}/ep{XX}/frame_YY.png
     dataset/labels/{train,val}/{skill}/ep{XX}/frame_YY.txt
5. `configs/dataset.yaml` を生成 (class names + train/val path)
6. Class balance + bbox size 分布 report を stdout に出す (README に転記想定)

Usage (model/yolo_obb sub-workspace 経由で叩く):
    # v1 (100 枚 / 90 train / 10 val 相当) — 従来通り
    cd model/yolo_obb
    pixi run build-yolo-dataset \\
        --out data/yolo_obb/dataset \\
        --val-ratio 0.10 \\
        --seed 42

    # v2 (400 枚 / 390 train / 10 val、v1 val 維持)
    pixi run build-yolo-dataset \\
        --out data/yolo_obb/dataset \\
        --val-ratio 0.10 \\
        --seed 42 \\
        --train-only-eps 62
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# 画像とラベルは同じ dataset repo (Issue #190 で元の 2 repo から要る物だけを移した)
HF_ANNOTATIONS_REPO = HF_FRAMES_REPO = "Team-RAMEN/ramen-g1-ikea-assembly-yolo-obb"
# 画像とラベルは数千の小さい file。1 つずつ落とすと 966 枚で 18 分かかったので並列に落とす (Issue #196)
DOWNLOAD_WORKERS = 16
# 並列に繋ぐと一時的な切断 (Server disconnected など) に当たるので、1 file ごとに間を空けて試し直す
DOWNLOAD_ATTEMPTS = 3

CLASS_NAMES = [
    "workspace",   # 0
    "leg",         # 1
    "leg_tip",     # 2
    "hole",        # 3
    "table_top",   # 4
    "hand_right",  # 5
    "hand_left",   # 6
]


# ------------------------------------------------------------
# HF pull
# ------------------------------------------------------------
def _dl(repo: str, filename: str, cache: Path, token: str | None) -> Path:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=str(cache),
                    token=token,
                )
            )
        except Exception:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(attempt)  # 1 s, 2 s
    raise AssertionError("unreachable")


def pull_annotations(cache: Path, token: str | None) -> list[dict]:
    """HF から yolo_obb/labels/**/*.txt を全部 pull。

    Returns:
        [{"skill": str, "ep": int, "frame": int, "txt_path": Path, "img_hf_path": str}]
    """
    api = HfApi(token=token)
    files = api.list_repo_files(HF_ANNOTATIONS_REPO, repo_type="dataset")
    txt_files = [f for f in files if f.startswith("yolo_obb/labels/") and f.endswith(".txt")]
    print(f"[hf] annotations repo: {len(txt_files)} txt files")

    # 先に path を全部確かめ、残った物だけを並列に落とす (map は入力の順に返す)
    entries: list[tuple[str, str, int, int, str]] = []
    for f in sorted(txt_files):
        # f = "yolo_obb/labels/{skill}/ep{XX}/frame_{YY}.txt"
        parts = Path(f).parts
        if len(parts) != 5 or parts[0] != "yolo_obb" or parts[1] != "labels":
            print(f"[warn] unexpected path {f} — skipping", file=sys.stderr)
            continue
        skill = parts[2]
        ep_str = parts[3]  # "ep0060"
        if not ep_str.startswith("ep"):
            print(f"[warn] non-ep dir {ep_str} — skipping", file=sys.stderr)
            continue
        ep = int(ep_str[2:])
        frame_stem = Path(parts[4]).stem  # "frame_00060"
        frame = int(frame_stem.split("_")[1])
        entries.append((f, skill, ep, frame, f"frames/{skill}/{ep_str}/{frame_stem}.png"))
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        txt_locals = list(
            pool.map(lambda e: _dl(HF_ANNOTATIONS_REPO, e[0], cache / "annotations", token), entries)
        )
    return [
        {
            "skill": skill,
            "ep": ep,
            "frame": frame,
            "txt_path": txt_local,
            "img_hf_path": img_hf_path,
        }
        for (_, skill, ep, frame, img_hf_path), txt_local in zip(entries, txt_locals)
    ]


def pull_images(entries: list[dict], cache: Path, token: str | None) -> list[dict]:
    """entries に対応する image を HF frames repo から pull。

    個別 file 失敗は warn で record、成功分だけ返す (全 pipeline abort させない)。
    entries は mutate せず、img_path 付きの新 list を返す。
    """
    def fetch(e: dict) -> tuple[dict, Path | None, str | None]:
        try:
            return e, _dl(HF_FRAMES_REPO, e["img_hf_path"], cache / "frames", token), None
        except Exception as err:  # 1 枚の失敗で全体を止めない
            return e, None, str(err)

    ok: list[dict] = []
    failed: list[tuple[dict, str]] = []
    # 並列に落とし、結果は入力の順に受け取る (map の順)
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        for i, (e, img_local, err) in enumerate(pool.map(fetch, entries), 1):
            if err is None:
                ok.append({**e, "img_path": img_local})
            else:
                failed.append((e, err))
            if i % 20 == 0 or i == len(entries):
                print(f"  [img] {i}/{len(entries)} DL 済 (ok={len(ok)}, fail={len(failed)})")
    if failed:
        print(f"[warn] {len(failed)} image DL failed, dropped from dataset:", file=sys.stderr)
        for e, err in failed[:5]:
            print(f"  - {e['img_hf_path']}: {err[:120]}", file=sys.stderr)
        if len(failed) > 5:
            print(f"  ... ({len(failed) - 5} more)", file=sys.stderr)
    return ok


# ------------------------------------------------------------
# split
# ------------------------------------------------------------
def stratified_split(
    entries: list[dict],
    val_ratio: float,
    seed: int,
    train_only_eps: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """skill 別に val_ratio 分ずつ val に振り分け (stratified random)。

    `train_only_eps` に含まれる ep の entry は split をバイパスして train に固定される。
    残った entry のみ stratified split の対象となるため、`seed` を同じにすれば
    v1 の (ep 60 のみ) split と bit-perfect 一致する。

    100 枚 / 6 skill × val_ratio=0.10 = 各 skill 約 2 枚 val (rounded)。
    """
    import numpy as np

    train_only_eps = train_only_eps or set()
    forced_train = [e for e in entries if e["ep"] in train_only_eps]
    to_split = [e for e in entries if e["ep"] not in train_only_eps]

    rng = np.random.default_rng(seed)
    by_skill: dict[str, list[dict]] = defaultdict(list)
    for e in to_split:
        by_skill[e["skill"]].append(e)

    train: list[dict] = list(forced_train)
    val: list[dict] = []
    for skill in sorted(by_skill.keys()):
        items = by_skill[skill]
        n_val = max(1, round(len(items) * val_ratio))  # 各 skill 最低 1 枚 val
        idxs = rng.permutation(len(items))
        for j, k in enumerate(idxs):
            (val if j < n_val else train).append(items[int(k)])
    return train, val


# ------------------------------------------------------------
# copy to ultralytics layout
# ------------------------------------------------------------
def copy_to_layout(
    entries: list[dict],
    out_root: Path,
    split_name: str,
) -> None:
    """dataset/images/{split}/{skill}/ep{XX}/frame_YY.png + labels/... に copy。"""
    for e in entries:
        skill = e["skill"]
        ep_dir = f"ep{e['ep']:04d}"
        img_dst = out_root / "images" / split_name / skill / ep_dir / Path(e["img_path"]).name
        txt_dst = out_root / "labels" / split_name / skill / ep_dir / Path(e["txt_path"]).name
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        txt_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e["img_path"], img_dst)
        shutil.copy2(e["txt_path"], txt_dst)


# ------------------------------------------------------------
# report
# ------------------------------------------------------------
def _read_boxes(txt_path: Path) -> list[tuple[int, list[float]]]:
    """YOLO OBB txt から (class_id, [x1,y1,x2,y2,x3,y3,x4,y4]) list を返す。"""
    out = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 9:
            continue
        out.append((int(parts[0]), [float(x) for x in parts[1:]]))
    return out


def _bbox_size(verts: list[float]) -> tuple[float, float]:
    """4 vertices から **axis-aligned 外接矩形** の w×h (normalized) を返す。

    OBB の実 w×h (rotated rectangle 自体の辺長) ではなく、その外接する
    軸沿い bbox の大きさ。rotation 大きい box では実 w×h より overestimate に
    なるが、class 別サイズ傾向を掴む report 用途としては十分。
    """
    xs = verts[0::2]
    ys = verts[1::2]
    return (max(xs) - min(xs), max(ys) - min(ys))


def class_balance_report(train: list[dict], val: list[dict]) -> None:
    """Class 別 box 数と bbox size 分布を stdout に出す。"""
    print("\n" + "=" * 50)
    print("Class balance report")
    print("=" * 50)
    for split_name, entries in [("train", train), ("val", val)]:
        cls_count: dict[int, int] = Counter()
        sizes: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for e in entries:
            for cid, verts in _read_boxes(e["txt_path"]):
                cls_count[cid] += 1
                sizes[cid].append(_bbox_size(verts))
        total_boxes = sum(cls_count.values())
        print(f"\n[{split_name}] {len(entries)} frames, {total_boxes} boxes")
        for cid in range(len(CLASS_NAMES)):
            n = cls_count.get(cid, 0)
            if n == 0:
                print(f"  {cid} {CLASS_NAMES[cid]:11}: 0")
                continue
            ss = sizes[cid]
            avg_w = sum(s[0] for s in ss) / len(ss)
            avg_h = sum(s[1] for s in ss) / len(ss)
            print(f"  {cid} {CLASS_NAMES[cid]:11}: {n:4} boxes  (avg bbox w={avg_w:.3f} h={avg_h:.3f} normalized)")


# ------------------------------------------------------------
# dataset.yaml
# ------------------------------------------------------------
def write_dataset_yaml(out_root: Path, yaml_path: Path, n_train: int, n_val: int) -> None:
    """ultralytics 標準 dataset.yaml を書く。frame 数は runtime に応じて注釈。"""
    lines = [
        f"# YOLO OBB dataset config (Issue #45 v1 / Issue #53 v2)",
        f"# {n_train + n_val} frames / {n_train} train / {n_val} val / skill stratified split",
        f"path: {out_root.resolve()}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"nc: {len(CLASS_NAMES)}",
        f"names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n")
    print(f"\n[yaml] wrote {yaml_path}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/yolo_obb/dataset"),
        help="ultralytics dataset 出力 root",
    )
    p.add_argument("--yaml", type=Path, default=Path("data/yolo_obb/configs/dataset.yaml"))
    p.add_argument("--val-ratio", type=float, default=0.10, help="val 側の比率 (skill 内)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-only-eps",
        type=str,
        default=None,
        help=(
            "指定した ep の entry は split を bypass して全部 train へ "
            "(v2 で ep 62 全部 train 用途、comma-separated: '62' or '62,64')"
        ),
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("data/yolo_obb/.hf_cache"),
        help="HF DL 一時 cache",
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] out={args.out}, yaml={args.yaml}, val_ratio={args.val_ratio}, seed={args.seed}")

    # 1. Annotations pull
    entries = pull_annotations(args.cache, token=None)
    print(f"[pull] {len(entries)} annotation txt")

    # 2. Images pull (個別 fail は skip、成功分だけ返る)
    entries = pull_images(entries, args.cache, token=None)
    if not entries:
        raise SystemExit("[FAIL] no images successfully pulled — check HF access")

    # 3. Split (stratified、train_only_eps は split bypass)
    train_only_eps: set[int] = set()
    if args.train_only_eps:
        train_only_eps = {int(x.strip()) for x in args.train_only_eps.split(",") if x.strip()}
        print(f"[split] train-only eps (bypass split): {sorted(train_only_eps)}")
    train, val = stratified_split(
        entries, args.val_ratio, args.seed, train_only_eps=train_only_eps
    )
    print(f"[split] train={len(train)} val={len(val)}")

    # 4. 既存 dataset dir を clean してから layout copy (seed / val_ratio 変更で
    #    旧 file が train/val 両方に残ると data 汚染するのを予防)
    for sub in ("images", "labels"):
        target = args.out / sub
        if target.exists():
            print(f"[clean] removing existing {target}")
            shutil.rmtree(target)

    for split_name, split_entries in [("train", train), ("val", val)]:
        copy_to_layout(split_entries, args.out, split_name)
    print(f"[copy] wrote images + labels to {args.out}")

    # 5. dataset.yaml
    write_dataset_yaml(args.out, args.yaml, n_train=len(train), n_val=len(val))

    # 6. Report
    class_balance_report(train, val)

    # 7. split manifest (再現性 + downstream ref)
    manifest = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train_only_eps": sorted(train_only_eps),
        "hf_annotations_repo": HF_ANNOTATIONS_REPO,
        "hf_frames_repo": HF_FRAMES_REPO,
        "train": [{"skill": e["skill"], "ep": e["ep"], "frame": e["frame"]} for e in train],
        "val": [{"skill": e["skill"], "ep": e["ep"], "frame": e["frame"]} for e in val],
    }
    (args.out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {args.out}/split_manifest.json")


if __name__ == "__main__":
    main()
