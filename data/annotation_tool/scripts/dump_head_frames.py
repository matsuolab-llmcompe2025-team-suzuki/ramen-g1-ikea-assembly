"""Skill balance sampling で LeRobot G1_WBT cam_0 640x480 PNG を抽出、HF に skill 別 upload する。

Epic #43 / Issue #44。

方針:
- frame ごとの skill の表 (`scripts/build_skill_labels.py` が作る `workspace/skill_labels.parquet`) の `skill_id` を使って
  各 skill から均等サンプリング (default 5 枚 x 6 skill = 30 枚)。
- HF から meta/episodes/*.parquet + cam_0 mp4 を **1 file ずつ** DL、
  抽出 → HF upload → 抽出後は不要 mp4 を削除 (--keep-mp4 で保持)。
- 抽出は resize しない (640x480 生 PNG)。ViT precompute の 384x384 とは別 stream。

Layout (local & HF 両方):
    frames/
    ├── insert_table_leg/
    │   ├── ep0004/
    │   │   └── frame_00123.png
    │   └── ep0007/
    │       └── frame_00089.png
    ├── move_to_table/
    ...

Usage:
    # まず dry-run で何を DL するか確認 (HF upload しない、HF DL もしない)
    pixi run python scripts/dump_head_frames.py --dry-run

    # 実行 (mp4 DL + 抽出 + HF upload + mp4 削除)。上げ先は必ず指定する (既定なし)
    pixi run python scripts/dump_head_frames.py --hf-upload-repo <your-dataset-repo>

    # local だけ (--hf-upload-repo を付けなければ HF には上げない)
    pixi run python scripts/dump_head_frames.py

    # 別 setting
    pixi run python scripts/dump_head_frames.py \\
        --labels-parquet workspace/skill_labels.parquet \\
        --per-skill 5 \\
        --ep-range 0-49 \\
        --split train \\
        --out workspace/images \\
        --seed 42 \\
        --keep-mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# ---- 定数 ----
HF_SOURCE_REPO = "BitRobot/G1_WBT_Dex1_Building-Children-Table"  # 抽出元 (LeRobot v3.0 mp4)
CAM_KEY = "observation.images.cam_0"

# skill_id → name (build_skill_labels.py の SKILL_REMAP の読み替え先と同じ 0-5 の 6 class)
SKILL_NAMES: dict[int, str] = {
    0: "insert_table_leg",
    1: "move_to_table",
    2: "flip_table",
    3: "finalize_leg",       # rotate_leg_to_tighten + rotate_table_base merge
    4: "pick_table_leg",
    5: "move_table_base",
}


# ------------------------------------------------------------
# sampling
# ------------------------------------------------------------
def parse_ep_range(spec: str) -> tuple[int, int]:
    """`0-49` → (0, 49)。"""
    lo, hi = spec.split("-")
    return int(lo), int(hi)


def _filter_labels(
    labels_parquet: Path,
    ep_range: tuple[int, int],
    split_filter: str,
) -> pa.Table:
    """ep_range + split で labels.parquet を filter して返す。"""
    t = pq.read_table(labels_parquet)
    lo, hi = ep_range
    mask = pc.and_(
        pc.and_(pc.greater_equal(t["episode_index"], lo), pc.less_equal(t["episode_index"], hi)),
        pc.equal(t["split"], split_filter),
    )
    return t.filter(mask)


def compute_min_skill_count(
    labels_parquet: Path,
    ep_range: tuple[int, int],
    split_filter: str,
) -> tuple[int, dict[int, int]]:
    """`--per-skill auto` 用: ep 範囲内の skill 別 frame 数と、その min を返す。

    Returns:
        (min_count, {skill_id: count}) — min_count は 0 skill を除外した最小値。
        存在しない skill (count=0) は auto 決定の分母に含めない。
    """
    t = _filter_labels(labels_parquet, ep_range, split_filter)
    if t.num_rows == 0:
        return 0, {}
    from collections import Counter as _Counter

    counts = dict(_Counter(t["skill_id"].to_pylist()))
    non_zero = [v for v in counts.values() if v > 0]
    if not non_zero:
        return 0, counts
    return min(non_zero), {int(k): int(v) for k, v in counts.items()}


def sample_full_skill_episodes(
    labels_parquet: Path,
    ep_range: tuple[int, int],
    split_filter: str,
    seed: int,
    n_skills_required: int = 6,
    skip_eps: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """全 skill 揃う ep のみ per-episode 独立 stratified sampling。

    - ep 範囲内 → 各 ep の skill_id 分布を計算
    - `n_skills_required` skill 揃う ep のみ選択 (揃わない ep は skip)
    - 各 ep で `per_skill = min skill count in that ep` を自動決定
    - stratified time-bin sampling で時間分散

    Args:
        skip_eps: 既に HF に上がっている ep を skip する resume 用。

    Returns:
        (samples, ep_stats)
          samples: [{"episode_index", "frame_index", "skill_id", "skill_name"}, ...]
          ep_stats: [{"episode_index", "per_skill", "total_frames", "skipped_reason"}, ...]
    """
    t = _filter_labels(labels_parquet, ep_range, split_filter)
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    stats: list[dict] = []
    from collections import Counter as _Counter

    eps_in_range = sorted(set(t["episode_index"].to_pylist()))
    for ep in eps_in_range:
        ep_t = t.filter(pc.equal(t["episode_index"], ep))
        skill_counts = _Counter(ep_t["skill_id"].to_pylist())
        non_zero_skills = [s for s, c in skill_counts.items() if c > 0]
        if len(non_zero_skills) < n_skills_required:
            stats.append(
                {
                    "episode_index": int(ep),
                    "per_skill": 0,
                    "total_frames": 0,
                    "skipped_reason": f"only {len(non_zero_skills)} skills present",
                }
            )
            continue
        if skip_eps and int(ep) in skip_eps:
            stats.append(
                {
                    "episode_index": int(ep),
                    "per_skill": 0,
                    "total_frames": 0,
                    "skipped_reason": "already uploaded (resume)",
                }
            )
            continue
        per_skill = min(c for c in skill_counts.values() if c > 0)
        for sid in sorted(skill_counts.keys()):
            n = int(skill_counts[sid])
            if n == 0:
                continue
            subset = ep_t.filter(pc.equal(ep_t["skill_id"], sid))
            frs = sorted(subset["frame_index"].to_pylist())
            k = per_skill
            for i in range(k):
                lo = i * n // k
                hi = max(lo + 1, (i + 1) * n // k)
                idx = int(rng.integers(lo, hi))
                fr = frs[idx]
                samples.append(
                    {
                        "episode_index": int(ep),
                        "frame_index": int(fr),
                        "skill_id": int(sid),
                        "skill_name": SKILL_NAMES.get(int(sid), f"skill_{sid}"),
                    }
                )
        stats.append(
            {
                "episode_index": int(ep),
                "per_skill": per_skill,
                "total_frames": per_skill * n_skills_required,
                "skipped_reason": None,
            }
        )
    return samples, stats


def query_uploaded_eps(repo_id: str, token: str | None) -> set[int]:
    """HF repo から `frames/<skill>/ep<XXXX>/...` パスの ep 番号集合を抽出 (resume 用)。"""
    import re

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        # repo が未作成なら空
        return set()
    pat = re.compile(r"frames/[^/]+/ep(\d{4})/")
    return {int(m.group(1)) for f in files if (m := pat.match(f))}


def sample_balanced(
    labels_parquet: Path,
    per_skill: int,
    ep_range: tuple[int, int],
    split_filter: str,
    seed: int,
) -> list[dict]:
    """skill_id ごとに **stratified time-bin sampling** で均等サンプリング。

    各 skill の frame を time 順に並べ、per_skill 等分の bin に分割し、
    各 bin から 1 個ランダム抽出。時間的分散が保証される
    (naive random では skill が narrow 区間に偏るとサンプルが密集する問題を回避)。

    Returns:
        list of {"episode_index", "frame_index", "skill_id", "skill_name"}
    """
    t = _filter_labels(labels_parquet, ep_range, split_filter)
    if t.num_rows == 0:
        print(
            f"[FAIL] no rows in ep {ep_range[0]}-{ep_range[1]} split={split_filter}",
            file=sys.stderr,
        )
        sys.exit(1)

    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    skill_ids = sorted(set(t["skill_id"].to_pylist()))
    for sid in skill_ids:
        subset = t.filter(pc.equal(t["skill_id"], sid))
        n = subset.num_rows
        if n == 0:
            continue
        k = min(per_skill, n)
        # time 順 sort: (episode_index, frame_index)
        eps = subset["episode_index"].to_pylist()
        frs = subset["frame_index"].to_pylist()
        pairs = sorted(zip(eps, frs))
        # per_skill 等分 bin
        for i in range(k):
            lo = i * n // k
            hi = max(lo + 1, (i + 1) * n // k)
            idx = int(rng.integers(lo, hi))
            ep, fr = pairs[idx]
            samples.append(
                {
                    "episode_index": int(ep),
                    "frame_index": int(fr),
                    "skill_id": int(sid),
                    "skill_name": SKILL_NAMES.get(int(sid), f"skill_{sid}"),
                }
            )
    return samples


# ------------------------------------------------------------
# HF meta / mp4 handling
# ------------------------------------------------------------
def dl_meta_files(lerobot_root: Path, token: str | None) -> None:
    """meta/episodes/chunk-000/file-*.parquet を HF から DL (存在すれば skip)。"""
    api = HfApi(token=token)
    files = api.list_repo_files(HF_SOURCE_REPO, repo_type="dataset")
    meta_files = [f for f in files if f.startswith("meta/episodes/") and f.endswith(".parquet")]
    if not meta_files:
        print(f"[FAIL] no meta/episodes/*.parquet in {HF_SOURCE_REPO}", file=sys.stderr)
        sys.exit(1)
    print(f"[meta] {len(meta_files)} meta parquet files to ensure locally")
    for f in meta_files:
        local = lerobot_root / f
        if local.exists():
            continue
        hf_hub_download(
            repo_id=HF_SOURCE_REPO,
            repo_type="dataset",
            filename=f,
            local_dir=str(lerobot_root),
            token=token,
        )


def load_ep_video_meta(lerobot_root: Path) -> dict[int, dict]:
    """ep_idx → {video_hf_path, from_timestamp}。

    `video_hf_path` は HF repo 内の相対パス (DL 時にそのまま filename に使える)。
    """
    meta_files = sorted((lerobot_root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"no meta parquet under {lerobot_root}/meta/episodes/")
    t = pa.concat_tables([pq.read_table(f) for f in meta_files])
    ep = t["episode_index"].to_pylist()
    chunk = t[f"videos/{CAM_KEY}/chunk_index"].to_pylist()
    file_ = t[f"videos/{CAM_KEY}/file_index"].to_pylist()
    from_ts = t[f"videos/{CAM_KEY}/from_timestamp"].to_pylist()
    result: dict[int, dict] = {}
    for i in range(len(ep)):
        e = int(ep[i])
        video_rel = f"videos/{CAM_KEY}/chunk-{int(chunk[i]):03d}/file-{int(file_[i]):03d}.mp4"
        result[e] = {"video_hf_path": video_rel, "from_timestamp": float(from_ts[i])}
    return result


def estimate_mp4_sizes(mp4_rel_paths: list[str], token: str | None) -> dict[str, int]:
    """HF Hub から mp4 の byte size を取得。unknown は 0。"""
    api = HfApi(token=token)
    info = api.dataset_info(HF_SOURCE_REPO, files_metadata=True, token=token)
    size_map = {s.rfilename: (s.size or 0) for s in info.siblings}
    return {p: size_map.get(p, 0) for p in mp4_rel_paths}


# ------------------------------------------------------------
# extract
# ------------------------------------------------------------
def extract_frames_from_mp4(
    mp4_path: Path,
    needs: list[tuple[int, int, int, str, float]],
    out_root: Path,
) -> list[dict]:
    """1 mp4 から needs 分の frame を PNG に落とす。

    出力レイアウト: `<out_root>/<skill_name>/ep{ep:04d}/frame_{fr:05d}.png`

    Args:
        mp4_path: local mp4
        needs: [(ep, frame_idx, sid, name, from_timestamp), ...]
        out_root: PNG 出力 root (skill folder はこの下に作る)

    Returns:
        [{rel_path, ep, frame, skill_id, skill_name, resolution}, ...]
        `rel_path` は out_root からの相対パス (HF upload の path_in_repo にも使う)。
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {mp4_path}")
    mp4_fps = cap.get(cv2.CAP_PROP_FPS)
    mp4_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written: list[dict] = []
    try:
        # abs_frame 昇順で seek すると mp4 の keyframe を逆行しないので decode 高速化
        needs = sorted(needs, key=lambda t: int(round(t[4] * mp4_fps)) + t[1])
        for ep, fr, sid, name, from_ts in needs:
            ts_offset = int(round(from_ts * mp4_fps))
            abs_frame = ts_offset + fr
            if not 0 <= abs_frame < mp4_total:
                print(
                    f"[warn] ep {ep} frame {fr} maps to {abs_frame}, mp4 has {mp4_total} — skipping",
                    file=sys.stderr,
                )
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(abs_frame))
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"[warn] ep {ep} frame {fr} read failed", file=sys.stderr)
                continue
            h, w = frame_bgr.shape[:2]
            rel_path = Path(name) / f"ep{ep:04d}" / f"frame_{fr:05d}.png"
            out_path = out_root / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame_bgr)  # PNG は resize なし、生 640x480
            written.append(
                {
                    "rel_path": str(rel_path),
                    "episode_index": ep,
                    "frame_index": fr,
                    "skill_id": sid,
                    "skill_name": name,
                    "resolution": f"{w}x{h}",
                }
            )
    finally:
        cap.release()
    return written


# ------------------------------------------------------------
# HF upload
# ------------------------------------------------------------
def ensure_dest_repo(repo_id: str, private: bool, token: str | None) -> None:
    """upload 先 dataset repo を idempotent に作成。既存なら noop。"""
    from huggingface_hub import create_repo

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )


def upload_frames_to_hf(
    written: list[dict],
    out_root: Path,
    repo_id: str,
    token: str | None,
) -> None:
    """extracted PNG 群を HF に **1 commit で batch upload** (per-file より数百倍速い)。

    HF path_in_repo は `frames/<skill>/ep<XXXX>/frame_<YYYYY>.png`。
    """
    from huggingface_hub import CommitOperationAdd

    api = HfApi(token=token)
    ops = []
    for row in written:
        local = out_root / row["rel_path"]
        remote = f"frames/{row['rel_path']}"
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
        row["hf_path"] = remote
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"add {len(written)} frames (upperpolicy annotation)",
        token=token,
    )


def upload_manifest_to_hf(
    manifest_path: Path,
    repo_id: str,
    token: str | None,
) -> None:
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="manifest.json",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    # HF_TOKEN を workspace root の .env から自動ロード (research/, iros_2026_ramen/ 共通ルート)
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--labels-parquet",
        type=Path,
        default=Path("workspace/skill_labels.parquet"),
        help="Skill 分布と (ep, frame) sampling の source",
    )
    p.add_argument(
        "--per-skill",
        type=lambda s: s if s == "auto" else int(s),
        default=5,
        help="skill_id ごとの抽出枚数。'auto' で ep 範囲内 min skill count を自動採用",
    )
    p.add_argument(
        "--ep-range",
        type=str,
        default="0-49",
        help="サンプリング対象 episode 範囲 (e.g. '0-49', DL する mp4 数を絞る目的)",
    )
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("workspace/lerobot_hf"),
        help="HF DL 先 (meta + mp4)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("workspace/images"),
        help="PNG 出力先",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("workspace/manifest.json"),
        help="抽出結果 metadata JSON 出力先",
    )
    p.add_argument(
        "--keep-mp4",
        action="store_true",
        help="抽出後も mp4 を残す (default: 削除して disk 節約)",
    )
    p.add_argument(
        "--hf-upload-repo",
        type=str,
        default=None,
        help="抽出 PNG の upload 先 dataset repo (指定したときだけ upload。既定なし、Issue #190)",
    )
    p.add_argument(
        "--no-hf-upload",
        dest="hf_upload",
        action="store_false",
        help="HF upload を skip (local 抽出のみ)",
    )
    p.add_argument(
        "--hf-public",
        action="store_true",
        help="upload repo を public で作成 (default: private)",
    )
    p.set_defaults(hf_upload=True)
    p.add_argument(
        "--limit-mp4",
        type=int,
        default=0,
        help="処理する mp4 file 数の上限 (0 = 無制限、smoke run 時に 1 指定)",
    )
    p.add_argument(
        "--full-skill-eps-only",
        action="store_true",
        help="全 6 skill 揃う ep のみ per-episode 独立 sampling (auto per_skill = ep 内 min skill)。--per-skill と --ep-range 内の unbalanced ep は skip される",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="HF に既に上がってる ep も再処理する (default: skip)",
    )
    p.set_defaults(resume=True)
    p.add_argument("--dry-run", action="store_true", help="サンプリング + DL 計画だけ表示")
    args = p.parse_args()
    if args.hf_upload and not args.hf_upload_repo:
        print("[hf] --hf-upload-repo が無いので HF には上げない (手元に抽出するだけ)")
        args.hf_upload = False

    ep_range = parse_ep_range(args.ep_range)
    token = None  # dotenv で環境変数に入っているので HfApi が拾う

    # 1. Sampling
    if args.full_skill_eps_only:
        # resume: HF から既に上がってる ep を skip
        skip_eps: set[int] = set()
        # 上げ先の repo に既にある ep を飛ばす (上げ先が無ければ何も飛ばさない)
        if args.resume and args.hf_upload_repo:
            print("[resume] querying HF for already-uploaded eps...")
            skip_eps = query_uploaded_eps(args.hf_upload_repo, token)
            print(f"[resume] {len(skip_eps)} eps already on HF → skip")
        samples, ep_stats = sample_full_skill_episodes(
            labels_parquet=args.labels_parquet,
            ep_range=ep_range,
            split_filter=args.split,
            seed=args.seed,
            skip_eps=skip_eps,
        )
        processed = [s for s in ep_stats if s["skipped_reason"] is None]
        skipped = [s for s in ep_stats if s["skipped_reason"] is not None]
        print(
            f"[full-skill-eps] processed {len(processed)} eps, skipped {len(skipped)} eps "
            f"({sum(1 for s in skipped if s['skipped_reason'] == 'already uploaded (resume)')} resume, "
            f"{sum(1 for s in skipped if 'only' in (s['skipped_reason'] or ''))} not-full)"
        )
        print(f"[full-skill-eps] total {len(samples)} frames selected")
    else:
        # per-skill auto 解決
        if args.per_skill == "auto":
            min_count, per_skill_dist = compute_min_skill_count(
                args.labels_parquet, ep_range, args.split
            )
            if min_count == 0:
                print(
                    f"[FAIL] no non-empty skill in ep {ep_range} split={args.split}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"[auto] ep {ep_range[0]}-{ep_range[1]} skill 別 count: "
                + " / ".join(
                    f"{SKILL_NAMES.get(k, '?')}={v}" for k, v in sorted(per_skill_dist.items())
                )
            )
            print(f"[auto] per_skill = min non-zero = {min_count}")
            per_skill_int = min_count
        else:
            per_skill_int = args.per_skill

        print(
            f"[sample] labels={args.labels_parquet}, per_skill={per_skill_int}, "
            f"ep_range={ep_range}, split={args.split}, seed={args.seed}"
        )
        samples = sample_balanced(
            labels_parquet=args.labels_parquet,
            per_skill=per_skill_int,
            ep_range=ep_range,
            split_filter=args.split,
            seed=args.seed,
        )
        ep_stats = []
    print(f"[sample] {len(samples)} frames selected:")
    # per-skill count
    per_skill_count: dict[int, int] = defaultdict(int)
    for s in samples:
        per_skill_count[s["skill_id"]] += 1
    for sid in sorted(per_skill_count):
        print(f"  skill_id={sid} ({SKILL_NAMES.get(sid, '?')}): {per_skill_count[sid]} frames")
    print()

    # 2. meta DL & load
    args.lerobot_root.mkdir(parents=True, exist_ok=True)
    dl_meta_files(args.lerobot_root, token)
    ep_meta = load_ep_video_meta(args.lerobot_root)

    # 3. Group by mp4
    by_mp4: dict[str, list[tuple[int, int, int, str, float]]] = defaultdict(list)
    for s in samples:
        m = ep_meta.get(s["episode_index"])
        if m is None:
            print(f"[warn] no meta for ep {s['episode_index']} — skipping", file=sys.stderr)
            continue
        by_mp4[m["video_hf_path"]].append(
            (s["episode_index"], s["frame_index"], s["skill_id"], s["skill_name"], m["from_timestamp"])
        )

    # 4. DL size estimate (--limit-mp4 で上限あればここで切り詰め)
    all_mp4_rels_sorted = sorted(by_mp4.keys())
    if args.limit_mp4 > 0:
        active_mp4_rels = all_mp4_rels_sorted[: args.limit_mp4]
        print(
            f"[limit] --limit-mp4={args.limit_mp4} → {len(active_mp4_rels)} / "
            f"{len(all_mp4_rels_sorted)} mp4 files を処理"
        )
    else:
        active_mp4_rels = all_mp4_rels_sorted
    sizes = estimate_mp4_sizes(active_mp4_rels, token)
    total_bytes = sum(sizes.values())
    print(f"[plan] {len(active_mp4_rels)} mp4 files to fetch, ~{total_bytes / 1e9:.2f} GB total:")
    for mp4_rel in active_mp4_rels:
        needs = by_mp4[mp4_rel]
        size_gb = sizes.get(mp4_rel, 0) / 1e9
        print(f"  {mp4_rel}: {len(needs)} frames needed, ~{size_gb:.2f} GB")
    print()

    if args.dry_run:
        if args.hf_upload:
            print(f"[dry-run] hf upload target: {args.hf_upload_repo} (private={not args.hf_public})")
        else:
            print("[dry-run] hf upload disabled")
        print("[dry-run] no download / no extract / no upload. Rerun without --dry-run to proceed.")
        return

    # 5. HF upload repo を事前確保
    if args.hf_upload:
        print(f"[hf] ensuring {args.hf_upload_repo} (private={not args.hf_public})")
        ensure_dest_repo(args.hf_upload_repo, private=not args.hf_public, token=token)

    # 6. Per-mp4: DL → extract → HF upload → (optionally) delete mp4
    args.out.mkdir(parents=True, exist_ok=True)
    all_written: list[dict] = []
    for mp4_rel in active_mp4_rels:
        needs = by_mp4[mp4_rel]
        print(f"[dl] fetching {mp4_rel}...")
        local_path = hf_hub_download(
            repo_id=HF_SOURCE_REPO,
            repo_type="dataset",
            filename=mp4_rel,
            local_dir=str(args.lerobot_root),
            token=token,
        )
        print(f"[extract] {mp4_rel} → {len(needs)} PNG")
        written = extract_frames_from_mp4(Path(local_path), needs, args.out)
        if args.hf_upload and written:
            print(f"[hf-up] {len(written)} PNG → {args.hf_upload_repo}")
            upload_frames_to_hf(written, args.out, args.hf_upload_repo, token)
        all_written.extend(written)
        if not args.keep_mp4:
            try:
                Path(local_path).unlink()
                print(f"[cleanup] removed {mp4_rel}")
            except OSError as e:
                print(f"[warn] failed to remove {local_path}: {e}", file=sys.stderr)

    # 7. Manifest (local & optionally HF)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(
            {
                "hf_source_repo": HF_SOURCE_REPO,
                "hf_dest_repo": args.hf_upload_repo if args.hf_upload else None,
                "cam_key": CAM_KEY,
                "sampling": {
                    "per_skill": args.per_skill,
                    "ep_range": args.ep_range,
                    "split": args.split,
                    "seed": args.seed,
                },
                "frames": all_written,
            },
            f,
            indent=2,
        )
    if args.hf_upload:
        print(f"[hf-up] manifest → {args.hf_upload_repo}")
        upload_manifest_to_hf(args.manifest, args.hf_upload_repo, token)
    print(f"[done] {len(all_written)} PNG under {args.out}, manifest at {args.manifest}")


if __name__ == "__main__":
    main()
