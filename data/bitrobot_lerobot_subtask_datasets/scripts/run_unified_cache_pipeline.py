"""overlay frame cache 作り直し pipeline の司令塔 (Issue #139)。

`configs/unified_frame_cache.yaml` の skill ごとに
  download → v0 展開 → YOLO 全件 → 焼き込み (token / overlay / common) → tar 化 → upload → verify
を、使う資源 (CPU / GPU / disk / network) が重ならないように並べて流す。

# 並列の組み方 (Sakura: 24 vCPU / H100 1 枚 / NVMe)

- download: 全 skill 同時 (network)
- v0 展開: 全 skill 同時。skill ごとの `decode_jobs` で CPU を配分
- YOLO: head cam の展開が揃った skill から 1 skill ずつ、`num_shards` process を 1 GPU に並べる
- 焼き込み: 1 skill ずつ。焼く種類は skill の `kinds` (未指定は token / overlay / common の 3 種)。
  worker は CPU / GPU 混成 (`bake.jobs` 個のうち `bake.cuda_workers` 個が
  GPU)。前の skill を焼いている間に次の skill の展開 (CPU) / YOLO (GPU) が進む
- `gpu.mps: true` なら CUDA MPS を起動し、YOLO shard と GPU worker の kernel を同時に走らせる
- tar 化: 焼き上がった mp4 から順に (低優先度)
- upload: tar ができた分から繰り返し (低優先度)。全 tar ができたら最後の 1 回で終わる

各段は `<work>/<skill>/.<stage>.done` を残し、再実行時は済んだ段を飛ばす。下位 script も
既存出力を skip するので、途中で止まっても同じ command で再開できる。
旧ファイルの削除 (cleanup) はここでは実行しない。verify 後に削除対象の一覧 (dry-run) を
log に出すので、確認してから `sync_frame_cache_to_hf.py cleanup --execute` を手で打つ。

# Usage (repo root から、main env で)

    pixi run --manifest-path model/subtask_policy_training/pixi.toml python \\
        data/bitrobot_lerobot_subtask_datasets/scripts/run_unified_cache_pipeline.py \\
        --config data/bitrobot_lerobot_subtask_datasets/configs/unified_frame_cache.yaml \\
        [--skills rotate_table_base] [--work-root /nvme/uc_smoke] [--dry-run]

    # smoke (本番 repo には上げない): 各 cam 先頭 1 mp4 の先頭 300 frame で最後まで流す
    ... --skills rotate_table_base --limit-files 1 --smoke-frames 300 --no-upload \\
        --work-root /nvme/uc_smoke

HF token は repo root の `.env` (HF_TOKEN) から読む (環境変数が既にあればそちらを優先)。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "data" / "bitrobot_lerobot_subtask_datasets" / "scripts"
DECODE_SCRIPT = SCRIPTS_DIR / "precompute_frame_cache.py"
PACK_SCRIPT = SCRIPTS_DIR / "pack_frame_cache_tars.py"
SYNC_SCRIPT = SCRIPTS_DIR / "sync_frame_cache_to_hf.py"
RESTORE_SCRIPT = SCRIPTS_DIR / "restore_frame_cache_tars.sh"
YOLO_SCRIPT = REPO_ROOT / "data" / "yolo_obb" / "scripts" / "precompute_yolo_obb.py"
BAKE_SCRIPT = REPO_ROOT / "model" / "ramen_ori" / "scripts" / "precompute_augmented_frame_cache_ramen_ori.py"
FRAME_COUNT_NAME = "_frame_count.txt"
CACHE_META_NAME = "_cache_meta.json"
V2_KINDS = ("overlay", "token", "common")


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 設定と path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillPaths:
    root: Path

    @property
    def src(self) -> Path:  # HF から落とした meta / data / videos
        return self.root / "src"

    @property
    def v0(self) -> Path:
        return self.root / "v0"

    @property
    def obb(self) -> Path:
        return self.root / "obb_yolo"

    @property
    def v2(self) -> Path:
        return self.root / "v2"

    @property
    def staging(self) -> Path:  # HF repo の path を写した upload 元
        return self.root / "staging"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def marker(self, stage: str) -> Path:
        return self.root / f".{stage}.done"

    def log(self, stage: str) -> Path:
        return self.logs / f"{stage}.log"


def load_config(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dotenv(path: Path) -> None:
    """KEY=VALUE 形式の .env を環境変数に読み込む (既にある key は上書きしない、値は表示しない)。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def yolo_cache_hash(cfg: dict[str, Any]) -> str:
    """YOLO cache dir 名 (precompute_yolo_obb.py と同じ計算)。"""
    spec = importlib.util.spec_from_file_location("precompute_yolo_obb", YOLO_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass の annotation 解決が sys.modules を参照するので exec 前に登録する
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    ref = module.YoloCkptRef(kind="hf", repo_id=cfg["yolo"]["ckpt"], revision=cfg["yolo"]["revision"])
    return ref.to_hash()


def all_cameras(cfg: dict[str, Any]) -> list[str]:
    return [cfg["cameras"]["head"], *cfg["cameras"]["wrist"]]


# ---------------------------------------------------------------------------
# command (純関数、test しやすいよう実行と分ける)
# ---------------------------------------------------------------------------


def _py(script: Path, *args: Any) -> list[str]:
    """main env (この script を動かしている python) で動かす command。"""
    return [sys.executable, str(script), *map(str, args)]


def decode_cmd(
    cfg: dict[str, Any], sp: SkillPaths, skill: dict[str, Any], smoke_frames: int | None = None
) -> list[str]:
    cmd = _py(
        DECODE_SCRIPT,
        "--lerobot-root", sp.src,
        "--cache-root", sp.v0,
        "--video-keys", ",".join(all_cameras(cfg)),
        "--keep-source-mp4",
        "--jobs", skill["decode_jobs"],
    )
    if smoke_frames is not None:
        cmd += ["--max-frames-per-mp4", str(smoke_frames)]
    return cmd


def yolo_cmd(cfg: dict[str, Any], sp: SkillPaths, shard: int) -> list[str]:
    y = cfg["yolo"]
    return [
        "pixi", "run", "--manifest-path", str(REPO_ROOT / cfg["envs"]["yolo"]),
        "python", str(YOLO_SCRIPT),
        "--frame-cache-root", str(sp.v0),
        "--yolo-ckpt", y["ckpt"],
        "--yolo-revision", y["revision"],
        "--output-root", str(sp.obb),
        "--cameras", cfg["cameras"]["head"],
        "--top-k", str(y["top_k"]),
        "--num-classes", str(y["num_classes"]),
        "--imgsz", str(y["imgsz"]),
        "--conf", str(y["conf"]),
        "--batch-size", str(y["batch_size"]),
        "--fail-on-truncation",
        "--num-shards", str(y["num_shards"]),
        "--shard-index", str(shard),
    ]


def skill_kinds(skill: dict[str, Any]) -> tuple[str, ...]:
    """skill の焼く種類 (`kinds`、未指定は V2_KINDS の 3 種)。V2_KINDS の順に並べて返す。

    wrist は common にしか書かれない (overlay の tar は head だけを詰める) ので common は必須、
    head の出力として overlay / token の少なくとも 1 つが要る。
    """
    kinds = set(skill.get("kinds", V2_KINDS))
    if kinds - set(V2_KINDS) or "common" not in kinds or not kinds & {"overlay", "token"}:
        raise PipelineError(
            f"{skill['name']}: kinds {sorted(kinds)} は使えない "
            f"({list(V2_KINDS)} から選び、common と overlay / token の少なくとも 1 つを含める)"
        )
    return tuple(k for k in V2_KINDS if k in kinds)


def bake_cmd(
    cfg: dict[str, Any], sp: SkillPaths, skill: dict[str, Any], config_path: Path, obb_hash: str
) -> list[str]:
    b = cfg["bake"]
    roots = [a for k in skill_kinds(skill) for a in (f"--output-{k}-root", sp.v2 / k)]
    cmd = _py(
        BAKE_SCRIPT,
        "--input-cache-root", sp.v0,
        *roots,
        "--aug-config-yaml", config_path,
        "--obb-root", sp.obb,
        "--obb-hash", obb_hash,
        "--obb-cameras", cfg["cameras"]["head"],
        "--obb-top-k", cfg["yolo"]["top_k"],
        "--obb-num-classes", cfg["yolo"]["num_classes"],
        "--num-variants", b["num_variants"],
        "--jobs", b["jobs"],
        "--cuda-workers", b["cuda_workers"],
        "--torch-threads", b["torch_threads"],
        "--chunksize", b["chunksize"],
        "--overlay-conf", b["overlay_conf"],
        "--overlay-thickness", b["overlay_thickness"],
    )
    if b.get("gpu_jpeg_encode"):
        cmd.append("--gpu-jpeg-encode")
    return cmd


def pack_v0_cmd(cfg: dict[str, Any], sp: SkillPaths) -> list[str]:
    return _py(
        PACK_SCRIPT,
        "--src-root", sp.v0,
        "--dest-dir", sp.staging / "frame_cache_tars",
        "--mode", "v0",
        "--jobs", cfg["pack"]["jobs"],
    )


def pack_v2_cmd(cfg: dict[str, Any], sp: SkillPaths, kind: str) -> list[str]:
    cams = cfg["cameras"]["wrist"] if kind == "common" else [cfg["cameras"]["head"]]
    return _py(
        PACK_SCRIPT,
        "--src-root", sp.v2 / kind,
        "--dest-dir", sp.staging / "frame_cache_v2_tars" / kind,
        "--mode", "variants",
        "--cameras", ",".join(cams),
        "--expect-from", sp.v0,
        "--watch",
        "--done-flag", sp.marker("bake"),
        "--jobs", cfg["pack"]["jobs"],
        "--poll-interval", cfg["pack"]["poll_interval_s"],
    )


def sync_cmd(cfg: dict[str, Any], sp: SkillPaths, repo_id: str, sub: str) -> list[str]:
    cmd = _py(SYNC_SCRIPT, sub, "--repo-id", repo_id, "--staging", sp.staging)
    if sub == "upload":
        cmd += [
            "--num-workers", str(cfg["upload"]["num_workers"]),
            "--until-flag", str(sp.marker("pack")),
            "--interval", str(cfg["upload"]["interval_s"]),
        ]
    return cmd


def download_patterns(cfg: dict[str, Any], limit_files: int | None) -> list[str]:
    """HF から落とす path。学習に使う 3 cam の video だけ (IR / head_right は落とさない)。"""
    patterns = ["meta/*", "data/*"]
    for cam in all_cameras(cfg):
        if limit_files is None:
            patterns.append(f"videos/{cam}/*")
        else:
            patterns += [f"videos/{cam}/chunk-000/file-{i:03d}.mp4" for i in range(limit_files)]
    return patterns


def mps_env(work_root: Path) -> dict[str, str]:
    """CUDA MPS の pipe / log dir (子 process はこの env で MPS daemon につながる)。"""
    return {
        "CUDA_MPS_PIPE_DIRECTORY": str(work_root / "mps" / "pipe"),
        "CUDA_MPS_LOG_DIRECTORY": str(work_root / "mps" / "log"),
    }


def start_mps(work_root: Path, runner: "Runner") -> Callable[[], None]:
    """MPS daemon を起動し、停止関数を返す。env は以降の子 process に引き継がれる。"""
    env = mps_env(work_root)
    for d in env.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    runner.note(f"CUDA MPS started ({env['CUDA_MPS_PIPE_DIRECTORY']})")

    def stop() -> None:
        subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", check=False)
        runner.note("CUDA MPS stopped")

    return stop


def cam_decoded(sp: SkillPaths, cam: str) -> bool:
    """cam の全 mp4 の展開が終わったか (mp4 ごとの _frame_count.txt で判定)。"""
    video_dir = sp.src / "videos" / cam
    mp4s = sorted(video_dir.rglob("*.mp4")) if video_dir.is_dir() else []
    if not mp4s:
        return False
    for mp4 in mp4s:
        rel = mp4.relative_to(video_dir)
        if not (sp.v0 / cam / rel.parent / rel.stem / FRAME_COUNT_NAME).is_file():
            return False
    return True


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


class Runner:
    """subprocess を log 付きで動かす。1 つでも失敗したら残りを止める。"""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.failed = threading.Event()
        self.errors: list[BaseException] = []
        self._lock = threading.Lock()

    def note(self, msg: str) -> None:
        with self._lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def run(self, cmd: list[str], log: Path, *, low_priority: bool = False) -> None:
        if self.dry_run:
            self.note(f"(dry-run) {' '.join(cmd)}  > {log}")
            return
        if self.failed.is_set():
            raise PipelineError("aborted (他の段が失敗)")
        if low_priority:
            prefix = ["nice", "-n", "19"]
            if shutil.which("ionice"):
                prefix += ["ionice", "-c3"]
            cmd = prefix + cmd
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
            while True:
                try:
                    rc = proc.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    if self.failed.is_set():
                        proc.terminate()
                        proc.wait()
                        raise PipelineError(f"aborted: {log.name}") from None
        if rc != 0:
            self.failed.set()
            raise PipelineError(f"failed (rc={rc}): see {log}")

    def start(self, fn: Callable[[], None], name: str) -> threading.Thread:
        """fn を background thread で動かす。例外は errors に積んで全体を止める。"""

        def target() -> None:
            try:
                fn()
            except BaseException as e:  # noqa: BLE001
                with self._lock:
                    self.errors.append(e)
                self.failed.set()
                self.note(f"ERROR in {name}: {e}")

        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        return t

    def join(self, *threads: threading.Thread) -> None:
        for t in threads:
            t.join()
        if self.failed.is_set():
            raise PipelineError("aborted (他の段が失敗)")

    def wait_until(self, predicate: Callable[[], bool], poll: float = 10.0) -> None:
        while not predicate():
            if self.failed.is_set():
                raise PipelineError("aborted (他の段が失敗)")
            time.sleep(poll)


@dataclass
class Context:
    cfg: dict[str, Any]
    config_path: Path
    work_root: Path
    obb_hash: str
    limit_files: int | None
    smoke_frames: int | None
    upload: bool
    runner: Runner
    gpu_lock: threading.Lock
    bake_lock: threading.Lock


def _stage(ctx: Context, sp: SkillPaths, name: str, fn: Callable[[], None]) -> None:
    """完了 marker が無ければ fn を実行して marker を置く。"""
    if sp.marker(name).exists():
        ctx.runner.note(f"{sp.root.name}: {name} skip (done)")
        return
    ctx.runner.note(f"{sp.root.name}: {name} start")
    fn()
    if not ctx.runner.dry_run:
        sp.marker(name).touch()
    ctx.runner.note(f"{sp.root.name}: {name} done")


def _download(ctx: Context, sp: SkillPaths, repo_id: str) -> None:
    patterns = download_patterns(ctx.cfg, ctx.limit_files)
    if ctx.runner.dry_run:
        ctx.runner.note(f"(dry-run) snapshot_download {repo_id} → {sp.src} {patterns}")
        return
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=sp.src,
        allow_patterns=patterns,
        max_workers=16,
    )


def _link_tree(src: Path, dst: Path) -> None:
    """src 配下の file を dst に hardlink (別 fs なら copy)。"""
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        try:
            os.link(path, out)
        except OSError:
            shutil.copy2(path, out)


def _stage_obb(ctx: Context, sp: SkillPaths) -> None:
    src = sp.obb / ctx.obb_hash
    if ctx.runner.dry_run:
        ctx.runner.note(f"(dry-run) link {src} → {sp.staging / 'obb_yolo' / ctx.obb_hash}")
        return
    _link_tree(src, sp.staging / "obb_yolo" / ctx.obb_hash)


def _stage_support_files(ctx: Context, sp: SkillPaths) -> None:
    """restore.sh と _cache_meta.json を tar 置き場に置く。"""
    if ctx.runner.dry_run:
        return
    for sub in ("frame_cache_tars", "frame_cache_v2_tars"):
        dest = sp.staging / sub
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RESTORE_SCRIPT, dest / "restore.sh")
        shutil.copy2(sp.v0 / CACHE_META_NAME, dest / CACHE_META_NAME)


def run_skill(ctx: Context, skill: dict[str, Any]) -> None:
    """1 skill の全段。skill 間の GPU / 焼き込みの取り合いは lock で 1 skill ずつにする。"""
    name, repo_id = skill["name"], skill["repo_id"]
    sp = SkillPaths(ctx.work_root / name)
    r = ctx.runner
    cfg = ctx.cfg

    _stage(ctx, sp, "download", lambda: _download(ctx, sp, repo_id))

    decode_t = r.start(
        lambda: _stage(
            ctx, sp, "decode",
            lambda: r.run(decode_cmd(cfg, sp, skill, ctx.smoke_frames), sp.log("decode")),
        ),
        f"{name}:decode",
    )

    # head cam が揃えば wrist の展開を待たずに YOLO を始める
    if not r.dry_run and not sp.marker("yolo").exists():
        r.wait_until(lambda: cam_decoded(sp, cfg["cameras"]["head"]))

    def yolo() -> None:
        shards = [
            r.start(
                lambda i=i: r.run(yolo_cmd(cfg, sp, i), sp.log(f"yolo_shard{i}")),
                f"{name}:yolo{i}",
            )
            for i in range(cfg["yolo"]["num_shards"])
        ]
        r.join(*shards)

    with ctx.gpu_lock:
        _stage(ctx, sp, "yolo", yolo)
    _stage(ctx, sp, "stage_obb", lambda: _stage_obb(ctx, sp))

    upload_t = None
    if ctx.upload:
        upload_t = r.start(
            lambda: _stage(
                ctx, sp, "upload",
                lambda: r.run(sync_cmd(cfg, sp, repo_id, "upload"), sp.log("upload"), low_priority=True),
            ),
            f"{name}:upload",
        )

    r.join(decode_t)
    _stage_support_files(ctx, sp)
    pack_v0_t = r.start(
        lambda: _stage(
            ctx, sp, "pack_v0",
            lambda: r.run(pack_v0_cmd(cfg, sp), sp.log("pack_v0"), low_priority=True),
        ),
        f"{name}:pack_v0",
    )

    with ctx.bake_lock:
        pack_v2_ts = []
        if not sp.marker("pack_v2").exists():
            pack_v2_ts = [
                r.start(
                    lambda k=k: r.run(pack_v2_cmd(cfg, sp, k), sp.log(f"pack_{k}"), low_priority=True),
                    f"{name}:pack_{k}",
                )
                for k in skill_kinds(skill)
            ]
        _stage(
            ctx, sp, "bake",
            lambda: r.run(bake_cmd(cfg, sp, skill, ctx.config_path, ctx.obb_hash), sp.log("bake")),
        )
    r.join(*pack_v2_ts)
    if not r.dry_run:
        sp.marker("pack_v2").touch()
    r.join(pack_v0_t)
    if not r.dry_run:
        sp.marker("pack").touch()  # uploader の最終回の合図

    if upload_t is not None:
        r.join(upload_t)
        _stage(ctx, sp, "verify", lambda: r.run(sync_cmd(cfg, sp, repo_id, "verify"), sp.log("verify")))
        # 削除はしない。削除対象の一覧だけ log に出す
        r.run(sync_cmd(cfg, sp, repo_id, "cleanup"), sp.log("cleanup_dry_run"))
        r.note(f"{name}: 旧ファイルの削除対象は {sp.log('cleanup_dry_run')} を確認")
    r.note(f"{name}: ALL DONE")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skills", default=None, help="対象 skill (comma 区切り)。未指定なら全 skill")
    p.add_argument("--work-root", type=Path, default=None, help="config の work_root を上書き")
    p.add_argument("--limit-files", type=int, default=None, help="cam ごとに先頭 N 個の mp4 だけ (smoke 用)")
    p.add_argument(
        "--smoke-frames",
        type=int,
        default=None,
        help="smoke 用: mp4 ごとに先頭 N frame だけ展開して最後まで流す (本番の repo に上げないこと)",
    )
    p.add_argument("--no-upload", action="store_true", help="upload / verify をしない")
    p.add_argument("--dry-run", action="store_true", help="実行せず command を表示")
    args = p.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    skills = cfg["skills"]
    if args.skills:
        wanted = {s.strip() for s in args.skills.split(",")}
        unknown = wanted - {s["name"] for s in skills}
        if unknown:
            p.error(f"unknown skills: {sorted(unknown)}")
        skills = [s for s in skills if s["name"] in wanted]
    for s in skills:  # 数時間後の焼き込みで気づかないよう、kinds の誤りは何も始める前に止める
        try:
            skill_kinds(s)
        except PipelineError as e:
            p.error(str(e))
    if args.smoke_frames is not None and not args.no_upload:
        p.error("--smoke-frames は途中までの cache を作るので --no-upload と一緒に使う")
    work_root = args.work_root or Path(cfg["work_root"])
    runner = Runner(dry_run=args.dry_run)
    ctx = Context(
        cfg=cfg,
        config_path=args.config.resolve(),
        work_root=work_root,
        obb_hash=yolo_cache_hash(cfg),
        limit_files=args.limit_files,
        smoke_frames=args.smoke_frames,
        upload=not args.no_upload,
        runner=runner,
        gpu_lock=threading.Lock(),
        bake_lock=threading.Lock(),
    )
    runner.note(
        f"work_root={work_root} skills={[s['name'] for s in skills]} obb_hash={ctx.obb_hash} "
        f"upload={ctx.upload} limit_files={args.limit_files} smoke_frames={args.smoke_frames}"
    )

    if not args.dry_run:
        # YOLO の weight は shard 間で取り合わないよう先に 1 回だけ落とす
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snapshot_download(
            repo_id=cfg["yolo"]["ckpt"], revision=cfg["yolo"]["revision"], repo_type="model"
        )

    stop_mps: Callable[[], None] | None = None
    if cfg.get("gpu", {}).get("mps"):
        if args.dry_run:
            runner.note(f"(dry-run) start CUDA MPS {mps_env(work_root)}")
        else:
            stop_mps = start_mps(work_root, runner)
    try:
        threads = [runner.start(lambda s=s: run_skill(ctx, s), s["name"]) for s in skills]
        for t in threads:
            t.join()
    finally:
        if stop_mps is not None:
            stop_mps()
    if runner.errors:
        for e in runner.errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    runner.note("pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
