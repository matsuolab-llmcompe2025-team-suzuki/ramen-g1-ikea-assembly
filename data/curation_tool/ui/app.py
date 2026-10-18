"""
app.py — FastAPI backend for curation UI。

- 11 REST endpoints (episodes / segments / tile.mp4 / labels / categories /
  metadata / hf preview & sync)
- lifespan で .env load + LeRobot meta cache + workspace 準備
- static file (`ui/static/`) を "/" と "/static/" で serve
- CLI entry: `python -m ui.app --port 8765`

Test 用の factory `create_app(workspace, meta_loader, load_env)` を提供し、
`app.state.meta` を mock で差し替え可能にしている。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ui import hf_sync, labels_io, segments


# .env は iros_2026_ramen/ 直下 (data/curation_tool/ui/ の 3 階層上)
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Prefetch state (module-level、単一プロセス想定) ---

# Current ep 専用 (highest priority、全 tile pre-build)
_full_prefetch_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="prefetch-full",
)
# +1..+5 ep 用 (partial tile pre-build)
_prefetch_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="prefetch-partial",
)
_prefetch_done: set[int] = set()          # partial or full どちらか完了
_full_prefetch_done: set[int] = set()     # full 完了 (current ep target)
_prefetch_pending: set[int] = set()
_prefetch_lock = threading.Lock()


# Partial prefetch (+1..+5 ep) の tile 上限。50 で 実質全 seg カバー想定 (typical ep ~20 seg)。
PREFETCH_TILE_COUNT = 50


def _prefetch_ep(
    meta: Any, ep_id: int, workspace: Path, full_tiles: bool = False
) -> None:
    """指定 ep の raw parquet + video download + tile mp4 pre-build を実行。

    Args:
        full_tiles: True なら全 seg 分 tile を build (current ep 用)。
                    False なら先頭 PREFETCH_TILE_COUNT seg (+1..+5 ep 用)。

    build 順序:
      1. data parquet download (~few MB)
      2. 4 RGB video mp4 download (~160MB)
      3. get_segments (task_index run 検出)
      4. seg list [:limit] 分 tile mp4 を build (既存 cache は skip)
    """
    try:
        segments.get_task_index_per_frame(meta, ep_id)
        segments.get_episode_video_paths(meta, ep_id)

        try:
            segs = segments.get_segments(meta, ep_id)
            limit = len(segs) if full_tiles else PREFETCH_TILE_COUNT
            for seg in segs[:limit]:
                cache = segments.segment_tile_cache_path(
                    workspace, ep_id, seg.segment_id,
                )
                if not cache.exists():
                    segments.build_tile_mp4(meta, seg, cache)
        except Exception as e:
            print(f"[prefetch] ep {ep_id} tile pre-build failed: {e}", file=sys.stderr)

        with _prefetch_lock:
            _prefetch_done.add(ep_id)
            if full_tiles:
                _full_prefetch_done.add(ep_id)
            _prefetch_pending.discard(ep_id)
    except Exception as e:
        with _prefetch_lock:
            _prefetch_pending.discard(ep_id)
        print(f"[prefetch] ep {ep_id} failed: {e}", file=sys.stderr)


def _merge_categories_from_hf(workspace: Path) -> int:
    """HF から latest failure_categories.json を fetch、local と union merge。

    Return: 追加された新 key の数 (local に無かった remote category)。
    """
    try:
        token = hf_sync.resolve_token()
    except RuntimeError:
        return 0

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        remote_path = hf_hub_download(
            repo_id=hf_sync.DEFAULT_REPO_ID,
            filename="failure_categories.json",
            repo_type="dataset",
            token=token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return 0
    except Exception as e:
        print(f"[merge-categories] fetch failed: {e}", file=sys.stderr)
        return 0

    with open(remote_path, encoding="utf-8") as f:
        remote_data = json.load(f)
    remote_cats = list(remote_data.get("categories", []))

    local_cats = labels_io.load_categories(workspace)
    local_keys = {c["key"] for c in local_cats}
    added = 0
    for rc in remote_cats:
        if rc.get("key") and rc["key"] not in local_keys:
            local_cats.append(rc)
            added += 1

    if added > 0:
        path = workspace / "failure_categories.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"categories": local_cats}, f, ensure_ascii=False, indent=2)
    return added


def _cleanup_prefetched_outside_window(
    meta: Any, workspace: Path, current_ep: int, window_size: int = 5,
) -> dict:
    """current_ep ± window_size 外の prefetched cache を全削除。

    削除対象の union:
      - _prefetch_done set (同 session の prefetch 済 ep)
      - workspace/videos_cache/ep{NNN}/ の実 dir (server restart 越しの orphan)
    どちらでも該当すれば window 外なら削除、restart 耐性を確保。

    Return: {"cleaned_eps": [ep_ids], "total_local_tiles": N, "total_hf_hub_videos": M}
    """
    keep_set = set(range(current_ep, current_ep + window_size + 1))

    # in-memory prefetch state
    with _prefetch_lock:
        known_eps = set(_prefetch_done) | set(_full_prefetch_done)

    # on-disk videos_cache scan (restart 越し orphan の回収)
    videos_cache_root = workspace / "videos_cache"
    disk_eps: set[int] = set()
    if videos_cache_root.exists():
        for entry in videos_cache_root.iterdir():
            if entry.is_dir() and entry.name.startswith("ep"):
                try:
                    disk_eps.add(int(entry.name[2:]))
                except ValueError:
                    pass

    to_delete = sorted((known_eps | disk_eps) - keep_set)

    result = {
        "cleaned_eps": [],
        "total_local_tiles": 0,
        "total_hf_hub_videos": 0,
    }
    for ep in to_delete:
        try:
            r = _delete_ep_video_cache(meta, ep, workspace)
            result["cleaned_eps"].append(ep)
            result["total_local_tiles"] += r["local_tiles"]
            result["total_hf_hub_videos"] += r["hf_hub_videos"]
        except Exception as e:
            print(f"[cleanup] ep {ep} failed: {e}", file=sys.stderr)
    return result


def _delete_ep_video_cache(
    meta: Any, ep_id: int, workspace: Path
) -> dict:
    """A: workspace の tile mp4 cache + B: HF hub の 4 RGB video mp4 を削除。

    Return: {"local_tiles": int, "hf_hub_videos": int}
    """
    result = {"local_tiles": 0, "hf_hub_videos": 0}

    # A: local tile cache (workspace/videos_cache/epNNN/*.mp4)
    ep_dir = workspace / "videos_cache" / f"ep{ep_id:03d}"
    if ep_dir.exists():
        n = sum(1 for p in ep_dir.iterdir() if p.is_file())
        shutil.rmtree(ep_dir)
        result["local_tiles"] = n
        # prefetch state からも外す (再度 prefetch されて再 download 可能に)
        with _prefetch_lock:
            _prefetch_done.discard(ep_id)
            _full_prefetch_done.discard(ep_id)

    # B: HF hub video cache (symlink + underlying blob)
    try:
        paths = segments.get_episode_video_paths(meta, ep_id)
        for _label, path in paths.items():
            if path.exists() or path.is_symlink():
                # symlink なら blob も削除 (disk 節約)
                blob = None
                if path.is_symlink():
                    try:
                        blob = path.resolve(strict=False)
                    except OSError:
                        blob = None
                try:
                    path.unlink()
                except OSError:
                    pass
                if blob is not None and blob.exists() and blob != path:
                    try:
                        blob.unlink()
                    except OSError:
                        pass
                result["hf_hub_videos"] += 1
    except Exception as e:
        print(f"[cache-delete] hf_hub failed for ep {ep_id}: {e}", file=sys.stderr)

    return result


# --- pydantic payload schemas ---

class LabelPayload(BaseModel):
    """POST /api/labels payload (Schema v4)。primary key = (episode_id, frame_start)。"""

    episode_id: int
    frame_start: int
    frame_end: int
    task_index: int  # user が確定させた最終値
    verdict: str
    failure_category: Optional[str] = None
    reviewer: str = "hara"
    reviewed_at: Optional[str] = None


class CategoryAddPayload(BaseModel):
    """POST /api/categories payload。"""

    key: str
    label: str
    added_ep: int


class MetadataPayload(BaseModel):
    """POST /api/metadata payload (reviewer state)。"""

    last_ep: Optional[int] = None
    last_frame_start: Optional[int] = None
    reviewer: str = "hara"
    schema_version: int = 4


class HfSyncPayload(BaseModel):
    """POST /api/hf/sync payload。"""

    commit_message: Optional[str] = None


# --- App factory ---

def create_app(
    workspace: Optional[Path] = None,
    meta_loader: Optional[Callable[[], Any]] = None,
    load_env: bool = True,
) -> FastAPI:
    """
    FastAPI app を作成。

    Args:
        workspace: labels 保存先 (None なら labels_io の default)
        meta_loader: LeRobot meta を返す factory (None なら segments.load_metadata)
        load_env: True なら DEFAULT_ENV_PATH の .env を load_dotenv
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_env and DEFAULT_ENV_PATH.exists():
            from dotenv import load_dotenv
            load_dotenv(DEFAULT_ENV_PATH)
        app.state.workspace = labels_io.workspace_dir(workspace)
        app.state.meta = (meta_loader or segments.load_metadata)()
        yield

    app = FastAPI(lifespan=lifespan, title="Curation Tool", version="0.1.0")

    # --- Static ---
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        index_html = STATIC_DIR / "index.html"
        if not index_html.exists():
            raise HTTPException(404, "index.html not yet implemented")
        return FileResponse(str(index_html))

    # --- Episodes / Segments ---

    @app.get("/api/episodes")
    def api_episodes():
        return {"episodes": segments.list_episodes(app.state.meta)}

    @app.get("/api/task_names")
    def api_task_names():
        """LeRobot task_index → inference-style skill name の full mapping。

        UI の corrections panel で「task_index を数字ではなく名前で選ぶ」ために使う。
        """
        return {"task_names": segments.LEROBOT_TASK_INDEX_TO_SKILL_NAME}

    @app.get("/api/episodes/{ep_id}/segments")
    def api_segments(ep_id: int):
        try:
            segs = segments.get_segments(app.state.meta, ep_id)
        except (IndexError, KeyError) as e:
            raise HTTPException(404, f"episode {ep_id} not found: {e}")
        return {"segments": [asdict(s) for s in segs]}

    @app.get("/api/episodes/{ep_id}/segments/{seg_id}/tile.mp4")
    def api_tile(ep_id: int, seg_id: int):
        ws = app.state.workspace
        cache = segments.segment_tile_cache_path(ws, ep_id, seg_id)
        if not cache.exists():
            try:
                segs = segments.get_segments(app.state.meta, ep_id)
            except (IndexError, KeyError) as e:
                raise HTTPException(404, f"episode {ep_id} not found: {e}")
            if seg_id >= len(segs):
                raise HTTPException(
                    404, f"segment {seg_id} not in ep {ep_id} (has {len(segs)} segments)"
                )
            try:
                segments.build_tile_mp4(app.state.meta, segs[seg_id], cache)
            except FileNotFoundError as e:
                raise HTTPException(500, f"ffmpeg missing: {e}")
            except RuntimeError as e:
                # ffmpeg 失敗時の stderr が e に含まれる (Fix #3 for PR #83 review)
                raise HTTPException(500, str(e))
        return FileResponse(str(cache), media_type="video/mp4")

    @app.get("/api/episodes/{ep_id}/tile_range.mp4")
    def api_tile_range(ep_id: int, frame_start: int, frame_end: int):
        """任意 frame 範囲の 4-tile mp4 (split/merge 後の unit 用)。

        LeRobot の segment 境界と一致しない frame 範囲を build_tile_mp4 に synthetic
        Segment として渡し、cache key に (frame_start, frame_end) を含める。
        """
        if frame_start < 0 or frame_end < frame_start:
            raise HTTPException(400, f"invalid range: {frame_start}-{frame_end}")
        ws = app.state.workspace
        cache_name = f"range_{frame_start:06d}_{frame_end:06d}_tile.mp4"
        cache = ws / "videos_cache" / f"ep{ep_id:03d}" / cache_name
        if not cache.exists():
            synthetic = segments.Segment(
                episode_id=ep_id,
                segment_id=-1,
                task_index=-1,
                frame_start=frame_start,
                frame_end=frame_end,
                frame_count=frame_end - frame_start + 1,
                task_name="",
            )
            try:
                segments.build_tile_mp4(app.state.meta, synthetic, cache)
            except FileNotFoundError as e:
                raise HTTPException(500, f"ffmpeg missing: {e}")
            except RuntimeError as e:
                # ffmpeg 失敗時の stderr が e に含まれる (Fix #3 for PR #83 review)
                raise HTTPException(500, str(e))
            except Exception as e:
                raise HTTPException(500, f"tile build failed: {e}")
        return FileResponse(str(cache), media_type="video/mp4")

    # --- Labels ---

    @app.get("/api/labels")
    def api_labels_get(ep_id: Optional[int] = None):
        df = labels_io.load_labels(app.state.workspace)
        if ep_id is not None:
            df = df[df["episode_id"] == ep_id]
        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        return {"labels": records}

    @app.post("/api/labels")
    def api_label_post(payload: LabelPayload):
        row = labels_io.make_row(
            episode_id=payload.episode_id,
            frame_start=payload.frame_start,
            frame_end=payload.frame_end,
            task_index=payload.task_index,
            verdict=payload.verdict,
            failure_category=payload.failure_category,
            reviewer=payload.reviewer,
            reviewed_at=payload.reviewed_at,
        )
        try:
            labels_io.upsert_label(row, workspace=app.state.workspace)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"status": "ok"}

    @app.delete("/api/labels")
    def api_label_delete(ep_id: int, frame_start: int):
        """(episode_id, frame_start) key で削除 (merge 操作で consumed unit を消す用途)。"""
        deleted = labels_io.delete_label(
            ep_id, frame_start, workspace=app.state.workspace,
        )
        if not deleted:
            raise HTTPException(404, f"no label at ep={ep_id} frame_start={frame_start}")
        return {"status": "ok"}

    @app.post("/api/episodes/{ep_id}/reset")
    def api_ep_reset(ep_id: int):
        """指定 ep の全 label を削除 (検証用、LeRobot 元状態リセット)。"""
        n = labels_io.reset_ep_labels(ep_id, workspace=app.state.workspace)
        return {"status": "ok", "deleted": n}

    @app.post("/api/episodes/{ep_id}/prefetch")
    def api_ep_prefetch(ep_id: int, full_tiles: bool = False):
        """指定 ep の parquet + video を背景 download + tile pre-build。

        full_tiles=True の場合、_full_prefetch_pool (高優先度、max_workers=1) に
        submit + 全 seg 分 tile を build。current ep 用。
        """
        if ep_id >= app.state.meta.total_episodes or ep_id < 0:
            raise HTTPException(400, f"ep {ep_id} out of range")
        with _prefetch_lock:
            # full 要求 → full_done 済ならスキップ、そうでなければ (partial done でも)
            # 追加 tile を build する必要あるので処理継続
            if full_tiles:
                if ep_id in _full_prefetch_done:
                    return {"status": "done", "ep_id": ep_id, "mode": "full"}
            else:
                if ep_id in _prefetch_done:
                    return {"status": "done", "ep_id": ep_id, "mode": "partial"}
            if ep_id in _prefetch_pending:
                return {"status": "pending", "ep_id": ep_id}
            _prefetch_pending.add(ep_id)
        pool = _full_prefetch_pool if full_tiles else _prefetch_pool
        pool.submit(
            _prefetch_ep, app.state.meta, ep_id, app.state.workspace, full_tiles,
        )
        return {
            "status": "queued",
            "ep_id": ep_id,
            "mode": "full" if full_tiles else "partial",
        }

    @app.get("/api/prefetch_status")
    def api_prefetch_status():
        """Prefetch の進行状況 (done / full_done / pending の ep_id list)。"""
        with _prefetch_lock:
            return {
                "done": sorted(_prefetch_done),
                "full_done": sorted(_full_prefetch_done),
                "pending": sorted(_prefetch_pending),
            }

    @app.post("/api/prefetch_cleanup/{current_ep}")
    def api_prefetch_cleanup(current_ep: int, window_size: int = 5):
        """current_ep ± window_size 外の prefetched cache を全削除 (jump 時 sliding window)。"""
        if current_ep < 0 or current_ep >= app.state.meta.total_episodes:
            raise HTTPException(400, f"ep {current_ep} out of range")
        result = _cleanup_prefetched_outside_window(
            app.state.meta, app.state.workspace, current_ep, window_size,
        )
        return {"status": "ok", **result}

    @app.post("/api/episodes/{ep_id}/push_and_delete")
    def api_push_and_delete(ep_id: int, payload: HfSyncPayload):
        """指定 ep の作業完了に伴い HF push → 成功なら local + HF hub video cache 削除。

        - push: hf_sync.sync_to_hf() で labels.parquet + labels.jsonl +
          failure_categories.json を 1 commit で upload
        - 成功時のみ cache 削除 (A local tile + B HF hub 4 video)
        - Push 失敗時は 500 raise、cache は残す (再挑戦可能)
        """
        if ep_id < 0 or ep_id >= app.state.meta.total_episodes:
            raise HTTPException(400, f"ep {ep_id} out of range")

        # 0. HF の failure_categories.json を先に merge (他 reviewer が追加した category
        # を取り込んでから push、conflict/上書き喪失を防ぐ)
        try:
            categories_merged = _merge_categories_from_hf(app.state.workspace)
        except Exception as e:
            print(f"[merge-categories] skipped: {e}", file=sys.stderr)
            categories_merged = 0

        try:
            commit_url = hf_sync.sync_to_hf(
                app.state.workspace,
                commit_message=payload.commit_message,
            )
        except RuntimeError as e:
            raise HTTPException(401, f"HF push failed: {e}")
        except Exception as e:
            raise HTTPException(500, f"HF push failed: {e}")

        del_result = _delete_ep_video_cache(
            app.state.meta, ep_id, app.state.workspace,
        )
        return {
            "status": "ok",
            "commit_url": commit_url,
            "deleted": del_result,
            "categories_merged": categories_merged,
        }

    # --- Categories ---

    @app.get("/api/categories")
    def api_categories_get():
        return {"categories": labels_io.load_categories(app.state.workspace)}

    @app.post("/api/categories")
    def api_categories_add(payload: CategoryAddPayload):
        try:
            labels_io.add_category(
                payload.key, payload.label, payload.added_ep,
                workspace=app.state.workspace,
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"status": "ok"}

    # --- Metadata (reviewer state) ---

    @app.get("/api/metadata")
    def api_metadata_get():
        return labels_io.load_metadata(app.state.workspace)

    @app.post("/api/metadata")
    def api_metadata_save(payload: MetadataPayload):
        labels_io.save_metadata(payload.model_dump(), app.state.workspace)
        return {"status": "ok"}

    # --- HF sync ---

    @app.post("/api/hf/preview")
    def api_hf_preview():
        try:
            preview = hf_sync.compute_preview(app.state.workspace)
        except RuntimeError as e:
            raise HTTPException(401, str(e))
        return asdict(preview)

    @app.post("/api/hf/sync")
    def api_hf_sync_endpoint(payload: HfSyncPayload):
        try:
            url = hf_sync.sync_to_hf(
                app.state.workspace, commit_message=payload.commit_message,
            )
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return {"commit_url": url}

    return app


# --- Module-level app (uvicorn `ui.app:app` 用) ---

app = create_app()


# --- CLI entry ---

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ui.app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
