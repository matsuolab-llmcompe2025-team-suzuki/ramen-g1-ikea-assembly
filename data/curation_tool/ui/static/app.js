"use strict";

/*
 * curation_tool UI (Schema v4)
 *
 * 単位: unit = 1 label = 1 skill (LeRobot の segment 概念とは分離)。
 * 表示は state.units array で管理し、split / merge で編集可能。
 * primary key = (episode_id, frame_start)、unit_id は表示上のみ (sort 順)。
 *
 * Shortcuts:
 *   1-3      即 save+next / 4 = failure (category 選択)
 *   [ / ]    現 frame を start / end に grab (今 unit の範囲変更)
 *   s        現 frame で split (今 unit を 2 unit に)
 *   m        merge with prev (前 unit と統合)
 *   c        corrections panel toggle
 *   f / +    failure category focus / add
 *   Space    play/pause / ←→ frame step / Shift+←→ 1sec step
 *   Backspace  前 unit に戻る
 */

const state = {
  episodes: [],
  currentEp: 0,
  segments: [],       // LeRobot の元 segments (unit 生成の base)
  units: [],          // 現在の view (labels 反映 + split/merge 結果)
  currentUnitIdx: 0,
  fps: 30,
  categories: [],
  taskNames: {},
  totalLabeled: 0,
  correctionsVisible: true,
  pendingVerdict: null,
  pendingDeletes: [], // save 前 に消したい frame_starts (merge で consumed unit)
};

// --- API ---

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

const fetchEpisodes = async () => (await api("/api/episodes")).episodes;
const fetchSegments = async (ep) => (await api(`/api/episodes/${ep}/segments`)).segments;
const fetchLabels = async (ep = null) => {
  const p = ep !== null ? `/api/labels?ep_id=${ep}` : "/api/labels";
  return (await api(p)).labels;
};
const fetchMetadata = async () => await api("/api/metadata");
const saveMetadata = async (meta) => await api("/api/metadata", {
  method: "POST", body: JSON.stringify(meta),
});
const postLabel = async (payload) => await api("/api/labels", {
  method: "POST", body: JSON.stringify(payload),
});
const deleteLabel = async (ep_id, frame_start) => {
  const res = await fetch(
    `/api/labels?ep_id=${ep_id}&frame_start=${frame_start}`,
    { method: "DELETE" },
  );
  return res.ok;
};
const fetchCategories = async () => (await api("/api/categories")).categories;
const postCategory = async (key, label, added_ep) => await api("/api/categories", {
  method: "POST", body: JSON.stringify({ key, label, added_ep }),
});
const fetchTaskNames = async () => (await api("/api/task_names")).task_names;
const postPrefetch = async (ep, fullTiles = false) => {
  const q = fullTiles ? "?full_tiles=true" : "";
  return await api(`/api/episodes/${ep}/prefetch${q}`, { method: "POST" });
};
const fetchPrefetchStatus = async () => await api("/api/prefetch_status");
const fetchHfPreview = async () => await api("/api/hf/preview", { method: "POST" });
const postHfSync = async (commit_message) => await api("/api/hf/sync", {
  method: "POST", body: JSON.stringify({ commit_message }),
});

// --- Utils ---

function el(id) { return document.getElementById(id); }

function setStatus(msg, kind = "info") {
  const s = el("status-msg");
  s.textContent = msg;
  s.style.color = kind === "error" ? "#ff8080" : "#b8ffb8";
  if (kind !== "error") {
    setTimeout(() => { if (s.textContent === msg) s.textContent = ""; }, 3000);
  }
}

function taskName(idx) {
  return state.taskNames[idx] || `t${idx}`;
}

function currentFrameOffset() {
  const video = el("tile-video");
  return Math.round(video.currentTime * state.fps);
}

function currentFrameAbsolute() {
  const unit = state.units[state.currentUnitIdx];
  return unit ? unit.frame_start + currentFrameOffset() : 0;
}

// --- Units 構築: LeRobot segments + saved labels の overlay ---

function buildUnits(segments, labels) {
  const units = [];
  // saved labels をまず入れる (task_index / frame は saved 側優先)
  labels.forEach((l) => {
    units.push({
      frame_start: l.frame_start,
      frame_end: l.frame_end,
      task_index: l.task_index,
      saved: true,
      verdict: l.verdict,
      failure_category: l.failure_category,
      original_frame_start: l.frame_start,
    });
  });
  // LeRobot の segment を「labels でカバーされない範囲」に分割して追加
  // (前実装は部分 overlap でも seg 全体 skip し frame 落ちのバグ。fix: 未 covered
  // range だけ切り出して unlabeled unit として追加)
  segments.forEach((seg) => {
    const uncovered = _subtractLabelRanges(seg, labels);
    uncovered.forEach((range) => {
      units.push({
        frame_start: range.start,
        frame_end: range.end,
        task_index: seg.task_index,
        saved: false,
        original_frame_start: null,
      });
    });
  });
  return units.sort((a, b) => a.frame_start - b.frame_start);
}

function _subtractLabelRanges(seg, labels) {
  // seg 範囲内で labels に覆われない frame 区間を返す。
  // 例: seg=[0,172], labels=[{0,122}] → [{123,172}]
  const relevant = labels
    .filter((l) => l.frame_start <= seg.frame_end && l.frame_end >= seg.frame_start)
    .sort((a, b) => a.frame_start - b.frame_start);
  const ranges = [];
  let cursor = seg.frame_start;
  for (const l of relevant) {
    const lStart = Math.max(l.frame_start, seg.frame_start);
    const lEnd = Math.min(l.frame_end, seg.frame_end);
    if (cursor < lStart) {
      ranges.push({ start: cursor, end: lStart - 1 });
    }
    cursor = Math.max(cursor, lEnd + 1);
  }
  if (cursor <= seg.frame_end) {
    ranges.push({ start: cursor, end: seg.frame_end });
  }
  return ranges;
}

// --- Render ---

function renderTopBar() {
  el("ep-info").textContent = `ep ${state.currentEp}/${state.episodes.length - 1}`;
  el("seg-info").textContent = state.units.length
    ? `unit ${state.currentUnitIdx + 1}/${state.units.length}`
    : "unit -/-";
  el("progress").textContent = `labeled: ${state.totalLabeled}`;
}

function renderUnitList() {
  const ul = el("segment-list");
  ul.innerHTML = "";
  state.units.forEach((u, i) => {
    const li = document.createElement("li");
    const marker = u.saved ? "✓" : (i === state.currentUnitIdx ? "▶" : " ");
    const name = taskName(u.task_index);
    li.textContent = `${marker} ${i}: ${name} (${u.frame_end - u.frame_start + 1}f)`;
    li.title = `task_index=${u.task_index}: ${name}\nframes ${u.frame_start}-${u.frame_end}`;
    if (u.saved) li.classList.add("done");
    if (u.saved && u.verdict) li.classList.add(`verdict-${u.verdict}`);
    if (i === state.currentUnitIdx) li.classList.add("current");
    // merge 可視化: 隣接で同 task_index + 同 verdict + 両方 saved
    // (同 skill でも verdict 違い = retry の失敗/成功等 は別 group 扱い)
    const prev = state.units[i - 1];
    if (prev && u.saved && prev.saved
        && prev.task_index === u.task_index
        && prev.verdict === u.verdict) {
      li.classList.add("merged-with-prev");
    }
    li.addEventListener("click", () => jumpToUnit(i));
    ul.appendChild(li);
  });
}

function renderUnitInfo() {
  const u = state.units[state.currentUnitIdx];
  if (!u) return;
  el("info-task-select").value = String(u.task_index);
  el("info-task").textContent = `[idx ${u.task_index}]`;
  const dur = u.frame_end - u.frame_start + 1;
  el("info-frames").textContent = `${u.frame_start} – ${u.frame_end} (${dur} frames)`;
  el("info-duration").textContent = `${(dur / state.fps).toFixed(2)}s @ ${state.fps}fps`;
  el("info-current-frame").textContent = `${u.frame_start} (offset 0)`;

  // Saved badge (verdict 色付き、未 save は hidden)
  const badge = el("info-saved-badge");
  badge.className = "";  // reset
  if (u.saved && u.verdict) {
    badge.textContent =
      `✓ SAVED (${u.verdict}${u.failure_category ? ": " + u.failure_category : ""})`;
    badge.classList.add(`v-${u.verdict}`);
  } else {
    badge.classList.add("hidden");
  }
}

function showSaveToast(unitIdx, verdict, failure_category) {
  const toast = document.createElement("div");
  toast.className = `save-toast v-${verdict}`;
  const jp = { optimal: "最良", success: "成功", ambiguous: "微妙", failure: "失敗" }[verdict] || verdict;
  const catSuffix = failure_category ? ` / ${failure_category}` : "";
  toast.innerHTML = `✓ SAVED unit ${unitIdx} = ${jp}<span class="toast-sub">${verdict}${catSuffix}</span>`;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 250);
  }, 1400);
}

// --- Timeline (focused: prev + current + next の 3 unit) ---

function getFocusedUnits() {
  // current の周辺 3 unit (index 付き) を返す。端では 2 or 1 に縮む。
  const idx = state.currentUnitIdx;
  const arr = [];
  if (idx - 1 >= 0) arr.push({ unit: state.units[idx - 1], globalIdx: idx - 1, role: "prev" });
  if (state.units[idx]) arr.push({ unit: state.units[idx], globalIdx: idx, role: "current" });
  if (idx + 1 < state.units.length) arr.push({ unit: state.units[idx + 1], globalIdx: idx + 1, role: "next" });
  return arr;
}

function getFocusedRange() {
  const focused = getFocusedUnits();
  if (focused.length === 0) return { start: 0, end: 1 };
  return {
    start: focused[0].unit.frame_start,
    end: focused[focused.length - 1].unit.frame_end,
  };
}

function renderTimeline() {
  const track = el("timeline-track");
  track.innerHTML = "";
  const focused = getFocusedUnits();
  const totalUnits = state.units.length;
  el("timeline-ep-info").textContent =
    `ep ${state.currentEp} · unit ${state.currentUnitIdx + 1}/${totalUnits} · focused ${focused.length}`;
  if (focused.length === 0) return;

  focused.forEach((f, i) => {
    const u = f.unit;
    const dur = u.frame_end - u.frame_start + 1;
    const div = document.createElement("div");
    div.className = "timeline-unit";
    div.style.flexGrow = dur;
    div.style.flexShrink = 1;
    div.style.flexBasis = "0";
    if (u.saved && u.verdict) div.classList.add(`verdict-${u.verdict}`);
    if (f.role === "current") div.classList.add("current");
    div.title =
      `unit ${f.globalIdx}: ${taskName(u.task_index)} (${u.frame_start}-${u.frame_end}` +
      (u.saved ? `, ${u.verdict}` : ", unlabeled") + ")";
    // 内部 label
    const inner = document.createElement("div");
    inner.className = "timeline-unit-label";
    inner.textContent = `${f.globalIdx}: ${taskName(u.task_index)}`;
    div.appendChild(inner);
    div.addEventListener("click", (e) => {
      if (e.target.classList.contains("boundary-handle")) return;
      jumpToUnit(f.globalIdx);
    });
    track.appendChild(div);

    // 境界 handle (次 unit との間)
    if (i < focused.length - 1) {
      const handle = document.createElement("div");
      handle.className = "boundary-handle";
      handle.dataset.leftIdx = f.globalIdx;
      handle.dataset.rightIdx = focused[i + 1].globalIdx;
      handle.title = "drag to move boundary";
      handle.addEventListener("mousedown", startBoundaryDrag);
      track.appendChild(handle);
    }
  });

  updateTimelinePlayhead();
}

function updateTimelinePlayhead() {
  const { start, end } = getFocusedRange();
  const total = Math.max(1, end - start + 1);
  const abs = currentFrameAbsolute();
  const pct = ((abs - start) / total) * 100;
  const playhead = el("timeline-playhead");
  if (pct < 0 || pct > 100) {
    playhead.style.display = "none";
  } else {
    playhead.style.display = "block";
    playhead.style.left = pct + "%";
  }
}

// --- Boundary drag ---

let boundaryDragState = null;

function startBoundaryDrag(e) {
  const handle = e.currentTarget;
  const leftIdx = parseInt(handle.dataset.leftIdx, 10);
  const rightIdx = parseInt(handle.dataset.rightIdx, 10);
  const rect = el("episode-timeline").getBoundingClientRect();
  const range = getFocusedRange();
  boundaryDragState = {
    leftIdx,
    rightIdx,
    timelineRect: rect,
    rangeStart: range.start,
    totalFrames: Math.max(1, range.end - range.start + 1),
  };
  document.body.style.cursor = "col-resize";
  e.preventDefault();
  e.stopPropagation();
}

document.addEventListener("mousemove", (e) => {
  if (!boundaryDragState) return;
  const rect = boundaryDragState.timelineRect;
  const relX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
  const pct = relX / rect.width;
  const targetFrame = Math.round(
    boundaryDragState.rangeStart + boundaryDragState.totalFrames * pct,
  );

  const leftU = state.units[boundaryDragState.leftIdx];
  const rightU = state.units[boundaryDragState.rightIdx];
  const minBound = leftU.frame_start + 1;
  const maxBound = rightU.frame_end;
  const clamped = Math.max(minBound, Math.min(maxBound, targetFrame));

  leftU.frame_end = clamped - 1;
  rightU.frame_start = clamped;

  renderTimeline();
  renderUnitInfo();
  renderUnitList();
  e.preventDefault();
});

document.addEventListener("mouseup", async () => {
  if (!boundaryDragState) return;
  const { leftIdx, rightIdx } = boundaryDragState;
  boundaryDragState = null;
  document.body.style.cursor = "";
  const leftU = state.units[leftIdx];
  const rightU = state.units[rightIdx];
  // saved side は auto-upsert (frame 変更を即 parquet に反映)
  if (leftU.saved && leftU.verdict) await autoUpsertUnit(leftU);
  if (rightU.saved && rightU.verdict) await autoUpsertUnit(rightU);
  // current unit の frame が変わっていれば video 再 load
  if (leftIdx === state.currentUnitIdx || rightIdx === state.currentUnitIdx) {
    loadVideo();
  }
  setStatus(`boundary updated → left end=${leftU.frame_end}, right start=${rightU.frame_start}`);
});

async function autoUpsertUnit(u) {
  // Post first, then delete old (Fix #1 for PR #83 review):
  // 前実装は delete → post の順で、post 失敗時に data loss (row 消失) の risk あり。
  // post を先に済ませてから old row を delete することで、worst case でも
  // duplicate row (1→2 row) までで data loss は発生しない。
  const oldFrameStart = u.original_frame_start;
  const needsDeleteOld =
    oldFrameStart !== null && oldFrameStart !== u.frame_start;
  try {
    await postLabel({
      episode_id: state.currentEp,
      frame_start: u.frame_start,
      frame_end: u.frame_end,
      task_index: u.task_index,
      verdict: u.verdict,
      failure_category: u.failure_category,
    });
    // Post 成功後にのみ old row 削除
    if (needsDeleteOld) {
      try {
        await deleteLabel(state.currentEp, oldFrameStart);
      } catch (e) {
        console.warn("auto-upsert: old row delete failed", e);
      }
    }
    u.original_frame_start = u.frame_start;
  } catch (e) {
    console.warn("auto-upsert failed", e);
    setStatus("boundary save failed, 再度 verdict save で復旧", "error");
  }
}

function renderVerdictSelection() {
  document.querySelectorAll(".verdict-btn").forEach((btn) => {
    btn.classList.toggle("selected", btn.dataset.verdict === state.pendingVerdict);
  });
}

function renderCategoryButtons() {
  const container = el("failure-category-buttons");
  container.innerHTML = "";
  // 候補 category 群
  state.categories.forEach((c) => {
    const btn = document.createElement("button");
    btn.className = "category-btn";
    btn.dataset.categoryKey = c.key;
    btn.title = `key: ${c.key}`;
    btn.innerHTML = `${c.label}<span class="cat-key">${c.key}</span>`;
    btn.addEventListener("click", () => selectCategoryAndSave(c));
    container.appendChild(btn);
  });
  // + 新規追加 button (末尾に、inline スタイル)
  const newBtn = document.createElement("button");
  newBtn.id = "cat-new-btn";
  newBtn.className = "cat-new-btn-inline";
  newBtn.textContent = "+ 新規追加";
  newBtn.title = "[+] new category";
  newBtn.addEventListener("click", openCatDialog);
  container.appendChild(newBtn);

  // 空 hint (candidates は無いが + 新規追加 は残す)
  if (state.categories.length === 0) {
    const hint = document.createElement("div");
    hint.className = "empty-hint";
    hint.textContent = "(まだ category 無し → 「+ 新規追加」で作成)";
    container.insertBefore(hint, newBtn);
  }
}

function selectCategoryAndSave(c) {
  // Chip の選択状態を視覚更新
  const chip = el("selected-category-chip");
  chip.classList.add("selected");
  el("selected-category-value").textContent = `${c.label} (${c.key})`;
  // 少し表示してから save+next (200ms、ユーザが選択を目視確認できる時間)
  if (state.pendingVerdict === "failure") {
    setTimeout(() => saveCurrentUnit("failure", c.key), 200);
  }
}

function resetSelectedChip() {
  const chip = el("selected-category-chip");
  chip.classList.remove("selected");
  el("selected-category-value").textContent = "—";
}

function renderTaskIndexOptions() {
  // info panel の task select を populate
  const select = el("info-task-select");
  select.innerHTML = "";
  Object.entries(state.taskNames)
    .map(([k, v]) => [parseInt(k, 10), v])
    .sort((a, b) => a[0] - b[0])
    .forEach(([idx, name]) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${idx}: ${name}`;
      select.appendChild(opt);
    });
}

function loadVideo() {
  const u = state.units[state.currentUnitIdx];
  if (!u) return;
  // frame_start/end が LeRobot の元 segment と一致すれば cached tile 再利用、
  // split/merge 後の任意範囲は tile_range endpoint で on-the-fly 生成
  const matchSeg = state.segments.find(
    (s) => s.frame_start === u.frame_start && s.frame_end === u.frame_end,
  );
  const url = matchSeg
    ? `/api/episodes/${state.currentEp}/segments/${matchSeg.segment_id}/tile.mp4`
    : `/api/episodes/${state.currentEp}/tile_range.mp4?frame_start=${u.frame_start}&frame_end=${u.frame_end}`;
  const video = el("tile-video");
  showVideoLoading();
  video.src = url;
  video.loop = true;
  video.load();
  video.play().catch(() => {});
}

function showVideoLoading() {
  el("video-loading-overlay").classList.remove("hidden");
}
function hideVideoLoading() {
  el("video-loading-overlay").classList.add("hidden");
}

function updateCurrentFrame() {
  const u = state.units[state.currentUnitIdx];
  if (!u) return;
  const segFrame = currentFrameOffset();
  const epFrame = u.frame_start + segFrame;
  el("info-current-frame").textContent = `${epFrame} (offset ${segFrame})`;
  updateTimelinePlayhead();
}

function updateForCurrentUnit() {
  state.pendingVerdict = null;
  hideFailureCategoryPanel();
  renderVerdictSelection();
  renderTopBar();
  renderUnitList();
  renderUnitInfo();
  renderTimeline();
  loadVideo();
}

// --- Split / Merge ---

function splitAtCurrentFrame() {
  const idx = state.currentUnitIdx;
  const unit = state.units[idx];
  if (!unit) return;
  const at = currentFrameAbsolute();
  if (at <= unit.frame_start || at > unit.frame_end) {
    setStatus(`split 位置 (${at}) が unit 範囲外`, "error");
    return;
  }
  const firstHalf = {
    ...unit,
    frame_end: at - 1,
    saved: false,
    verdict: null,
  };
  const secondHalf = {
    ...unit,
    frame_start: at,
    saved: false,
    verdict: null,
    original_frame_start: null,
  };
  // 元 unit が saved なら、既存 row (original_frame_start) の frame_end 変更 = upsert で OK
  // secondHalf は新 row、pending は無し
  state.units.splice(idx, 1, firstHalf, secondHalf);
  updateForCurrentUnit();
  setStatus(`split at frame ${at}: 2 units created`);
}

function mergeWithPrev() {
  const idx = state.currentUnitIdx;
  if (idx === 0) {
    setStatus("前 unit が無い、merge 不可", "error");
    return;
  }
  const prev = state.units[idx - 1];
  const curr = state.units[idx];
  const merged = {
    frame_start: prev.frame_start,
    frame_end: curr.frame_end,
    task_index: prev.task_index,  // 前 unit の task を継承 (user が変えたければ dropdown で変更)
    saved: false,
    original_frame_start: prev.original_frame_start,  // prev の row を継続 upsert
  };
  // curr が saved なら、その row は不要になるので pendingDeletes に
  if (curr.saved && curr.original_frame_start !== null) {
    state.pendingDeletes.push(curr.original_frame_start);
  }
  state.units.splice(idx - 1, 2, merged);
  state.currentUnitIdx = idx - 1;
  updateForCurrentUnit();
  setStatus(`merged with prev unit`);
}

// --- Save flow ---

async function saveCurrentUnit(verdict, failure_category = null) {
  const idx = state.currentUnitIdx;
  const u = state.units[idx];
  if (!u) return;

  // task_index は info-task-select の change handler で u.task_index に反映済のため直接使用
  // (前 refactor で corrections panel 削除、旧 corr-task-input への参照バグを fix)
  const finalTaskIndex = u.task_index;

  const payload = {
    episode_id: state.currentEp,
    frame_start: u.frame_start,
    frame_end: u.frame_end,
    task_index: finalTaskIndex,
    verdict: verdict,
    failure_category: failure_category,
  };

  // Pending delete (merge で consumed unit の row) を先に処理
  for (const fs of state.pendingDeletes) {
    try {
      await deleteLabel(state.currentEp, fs);
    } catch (e) {
      console.warn("delete failed", e);
    }
  }
  state.pendingDeletes = [];

  // primary key 変更 (merge で frame_start が変わったケース) の場合、古い row を delete
  if (u.original_frame_start !== null && u.original_frame_start !== u.frame_start) {
    try {
      await deleteLabel(state.currentEp, u.original_frame_start);
    } catch (e) {
      console.warn("delete old row failed", e);
    }
  }

  try {
    await postLabel(payload);
    if (!u.saved) state.totalLabeled++;
    u.saved = true;
    u.task_index = finalTaskIndex;
    u.verdict = verdict;
    u.failure_category = failure_category;
    u.original_frame_start = u.frame_start;
    setStatus(`✓ saved: unit ${idx} = ${verdict}${failure_category ? " (" + failure_category + ")" : ""}`);
    // 画面右上に大 toast (advance と関係なく確実に見える)
    showSaveToast(idx, verdict, failure_category);
  } catch (e) {
    setStatus(`save failed: ${e.message}`, "error");
    return;
  }

  saveMetadata({
    last_ep: state.currentEp,
    last_frame_start: u.frame_start,
    reviewer: "hara",
    schema_version: 4,
  }).catch((e) => console.warn("metadata save failed", e));

  advanceToNextUnit();
}

function advanceToNextUnit() {
  if (state.currentUnitIdx < state.units.length - 1) {
    state.currentUnitIdx++;
    updateForCurrentUnit();
  } else {
    setStatus(`ep ${state.currentEp} 完走! (${state.units.length} units)`);
    renderUnitList();
  }
}

function prevUnit() {
  if (state.currentUnitIdx > 0) {
    state.currentUnitIdx--;
    updateForCurrentUnit();
  }
}

function jumpToUnit(i) {
  state.currentUnitIdx = i;
  updateForCurrentUnit();
}

// --- Failure category panel ---

function showFailureCategoryPanel() {
  resetSelectedChip();
  el("failure-category-panel").classList.remove("hidden");
  setTimeout(() => {
    const first = el("failure-category-buttons").querySelector(".category-btn");
    if (first) first.focus();
    else el("cat-new-btn").focus();
  }, 50);
}
function hideFailureCategoryPanel() {
  el("failure-category-panel").classList.add("hidden");
  resetSelectedChip();
}

// --- HF modal ---

async function openHfSyncModal() {
  el("hf-modal").classList.remove("hidden");
  el("hf-preview").innerHTML = "loading…";
  el("hf-push").disabled = true;
  el("hf-push").textContent = "Push to HF";
  el("hf-commit-msg").value = `curation: sync at ${new Date().toISOString()}`;
  try {
    const preview = await fetchHfPreview();
    const cats = (preview.categories_added || []).map((c) => `${c.key}: ${c.label}`).join(", ") || "(none)";
    const files = (preview.files_to_upload || []).join(", ") || "(none)";
    el("hf-preview").innerHTML = `
      <div>remote_exists: <strong>${preview.remote_exists}</strong></div>
      <div>labels new rows: <strong>${preview.labels_new_rows}</strong> (local total: ${preview.labels_total_rows_local})</div>
      <div>jsonl lines (local): <strong>${preview.jsonl_new_lines}</strong></div>
      <div>categories added: <strong>${cats}</strong></div>
      <div>files: ${files}</div>
    `;
    el("hf-push").disabled = false;
  } catch (e) {
    el("hf-preview").innerHTML = `<div style="color:red">preview failed: ${e.message}</div>`;
  }
}
function closeHfSyncModal() { el("hf-modal").classList.add("hidden"); }

async function doHfPush() {
  const msg = el("hf-commit-msg").value.trim() || null;
  el("hf-push").disabled = true;
  el("hf-push").textContent = "pushing…";
  try {
    const r = await postHfSync(msg);
    setStatus(`HF push OK: ${r.commit_url}`);
    closeHfSyncModal();
  } catch (e) {
    setStatus(`HF push failed: ${e.message}`, "error");
    el("hf-push").disabled = false;
    el("hf-push").textContent = "Push to HF";
  }
}

// --- New category dialog ---

function openCatDialog() {
  el("cat-dialog").classList.remove("hidden");
  el("cat-key").value = "";
  el("cat-label-input").value = "";
  setTimeout(() => el("cat-key").focus(), 50);
}
function closeCatDialog() { el("cat-dialog").classList.add("hidden"); }

async function addNewCategory() {
  const key = el("cat-key").value.trim();
  const label = el("cat-label-input").value.trim();
  if (!key || !label) {
    setStatus("key と label 両方必須", "error");
    return;
  }
  try {
    await postCategory(key, label, state.currentEp);
    state.categories = await fetchCategories();
    renderCategoryButtons();
    closeCatDialog();
    setStatus(`category added: ${key}`);
    if (state.pendingVerdict === "failure") {
      await saveCurrentUnit("failure", key);
    }
  } catch (e) {
    setStatus(`category add failed: ${e.message}`, "error");
  }
}

// --- Corrections manual actions ---

function grabStartAsFrameStart() {
  const u = state.units[state.currentUnitIdx];
  if (!u) return;
  const at = currentFrameAbsolute();
  if (at > u.frame_end) {
    setStatus(`start (${at}) > end (${u.frame_end}) 不可`, "error");
    return;
  }
  u.frame_start = at;
  renderUnitInfo();
  renderUnitList();
  renderTimeline();
  setStatus(`[ frame_start = ${at}`);
}

function grabEndAsFrameEnd() {
  const u = state.units[state.currentUnitIdx];
  if (!u) return;
  const at = currentFrameAbsolute();
  if (at < u.frame_start) {
    setStatus(`end (${at}) < start (${u.frame_start}) 不可`, "error");
    return;
  }
  u.frame_end = at;
  renderUnitInfo();
  renderUnitList();
  renderTimeline();
  setStatus(`] frame_end = ${at}`);
}

// --- Keyboard ---

const VERDICT_KEYS = { "1": "optimal", "2": "success", "3": "ambiguous", "4": "failure" };

function handleKey(e) {
  const target = e.target;
  const isTyping = target.tagName === "INPUT" || target.tagName === "TEXTAREA";
  if (isTyping) return;

  const video = el("tile-video");

  if (e.key === " ") {
    e.preventDefault();
    if (video.paused) video.play(); else video.pause();
    return;
  }
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    const step = e.shiftKey ? 1.0 : (1.0 / state.fps);
    video.currentTime = Math.max(0, video.currentTime - step);
    return;
  }
  if (e.key === "ArrowRight") {
    e.preventDefault();
    const step = e.shiftKey ? 1.0 : (1.0 / state.fps);
    video.currentTime = Math.min(video.duration || 1e9, video.currentTime + step);
    return;
  }
  if (e.key === "[") { e.preventDefault(); grabStartAsFrameStart(); return; }
  if (e.key === "]") { e.preventDefault(); grabEndAsFrameEnd(); return; }
  if (e.key === "s") { e.preventDefault(); splitAtCurrentFrame(); return; }
  if (e.key === "m") { e.preventDefault(); mergeWithPrev(); return; }
  // "c" (corrections toggle) は corrections panel 廃止に伴い削除
  if (e.key === "f") {
    e.preventDefault();
    if (state.pendingVerdict === "failure") {
      const first = el("failure-category-buttons").querySelector(".category-btn");
      if (first) first.focus();
      else document.querySelector(".cat-new-btn-inline")?.focus();
    }
    return;
  }
  if (e.key === "+") {
    e.preventDefault();
    if (state.pendingVerdict === "failure") openCatDialog();
    return;
  }
  if (e.key === "Backspace") { e.preventDefault(); prevUnit(); return; }
  // Next ep は button のみで発動 (誤爆防止のため n shortcut は廃止)
  if (e.key === "g" || e.key === "G") {
    e.preventDefault();
    el("ep-jump-input").focus();
    el("ep-jump-input").select();
    return;
  }

  const verdict = VERDICT_KEYS[e.key];
  if (verdict) {
    e.preventDefault();
    if (verdict === "failure") {
      state.pendingVerdict = "failure";
      renderVerdictSelection();
      showFailureCategoryPanel();
      setStatus("failure: category 選択 [f] focus, [+] new");
    } else {
      state.pendingVerdict = verdict;
      renderVerdictSelection();
      saveCurrentUnit(verdict);
    }
  }
}

// --- Next ep + Prefetch ---

const PREFETCH_WINDOW = 5;
let _prefetchPollTimer = null;

async function jumpToEpisode(target = null) {
  const input = el("ep-jump-input");
  const parsed = target !== null ? target : parseInt(input.value, 10);
  if (isNaN(parsed) || parsed < 0 || parsed >= state.episodes.length) {
    setStatus(
      `ep ${input.value} は範囲外 (0-${state.episodes.length - 1})`,
      "error",
    );
    return;
  }
  input.value = "";
  input.blur();
  await loadEpisode(parsed);
  // Sliding window cleanup: 新 window (target..target+5) 外の prefetched cache を削除
  cleanupPrefetchedOutsideWindow(parsed);
  triggerPrefetchWindow(parsed);
  // URL 更新 (bookmark 可能に)
  const url = new URL(window.location);
  url.searchParams.set("ep", String(parsed));
  window.history.replaceState({}, "", url.toString());
  setStatus(`jumped to ep ${parsed}`);
}

async function cleanupPrefetchedOutsideWindow(currentEp) {
  try {
    const res = await fetch(`/api/prefetch_cleanup/${currentEp}`, {
      method: "POST",
    });
    if (!res.ok) return;
    const body = await res.json();
    const nEps = body.cleaned_eps.length;
    if (nEps > 0) {
      const tiles = body.total_local_tiles;
      const videos = body.total_hf_hub_videos;
      setStatus(
        `cleanup: ${nEps} eps (${tiles} tiles + ${videos} videos) 削除`,
      );
    }
  } catch (e) {
    console.warn("prefetch cleanup failed", e);
  }
}

async function nextEpisode() {
  const target = state.currentEp + 1;
  if (target >= state.episodes.length) {
    setStatus("最後の ep です", "error");
    return;
  }
  const current = state.currentEp;
  const toggle = el("auto-push-delete-toggle");
  const autoPushDelete = toggle && toggle.checked;
  // 現 ep に saved label があるかチェック
  const savedCount = state.units.filter((u) => u.saved).length;

  if (autoPushDelete && savedCount > 0) {
    setStatus(`ep ${current}: HF push + cache 削除...`);
    try {
      const res = await fetch(`/api/episodes/${current}/push_and_delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          commit_message: `curation: ep${current} (${savedCount} units)`,
        }),
      });
      if (!res.ok) {
        throw new Error(`${res.status}: ${await res.text()}`);
      }
      const body = await res.json();
      const d = body.deleted;
      const merged = body.categories_merged || 0;
      const mergeSuffix = merged > 0 ? ` / ${merged} category 追加` : "";
      setStatus(
        `ep ${current} pushed → ${d.local_tiles} tile + ${d.hf_hub_videos} video 削除${mergeSuffix}`,
      );
      // Categories が merge されていたら UI reload (他 reviewer 追加分を反映)
      if (merged > 0) {
        state.categories = await fetchCategories();
        renderCategoryButtons();
      }
    } catch (e) {
      setStatus(`push+delete failed: ${e.message}`, "error");
      return;  // 次 ep には進まない
    }
  } else if (autoPushDelete && savedCount === 0) {
    setStatus(`ep ${current}: label 無し、push+delete skip`);
  }

  await loadEpisode(target);
  // Sliding cleanup を追加 (Fix #90): Next 連打で prefetch した +1..+5 の
  // window shift 分が積み上がる bug を防ぐ。jumpToEpisode と同じ pattern。
  cleanupPrefetchedOutsideWindow(target);
  triggerPrefetchWindow(target);
}

function triggerPrefetchWindow(currentEp) {
  // 現 ep は full_tiles=true で高優先度 pool へ (全 seg の tile 事前 build)
  // +1..+5 ep は partial pool へ (先頭 PREFETCH_TILE_COUNT 分の tile)
  postPrefetch(currentEp, true).catch(() => {});
  for (let i = 1; i <= PREFETCH_WINDOW; i++) {
    const t = currentEp + i;
    if (t < state.episodes.length) {
      postPrefetch(t, false).catch(() => {});
    }
  }
  startPrefetchPolling();
}

async function startPrefetchPolling() {
  if (_prefetchPollTimer) return;
  const poll = async () => {
    try {
      const st = await fetchPrefetchStatus();
      // 現 ep の +1..+N の状態を badge に表示
      const window = [];
      for (let i = 1; i <= PREFETCH_WINDOW; i++) {
        const t = state.currentEp + i;
        if (t < state.episodes.length) window.push(t);
      }
      const doneSet = new Set(st.done);
      const pendingSet = new Set(st.pending);
      const doneCount = window.filter((t) => doneSet.has(t)).length;
      const pendingCount = window.filter((t) => pendingSet.has(t)).length;
      const badge = el("prefetch-badge");
      if (pendingCount === 0 && doneCount === window.length) {
        badge.textContent = `⟳ buffer: ${doneCount}/${window.length} ready`;
        badge.classList.remove("hidden");
        // 完了なので停止
        clearInterval(_prefetchPollTimer);
        _prefetchPollTimer = null;
        setTimeout(() => badge.classList.add("hidden"), 2000);
      } else if (window.length > 0) {
        badge.textContent = `⟳ buffer: ${doneCount} done / ${pendingCount} pending`;
        badge.classList.remove("hidden");
      } else {
        badge.classList.add("hidden");
      }
    } catch (e) {
      console.warn("prefetch poll failed", e);
    }
  };
  await poll();  // 即時 1 回
  _prefetchPollTimer = setInterval(poll, 3000);  // 以降 3s ごと
}

// --- Reset ep ---

async function resetEp() {
  const ep = state.currentEp;
  const ok = window.confirm(
    `ep ${ep} の全 label を削除して LeRobot 元状態に戻します。よろしいですか?`,
  );
  if (!ok) return;
  try {
    const res = await fetch(`/api/episodes/${ep}/reset`, { method: "POST" });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    const body = await res.json();
    setStatus(`ep ${ep} reset OK (${body.deleted} labels 削除)`);
    // 現在の ep を reload
    await loadEpisode(ep);
    // totalLabeled 更新
    const allLabels = await fetchLabels();
    state.totalLabeled = allLabels.length;
    renderTopBar();
  } catch (e) {
    setStatus(`reset failed: ${e.message}`, "error");
  }
}

// --- Resize handles (JS drag) ---

function setupResize() {
  function makeHandler(handleId, cssVar, minW, maxW) {
    const handle = el(handleId);
    if (!handle) return;
    let dragging = false, startX = 0, startW = 0;
    handle.addEventListener("mousedown", (e) => {
      dragging = true;
      startX = e.clientX;
      const cur = getComputedStyle(document.documentElement)
        .getPropertyValue(cssVar).trim();
      startW = parseInt(cur, 10) || minW;
      document.body.style.cursor = "col-resize";
      e.preventDefault();
    });
    document.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      // handleId によって方向が異なる: 左 handle は右にドラッグで拡大、
      // 右 handle は左にドラッグで拡大
      const delta = handleId === "resize-left"
        ? (e.clientX - startX)
        : -(e.clientX - startX);
      const newW = Math.max(minW, Math.min(maxW, startW + delta));
      document.documentElement.style.setProperty(cssVar, newW + "px");
    });
    document.addEventListener("mouseup", () => {
      if (dragging) {
        dragging = false;
        document.body.style.cursor = "";
      }
    });
  }
  makeHandler("resize-left", "--left-w", 120, 500);
  makeHandler("resize-right", "--right-w", 180, 500);
}

// --- Bootstrap ---

async function loadEpisode(ep) {
  state.currentEp = ep;
  state.segments = await fetchSegments(ep);
  const labels = await fetchLabels(ep);
  state.units = buildUnits(state.segments, labels);
  state.currentUnitIdx = 0;
  updateForCurrentUnit();
  // Resume 用に metadata.last_ep を即更新 (browsing だけで save 無しでも次回起動で
  // この ep から再開される)
  saveMetadata({
    last_ep: ep,
    last_frame_start: state.units.length ? state.units[0].frame_start : null,
    reviewer: "hara",
    schema_version: 4,
  }).catch((e) => console.warn("metadata save failed", e));
}

async function init() {
  try {
    setStatus("loading…");
    state.episodes = await fetchEpisodes();

    const [meta, allLabels, categories, taskNames] = await Promise.all([
      fetchMetadata(),
      fetchLabels(),
      fetchCategories(),
      fetchTaskNames(),
    ]);
    state.totalLabeled = allLabels.length;
    state.categories = categories;
    state.taskNames = taskNames;
    renderCategoryButtons();
    renderTaskIndexOptions();

    // 起動 ep 決定: URL param ?ep=N > metadata.last_ep > 0
    const urlEp = new URLSearchParams(window.location.search).get("ep");
    const urlEpParsed = urlEp !== null ? parseInt(urlEp, 10) : NaN;
    let startEp;
    if (!isNaN(urlEpParsed) && urlEpParsed >= 0 && urlEpParsed < state.episodes.length) {
      startEp = urlEpParsed;
    } else if (meta.last_ep !== null && meta.last_ep !== undefined) {
      startEp = meta.last_ep;
    } else {
      startEp = 0;
    }
    await loadEpisode(startEp);

    // 起動時にも prefetch window を kick (+1..+5)
    triggerPrefetchWindow(startEp);

    el("tile-video").addEventListener("timeupdate", updateCurrentFrame);
    // Video loading overlay の event 制御
    // loadstart / waiting → 表示、canplay / canplaythrough / error → 隠す
    const _v = el("tile-video");
    _v.addEventListener("loadstart", showVideoLoading);
    _v.addEventListener("waiting", showVideoLoading);
    _v.addEventListener("canplay", hideVideoLoading);
    _v.addEventListener("canplaythrough", hideVideoLoading);
    _v.addEventListener("error", hideVideoLoading);

    // Resize handles
    setupResize();

    // wire buttons
    el("hf-sync-btn").addEventListener("click", openHfSyncModal);
    el("reset-ep-btn").addEventListener("click", resetEp);
    el("next-ep-btn").addEventListener("click", nextEpisode);
    el("ep-jump-btn").addEventListener("click", () => jumpToEpisode());
    el("ep-jump-input").addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); jumpToEpisode(); }
    });
    el("hf-cancel").addEventListener("click", closeHfSyncModal);
    el("hf-push").addEventListener("click", doHfPush);
    el("cat-cancel").addEventListener("click", closeCatDialog);
    el("cat-add").addEventListener("click", addNewCategory);
    // cat-new-btn は renderCategoryButtons 内で動的生成、listener もそこで bind 済
    // info panel の task select (旧 corrections から移設)
    el("info-task-select").addEventListener("change", (e) => {
      const u = state.units[state.currentUnitIdx];
      if (!u) return;
      const v = e.target.value;
      if (v !== "") {
        u.task_index = parseInt(v, 10);
        renderUnitInfo();
        renderUnitList();
        renderTimeline();
      }
    });

    document.querySelectorAll(".verdict-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const v = btn.dataset.verdict;
        state.pendingVerdict = v;
        renderVerdictSelection();
        if (v === "failure") {
          showFailureCategoryPanel();
          setStatus("failure: category 選択");
        } else {
          saveCurrentUnit(v);
        }
      });
    });

    setStatus(`ready: ${state.episodes.length} episodes`);
  } catch (e) {
    setStatus(`init failed: ${e.message}`, "error");
    console.error(e);
  }
}

document.addEventListener("keydown", handleKey);
init();
