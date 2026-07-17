"use strict";

// =========================================================
// Backend API contract (Flask record_server.py)
//
// SSE  GET /state/stream
//      data: { timestamp, arms:[{id,active,q_actual[7],q_cmd[7]|null,connected},...],
//              recording, episode_frame_count, current_task, total_episodes, total_frames }
//
// GET  /stream/0|1|2|3          MJPEG camera streams
// GET  /status                  {arms:[...], cameras:[...], dataset_path, total_episodes, total_frames}
//
// POST /arm/<id>/engage         → {success, error}
// POST /arm/<id>/disengage      → {success, error}
//
// POST /recording/start         body:{task} → {success, error}
// POST /recording/stop          → {success, episode_index, frames}
// POST /recording/discard       → {success}
//
// GET  /episodes                → [{episode_index, task, length, timestamp, duration_s,
//                                    has_video:{wrist_0,wrist_1,wrist_2,scene}, notes_count}]
// GET  /episodes/<idx>          → {episode_index, task, length, fps, timestamp, duration_s,
//                                    notes:[{id,text,timestamp_s,created_at}]}
// GET  /episodes/<idx>/video/<cam_key>   MP4 stream
//
// POST   /episodes/<idx>/notes             body:{text,timestamp_s} → Note
// PUT    /episodes/<idx>/notes/<note_id>   body:{text}             → Note
// DELETE /episodes/<idx>/notes/<note_id>   → {success}
// DELETE /episodes/<idx>                   → {success}
// =========================================================

// ── Global state ────────────────────────────────────────
const state = {
  // SSE
  sseConnected: false,

  // Arms (up to 3): [{id, active, q_actual[7], q_cmd[7]|null, connected}]
  arms: [],

  // Recording
  recording: false,
  episodeFrameCount: 0,
  currentTask: "",
  totalEpisodes: 0,
  totalFrames: 0,

  // Dataset
  datasetPath: "",

  // Save tracking (for SSE dedup)
  lastSavedEpisodeTs: null,

  // Recordings view
  episodes: [],
  selectedEpisodeIdx: null,
  selectedEpisode: null,
};

// ── Helpers ──────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

function fmt(v, d = 3) {
  return v == null || isNaN(v) ? "—" : Number(v).toFixed(d);
}

function formatDuration(secs) {
  if (secs == null || isNaN(secs)) return "—";
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(1);
  return m > 0 ? `${m}m ${Math.floor(secs % 60)}s` : `${s}s`;
}

function formatVideoTime(secs) {
  if (secs == null || secs < 0) return null;
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function formatDate(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleString();
}

function showToast(message, type = "success") {
  const container = $("toast-container");
  if (!container) return;
  const el = document.createElement("div");
  el.className = `toast toast-${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => el.classList.add("toast-show"), 10);
  setTimeout(() => {
    el.classList.remove("toast-show");
    setTimeout(() => el.remove(), 400);
  }, 4000);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function apiFetch(url, opts = {}) {
  const defaultHeaders = { "Content-Type": "application/json" };
  const resp = await fetch(url, {
    headers: defaultHeaders,
    ...opts,
    headers: { ...defaultHeaders, ...(opts.headers || {}) },
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  const ct = resp.headers.get("Content-Type") || "";
  return ct.includes("application/json") ? resp.json() : resp.text();
}

// ── SSE ──────────────────────────────────────────────────
let sseSource = null;
let sseReconnectTimer = null;

function setupSSE() {
  if (sseSource) { sseSource.close(); sseSource = null; }

  sseSource = new EventSource("/state/stream");

  sseSource.onopen = () => {
    state.sseConnected = true;
    updateDot($("sse-dot"), "ok");
    $("sse-label").textContent = "Live";
    clearTimeout(sseReconnectTimer);
  };

  sseSource.onmessage = e => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    mergeServerState(data);
    renderControlView();
  };

  sseSource.onerror = () => {
    state.sseConnected = false;
    updateDot($("sse-dot"), "error");
    $("sse-label").textContent = "Reconnecting…";
    sseSource.close();
    sseSource = null;
    sseReconnectTimer = setTimeout(setupSSE, 3000);
  };
}

function mergeServerState(data) {
  // Arms array from backend: [{id, active, q_actual, q_cmd, connected}]
  if (Array.isArray(data.arms)) {
    state.arms = data.arms;
  }

  if (data.recording != null)           state.recording          = data.recording;
  if (data.episode_frame_count != null) state.episodeFrameCount  = data.episode_frame_count;
  if (data.current_task != null)        state.currentTask        = data.current_task;
  if (data.total_episodes != null)      state.totalEpisodes      = data.total_episodes;
  if (data.total_frames != null)        state.totalFrames        = data.total_frames;
  if (data.dataset_path != null)        state.datasetPath        = data.dataset_path;

  // Detect newly saved episode via SSE and show toast
  if (data.last_saved_episode && data.last_saved_episode.ts) {
    const prev = state.lastSavedEpisodeTs;
    if (!prev || data.last_saved_episode.ts > prev) {
      state.lastSavedEpisodeTs = data.last_saved_episode.ts;
      const ep = data.last_saved_episode;
      showToast(`Episode #${ep.episode_index} saved — ${ep.frames.toLocaleString()} frames`, "success");
    }
  }
}

// ── Status poll (fallback / initial load) ───────────────
async function loadStatus() {
  try {
    const data = await apiFetch("/status");
    // /status returns {arms:[{id,so101_port,yam_channel,connected}], cameras:[...], ...}
    if (data.total_episodes != null) state.totalEpisodes = data.total_episodes;
    if (data.total_frames   != null) state.totalFrames   = data.total_frames;
    if (data.dataset_path   != null) state.datasetPath   = data.dataset_path;

    // Update camera status dots from /status
    if (Array.isArray(data.cameras)) {
      const connected = data.cameras.filter(c => c.connected).length;
      const total     = data.cameras.length;
      updateDot($("cam-dot"), connected > 0 ? "ok" : "warn");
      $("cam-label").textContent = `${connected}/${total} cams`;
    }

    renderControlView();
  } catch {
    // Server not ready — ignore
  }
}

// ── Render helpers ───────────────────────────────────────
function updateDot(el, cls) {
  if (!el) return;
  el.className = "dot" + (cls ? " " + cls : "");
}

const JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"];

function renderArmPanel(armIdx) {
  const arm = state.arms.find(a => a.id === armIdx);
  const panelEl  = $(`arm${armIdx}-panel`);
  const badgeEl  = $(`arm${armIdx}-badge`);
  const dotEl    = $(`arm${armIdx}-dot`);
  const labelEl  = $(`arm${armIdx}-label`);
  const tbodyEl  = $(`arm${armIdx}-joints`);
  const engageBtn    = $(`arm${armIdx}-engage-btn`);
  const disengageBtn = $(`arm${armIdx}-disengage-btn`);
  const gripBar  = $(`arm${armIdx}-grip-bar`);
  const gripVal  = $(`arm${armIdx}-grip-val`);

  if (!arm || !arm.connected) {
    updateDot(dotEl, "error");
    if (labelEl) labelEl.textContent = `Leader ${armIdx} (offline)`;
    if (badgeEl) { badgeEl.textContent = "OFFLINE"; badgeEl.className = "arm-status-badge offline"; }
    if (tbodyEl) tbodyEl.innerHTML = `<tr><td colspan="3" class="dim">Not connected</td></tr>`;
    if (engageBtn)    engageBtn.disabled    = true;
    if (disengageBtn) disengageBtn.disabled = true;
    if (gripBar) gripBar.style.width = "0%";
    if (gripVal) gripVal.textContent = "—";
    if (panelEl) { panelEl.classList.remove("arm-active"); panelEl.classList.remove("arm-idle"); }
    return;
  }

  updateDot(dotEl, "ok");
  if (labelEl) labelEl.textContent = `Leader ${armIdx}`;

  const active = arm.active;
  if (panelEl) {
    panelEl.classList.toggle("arm-active", active);
    panelEl.classList.toggle("arm-idle", !active);
  }
  if (badgeEl) {
    badgeEl.textContent  = active ? "ACTIVE" : "IDLE";
    badgeEl.className    = "arm-status-badge " + (active ? "active" : "idle");
  }
  if (engageBtn)    engageBtn.disabled    = active;
  if (disengageBtn) disengageBtn.disabled = !active;

  // Joint table
  if (tbodyEl) {
    const actual = arm.q_actual || [];
    const cmd    = arm.q_cmd   || [];
    tbodyEl.innerHTML = JOINT_NAMES.map((name, i) =>
      `<tr><td>${name}</td><td class="mono">${fmt(actual[i])}</td><td class="mono">${cmd[i] != null ? fmt(cmd[i]) : "—"}</td></tr>`
    ).join("");
  }

  // Gripper bar (last element, index 6)
  const gripNorm = arm.q_actual && arm.q_actual[6] != null ? arm.q_actual[6] : 0;
  if (gripBar) gripBar.style.width = Math.max(0, Math.min(100, gripNorm * 100)).toFixed(0) + "%";
  if (gripVal) gripVal.textContent = fmt(gripNorm, 2);
}

function renderControlView() {
  // Render all arm panels
  renderArmPanel(0);
  renderArmPanel(1);
  renderArmPanel(2);

  // Session stats
  const secsTotal = state.totalFrames ? (state.totalFrames / 30) : 0;
  $("stat-dataset").textContent  = state.datasetPath ? state.datasetPath.split("/").pop() : "—";
  $("stat-episodes").textContent = state.totalEpisodes ?? "—";
  $("stat-frames").textContent   = state.totalFrames   ?? "—";
  $("stat-duration").textContent = secsTotal ? formatDuration(secsTotal) : "—";

  // Recording controls
  const recBtn     = $("record-btn");
  const discardBtn = $("discard-btn");
  const recDot     = $("rec-indicator");
  const recText    = $("rec-status-text");
  const recFrames  = $("rec-frame-count");

  if (state.recording) {
    recBtn.textContent = "⏹ Stop Recording";
    recBtn.classList.add("recording");
    discardBtn.disabled = false;
    recDot.classList.remove("hidden");
    recText.textContent = "Recording";
    recFrames.textContent = `${state.episodeFrameCount.toLocaleString()} frames`;
  } else {
    recBtn.textContent = "⏺ Start Recording";
    recBtn.classList.remove("recording");
    discardBtn.disabled = true;
    recDot.classList.add("hidden");
    recText.textContent = "Stopped";
    recFrames.textContent = "";
  }

  // Recording banner
  const banner = $("recording-banner");
  if (banner) {
    banner.classList.toggle("hidden", !state.recording);
    if (state.recording) {
      const taskEl = $("rec-banner-task");
      const framesEl = $("rec-banner-frames");
      if (taskEl) taskEl.textContent = state.currentTask || "";
      if (framesEl) framesEl.textContent = `${state.episodeFrameCount.toLocaleString()} frames`;
    }
  }

  // SSE connection indicator
  if (!state.sseConnected) {
    updateDot($("sse-dot"), "warn");
  }
}

// ── Tab navigation ───────────────────────────────────────
function setupTabs() {
  $$(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      $$(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".view").forEach(v => {
        v.hidden = v.id !== "view-" + tab;
      });
      if (tab === "recordings") loadEpisodes();
    });
  });
}

// ── Camera image error handling ──────────────────────────
function setupCameraFeeds() {
  [0, 1, 2, 3].forEach(i => {
    const img     = $(`cam-feed-${i}`);
    const overlay = $(`cam-overlay-${i}`);
    if (!img) return;
    img.addEventListener("error", ()  => { if (overlay) overlay.classList.add("visible"); });
    img.addEventListener("load",  ()  => { if (overlay) overlay.classList.remove("visible"); });
  });
}

// ── Arm engage / disengage buttons ──────────────────────
function setupArmControls() {
  [0, 1, 2].forEach(armIdx => {
    const engBtn  = $(`arm${armIdx}-engage-btn`);
    const disBtn  = $(`arm${armIdx}-disengage-btn`);

    if (engBtn) {
      engBtn.addEventListener("click", async () => {
        engBtn.disabled = true;
        try {
          const res = await apiFetch(`/arm/${armIdx}/engage`, { method: "POST" });
          if (!res.success) throw new Error(res.error || "engage failed");
        } catch (err) {
          console.error(`arm${armIdx} engage error:`, err);
          alert(`Engage arm ${armIdx} failed: ${err.message}`);
        } finally {
          engBtn.disabled = false;
        }
      });
    }

    if (disBtn) {
      disBtn.addEventListener("click", async () => {
        disBtn.disabled = true;
        try {
          const res = await apiFetch(`/arm/${armIdx}/disengage`, { method: "POST" });
          if (!res.success) throw new Error(res.error || "disengage failed");
        } catch (err) {
          console.error(`arm${armIdx} disengage error:`, err);
          alert(`Disengage arm ${armIdx} failed: ${err.message}`);
        } finally {
          disBtn.disabled = false;
        }
      });
    }
  });
}

// ── Recording controls ───────────────────────────────────
function setupRecordingControls() {
  $("record-btn").addEventListener("click", async () => {
    const btn = $("record-btn");
    btn.disabled = true;
    try {
      if (state.recording) {
        const res = await apiFetch("/recording/stop", { method: "POST" });
        state.recording = false;
        state.episodeFrameCount = 0;
        if (res.episode_index >= 0) {
          showToast(`Episode #${res.episode_index} saved — ${res.frames.toLocaleString()} frames`, "success");
        }
        // Refresh episodes list if on that tab
        if (!$("view-recordings").hidden) await loadEpisodes();
      } else {
        const task = $("task-input").value.trim() || "manipulation task";
        const res  = await apiFetch("/recording/start", {
          method: "POST",
          body: JSON.stringify({ task }),
        });
        if (!res.success) throw new Error(res.error || "start failed");
        state.recording = true;
        state.episodeFrameCount = 0;
        state.currentTask = task;
      }
      renderControlView();
    } catch (err) {
      console.error("Recording toggle failed:", err);
      alert("Recording error: " + err.message);
    } finally {
      setTimeout(() => { btn.disabled = false; }, 400);
    }
  });

  $("discard-btn").addEventListener("click", async () => {
    if (!state.recording) return;
    if (!confirm("Discard the current episode? All recorded frames will be lost.")) return;
    const btn = $("discard-btn");
    btn.disabled = true;
    try {
      await apiFetch("/recording/discard", { method: "POST" });
      state.recording = false;
      state.episodeFrameCount = 0;
      renderControlView();
    } catch (err) {
      console.error("Discard failed:", err);
      alert("Discard failed: " + err.message);
    } finally {
      setTimeout(() => { btn.disabled = false; }, 400);
    }
  });
}

// ── Episode list (Recordings view) ───────────────────────
async function loadEpisodes() {
  const list = $("episode-list");
  list.innerHTML = '<li class="episode-item dim">Loading…</li>';
  try {
    const data = await apiFetch("/episodes");
    state.episodes = Array.isArray(data) ? data : [];
    renderEpisodeList();
  } catch (err) {
    console.error("loadEpisodes failed:", err);
    list.innerHTML = '<li class="episode-item dim">Failed to load.</li>';
  }
}

function renderEpisodeList() {
  const list = $("episode-list");
  if (!state.episodes.length) {
    list.innerHTML = '<li class="episode-item dim">No episodes yet.</li>';
    return;
  }

  list.innerHTML = state.episodes.map(ep => {
    const idx       = ep.episode_index;
    const isSelected = idx === state.selectedEpisodeIdx;
    const task      = escapeHtml(ep.task || `Episode ${idx}`);
    const duration  = formatDuration(ep.duration_s);
    const frames    = ep.length ? `${ep.length.toLocaleString()} frames` : "";
    const notes     = ep.notes_count ? `💬 ${ep.notes_count}` : "";
    const date      = formatDate(ep.timestamp);

    return `<li class="episode-item${isSelected ? " selected" : ""}" data-idx="${idx}">
      <div class="episode-item-title">${task}</div>
      <div class="episode-item-meta">${date} · ${duration} · ${frames} ${notes}</div>
    </li>`;
  }).join("");

  list.querySelectorAll(".episode-item[data-idx]").forEach(el => {
    el.addEventListener("click", () => selectEpisode(parseInt(el.dataset.idx, 10)));
  });
}

async function selectEpisode(idx) {
  state.selectedEpisodeIdx = idx;
  renderEpisodeList();

  $("detail-placeholder").hidden = false;
  $("detail-content").hidden = true;

  try {
    const ep = await apiFetch(`/episodes/${idx}`);
    state.selectedEpisode = ep;
    renderEpisodeDetail(ep);
  } catch (err) {
    console.error("selectEpisode failed:", err);
  }
}

const CAM_KEYS = ['wrist_0', 'wrist_1', 'wrist_2', 'scene'];

function getVideoEl(key) { return document.getElementById('video-' + key); }

function loadEpisodeVideos(epIdx, hasVideo) {
  CAM_KEYS.forEach(key => {
    const v = getVideoEl(key);
    if (!v) return;
    if (hasVideo && hasVideo[key] !== false) {
      v.src = '/episodes/' + epIdx + '/video/' + key;
      v.load();
    } else {
      v.removeAttribute('src');
      v.load();
    }
  });
  resetVideoControls();
}

function getPrimaryVideo() {
  return CAM_KEYS.map(getVideoEl).find(v => v && v.src && v.readyState > 0) || getVideoEl('wrist_0');
}

function seekAll(t) {
  CAM_KEYS.forEach(key => {
    const v = getVideoEl(key);
    if (v && v.src) { try { v.currentTime = t; } catch(e) {} }
  });
}

function resetVideoControls() {
  const slider = document.getElementById('video-slider');
  const timeEl = document.getElementById('video-time-display');
  const playBtn = document.getElementById('video-play-btn');
  if (slider) slider.value = 0;
  if (timeEl) timeEl.textContent = '0:00';
  if (playBtn) playBtn.textContent = '▶';
}

function setupVideoControls() {
  const playBtn = document.getElementById('video-play-btn');
  const slider  = document.getElementById('video-slider');
  const timeEl  = document.getElementById('video-time-display');
  if (!playBtn || !slider || !timeEl) return;

  let isPaused = true;

  playBtn.addEventListener('click', () => {
    const primary = getPrimaryVideo();
    if (!primary || !primary.src) return;
    if (primary.paused) {
      CAM_KEYS.forEach(k => { const v = getVideoEl(k); if (v && v.src) v.play().catch(() => {}); });
      playBtn.textContent = '⏸';
      isPaused = false;
    } else {
      CAM_KEYS.forEach(k => { const v = getVideoEl(k); if (v && v.src) v.pause(); });
      playBtn.textContent = '▶';
      isPaused = true;
    }
  });

  slider.addEventListener('input', () => {
    const primary = getPrimaryVideo();
    if (!primary || !primary.duration) return;
    const t = (parseInt(slider.value, 10) / 1000) * primary.duration;
    seekAll(t);
  });

  CAM_KEYS.forEach(key => {
    const v = getVideoEl(key);
    if (!v) return;
    v.addEventListener('timeupdate', () => {
      const primary = getPrimaryVideo();
      if (!primary || !primary.duration || primary !== v) return;
      slider.value = Math.round((v.currentTime / v.duration) * 1000);
      const m = Math.floor(v.currentTime / 60);
      const s = (v.currentTime % 60).toFixed(1).padStart(4, '0');
      timeEl.textContent = m + ':' + s;
      const noteTimeEl = document.getElementById('add-note-time');
      if (noteTimeEl) noteTimeEl.textContent = m + ':' + s;
    });
    v.addEventListener('ended', () => {
      const playBtn = document.getElementById('video-play-btn');
      if (playBtn) playBtn.textContent = '▶';
    });
  });
}

function renderEpisodeDetail(ep) {
  $("detail-placeholder").hidden = true;
  $("detail-content").hidden = false;

  const idx = ep.episode_index;

  $("detail-title").textContent    = `Episode ${idx}`;
  $("detail-date").textContent     = formatDate(ep.timestamp);
  $("detail-duration").textContent = formatDuration(ep.duration_s);
  $("detail-frames").textContent   = ep.length ? `${ep.length.toLocaleString()} frames` : "";
  $("detail-task").textContent     = ep.task || "—";

  loadEpisodeVideos(idx, ep.has_video || {});

  renderNotes(ep.notes || [], idx);

  $("delete-episode-btn").onclick = async () => {
    if (!confirm(`Delete episode ${idx}? This cannot be undone.`)) return;
    try {
      await apiFetch(`/episodes/${idx}`, { method: "DELETE" });
      state.selectedEpisodeIdx = null;
      state.selectedEpisode    = null;
      $("detail-placeholder").hidden = false;
      $("detail-content").hidden     = true;
      await loadEpisodes();
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  };
}

// ── Notes ────────────────────────────────────────────────
function renderNotes(notes, episodeIdx) {
  const list = $("notes-list");
  if (!notes.length) {
    list.innerHTML = '<li class="dim" style="padding:8px 12px;font-size:13px">No notes yet. Add one below.</li>';
    return;
  }

  // Sort: episode-level notes (timestamp_s < 0) first, then by timestamp
  const sorted = [...notes].sort((a, b) => {
    if (a.timestamp_s < 0 && b.timestamp_s >= 0) return -1;
    if (b.timestamp_s < 0 && a.timestamp_s >= 0) return  1;
    return a.timestamp_s - b.timestamp_s;
  });

  list.innerHTML = sorted.map(note => {
    const timeLabel = note.timestamp_s >= 0
      ? `<span class="note-time">${formatVideoTime(note.timestamp_s)}</span>`
      : `<span class="note-time episode-note">📌</span>`;
    return `<li class="note-item" data-nid="${note.id}" data-ts="${note.timestamp_s}">
      ${timeLabel}
      <span class="note-text">${escapeHtml(note.text)}</span>
      <button class="note-delete" title="Delete">&times;</button>
    </li>`;
  }).join("");

  list.querySelectorAll(".note-item").forEach(el => {
    el.addEventListener("click", e => {
      if (e.target.classList.contains("note-delete")) return;
      const ts = parseFloat(el.dataset.ts);
      if (ts >= 0) seekAll(ts);
    });

    // Delete note
    el.querySelector(".note-delete").addEventListener("click", async e => {
      e.stopPropagation();
      if (!confirm("Delete this note?")) return;
      const noteId = el.dataset.nid;
      try {
        await apiFetch(`/episodes/${episodeIdx}/notes/${noteId}`, { method: "DELETE" });
        const ep = await apiFetch(`/episodes/${episodeIdx}`);
        state.selectedEpisode = ep;
        renderNotes(ep.notes || [], episodeIdx);
      } catch (err) {
        alert("Delete note failed: " + err.message);
      }
    });
  });
}

function setupAddNote() {
  $("add-note-btn").addEventListener("click", async () => {
    const idx = state.selectedEpisodeIdx;
    if (idx == null) return;

    const text = $("add-note-input").value.trim();
    if (!text) { alert("Please enter note text."); return; }

    const primary     = getPrimaryVideo();
    const timestamp_s = primary && primary.readyState ? primary.currentTime : -1.0;

    const btn = $("add-note-btn");
    btn.disabled = true;
    try {
      await apiFetch(`/episodes/${idx}/notes`, {
        method: "POST",
        body: JSON.stringify({ text, timestamp_s }),
      });
      $("add-note-input").value = "";
      const ep = await apiFetch(`/episodes/${idx}`);
      state.selectedEpisode = ep;
      renderNotes(ep.notes || [], idx);
    } catch (err) {
      alert("Add note failed: " + err.message);
    } finally {
      btn.disabled = false;
    }
  });

  // Also allow Enter key in note input
  $("add-note-input").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("add-note-btn").click();
    }
  });
}

// ── Refresh episodes button ───────────────────────────────
function setupRefreshEpisodes() {
  $("refresh-episodes-btn").addEventListener("click", loadEpisodes);
}

// ── Camera setup panel ───────────────────────────────────
let _detectedCameras = [];
let _camSlotAssignment = { wrist_0: null, wrist_1: null, wrist_2: null, scene: null };

const SLOT_KEYS = ['wrist_0', 'wrist_1', 'wrist_2', 'scene'];

function setupCameraPanel() {
  const toggleBtn = $('cam-setup-toggle');
  const body      = $('cam-setup-body');
  const chevron   = $('cam-setup-chevron');
  const detectBtn = $('cam-detect-btn');
  const saveBtn   = $('cam-save-btn');
  const copyBtn   = $('cam-cli-copy-btn');

  if (!toggleBtn) return;

  toggleBtn.addEventListener('click', e => {
    if (e.target === detectBtn || detectBtn.contains(e.target)) return;
    const open = !body.hidden;
    body.hidden = open;
    chevron.innerHTML = open ? '&#9660;' : '&#9650;';
  });

  detectBtn.addEventListener('click', async e => {
    e.stopPropagation();
    detectBtn.disabled = true;
    detectBtn.textContent = 'Detecting…';
    try {
      const data = await apiFetch('/cameras/detect');
      _detectedCameras = data.detected || [];
      _camSlotAssignment = data.current_slots || { wrist_0: null, wrist_1: null, wrist_2: null, scene: null };
      _populateCameraDropdowns();
      if (data.cli_args) _showCliCommand(data.cli_args);
      body.hidden = false;
      chevron.innerHTML = '&#9650;';
      $('cam-setup-status').textContent = `${_detectedCameras.length} camera(s) found`;
      $('cam-setup-status').style.color = 'var(--success)';
    } catch (err) {
      $('cam-setup-status').textContent = 'Detection failed: ' + err.message;
      $('cam-setup-status').style.color = 'var(--danger)';
    } finally {
      detectBtn.disabled = false;
      detectBtn.textContent = '📷 Detect Cameras';
    }
  });

  saveBtn.addEventListener('click', async () => {
    const slots = _readSlotSelections();
    saveBtn.disabled = true;
    try {
      const res = await apiFetch('/cameras/save-config', {
        method: 'POST',
        body: JSON.stringify({ slots }),
      });
      $('cam-setup-status').textContent = 'Config saved — restart server to apply';
      $('cam-setup-status').style.color = 'var(--success)';
      if (res.cli_args) _showCliCommand(res.cli_args);
    } catch (err) {
      $('cam-setup-status').textContent = 'Save failed: ' + err.message;
      $('cam-setup-status').style.color = 'var(--danger)';
    } finally {
      saveBtn.disabled = false;
    }
  });

  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const code = $('cam-cli-code');
      if (!code) return;
      navigator.clipboard.writeText(code.textContent).then(() => {
        copyBtn.textContent = '✓';
        setTimeout(() => { copyBtn.innerHTML = '&#128203;'; }, 1500);
      });
    });
  }

  const reconnectBtn = $('cam-reconnect-scene-btn');
  if (reconnectBtn) {
    reconnectBtn.addEventListener('click', async e => {
      e.stopPropagation();
      reconnectBtn.disabled = true;
      reconnectBtn.textContent = 'Searching…';
      const statusEl = $('cam-setup-status');
      try {
        const res = await apiFetch('/cameras/reconnect-scene', { method: 'POST' });
        if (res.success) {
          if (statusEl) { statusEl.textContent = `Scene connected: ${res.camera.label}`; statusEl.style.color = 'var(--success)'; }
          showToast(`Scene camera connected: ${res.camera.label}`, 'success');
          body.hidden = false;
          chevron.innerHTML = '&#9650;';
        } else {
          if (statusEl) { statusEl.textContent = res.error || 'Not found'; statusEl.style.color = 'var(--danger)'; }
        }
      } catch (err) {
        if (statusEl) { statusEl.textContent = 'Error: ' + err.message; statusEl.style.color = 'var(--danger)'; }
      } finally {
        reconnectBtn.disabled = false;
        reconnectBtn.innerHTML = '&#x21bb; Reconnect Scene Camera';
      }
    });
  }
}

function _populateCameraDropdowns() {
  SLOT_KEYS.forEach(key => {
    const sel = $('slot-select-' + key);
    if (!sel) return;
    sel.innerHTML = '<option value="">— none —</option>';
    _detectedCameras.forEach(cam => {
      const opt = document.createElement('option');
      opt.value = JSON.stringify({ type: cam.type, id: cam.id });
      opt.textContent = cam.label;
      sel.appendChild(opt);
    });
    const current = _camSlotAssignment[key];
    if (current) {
      const val = JSON.stringify({ type: current.type, id: current.id });
      sel.value = val;
      if (!sel.value) sel.value = '';
    }
  });
}

function _readSlotSelections() {
  const slots = {};
  SLOT_KEYS.forEach(key => {
    const sel = $('slot-select-' + key);
    if (!sel || !sel.value) {
      slots[key] = null;
    } else {
      try { slots[key] = JSON.parse(sel.value); } catch { slots[key] = null; }
    }
  });
  return slots;
}

function _showCliCommand(cliArgs) {
  const row  = $('cam-cli-row');
  const code = $('cam-cli-code');
  if (!row || !code) return;
  const base = 'python record_server.py --arm0-port /dev/ttyACM0 --arm0-channel can0 --execute';
  code.textContent = cliArgs ? base + ' ' + cliArgs : base;
  row.hidden = false;
}

// ── Init ─────────────────────────────────────────────────
async function init() {
  setupTabs();
  setupCameraFeeds();
  setupArmControls();
  setupRecordingControls();
  setupAddNote();
  setupRefreshEpisodes();
  setupCameraPanel();
  setupVideoControls();

  await loadStatus();
  setupSSE();

  // Fallback poll every 5 s (SSE is primary)
  setInterval(loadStatus, 5000);
}

document.addEventListener("DOMContentLoaded", init);
