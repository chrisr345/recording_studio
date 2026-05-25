#!/usr/bin/env python3
"""Flask web server for robot teleoperation recording.

Serves a browser UI, runs SO-101→YAM teleop at 30 Hz, streams RealSense
camera MJPEG feeds, and saves demonstrations in LeRobot dataset format.

Usage:
    python record_server.py --arm0-port /dev/ttyUSB0 --arm0-channel can0 \
        --camera0-serial 353322271521 --execute
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure lerobot/src is importable
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_LEROBOT_SRC = os.path.join(_HERE, "lerobot", "src")
if _LEROBOT_SRC not in sys.path:
    sys.path.insert(0, _LEROBOT_SRC)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from flask import Flask, Response, jsonify, request, send_from_directory

from so101_yam_teleop import SO101Reader, SO101YamTeleop, YAM_JOINT_LIMITS, _now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("record_server")

# ---------------------------------------------------------------------------
# Placeholder JPEG (1×1 grey pixel) for cameras that have no frame yet
# ---------------------------------------------------------------------------

def _make_placeholder_jpeg() -> bytes:
    img = np.full((120, 160, 3), 80, dtype=np.uint8)
    cv2.putText(img, "No signal", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return bytes(buf) if ok else b""


PLACEHOLDER_JPEG = _make_placeholder_jpeg()

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class SharedState:
    arms_state: list[dict] = field(default_factory=list)
    # Last SO-101 readings per arm, updated by control loop. Engage endpoint reads
    # these instead of calling reader.read() directly (avoids serial port race).
    last_arm_readings: list[tuple] = field(default_factory=list)  # [(arm_rad, grip)|None]
    camera_jpegs: list[bytes] = field(default_factory=lambda: [PLACEHOLDER_JPEG] * 3)
    recording: bool = False
    episode_frame_count: int = 0
    current_task: str = ""
    total_episodes: int = 0
    total_frames: int = 0
    last_saved_episode: dict | None = None  # {episode_index, frames, ts}
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Episode metadata helpers
# ---------------------------------------------------------------------------

def _notes_dir(dataset_path: str) -> Path:
    p = Path(dataset_path) / "notes"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _notes_file(dataset_path: str, episode_index: int) -> Path:
    return _notes_dir(dataset_path) / f"episode_{episode_index:06d}.json"


def _load_notes(dataset_path: str, episode_index: int) -> list[dict]:
    p = _notes_file(dataset_path, episode_index)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _save_notes(dataset_path: str, episode_index: int, notes: list[dict]) -> None:
    _notes_file(dataset_path, episode_index).write_text(json.dumps(notes, indent=2))


def _deleted_episodes_path(dataset_path: str) -> Path:
    return Path(dataset_path) / "deleted_episodes.json"


def _load_deleted_episodes(dataset_path: str) -> set[int]:
    p = _deleted_episodes_path(dataset_path)
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except Exception:
        return set()


def _save_deleted_episodes(dataset_path: str, deleted: set[int]) -> None:
    _deleted_episodes_path(dataset_path).write_text(json.dumps(sorted(deleted), indent=2))


def _find_video_for_episode(dataset_path: str, episode_index: int, camera_key: str) -> Path | None:
    """Return the MP4 path for a given episode and camera key.

    LeRobot stores videos as:
      videos/observation.images.<cam_key>/chunk-<chunk:03d>/file-<file:03d>.mp4

    The chunk and file indices are stored in the episodes parquet under columns
    named "videos/observation.images.<cam_key>/chunk_index" and
    "videos/observation.images.<cam_key>/file_index".
    """
    full_key = f"observation.images.{camera_key}"
    chunk_col = f"videos/{full_key}/chunk_index"
    file_col = f"videos/{full_key}/file_index"

    rows = _read_episodes_parquet(dataset_path)
    row = next((r for r in rows if int(r.get("episode_index", -1)) == episode_index), None)
    if row is None:
        return None

    if chunk_col in row and file_col in row:
        chunk_idx = int(row[chunk_col])
        file_idx = int(row[file_col])
        path = (
            Path(dataset_path)
            / "videos"
            / full_key
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )
        return path if path.exists() else None

    # Fallback: glob for any mp4 under the camera key directory
    pattern = str(Path(dataset_path) / "videos" / full_key / "**" / "*.mp4")
    matches = sorted(glob.glob(pattern, recursive=True))
    if matches:
        return Path(matches[episode_index]) if episode_index < len(matches) else None
    return None


def _read_episodes_parquet(dataset_path: str) -> list[dict]:
    """Read episode rows from LeRobot parquet metadata. Returns list of dicts."""
    try:
        import pandas as pd
    except ImportError:
        return []

    episodes_glob = str(Path(dataset_path) / "meta" / "episodes" / "**" / "*.parquet")
    files = glob.glob(episodes_glob, recursive=True)
    if not files:
        # try flat path
        alt = Path(dataset_path) / "meta" / "episodes.parquet"
        if alt.exists():
            files = [str(alt)]
    if not files:
        return []

    try:
        import pandas as pd
        dfs = [pd.read_parquet(f) for f in sorted(files)]
        df = pd.concat(dfs, ignore_index=True)
        return df.to_dict("records")
    except Exception as exc:
        log.warning("Could not read episodes parquet: %s", exc)
        return []


def _read_tasks_parquet(dataset_path: str) -> dict[int, str]:
    """Returns {task_index: task_string}."""
    try:
        import pandas as pd
    except ImportError:
        return {}

    tasks_file = Path(dataset_path) / "meta" / "tasks.parquet"
    if not tasks_file.exists():
        # try chunk path
        tasks_glob = str(Path(dataset_path) / "meta" / "tasks" / "**" / "*.parquet")
        files = glob.glob(tasks_glob, recursive=True)
        if files:
            tasks_file = Path(sorted(files)[0])
        else:
            return {}
    try:
        df = pd.read_parquet(tasks_file).reset_index()
        return {int(row["task_index"]): row["task"] for _, row in df.iterrows()}
    except Exception as exc:
        log.warning("Could not read tasks parquet: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Camera detection and config persistence
# ---------------------------------------------------------------------------

def _detect_cameras() -> list[dict]:
    """Detect available RealSense devices and non-RealSense V4L2 cameras."""
    found = []

    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        for dev in ctx.query_devices():
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
                name = dev.get_info(rs.camera_info.name)
                found.append({"type": "realsense", "id": serial, "label": f"{name} (serial {serial})"})
            except Exception:
                pass
    except Exception as exc:
        log.debug("RealSense detection failed: %s", exc)

    for i in range(12):
        sys_name = Path(f"/sys/class/video4linux/video{i}/name")
        if not sys_name.exists():
            continue
        device_name = sys_name.read_text().strip()
        if "RealSense" in device_name or "Intel(R) RealSense" in device_name:
            continue
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    found.append({"type": "opencv", "id": str(i), "label": f"{device_name} (/dev/video{i})"})
            else:
                cap.release()
        except Exception:
            pass

    return found


CAMERA_SLOT_NAMES = ["wrist_0", "wrist_1", "scene"]


def _camera_config_path(dataset_path: str) -> Path:
    return Path(dataset_path) / "camera_config.json"


def _load_camera_config(dataset_path: str) -> dict:
    """Load {slots: {wrist_0: {type,id}|null, wrist_1: ..., scene: ...}} from disk."""
    p = _camera_config_path(dataset_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        log.warning("Could not read camera_config.json: %s", exc)
        return {}


def _save_camera_config(dataset_path: str, slots: dict) -> None:
    _camera_config_path(dataset_path).write_text(json.dumps({"slots": slots}, indent=2))


def _slots_to_cli_args(slots: dict) -> str:
    """Convert slot assignments to the CLI args string the user needs to pass."""
    parts = []
    w0 = slots.get("wrist_0")
    w1 = slots.get("wrist_1")
    sc = slots.get("scene")
    if w0 and w0.get("type") == "realsense":
        parts += ["--camera0-serial", w0["id"]]
    if w1 and w1.get("type") == "realsense":
        parts += ["--camera1-serial", w1["id"]]
    if sc:
        if sc.get("type") == "realsense":
            parts += ["--camera2-serial", sc["id"]]
        elif sc.get("type") == "opencv":
            parts += ["--camera2-index", sc["id"]]
    return " ".join(parts)


def _current_slot_config() -> dict:
    """Return current camera slot config from server_args."""
    if server_args is None:
        return {k: None for k in CAMERA_SLOT_NAMES}
    slots = {}
    s0 = getattr(server_args, "camera0_serial", None)
    s1 = getattr(server_args, "camera1_serial", None)
    s2 = getattr(server_args, "camera2_serial", None)
    cv_idx = getattr(server_args, "camera2_index", -1)
    slots["wrist_0"] = {"type": "realsense", "id": s0} if s0 else None
    slots["wrist_1"] = {"type": "realsense", "id": s1} if s1 else None
    if s2:
        slots["scene"] = {"type": "realsense", "id": s2}
    elif cv_idx is not None and cv_idx >= 0:
        slots["scene"] = {"type": "opencv", "id": str(cv_idx)}
    else:
        slots["scene"] = None
    return slots


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

def control_loop(
    readers: list[SO101Reader | None],
    teleops: list[SO101YamTeleop | None],
    cameras_rs: list[Any],   # RealSenseCamera or None
    cameras_cv: list[Any],   # cv2.VideoCapture or None
    dataset_ref: list[Any],  # [dataset | None]  — mutable so we can replace
    state: SharedState,
    stop_event: threading.Event,
    dataset_lock: threading.Lock,
    args: argparse.Namespace,
) -> None:
    period = 1.0 / args.hz
    next_tick = time.perf_counter()
    # Last non-None frame per RealSense slot — used as fallback when read_latest
    # returns None (e.g. brief USB hiccup or warm-up delay at startup).
    rs_frame_cache: list[Any] = [None] * max(len(cameras_rs), 1)

    log.info("Control loop started at %.1f Hz", args.hz)

    while not stop_event.is_set():
        now = time.perf_counter()
        sleep_s = next_tick - now
        if sleep_s > 0.001:
            time.sleep(sleep_s)
        next_tick = max(next_tick + period, time.perf_counter())

        # --- Read each arm ---
        arm_readings: list[tuple[np.ndarray | None, float | None]] = []
        for i, reader in enumerate(readers):
            if reader is None:
                arm_readings.append((None, None))
                continue
            try:
                arm_rad, grip = reader.read()
                arm_readings.append((arm_rad, grip))
            except Exception as exc:
                log.debug("Arm %d read error: %s", i, exc)
                arm_readings.append((None, None))

        # Cache readings so engage endpoint can use them without touching the serial port
        with state.lock:
            state.last_arm_readings = list(arm_readings)

        # --- Step each teleop ---
        arms_state_snapshot: list[dict] = []
        for i, teleop in enumerate(teleops):
            arm_rad, grip = arm_readings[i]
            connected = teleop is not None

            if teleop is None or arm_rad is None:
                arms_state_snapshot.append({
                    "id": i,
                    "active": False,
                    "q_actual": [0.0] * 7,
                    "q_cmd": None,
                    "connected": False,
                })
                continue

            try:
                info = teleop.step(arm_rad, grip)
                q_actual = info["q_actual"].tolist() if isinstance(info["q_actual"], np.ndarray) else list(info["q_actual"])
                q_cmd = None
                if info.get("q_cmd") is not None:
                    q_cmd = info["q_cmd"].tolist() if isinstance(info["q_cmd"], np.ndarray) else list(info["q_cmd"])
                arms_state_snapshot.append({
                    "id": i,
                    "active": bool(info["active"]),
                    "q_actual": q_actual,
                    "q_cmd": q_cmd,
                    "connected": True,
                })
            except Exception as exc:
                log.warning("Arm %d step error: %s", i, exc, exc_info=True)
                arms_state_snapshot.append({
                    "id": i,
                    "active": False,
                    "q_actual": [0.0] * 7,
                    "q_cmd": None,
                    "connected": True,
                })

        # --- Grab camera frames, encode to JPEG ---
        camera_jpegs: list[bytes] = []
        camera_frames_rgb: list[np.ndarray | None] = []

        # cameras 0 and 1: RealSense wrist cameras
        for cam_idx, cam in enumerate(cameras_rs):
            if cam is None:
                camera_jpegs.append(PLACEHOLDER_JPEG)
                camera_frames_rgb.append(None)
                continue
            try:
                frame = cam.read_latest(max_age_ms=1000)
                if frame is not None:
                    rs_frame_cache[cam_idx] = frame
                else:
                    frame = rs_frame_cache[cam_idx]  # fall back to last good frame
                if frame is not None:
                    # frame is RGB; convert to BGR for imencode
                    bgr = frame[:, :, ::-1]
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    camera_jpegs.append(bytes(buf) if ok else PLACEHOLDER_JPEG)
                    camera_frames_rgb.append(frame)
                else:
                    camera_jpegs.append(PLACEHOLDER_JPEG)
                    camera_frames_rgb.append(None)
            except Exception as exc:
                log.debug("RealSense camera read error: %s", exc)
                camera_jpegs.append(PLACEHOLDER_JPEG)
                camera_frames_rgb.append(None)

        # camera 2: scene camera (OpenCV or RealSense)
        scene_cam = cameras_cv[0] if cameras_cv else None
        if scene_cam is not None:
            try:
                ret, frame_bgr = scene_cam.read()
                if ret and frame_bgr is not None:
                    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    camera_jpegs.append(bytes(buf) if ok else PLACEHOLDER_JPEG)
                    camera_frames_rgb.append(frame_bgr[:, :, ::-1])  # BGR→RGB
                else:
                    camera_jpegs.append(PLACEHOLDER_JPEG)
                    camera_frames_rgb.append(None)
            except Exception as exc:
                log.debug("Scene camera read error: %s", exc)
                camera_jpegs.append(PLACEHOLDER_JPEG)
                camera_frames_rgb.append(None)
        else:
            camera_jpegs.append(PLACEHOLDER_JPEG)
            camera_frames_rgb.append(None)

        # Pad to 3 total cameras
        while len(camera_jpegs) < 3:
            camera_jpegs.append(PLACEHOLDER_JPEG)
            camera_frames_rgb.append(None)

        # --- If recording, build and add frame ---
        with state.lock:
            recording = state.recording
            current_task = state.current_task

        if recording:
            dataset = dataset_ref[0]
            if dataset is not None and arms_state_snapshot:
                # Build observation.state and action arrays
                q_actual_parts = []
                q_cmd_parts = []
                for arm_s in arms_state_snapshot:
                    q_actual_parts.append(arm_s["q_actual"][:7])
                    q_cmd_parts.append(arm_s["q_cmd"][:7] if arm_s["q_cmd"] else arm_s["q_actual"][:7])

                obs_state = np.concatenate(q_actual_parts, axis=0).astype(np.float32)
                action = np.concatenate(q_cmd_parts, axis=0).astype(np.float32)

                frame_dict: dict[str, Any] = {
                    "task": current_task,
                    "observation.state": obs_state,
                    "action": action,
                }

                # Camera keys in order: wrist_0, wrist_1, scene
                cam_keys = ["wrist_0", "wrist_1", "scene"]
                for cidx, rgb_frame in enumerate(camera_frames_rgb[:3]):
                    key = f"observation.images.{cam_keys[cidx]}"
                    if rgb_frame is not None:
                        # Ensure uint8 RGB (480, 640, 3)
                        frame_dict[key] = rgb_frame.astype(np.uint8)

                # LeRobot requires an exact feature match — drop any keys the
                # dataset wasn't created with (e.g. a newly added scene camera).
                ds_features = set(dataset.features.keys())
                frame_dict = {k: v for k, v in frame_dict.items()
                              if k in ds_features or k == "task"}

                try:
                    with dataset_lock:
                        dataset.add_frame(frame_dict)
                    with state.lock:
                        state.episode_frame_count += 1
                except Exception as exc:
                    log.warning("dataset.add_frame error: %s", exc)

        # --- Update shared state ---
        with state.lock:
            state.arms_state = arms_state_snapshot
            state.camera_jpegs = camera_jpegs

    log.info("Control loop stopped.")


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

STATIC_DIR = os.path.join(_HERE, "static")

app = Flask(__name__)

# These globals are set in main() before the server starts
shared_state: SharedState = SharedState()
stop_event: threading.Event = threading.Event()
dataset_ref: list[Any] = [None]
dataset_lock: threading.Lock = threading.Lock()
readers: list[SO101Reader | None] = []
teleops: list[SO101YamTeleop | None] = []
cameras_rs: list[Any] = []
cameras_cv: list[Any] = []
server_args: argparse.Namespace | None = None

# Recording episode tracking
_episode_start_time: float = 0.0


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    try:
        return send_from_directory(STATIC_DIR, "index.html")
    except Exception:
        return "<h1>record_server running</h1><p>Place index.html in ./static/</p>", 200


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


# ---------------------------------------------------------------------------
# MJPEG camera streaming
# ---------------------------------------------------------------------------

def generate_mjpeg(camera_id: int, state: SharedState):
    while True:
        with state.lock:
            jpegs = state.camera_jpegs
            jpeg = jpegs[camera_id] if camera_id < len(jpegs) else PLACEHOLDER_JPEG
        if not jpeg:
            jpeg = PLACEHOLDER_JPEG
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.033)


@app.route("/stream/<int:camera_id>")
def stream(camera_id: int):
    if camera_id not in (0, 1, 2):
        return jsonify({"error": "camera_id must be 0, 1, or 2"}), 400
    return Response(
        generate_mjpeg(camera_id, shared_state),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# SSE state stream
# ---------------------------------------------------------------------------

@app.route("/state/stream")
def state_stream():
    def generate():
        while True:
            with shared_state.lock:
                data = {
                    "timestamp": time.time(),
                    "arms": shared_state.arms_state,
                    "recording": shared_state.recording,
                    "episode_frame_count": shared_state.episode_frame_count,
                    "current_task": shared_state.current_task,
                    "total_episodes": shared_state.total_episodes,
                    "total_frames": shared_state.total_frames,
                    "last_saved_episode": shared_state.last_saved_episode,
                }
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Arm control
# ---------------------------------------------------------------------------

@app.route("/arm/<int:arm_id>/engage", methods=["POST"])
def arm_engage(arm_id: int):
    if arm_id >= len(teleops) or teleops[arm_id] is None:
        return jsonify({"success": False, "error": f"Arm {arm_id} not connected"}), 404
    if arm_id >= len(readers) or readers[arm_id] is None:
        return jsonify({"success": False, "error": f"SO-101 reader for arm {arm_id} not connected"}), 404
    try:
        # Use the last reading cached by the control loop — avoids concurrent
        # serial port access which corrupts reads on the Feetech bus.
        with shared_state.lock:
            readings = list(shared_state.last_arm_readings)
        if arm_id < len(readings) and readings[arm_id][0] is not None:
            arm_rad, grip = readings[arm_id]
        else:
            return jsonify({"success": False, "error": "No arm reading available yet — is the SO-101 connected?"}), 503
        teleops[arm_id].engage(arm_rad, grip)
        return jsonify({"success": True, "error": None})
    except Exception as exc:
        log.exception("arm_engage error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/arm/<int:arm_id>/disengage", methods=["POST"])
def arm_disengage(arm_id: int):
    if arm_id >= len(teleops) or teleops[arm_id] is None:
        return jsonify({"success": False, "error": f"Arm {arm_id} not connected"}), 404
    try:
        teleops[arm_id].disengage()
        return jsonify({"success": True, "error": None})
    except Exception as exc:
        log.exception("arm_disengage error")
        return jsonify({"success": False, "error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Recording control
# ---------------------------------------------------------------------------

@app.route("/recording/start", methods=["POST"])
def recording_start():
    global _episode_start_time

    body = request.get_json(silent=True) or {}
    task = body.get("task", server_args.task if server_args else "")

    with shared_state.lock:
        if shared_state.recording:
            return jsonify({"success": False, "error": "Already recording"}), 400

    if dataset_ref[0] is None:
        return jsonify({"success": False, "error": "Dataset not initialized"}), 500

    with shared_state.lock:
        shared_state.recording = True
        shared_state.episode_frame_count = 0
        shared_state.current_task = task
    _episode_start_time = time.time()

    log.info("Recording started: task=%r", task)
    return jsonify({"success": True, "error": None})


@app.route("/recording/stop", methods=["POST"])
def recording_stop():
    with shared_state.lock:
        if not shared_state.recording:
            return jsonify({"success": False, "error": "Not recording"}), 400
        shared_state.recording = False
        frame_count = shared_state.episode_frame_count
        shared_state.episode_frame_count = 0

    dataset = dataset_ref[0]
    if dataset is None:
        return jsonify({"success": False, "error": "Dataset not available"}), 500

    episode_index = -1
    try:
        with dataset_lock:
            if dataset.has_pending_frames():
                dataset.save_episode()
                # Force-flush the episodes parquet immediately — resume() doesn't
                # accept metadata_buffer_size so the default is 10; without this
                # the parquet file isn't written until 10 episodes accumulate.
                try:
                    from lerobot.datasets.utils import update_chunk_file_indices
                    dataset.meta._flush_metadata_buffer()
                    if hasattr(dataset.meta, "_pq_writer") and dataset.meta._pq_writer:
                        dataset.meta._pq_writer.close()
                        dataset.meta._pq_writer = None
                    # Advance the parquet file indices so the NEXT episode is
                    # written to a new file. Without this, _save_episode_metadata()
                    # sees latest_episode pointing to the same chunk/file and
                    # creates a new ParquetWriter at the same path, overwriting
                    # all previous episodes in that file.
                    if dataset.meta.latest_episode is not None:
                        old_chunk = dataset.meta.latest_episode["meta/episodes/chunk_index"][0]
                        old_file = dataset.meta.latest_episode["meta/episodes/file_index"][0]
                        new_chunk, new_file = update_chunk_file_indices(
                            old_chunk, old_file, dataset.meta.chunks_size
                        )
                        dataset.meta.latest_episode["meta/episodes/chunk_index"] = [new_chunk]
                        dataset.meta.latest_episode["meta/episodes/file_index"] = [new_file]
                except Exception as flush_exc:
                    log.warning("metadata flush error (non-fatal): %s", flush_exc)
                episode_index = dataset.meta.total_episodes - 1
                with shared_state.lock:
                    shared_state.total_episodes = dataset.meta.total_episodes
                    shared_state.total_frames = dataset.meta.total_frames
                    shared_state.last_saved_episode = {
                        "episode_index": episode_index,
                        "frames": frame_count,
                        "ts": time.time(),
                    }
                log.info("Episode %d saved (%d frames)", episode_index, frame_count)
            else:
                dataset.clear_episode_buffer()
                log.info("No frames to save — episode buffer cleared")
    except Exception as exc:
        log.exception("save_episode error")
        return jsonify({"success": False, "error": str(exc)}), 500

    return jsonify({"success": True, "episode_index": episode_index, "frames": frame_count})


@app.route("/recording/discard", methods=["POST"])
def recording_discard():
    with shared_state.lock:
        shared_state.recording = False
        shared_state.episode_frame_count = 0

    dataset = dataset_ref[0]
    if dataset is not None:
        try:
            with dataset_lock:
                dataset.clear_episode_buffer()
        except Exception as exc:
            log.warning("clear_episode_buffer error: %s", exc)

    log.info("Recording discarded")
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Episode listing and detail
# ---------------------------------------------------------------------------

def _resolve_task_from_row(row: dict, tasks: dict[int, str]) -> str:
    """Extract task string from an episode row.

    The 'tasks' column is a list/array of task strings (e.g. ["manipulation task"]).
    Fall back to looking up task_index in the tasks dict.
    """
    tasks_val = row.get("tasks")
    if tasks_val is not None:
        # numpy array, list, or similar sequence
        try:
            if hasattr(tasks_val, "__iter__") and not isinstance(tasks_val, str):
                items = list(tasks_val)
                if items:
                    return str(items[0])
        except Exception:
            pass
        if isinstance(tasks_val, str):
            return tasks_val

    task_index = row.get("task_index", 0)
    return tasks.get(int(task_index), "")


def _resolve_timestamps(row: dict, length: int) -> tuple[float, float]:
    """Return (timestamp, duration_s) from an episode row.

    Tries the video-prefixed timestamp columns first (LeRobot v2 format),
    then bare from_timestamp/to_timestamp, then falls back to length/fps.
    """
    # Try prefixed columns for any camera key found in the row
    import re
    for col in row:
        m = re.match(r"^videos/(.+)/from_timestamp$", col)
        if m:
            to_col = f"videos/{m.group(1)}/to_timestamp"
            from_ts = row.get(col)
            to_ts = row.get(to_col)
            if from_ts is not None and to_ts is not None:
                return float(from_ts), float(to_ts) - float(from_ts)

    # Try bare columns (older format)
    from_ts = row.get("from_timestamp")
    to_ts = row.get("to_timestamp")
    if from_ts is not None and to_ts is not None:
        return float(from_ts), float(to_ts) - float(from_ts)

    return 0.0, length / 30.0


def _build_episode_summary(row: dict, dataset_path: str, tasks: dict[int, str], deleted: set[int]) -> dict | None:
    idx = int(row.get("episode_index", -1))
    if idx < 0 or idx in deleted:
        return None

    task = _resolve_task_from_row(row, tasks)
    length = int(row.get("length", 0))
    timestamp, duration_s = _resolve_timestamps(row, length)

    if timestamp == 0.0:
        nf = _notes_file(dataset_path, idx)
        if nf.exists():
            timestamp = nf.stat().st_mtime
        else:
            for cam_key in ("wrist_0", "wrist_1", "scene"):
                vp = _find_video_for_episode(dataset_path, idx, cam_key)
                if vp is not None:
                    timestamp = vp.stat().st_mtime
                    break

    has_video: dict[str, bool] = {}
    for cam_key in ("wrist_0", "wrist_1", "scene"):
        has_video[cam_key] = _find_video_for_episode(dataset_path, idx, cam_key) is not None

    notes = _load_notes(dataset_path, idx)

    return {
        "episode_index": idx,
        "task": task,
        "length": length,
        "timestamp": timestamp,
        "duration_s": duration_s,
        "has_video": has_video,
        "notes_count": len(notes),
    }


@app.route("/episodes")
def episodes_list():
    ds_path = server_args.dataset_path if server_args else "./recordings"
    rows = _read_episodes_parquet(ds_path)
    tasks = _read_tasks_parquet(ds_path)
    deleted = _load_deleted_episodes(ds_path)

    summaries = []
    for row in rows:
        s = _build_episode_summary(row, ds_path, tasks, deleted)
        if s is not None:
            summaries.append(s)

    # Newest first
    summaries.sort(key=lambda x: x["episode_index"], reverse=True)
    return jsonify(summaries)


@app.route("/episodes/<int:idx>")
def episode_detail(idx: int):
    ds_path = server_args.dataset_path if server_args else "./recordings"
    deleted = _load_deleted_episodes(ds_path)
    if idx in deleted:
        return jsonify({"error": "Episode deleted"}), 404

    rows = _read_episodes_parquet(ds_path)
    tasks = _read_tasks_parquet(ds_path)

    row = next((r for r in rows if int(r.get("episode_index", -1)) == idx), None)

    if row is None:
        return jsonify({"error": f"Episode {idx} not found"}), 404

    task = _resolve_task_from_row(row, tasks)
    length = int(row.get("length", 0))
    timestamp, duration_s = _resolve_timestamps(row, length)

    if timestamp == 0.0:
        nf = _notes_file(ds_path, idx)
        if nf.exists():
            timestamp = nf.stat().st_mtime
        else:
            for cam_key in ("wrist_0", "wrist_1", "scene"):
                vp = _find_video_for_episode(ds_path, idx, cam_key)
                if vp is not None:
                    timestamp = vp.stat().st_mtime
                    break

    notes = _load_notes(ds_path, idx)

    return jsonify({
        "episode_index": idx,
        "task": task,
        "length": length,
        "fps": 30,
        "timestamp": timestamp,
        "duration_s": duration_s,
        "notes": notes,
    })


@app.route("/episodes/<int:idx>/video/<camera_key>")
def episode_video(idx: int, camera_key: str):
    if camera_key not in ("wrist_0", "wrist_1", "scene"):
        return jsonify({"error": "Invalid camera_key"}), 400

    ds_path = server_args.dataset_path if server_args else "./recordings"
    deleted = _load_deleted_episodes(ds_path)
    if idx in deleted:
        return jsonify({"error": "Episode deleted"}), 404

    video_path = _find_video_for_episode(ds_path, idx, camera_key)
    if video_path is None or not video_path.exists():
        return jsonify({"error": "Video not found"}), 404

    from flask import send_file
    return send_file(str(video_path), mimetype="video/mp4", conditional=True)


# ---------------------------------------------------------------------------
# Episode notes
# ---------------------------------------------------------------------------

@app.route("/episodes/<int:idx>/notes", methods=["POST"])
def add_note(idx: int):
    ds_path = server_args.dataset_path if server_args else "./recordings"
    deleted = _load_deleted_episodes(ds_path)
    if idx in deleted:
        return jsonify({"error": "Episode deleted"}), 404

    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    timestamp_s = float(body.get("timestamp_s", -1.0))

    note = {
        "id": uuid.uuid4().hex,
        "text": text,
        "timestamp_s": timestamp_s,
        "created_at": time.time(),
    }
    notes = _load_notes(ds_path, idx)
    notes.append(note)
    _save_notes(ds_path, idx, notes)
    return jsonify(note), 201


@app.route("/episodes/<int:idx>/notes/<note_id>", methods=["PUT"])
def update_note(idx: int, note_id: str):
    ds_path = server_args.dataset_path if server_args else "./recordings"
    body = request.get_json(silent=True) or {}
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    notes = _load_notes(ds_path, idx)
    for note in notes:
        if note["id"] == note_id:
            note["text"] = text
            _save_notes(ds_path, idx, notes)
            return jsonify(note)
    return jsonify({"error": "Note not found"}), 404


@app.route("/episodes/<int:idx>/notes/<note_id>", methods=["DELETE"])
def delete_note(idx: int, note_id: str):
    ds_path = server_args.dataset_path if server_args else "./recordings"
    notes = _load_notes(ds_path, idx)
    original_len = len(notes)
    notes = [n for n in notes if n["id"] != note_id]
    if len(notes) == original_len:
        return jsonify({"error": "Note not found"}), 404
    _save_notes(ds_path, idx, notes)
    return jsonify({"success": True})


@app.route("/episodes/<int:idx>", methods=["DELETE"])
def delete_episode(idx: int):
    ds_path = server_args.dataset_path if server_args else "./recordings"
    deleted = _load_deleted_episodes(ds_path)
    deleted.add(idx)
    _save_deleted_episodes(ds_path, deleted)
    # Also remove notes file
    nf = _notes_file(ds_path, idx)
    if nf.exists():
        try:
            nf.unlink()
        except Exception:
            pass
    log.info("Episode %d marked as deleted", idx)
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.route("/status")
def status():
    ds_path = server_args.dataset_path if server_args else "./recordings"

    arms_info = []
    for i, (reader, teleop) in enumerate(zip(readers, teleops)):
        arms_info.append({
            "id": i,
            "so101_port": getattr(server_args, f"arm{i}_port", None) if server_args else None,
            "yam_channel": getattr(server_args, f"arm{i}_channel", None) if server_args else None,
            "connected": reader is not None and teleop is not None,
        })

    cam_names = ["wrist_0", "wrist_1", "scene"]
    cameras_info = []
    for i, cam in enumerate(cameras_rs):
        cameras_info.append({
            "id": i,
            "name": cam_names[i],
            "connected": cam is not None,
        })
    # Scene camera
    scene_connected = len(cameras_cv) > 0 and cameras_cv[0] is not None
    if len(cameras_rs) < 3:
        cameras_info.append({
            "id": 2,
            "name": "scene",
            "connected": scene_connected,
        })

    with shared_state.lock:
        total_episodes = shared_state.total_episodes
        total_frames = shared_state.total_frames

    return jsonify({
        "arms": arms_info,
        "cameras": cameras_info,
        "dataset_path": str(Path(ds_path).resolve()),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    })


@app.route("/cameras/detect")
def cameras_detect():
    detected = _detect_cameras()
    current = _current_slot_config()
    cli = _slots_to_cli_args(current)
    return jsonify({"detected": detected, "current_slots": current, "cli_args": cli})


@app.route("/cameras/save-config", methods=["POST"])
def cameras_save_config():
    ds_path = server_args.dataset_path if server_args else "./recordings"
    body = request.get_json(silent=True) or {}
    slots = body.get("slots")
    if not isinstance(slots, dict):
        return jsonify({"error": "slots must be an object"}), 400
    normalized = {}
    for key in CAMERA_SLOT_NAMES:
        val = slots.get(key)
        if val and val.get("type") and val.get("id"):
            normalized[key] = {"type": val["type"], "id": str(val["id"])}
        else:
            normalized[key] = None
    _save_camera_config(ds_path, normalized)
    cli = _slots_to_cli_args(normalized)
    log.info("Camera config saved: %s (cli: %s)", normalized, cli)
    return jsonify({"success": True, "cli_args": cli})


# ---------------------------------------------------------------------------
# Dataset initialization
# ---------------------------------------------------------------------------

def _has_recorded_data(dataset_path: str) -> bool:
    """Return True if the directory contains any video or parquet files."""
    import glob
    p = Path(dataset_path)
    return bool(
        glob.glob(str(p / "**" / "*.mp4"), recursive=True) or
        glob.glob(str(p / "**" / "*.parquet"), recursive=True)
    )


def _quarantine_corrupt_parquets(dataset_path: str) -> list[str]:
    """Find unreadable parquet files and rename them to .corrupt.<ts>.
    Returns list of paths that were quarantined.
    """
    import glob as _glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return []

    quarantined = []
    ts = int(time.time())
    for pattern in ["data/**/*.parquet", "meta/**/*.parquet"]:
        for f in sorted(_glob.glob(str(Path(dataset_path) / pattern), recursive=True)):
            try:
                pq.read_schema(f)  # lightweight header-only check
            except Exception:
                dest = f + f".corrupt.{ts}"
                try:
                    os.rename(f, dest)
                    quarantined.append(f)
                    log.warning("Quarantined corrupt parquet: %s → %s", f, dest)
                except Exception as mv_exc:
                    log.error("Could not quarantine %s: %s", f, mv_exc)
    return quarantined


def _init_dataset(args: argparse.Namespace, num_arms: int, camera_keys: list[str]) -> Any:
    """Create or resume a LeRobot dataset. Returns dataset or None on failure."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        log.error("Cannot import LeRobotDataset: %s", exc)
        return None

    dataset_path = args.dataset_path

    import shutil

    # Explicit reset requested — rename existing data to a timestamped backup first.
    if getattr(args, "reset_dataset", False) and Path(dataset_path).exists():
        backup = f"{dataset_path}.bak.{int(time.time())}"
        try:
            shutil.move(dataset_path, backup)
            log.info("--reset-dataset: moved existing data to %s", backup)
        except Exception as exc:
            log.error("Could not back up dataset directory: %s", exc)
            return None

    info_file = Path(dataset_path) / "meta" / "info.json"
    if info_file.exists():
        try:
            log.info("Resuming existing dataset at %s", dataset_path)
            # Note: resume() does NOT accept metadata_buffer_size — that's create()-only.
            dataset = LeRobotDataset.resume(
                repo_id=args.dataset_name,
                root=Path(dataset_path),
                streaming_encoding=True,
            )
            _notes_dir(dataset_path)
            return dataset
        except Exception as exc:
            log.warning("Failed to resume dataset at %s: %s — scanning for corrupt parquets ...", dataset_path, exc)
            quarantined = _quarantine_corrupt_parquets(dataset_path)
            if quarantined:
                log.info("Quarantined %d corrupt file(s), retrying resume ...", len(quarantined))
                try:
                    dataset = LeRobotDataset.resume(
                        repo_id=args.dataset_name,
                        root=Path(dataset_path),
                        streaming_encoding=True,
                    )
                    _notes_dir(dataset_path)
                    log.info("Resume succeeded after quarantine. Lost episode data: %s", quarantined)
                    return dataset
                except Exception as retry_exc:
                    log.error(
                        "Resume still failed after quarantine: %s\n"
                        "Your recordings are intact. Inspect %s and restart.",
                        retry_exc, dataset_path,
                    )
                    return None
            else:
                log.error(
                    "Failed to resume dataset at %s: %s\n"
                    "Your recordings are intact. To start fresh run with --reset-dataset.",
                    dataset_path, exc,
                )
                return None
    elif Path(dataset_path).exists():
        if _has_recorded_data(dataset_path):
            # Directory has data but no info.json — corrupted or partial write.
            log.error(
                "Dataset directory %s exists with data files but no meta/info.json. "
                "Your recordings may be recoverable. Refusing to overwrite. "
                "Move or rename the directory manually, then restart.",
                dataset_path,
            )
            return None
        # Genuinely empty leftover from a previous failed create() — safe to remove.
        try:
            shutil.rmtree(dataset_path)
            log.info("Removed empty dataset directory (no info.json, no data): %s", dataset_path)
        except Exception as rm_exc:
            log.error("Could not remove empty dataset directory %s: %s", dataset_path, rm_exc)
            return None

    # Build features dict
    state_shape = (num_arms * 7,)
    state_names = []
    for arm_i in range(num_arms):
        prefix = f"arm{arm_i}_" if num_arms > 1 else ""
        state_names += [f"{prefix}{n}" for n in ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]]

    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": state_shape,
            "names": state_names,
        },
    }

    for cam_key in camera_keys:
        features[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    try:
        log.info("Creating new dataset at %s (repo_id=%s)", dataset_path, args.dataset_name)
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_name,
            fps=int(args.hz),
            features=features,
            root=Path(dataset_path),
            use_videos=True,
            streaming_encoding=True,
            metadata_buffer_size=1,
        )
        _notes_dir(dataset_path)
        return dataset
    except Exception as exc:
        log.error("Failed to create dataset: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Hardware initialization
# ---------------------------------------------------------------------------

def _init_readers_and_teleops(args: argparse.Namespace):
    """Initialize SO-101 readers and YAM teleop objects. Returns (readers, teleops)."""
    _readers: list[SO101Reader | None] = []
    _teleops: list[SO101YamTeleop | None] = []

    arm_configs = []
    if args.arm0_port:
        arm_configs.append((0, args.arm0_port, args.arm0_channel))
    if getattr(args, "arm1_port", None):
        arm_configs.append((1, args.arm1_port, args.arm1_channel))

    # Ensure lists are sized for both possible arms (pad with None)
    max_arm = max((ac[0] for ac in arm_configs), default=-1)
    for arm_idx in range(max_arm + 1):
        _readers.append(None)
        _teleops.append(None)

    for arm_i, port, channel in arm_configs:
        # SO-101 reader
        try:
            log.info("Connecting SO-101 arm %d on %s ...", arm_i, port)
            reader = SO101Reader(port=port)
            _readers[arm_i] = reader
            log.info("SO-101 arm %d connected.", arm_i)
        except Exception as exc:
            log.warning("SO-101 arm %d failed to connect on %s: %s", arm_i, port, exc)

        # YAM teleop
        try:
            log.info("Connecting YAM arm %d on %s ...", arm_i, channel)
            gains = args.gains
            gripper_invert = args.gripper_invert
            if arm_i == 1:
                if getattr(args, "arm1_gains", None) is not None:
                    gains = args.arm1_gains
                if getattr(args, "arm1_gripper_invert", None) is not None:
                    gripper_invert = args.arm1_gripper_invert
            arm_args = argparse.Namespace(
                channel=channel,
                gains=gains,
                gripper_invert=gripper_invert,
                max_joint_step=args.max_joint_step,
                execute=args.execute,
            )
            teleop = SO101YamTeleop(arm_args)
            _teleops[arm_i] = teleop
            log.info("YAM arm %d connected.", arm_i)
        except Exception as exc:
            log.warning("YAM arm %d failed to connect on %s: %s", arm_i, channel, exc)

    # Ensure at least one slot so arm 0 exists
    if not _readers:
        _readers.append(None)
        _teleops.append(None)

    return _readers, _teleops


def _init_cameras(args: argparse.Namespace):
    """Initialize RealSense and OpenCV cameras. Returns (cameras_rs, cameras_cv, camera_keys)."""
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        has_rs_lib = True
    except ImportError:
        log.warning("lerobot RealSense camera not available")
        has_rs_lib = False

    _cameras_rs: list[Any] = []
    _camera_keys: list[str] = []

    # Wrist cameras 0 and 1 (RealSense)
    for cam_i, serial_attr in enumerate(["camera0_serial", "camera1_serial"]):
        serial = getattr(args, serial_attr, None)
        if serial and has_rs_lib:
            try:
                log.info("Connecting RealSense camera %d (serial %s) ...", cam_i, serial)
                config = RealSenseCameraConfig(
                    serial_number_or_name=serial,
                    fps=30,
                    width=640,
                    height=480,
                )
                cam = RealSenseCamera(config)
                cam.connect()
                _cameras_rs.append(cam)
                key = f"wrist_{cam_i}"
                _camera_keys.append(key)
                log.info("RealSense camera %d connected.", cam_i)
            except Exception as exc:
                log.warning("RealSense camera %d (serial %s) failed: %s", cam_i, serial, exc)
                _cameras_rs.append(None)
        else:
            _cameras_rs.append(None)

    # Scene camera: RealSense serial takes precedence over OpenCV index
    _cameras_cv: list[Any] = []
    scene_serial = getattr(args, "camera2_serial", None)
    scene_index = getattr(args, "camera2_index", -1)

    if scene_serial and has_rs_lib:
        try:
            log.info("Connecting scene RealSense (serial %s) ...", scene_serial)
            config = RealSenseCameraConfig(
                serial_number_or_name=scene_serial,
                fps=30,
                width=640,
                height=480,
            )
            cam = RealSenseCamera(config)
            cam.connect()
            _cameras_rs.append(cam)
            _camera_keys.append("scene")
            log.info("Scene RealSense connected.")
        except Exception as exc:
            log.warning("Scene RealSense (serial %s) failed: %s", scene_serial, exc)
            _cameras_rs.append(None)
    elif scene_index >= 0:
        try:
            log.info("Opening scene camera OpenCV index %d ...", scene_index)
            cap = cv2.VideoCapture(scene_index)
            if cap.isOpened():
                _cameras_cv.append(cap)
                _camera_keys.append("scene")
                log.info("Scene OpenCV camera connected.")
            else:
                cap.release()
                log.warning("Scene OpenCV camera index %d not available", scene_index)
                _cameras_cv.append(None)
        except Exception as exc:
            log.warning("Scene OpenCV camera failed: %s", exc)
            _cameras_cv.append(None)
    else:
        # No scene camera
        pass

    return _cameras_rs, _cameras_cv, _camera_keys


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------

def _shutdown(dataset: Any, _readers: list, _teleops: list, _cameras_rs: list, _cameras_cv: list):
    log.info("Shutting down ...")

    if dataset is not None:
        try:
            with dataset_lock:
                # Rescue any in-progress recording before finalizing
                with shared_state.lock:
                    was_recording = shared_state.recording
                if was_recording and dataset.has_pending_frames():
                    log.info("Server shutdown during recording — auto-saving episode")
                    dataset.save_episode()
                dataset.finalize()
            log.info("Dataset finalized.")
        except Exception as exc:
            log.warning("dataset.finalize() error: %s", exc)

    for i, reader in enumerate(_readers):
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:
                log.warning("SO-101 reader %d close error: %s", i, exc)

    for i, teleop in enumerate(_teleops):
        if teleop is not None:
            try:
                teleop.close()
            except Exception as exc:
                log.warning("YAM teleop %d close error: %s", i, exc)

    for i, cam in enumerate(_cameras_rs):
        if cam is not None:
            try:
                cam.disconnect()
            except Exception as exc:
                log.warning("RealSense camera %d disconnect error: %s", i, exc)

    for cap in _cameras_cv:
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                log.warning("OpenCV camera release error: %s", exc)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flask web server for robot teleoperation recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Arm 0
    parser.add_argument("--arm0-port", default="/dev/ttyACM0",
                        help="SO-101 arm 0 serial port")
    parser.add_argument("--arm0-channel", default="can0",
                        help="YAM arm 0 CAN channel")

    # Arm 1 (optional)
    parser.add_argument("--arm1-port", default=None,
                        help="SO-101 arm 1 serial port (omit for single arm)")
    parser.add_argument("--arm1-channel", default="can1",
                        help="YAM arm 1 CAN channel")
    parser.add_argument(
        "--arm1-gains", type=float, nargs=5,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="Per-joint gains for arm 1 (defaults to --gains if omitted)",
    )
    parser.add_argument("--arm1-gripper-invert", action="store_true", default=None,
                        help="Invert gripper on arm 1 (defaults to --gripper-invert if omitted)")
    parser.add_argument("--no-arm1-gripper-invert", dest="arm1_gripper_invert", action="store_false",
                        help="Disable gripper inversion on arm 1")

    # Cameras
    parser.add_argument("--camera0-serial", default="353322271521",
                        help="Wrist camera 0 RealSense serial")
    parser.add_argument("--camera1-serial", default=None,
                        help="Wrist camera 1 RealSense serial (optional)")
    parser.add_argument("--camera2-serial", default=None,
                        help="Scene camera RealSense serial (optional)")
    parser.add_argument("--camera2-index", type=int, default=-1,
                        help="Scene camera OpenCV index (-1 = none)")

    # Robot control
    parser.add_argument("--execute", action="store_true",
                        help="Actually send commands to robot")
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency (Hz)")
    parser.add_argument(
        "--gains", type=float, nargs=5,
        default=[-1.0, 1.0, -1.0, -1.0, 1.0],
        metavar=("J1", "J2", "J3", "J4", "J5"),
        help="Per-joint gains for SO-101→YAM mapping",
    )
    parser.add_argument("--max-joint-step", type=float, default=0.05,
                        help="Rate limit per tick (rad). 0 = disabled.")
    parser.add_argument("--gripper-invert", action="store_true", default=True,
                        help="Invert gripper direction (default: on)")
    parser.add_argument("--no-gripper-invert", dest="gripper_invert", action="store_false",
                        help="Disable gripper inversion")

    # Dataset
    parser.add_argument("--dataset-path", default="./recordings",
                        help="Where to save the LeRobot dataset")
    parser.add_argument("--dataset-name", default="local/yam_demos",
                        help="Dataset repo_id")
    parser.add_argument("--task", default="manipulation task",
                        help="Default task description")
    parser.add_argument("--reset-dataset", action="store_true",
                        help="Move existing dataset to a timestamped backup and start fresh")

    # Server
    parser.add_argument("--web-port", type=int, default=5000,
                        help="Web server port")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    global shared_state, stop_event, dataset_ref, dataset_lock
    global readers, teleops, cameras_rs, cameras_cv, server_args

    parser = _build_parser()
    args = parser.parse_args()
    server_args = args

    _cam_cfg = _load_camera_config(args.dataset_path)
    _cam_slots = _cam_cfg.get("slots", {})
    _parser_defaults = vars(parser.parse_args([]))
    def _arg_was_default(attr: str) -> bool:
        return getattr(args, attr, None) == _parser_defaults.get(attr)

    if _cam_slots.get("wrist_0") and _arg_was_default("camera0_serial"):
        s = _cam_slots["wrist_0"]
        if s and s.get("type") == "realsense":
            args.camera0_serial = s["id"]
            log.info("camera_config.json: wrist_0 → RealSense %s", s["id"])

    if _cam_slots.get("wrist_1") and _arg_was_default("camera1_serial"):
        s = _cam_slots["wrist_1"]
        if s and s.get("type") == "realsense":
            args.camera1_serial = s["id"]
            log.info("camera_config.json: wrist_1 → RealSense %s", s["id"])

    if _cam_slots.get("scene") and _arg_was_default("camera2_serial") and _arg_was_default("camera2_index"):
        s = _cam_slots["scene"]
        if s and s.get("type") == "realsense":
            args.camera2_serial = s["id"]
            log.info("camera_config.json: scene → RealSense %s", s["id"])
        elif s and s.get("type") == "opencv":
            args.camera2_index = int(s["id"])
            log.info("camera_config.json: scene → OpenCV index %s", s["id"])

    # Ensure static dir exists (don't fail if absent)
    os.makedirs(STATIC_DIR, exist_ok=True)

    # --- Initialize hardware ---
    readers, teleops = _init_readers_and_teleops(args)
    cameras_rs, cameras_cv, camera_keys = _init_cameras(args)

    # Number of configured arms
    num_arms = len([t for t in teleops if t is not None]) or len(readers)
    if num_arms == 0:
        num_arms = 1  # default to single arm even if not connected

    log.info("Configured: %d arm(s), camera_keys=%s", num_arms, camera_keys)

    # --- Initialize dataset ---
    dataset = _init_dataset(args, num_arms, camera_keys)
    dataset_ref[0] = dataset

    if dataset is not None:
        with shared_state.lock:
            shared_state.total_episodes = dataset.meta.total_episodes
            shared_state.total_frames = dataset.meta.total_frames
            shared_state.current_task = args.task

    # --- Expand shared state camera slots ---
    total_cameras = max(3, len(cameras_rs) + len(cameras_cv))
    with shared_state.lock:
        shared_state.camera_jpegs = [PLACEHOLDER_JPEG] * total_cameras

    # Initialize arms_state and last_arm_readings
    with shared_state.lock:
        shared_state.arms_state = [
            {"id": i, "active": False, "q_actual": [0.0] * 7, "q_cmd": None, "connected": False}
            for i in range(len(teleops))
        ]
        shared_state.last_arm_readings = [(None, None)] * len(readers)

    # --- Start control loop thread ---
    stop_event = threading.Event()
    ctrl_thread = threading.Thread(
        target=control_loop,
        args=(readers, teleops, cameras_rs, cameras_cv, dataset_ref, shared_state, stop_event, dataset_lock, args),
        daemon=True,
        name="control-loop",
    )
    ctrl_thread.start()

    # --- Register shutdown ---
    import atexit
    def _atexit():
        stop_event.set()
        ctrl_thread.join(timeout=3.0)
        _shutdown(dataset_ref[0], readers, teleops, cameras_rs, cameras_cv)

    atexit.register(_atexit)

    # --- Start Flask server ---
    print(f"Server running at http://localhost:{args.web_port}")
    log.info(
        "Dataset: %s  |  execute=%s  |  hz=%.0f",
        args.dataset_path, args.execute, args.hz,
    )

    try:
        app.run(host="0.0.0.0", port=args.web_port, threaded=True)
    finally:
        stop_event.set()
        ctrl_thread.join(timeout=3.0)
        _shutdown(dataset_ref[0], readers, teleops, cameras_rs, cameras_cv)


if __name__ == "__main__":
    main()
