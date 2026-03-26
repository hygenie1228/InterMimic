import argparse
import glob
import os
import signal
import subprocess
import sys
import time
import shutil
from datetime import datetime
from typing import List

import torch

from intermimic.utils.path_utils import resolve_data_path, resolve_repo_path


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v.strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v.strip() == "":
        return default
    return float(v)


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    # Print the command for easier debugging.
    print("[data_replay_video] Running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=20)
        return
    except Exception:
        pass
    try:
        proc.send_signal(signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _pick_image_dir(frames: List[str]) -> str:
    # Choose the directory containing the newest frame.
    newest = max(frames, key=lambda p: os.path.getmtime(p))
    return os.path.dirname(newest)


def _normalize_exp_dir(exp_dir: str) -> str:
    # Allow caller to pass "exp/foo" or "/foo/".
    exp_dir = exp_dir.strip().strip("/")
    # If an absolute path like ".../exp/debug" is provided, keep only "debug".
    if "/exp/" in exp_dir:
        exp_dir = exp_dir.split("/exp/", 1)[1]
    if exp_dir.startswith("exp/"):
        exp_dir = exp_dir[len("exp/") :]
    return exp_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Render InterMimic dataset frames and encode to video.")
    parser.add_argument("--task", default="InterMimic")
    parser.add_argument(
        "--cfg_env",
        default="isaacgym/src/intermimic/data/cfg/omomo_test.yaml",
        help="Path (relative to repo root or absolute) for env cfg.",
    )
    parser.add_argument(
        "--cfg_train",
        default="isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml",
        help="Path (relative to repo root or absolute) for train cfg.",
    )
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--env_id", type=int, default=int(os.environ.get("ENV_ID", 0)))

    parser.add_argument(
        "--motion_file",
        default="",
        help="Optional single motion .pt file path (relative to repo root or absolute). When set, only this motion is replayed.",
    )
    parser.add_argument("--min_frames", type=int, default=_env_int("MIN_FRAMES", 300))
    parser.add_argument("--frame_timeout_sec", type=int, default=_env_int("FRAME_TIMEOUT_SEC", 180))
    parser.add_argument("--fps", type=float, default=_env_float("FPS", 30.0))
    parser.add_argument("--clear_old", action="store_true", default=os.environ.get("CLEAR_OLD", "1") != "0")
    parser.add_argument(
        "--exp_dir",
        default=os.environ.get("EXP_DIR", ""),
        help="Save frames/videos under repo-root exp/{exp_dir} (e.g., /home/.../InterMimic/exp/{exp_dir}).",
    )
    parser.add_argument("--out_video", default="", help="Optional explicit output mp4 path.")

    args = parser.parse_args()

    expected_motion_frames = None
    if args.motion_file.strip():
        motion_path = resolve_repo_path(args.motion_file.strip(), must_exist=True)
        # Ensure downstream code (task) sees the correct absolute path.
        args.motion_file = str(motion_path)
        motion_obj = torch.load(str(motion_path), map_location="cpu")
        # InterMimic expects the motion file to be a tensor-like object where shape[0] is #frames.
        if hasattr(motion_obj, "shape") and motion_obj.shape:
            expected_motion_frames = int(motion_obj.shape[0])
        else:
            raise RuntimeError(f"[data_replay_video] Unsupported motion file contents: {motion_path}")

        if expected_motion_frames <= 0:
            raise RuntimeError(f"[data_replay_video] Invalid motion frame count: {expected_motion_frames} for {motion_path}")

        # When a single motion is requested, always render until its last frame.
        args.min_frames = expected_motion_frames
        # Add extra slack for simulator startup, saving, etc.
        args.frame_timeout_sec = max(args.frame_timeout_sec, int(expected_motion_frames / max(args.fps, 1e-6) + 60))
        print(
            f"[data_replay_video] Single-motion mode enabled: {motion_path} frames={expected_motion_frames}",
            flush=True,
        )

    # Save output under repo-root `exp/` (e.g., /home/.../InterMimic/exp/debug).
    exp_root = resolve_repo_path("exp", must_exist=False)
    exp_dir = _normalize_exp_dir(args.exp_dir.strip())

    # Frames are produced by the task under either:
    #   exp/<exp_dir>/images/rgb_env{env_id}_frame*.png  (when EXP_DIR is set)
    # or:
    #   images/<dataname>/rgb_env{env_id}_frame*.png     (legacy fallback)
    images_root = resolve_data_path("images", must_exist=False)
    images_frame_glob = os.path.join(str(images_root), "**", f"rgb_env{args.env_id}_frame*.png")

    exp_images_dir = None
    exp_frame_glob = ""
    if exp_dir:
        exp_images_dir = exp_root / exp_dir / "images"
        exp_frame_glob = os.path.join(str(exp_images_dir), f"rgb_env{args.env_id}_frame*.png")

    frame_glob = exp_frame_glob if exp_frame_glob else images_frame_glob

    if args.clear_old:
        globs_to_clear = [images_frame_glob]
        if exp_frame_glob:
            globs_to_clear.append(exp_frame_glob)
        for g in globs_to_clear:
            old_frames = glob.glob(g, recursive=True)
            if old_frames:
                print(f"[data_replay_video] Clearing {len(old_frames)} old frames under: {g}", flush=True)
                for p in old_frames:
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    # Start the replay process (it runs an infinite loop in --play_dataset mode).
    cmd = [
        sys.executable,
        "-m",
        "intermimic.run",
        "--task",
        args.task,
        "--cfg_env",
        args.cfg_env,
        "--cfg_train",
        args.cfg_train,
        "--test",
        "--play_dataset",
        "--visualize",
        "--num_envs",
        str(args.num_envs),
    ]
    if args.motion_file.strip():
        cmd += ["--motion_file", args.motion_file.strip()]
        cmd += ["--num_episode", "1"]

    print("[data_replay_video] Starting replay subprocess ...", flush=True)
    child_env = os.environ.copy()
    if args.exp_dir.strip():
        # `intermimic.py` (task render) uses EXP_DIR env var to decide where to save frames.
        child_env["EXP_DIR"] = args.exp_dir.strip()
    result = subprocess.run(cmd, env=child_env)
    if result.returncode != 0:
        raise RuntimeError(f"[data_replay_video] Replay subprocess failed with code {result.returncode}")


if __name__ == "__main__":
    main()

