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
        "--save_images",
        "--num_envs",
        str(args.num_envs),
    ]

    print("[data_replay_video] Starting replay subprocess ...", flush=True)
    child_env = os.environ.copy()
    proc = subprocess.Popen(cmd, stdout=None, stderr=None, env=child_env)

    start = time.time()
    try:
        selected_dir = ""
        while True:
            frames = glob.glob(frame_glob, recursive=True)
            if frames:
                # Group frames by directory to avoid mixing multiple sequences.
                counts_by_dir = {}
                for p in frames:
                    d = os.path.dirname(p)
                    counts_by_dir[d] = counts_by_dir.get(d, 0) + 1
                selected_dir = max(counts_by_dir.keys(), key=lambda k: counts_by_dir[k])
                if counts_by_dir[selected_dir] >= args.min_frames:
                    print(
                        "[data_replay_video] Selected directory for video encoding:",
                        selected_dir,
                        "frames:",
                        counts_by_dir[selected_dir],
                        flush=True,
                    )
                    break
            if (time.time() - start) >= args.frame_timeout_sec:
                print(
                    f"[data_replay_video] Timeout after {args.frame_timeout_sec}s with {len(frames)} frames.",
                    flush=True,
                )
                break
            time.sleep(2.0)
    finally:
        _kill_process_tree(proc)

    # After stopping, ensure we have a selected directory with frames.
    if not selected_dir:
        frames = sorted(glob.glob(frame_glob, recursive=True))
        if not frames:
            raise RuntimeError(f"[data_replay_video] ERROR: No frames found: {frame_glob}")
        selected_dir = os.path.dirname(max(frames, key=lambda p: os.path.getmtime(p)))

    image_dir = selected_dir
    print(f"[data_replay_video] Encoding video from directory: {image_dir}", flush=True)

    # Restrict to frames from the chosen directory.
    image_dir_frames = sorted(glob.glob(os.path.join(image_dir, f"rgb_env{args.env_id}_frame*.png")))
    if not image_dir_frames:
        raise RuntimeError(f"[data_replay_video] ERROR: No frames found in picked directory: {image_dir}")

    # Move frames into exp/{exp_dir} for user-controlled output location.
    if exp_dir:
        dest_dir = os.path.join(str(exp_root), exp_dir, "images")
    else:
        # If exp_dir isn't provided, keep frames where they are.
        dest_dir = image_dir

    print(
        "[data_replay_video] Moving frames to:",
        dest_dir,
        "from:",
        image_dir,
        flush=True,
    )

    os.makedirs(dest_dir, exist_ok=True)
    if dest_dir != image_dir:
        if args.clear_old:
            # Remove any previous frames in the destination folder so ffmpeg
            # doesn't encode stale images.
            for p in glob.glob(os.path.join(dest_dir, f"rgb_env{args.env_id}_frame*.png")):
                try:
                    os.remove(p)
                except Exception:
                    pass
        for src in image_dir_frames:
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.move(src, dst)
        image_dir = dest_dir

    # Cleanup any leftover frames from legacy `images/` so output only remains under exp/{exp_dir}.
    leftover = glob.glob(images_frame_glob, recursive=True)
    if leftover:
        print(f"[data_replay_video] Cleaning leftover frames under legacy images/: {len(leftover)}", flush=True)
        for p in leftover:
            try:
                os.remove(p)
            except Exception:
                pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Store the video alongside the frames directory.
    out_video = args.out_video or os.path.join(image_dir, f"data_replay_{timestamp}.mp4")

    # Encode with ffmpeg using glob input.
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(args.fps),
        "-pattern_type",
        "glob",
        "-i",
        os.path.join(image_dir, f"rgb_env{args.env_id}_frame*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_video,
    ]
    # Use stdout/stderr directly (not captured) so ffmpeg errors are visible.
    print("[data_replay_video] Running ffmpeg ...", flush=True)
    completed = subprocess.run(ffmpeg_cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"[data_replay_video] ffmpeg failed with code {completed.returncode}")

    print(f"[data_replay_video] Video saved to: {out_video}", flush=True)


if __name__ == "__main__":
    main()

