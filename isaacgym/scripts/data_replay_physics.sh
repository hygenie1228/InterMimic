#!/bin/sh
set -e

export CUDA_VISIBLE_DEVICES=1
export ISAACGYM_ROOT=/home/namhj/Isaac/IsaacGym
export PYTHONPATH="$ISAACGYM_ROOT/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ISAACGYM_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

SEQ_NAME="sub2_largetable_000"
MOTION_FILE="/home/namhj/InterMimic/InterAct/OMOMO_new/${SEQ_NAME}.pt"
EXP_DIR="/home/namhj/InterMimic/exp/${SEQ_NAME}"
OBJECT_MODE="${OBJECT_MODE:-dynamic}"      # kinematic | dynamic
ROOT_TRACK_MODE="${ROOT_TRACK_MODE:-hard}" # hard | off

xvfb-run -a \
  python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_test.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --test \
    --play_dataset \
    --play_dataset_physics \
    --object_mode "${OBJECT_MODE}" \
    --root_track_mode "${ROOT_TRACK_MODE}" \
    --num_envs 1 \
    --exp_dir "${EXP_DIR}" \
    --motion_file "${MOTION_FILE}" \
    --visualize \
    --disable_dataset_contact_overlay \
    --num_episode 1
