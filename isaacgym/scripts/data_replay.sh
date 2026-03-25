#!/bin/sh
set -e

export ISAACGYM_ROOT=/home/namhj/Isaac/IsaacGym
export PYTHONPATH="$ISAACGYM_ROOT/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ISAACGYM_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

SEQ_NAME="sub2_largetable_000"
EXP_DIR="/home/namhj/InterMimic/exp/${SEQ_NAME}"
MOTION_FILE="/home/namhj/InterMimic/InterAct/OMOMO/${SEQ_NAME}.pt"

xvfb-run -a \
  python -m intermimic.data_replay_video \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_test.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --num_envs 16 \
    --env_id 0 \
    --exp_dir "${EXP_DIR}" \
    --motion_file "$MOTION_FILE"