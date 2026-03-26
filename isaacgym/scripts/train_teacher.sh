#!/bin/sh
set -e

export CUDA_VISIBLE_DEVICES=4
export ISAACGYM_ROOT=/home/namhj/Isaac/IsaacGym
export PYTHONPATH="$ISAACGYM_ROOT/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ISAACGYM_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

SEQ_NAME="sub2-iris"
EXP_DIR="/home/namhj/InterMimic/exp/${SEQ_NAME}"
ENV_CONFIG_FILE="configs/env_small.yaml"
TRAIN_CONFIG_FILE="configs/train_small.yaml"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env "${ENV_CONFIG_FILE}" \
    --cfg_train "${TRAIN_CONFIG_FILE}" \
    --exp_dir "${EXP_DIR}" \
    --output "${EXP_DIR}/checkpoints" \
    --headless \
    
