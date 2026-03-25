#!/bin/sh
set -e

export ISAACGYM_ROOT=/home/namhj/Isaac/IsaacGym
export PYTHONPATH="$ISAACGYM_ROOT/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$ISAACGYM_ROOT/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"


SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --headless \
    --output checkpoints
