#!/bin/bash
# Launch the Wan voxel-conditioning work on the persist TPU via scripts/tpu/run.py.
# Runs the conditioning smoke test by default; swap --run-command for training (train_wan.py)
# once the data pipeline + config are wired.
set -e
cd "$(dirname "$0")/../../.."   # repo root
. scripts/tpu/set_env_local.sh
python scripts/tpu/run.py \
  --resource-name persist-v5p-8-central1a-2 \
  --gcp-zone us-central1-a \
  --gcp-project nyu-vision-lab \
  --storage_bucket solaris-central1 \
  --git-repo-url git@github.com:gikees/maxdiffusion.git \
  --git-branch egor/wan-conditioning \
  --conda-env-name maxdiffusion \
  --skip-tests \
  --run-command "python scripts/tpu/smoke_voxel_cond.py"
