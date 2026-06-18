#!/bin/bash
# (Re)create the persist conditioning TPU — matches the one we run on:
#   persist-v5p-8-central1a-2 : single-host v5p-8, us-central1-a, project nyu-vision-lab, spot.
# Name / zone / project are overridable as positional args (defaults match our TPU).
#   bash create_tpu.sh [name] [zone] [project]
set -e

TPU_NAME="${1:-persist-v5p-8-central1a-2}"
ZONE="${2:-us-central1-a}"
PROJECT="${3:-nyu-vision-lab}"
ACCELERATOR_TYPE="v5p-8"
RUNTIME_VERSION="v2-alpha-tpuv5"

echo "Creating queued TPU resource '$TPU_NAME' ($ACCELERATOR_TYPE / $RUNTIME_VERSION, spot) in $ZONE / $PROJECT ..."
gcloud compute tpus queued-resources create "$TPU_NAME" \
  --node-id "$TPU_NAME" \
  --project "$PROJECT" \
  --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" \
  --runtime-version "$RUNTIME_VERSION" \
  --spot

echo "Queued. Watch status with:"
echo "  gcloud compute tpus queued-resources describe $TPU_NAME --zone=$ZONE --project=$PROJECT"
echo "  gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT | grep $TPU_NAME"
