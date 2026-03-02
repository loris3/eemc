#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <job-file.sbatch> [script args...]"
  exit 1
fi

JOB_FILE="$1"
shift  # Remaining arguments will be passed to sbatch as script args

# Load environment
source .env

SBATCH_ARGS=()
SBATCH_ARGS+=(--container-image="$CONTAINER_IMAGE")
SBATCH_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")

if [[ "$JOB_FILE" != *jupyter* ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
else
  SBATCH_ARGS+=(--nodelist="galadriel")
fi

SBATCH_ARGS+=(--container-mount-home)

# Submit job with the script args
sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE" "$@"