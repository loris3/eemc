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

if [[ "$JOB_FILE" == *jupyter* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel")
elif [[ "$JOB_FILE" == *Representations* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel,shelob,dgx-h100-em2")
elif [[ "$JOB_FILE" == *Similarity* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel,shelob,dgx-h100-em2")
else
    SBATCH_ARGS+=(--nodelist="$NODELIST")
fi
SBATCH_ARGS+=(--container-mount-home)

# Submit job with the script args
sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE" "$@"