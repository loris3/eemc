#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <job-file.sbatch>"
  exit 1
fi

JOB_FILE="$1"




source .env
SBATCH_ARGS=()

SBATCH_ARGS+=(--container-image="$CONTAINER_IMAGE")
SBATCH_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")
if [[ "$JOB_FILE" != *jupyter* ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
else
  SBATCH_ARGS+=(--nodelist="dgx1")
fi
SBATCH_ARGS+=(--container-mount-home)

sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE"
