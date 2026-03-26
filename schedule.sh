#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <job-file.sbatch> [--job-name NAME] [script args...]"
  exit 1
fi

JOB_FILE="$1"
shift

JOB_NAME=""  # default empty
SCRIPT_ARGS=()

# Parse optional --job-name flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job-name)
            if [[ -n "$2" ]]; then
                JOB_NAME="$2"
                shift 2
            else
                echo "Error: --job-name requires a value"
                exit 1
            fi
            ;;
        *)
            SCRIPT_ARGS+=("$1")
            shift
            ;;
    esac
done

# Fallback: derive job name from file if not set
if [[ -z "$JOB_NAME" ]]; then
    SAFE_JOB_FILE="${JOB_FILE,,}"             # lowercase
    SAFE_JOB_FILE="${SAFE_JOB_FILE//[\[\]]/}" # remove brackets
    JOB_NAME="$(basename "$SAFE_JOB_FILE" .sbatch)"
fi

# Load environment
source .env

SBATCH_ARGS=()
SBATCH_ARGS+=(--container-image="$CONTAINER_IMAGE")
SBATCH_ARGS+=(--container-workdir="$CONTAINER_WORKDIR")
SBATCH_ARGS+=(--job-name="$JOB_NAME")

# Node selection based on filename
SAFE_JOB_FILE="${JOB_FILE,,}"
SAFE_JOB_FILE="${SAFE_JOB_FILE//[\[\]]/}"
if [[ "$SAFE_JOB_FILE" == *jupyter* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel")
elif [[ "$SAFE_JOB_FILE" == *representations* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel,shelob,dgx-h100-em2,vader")
elif [[ "$SAFE_JOB_FILE" == *similarity* ]]; then
    SBATCH_ARGS+=(--nodelist="galadriel,shelob,dgx-h100-em2")
else
    SBATCH_ARGS+=(--nodelist="$NODELIST")
fi
SBATCH_ARGS+=(--container-mount-home)

# Submit job with remaining script args untouched
sbatch "${SBATCH_ARGS[@]}" "$JOB_FILE" "${SCRIPT_ARGS[@]}"