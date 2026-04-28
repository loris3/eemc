#!/bin/bash
#SBATCH --job-name="[EEMC] Similarity"
#SBATCH --cpus-per-task=8
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH --account=datamining
python3 --version
df -h


# Script path is the first argument
SCRIPT_PATH="$1"
shift
ARGS="$@"

# Run the Python script
pwd
python3 -m "$SCRIPT_PATH" $ARGS