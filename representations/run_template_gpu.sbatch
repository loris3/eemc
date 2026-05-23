#!/bin/bash
#SBATCH --job-name="[EEMC] Representations"
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-16:00:00
#SBATCH --nodes=1
#SBATCH --partition=p_low

python3 --version
df -h


SCRIPT_PATH="$1"
shift
ARGS="$@"

# Run the Python script
pwd
python3 -m "$SCRIPT_PATH" $ARGS