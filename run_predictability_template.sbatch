#!/bin/bash
#SBATCH --job-name="[EEMC] Predictability"
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:20:00
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH --account=datamining
#SBATCH --requeue 
python3 --version
df -h



SCRIPT_PATH="$1"
shift
ARGS="$@"

pwd
python3 -m "$SCRIPT_PATH" $ARGS