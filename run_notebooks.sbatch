#!/bin/bash
#SBATCH --job-name="[CEELM] Notebooks"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --time=0-01:00:00
#SBATCH --array=0-6

mkdir -p ./out


source .env
ulimit -n 65535



python3 --version
df -h

NOTEBOOKS=(
  "experiment_1.ipynb"
  "experiment_2.ipynb"
  "experiment_3.ipynb"
  "finetuning_eval.ipynb"
  "heatmaps.ipynb"
  "qualitative_analysis.ipynb"
  "figure_1.ipynb"
)

NB=${NOTEBOOKS[$SLURM_ARRAY_TASK_ID]}
OUT="./out/${NB%.ipynb}.ipynb"

echo "Running notebook: $NB"
echo "Output notebook: $OUT"


python3.10 -m jupyter nbconvert \
  --to notebook \
  --execute "$NB" \
  --output "$OUT"
