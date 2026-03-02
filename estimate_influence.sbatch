#!/bin/bash
#SBATCH --job-name="[EEMC] Influence Estimation"
#SBATCH --array=0-0%1
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:1
#SBATCH --time=1-7:59:00
#SBATCH --nodes=1
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
source .env
python3 --version

df -h
ulimit -n 65535
pip show datasets

MODELS=(
  "allenai/OLMo-2-0425-1B ./models/OLMo-2-0425-1B_ftrace_date-lm_conversational_lr1e-05_seed42"
)

# Extract model + directory based on task ID
TASK="${MODELS[$SLURM_ARRAY_TASK_ID]}"
BASE_MODEL=$(echo $TASK | awk '{print $1}')
MODEL_DIR=$(echo $TASK | awk '{print $2}')

echo "Running task $SLURM_ARRAY_TASK_ID with base_model=$BASE_MODEL, model_dir=$MODEL_DIR"

python3 estimate_influence.py \
  --base_model "$BASE_MODEL" \
  --model_dir "$MODEL_DIR" \
  --train_dataset loris3/ftrace_date-lm_conversational \
  --test_dataset loris3/ftrace_date-lm_conversational \
  --train_dataset_split=train \
  --test_dataset_split=train

rm -rf /tmp/cache/gradients
