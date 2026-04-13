#!/bin/bash
#SBATCH --job-name="[EEMC] Influence Estimation"
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-7:59:00
#SBATCH --nodes=1

source .env
python3 --version

df -h
ulimit -n 65535
pip show datasets


CONFIGS=(
  "Qwen3-0.6-Base ./models/Qwen3-0.6B-Base_ftrace_date-lm_conversational_lr0.0001_seed42 loris3/ftrace_date-lm_conversational-sample" 
  "Qwen3-0.6-Base ./models/Qwen3-0.6B-Base_tulu-3-sft-personas-math_lr0.0001_seed42 loris3/tulu-3-sft-personas-math-sample" 
  "Qwen3-0.6-Base ./models/Qwen3-0.6B-Base_tulu-3-sft-personas-code_lr0.0001_seed42 loris3/tulu-3-sft-personas-code-sample" 
  "Qwen3-0.6-Base ./models/Qwen3-0.6B-Base_squad_v2_conversational_lr0.0001_seed42 loris3/squad_v2_conversational-sample" 
  "Qwen3-0.6-Base ./models/Qwen3-0.6B-Base_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42 loris3/tulu-3-sft-olmo-2-mixture-0225-sample" 
)

# Loop over triplets
for CONFIG in "${CONFIGS[@]}"; do
  BASE_MODEL=$(echo $CONFIG | awk '{print $1}')
  MODEL_DIR=$(echo $CONFIG | awk '{print $2}')
  DATASET=$(echo $CONFIG | awk '{print $3}')

  echo "Estimating influence for base_model=$BASE_MODEL model_dir=$MODEL_DIR dataset=$DATASET"

  python3 estimate_influence.py \
    --base_model "$BASE_MODEL" \
    --model_dir "$MODEL_DIR" \
    --train_dataset "$DATASET" \
    --test_dataset "$DATASET" \
    --train_dataset_split=test \
    --test_dataset_split=test

  # Clear cache after each run
  rm -rf /tmp/cache/gradients
done