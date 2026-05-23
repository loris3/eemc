#!/bin/bash
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint="nvidia-h100|nvidia-h200"
#SBATCH --time=2-16:00:00
#SBATCH --nodes=1  
#SBATCH --container-writable 
#SBATCH --partition=p_datamining
#SBATCH --account=datamining

python3 --version
df -h





export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="eemc_finetuning"

set -e

# pip freeze



BASE_MODELS=(
  # "Qwen/Qwen3-0.6B-Base 8"
  # "Qwen/Qwen3-1.7B-Base 8"
  #  "Qwen/Qwen3-4B-Base 4"
  "Qwen/Qwen3-8B-Base 4"
  # "allenai/Olmo-3-1025-7B 4" 
  # "allenai/OLMo-2-0425-1B 8"
  # "meta-llama/Llama-3.2-1B 8"
)
  # "allenai/OLMo-2-1124-13B 2"
DATASETS=(
   "loris3/ftrace_date-lm_conversational"
  #  "loris3/squad_v2_conversational" 
   "allenai/tulu-3-sft-personas-math"
   "allenai/tulu-3-sft-personas-code"
  #  "allenai/tulu-3-sft-olmo-2-mixture-0225"
)

NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}
TARGET_EFF_BS=64

for ENTRY in "${BASE_MODELS[@]}"; do
  read -r BASE_MODEL BATCH_SIZE <<< "$ENTRY"

  TOTAL_PER_STEP=$(( BATCH_SIZE * NUM_PROCESSES ))
  GRAD_ACC=$(( (TARGET_EFF_BS + TOTAL_PER_STEP - 1) / TOTAL_PER_STEP ))

  EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACC * NUM_PROCESSES))

  if [[ "$EFFECTIVE_BATCH_SIZE" -ne "$TARGET_EFF_BS" ]]; then
    echo "ERROR: batch size mismatch" >&2
    echo "  computed: $EFFECTIVE_BATCH_SIZE" >&2
    echo "  target:   $TARGET_EFF_BS" >&2
    exit 1
  fi
  echo "Model: $BASE_MODEL"
  echo "Per-device batch size: $BATCH_SIZE"
  echo "Gradient accumulation steps: $GRAD_ACC"
  echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"

  for DATASET in "${DATASETS[@]}"; do
    echo "Fine-tuning base_model=$BASE_MODEL dataset=$DATASET"

    python3 -m accelerate.commands.launch finetune.py \
      --base_model "$BASE_MODEL" \
      --train_dataset "$DATASET" \
      --train_split "train" \
      --learning_rate "1e-4" \
      --seed "42" \
      --lora_r "8" \
      --num_train_epochs "1" \
      --per_device_train_batch_size "$BATCH_SIZE" \
      --gradient_accumulation_steps "$GRAD_ACC" \
      --logging_steps "50" \
      --save_total_limit "2" \
      --bf16 "true" \
      --report_to "wandb"


  done
done