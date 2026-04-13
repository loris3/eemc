#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --constraint="nvidia-h100|nvidia-h200"
#SBATCH --time=1-16:00:00
#SBATCH --nodes=1  


# Print environment info
python3 --version
df -h





export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="eemc_finetuning"

set -e


BATCH_SIZE_32B=8
GRAD_ACC_32B=2

BATCH_SIZE_13B=8
GRAD_ACC_13B=2



BASE_MODELS=(
  # "Qwen/Qwen3-0.6B-Base"
  # "Qwen/Qwen3-1.7B-Base"
  # # "Qwen/Qwen3-4B-Base"
  # # "Qwen/Qwen3-8B-Base"
  # "allenai/OLMo-2-1124-13B"
  # "allenai/Olmo-3-1025-7B"
  # "allenai/OLMo-2-0425-1B"
  # "meta-llama/Llama-3.2-1B"
)

DATASETS=(
    "loris3/ftrace_date-lm_conversational"
  # "allenai/tulu-3-sft-personas-math"
  #  "allenai/tulu-3-sft-personas-code"
  #  "loris3/squad_v2_conversational"
  #  "allenai/tulu-3-sft-olmo-2-mixture-0225"

)



NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}


for BASE_MODEL in "${BASE_MODELS[@]}"; do
  if [[ "$BASE_MODEL" == *"32B"* ]]; then
      BATCH_SIZE=$BATCH_SIZE_32B
      GRAD_ACC=$GRAD_ACC_32B

  elif [[ "$BASE_MODEL" == *"13B"* ]]; then
      BATCH_SIZE=$BATCH_SIZE_13B
      GRAD_ACC=$GRAD_ACC_13B
  else
      echo "Unknown model size in $BASE_MODEL, using safe defaults"
      BATCH_SIZE=8
      GRAD_ACC=2
  fi

    EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACC * NUM_PROCESSES))
    echo "Base model: $BASE_MODEL"
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
      --report_to "wandb"\
      --fsdp "full_shard auto_wrap"


  done
done