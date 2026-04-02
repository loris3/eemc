#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4 
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1  
#SBATCH --account=datamining
#SBATCH --partition=p_datamining

# Print environment info
python3 --version
df -h





export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT="eemc_finetuning"

set -e


BATCH_SIZE=8
GRAD_ACC=2


echo Effective batch size $EFFECTIVE_BATCH_SIZE
BASE_MODELS=(
  "allenai/Olmo-3-1025-7B"
  "allenai/OLMo-2-0425-1B"
  "meta-llama/Llama-3.2-1B"
  "Qwen/Qwen2.5-0.5B"
)

DATASETS=(
    "loris3/ftrace_date-lm_conversational"
  # "allenai/tulu-3-sft-personas-math"
  #  "allenai/tulu-3-sft-personas-code"
  #  "loris3/squad_v2_conversational"
  #  "allenai/tulu-3-sft-olmo-2-mixture-0225"

)


NUM_PROCESSES=${SLURM_GPUS_ON_NODE:-1}
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACC * NUM_PROCESSES))

echo "Effective batch size: $EFFECTIVE_BATCH_SIZE"

for BASE_MODEL in "${BASE_MODELS[@]}"; do
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