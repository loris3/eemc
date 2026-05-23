#!/bin/bash
#SBATCH --job-name="[EEMC] Influence Estimation"
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:1
#SBATCH --time=1-7:59:00
#SBATCH --nodes=1
#SBATCH --partition=p_datamining
#SBATCH --account=datamining
source .env
python3 --version

df -h
ulimit -n 65535
pip show datasets


readarray -t MODEL_IE_CONFIGS < <(
python3 - << 'EOF'
import yaml
with open("experiments.yaml") as f:
    cfg = yaml.safe_load(f)
for base in cfg["base_models"]:
    for exp in cfg["experiments"]:
        base_model = base.split("/")[1]
        model_dir = f"./models/{base_model}{exp['model']}"
        dataset = exp["dataset"]       
        print(f"{base_model} {model_dir} {dataset}")
EOF
)
echo "Estimating for models:"
for cfg in "${MODEL_IE_CONFIGS[@]}"; do
  echo "$cfg"
done

for CONFIG in "${MODEL_IE_CONFIGS[@]}"; do
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

  # Clear ramdisk after each run
  rm -rf /tmp/cache/gradients
done