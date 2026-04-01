#!/bin/bash
#SBATCH --job-name="[EEMC] Influence Estimation"
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64  
#SBATCH --gres=gpu:4
#SBATCH --time=1-7:59:00
#SBATCH --nodes=1
#SBATCH --account=datamining
#SBATCH --partition=p_datamining
source .env
python3 --version

df -h
ulimit -n 65535
pip show datasets



python3 estimate_influence.py \
  --base_model "allenai/OLMo-2-0425-1B" \
  --model_dir "./models/OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42" \
  --train_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample \
  --test_dataset loris3/tulu-3-sft-olmo-2-mixture-0225-sample \
  --train_dataset_split=test \
  --test_dataset_split=test
rm -rf /tmp/cache/gradients


python3 estimate_influence.py \
  --base_model "allenai/OLMo-2-0425-1B" \
  --model_dir "./models/OLMo-2-0425-1B_ftrace_date-lm_conversational_lr1e-05_seed42" \
  --train_dataset loris3/ftrace_date-lm_conversational-sample \
  --test_dataset loris3/ftrace_date-lm_conversational-sample \
  --train_dataset_split=test \
  --test_dataset_split=test
rm -rf /tmp/cache/gradients



python3 estimate_influence.py \
  --base_model "allenai/OLMo-2-0425-1B" \
  --model_dir "./models/OLMo-2-0425-1B_tulu-3-sft-personas-code_lr1e-05_seed42" \
  --train_dataset loris3/tulu-3-sft-personas-code-sample \
  --test_dataset loris3/tulu-3-sft-personas-code-sample \
  --train_dataset_split=test \
  --test_dataset_split=test
rm -rf /tmp/cache/gradients

python3 estimate_influence.py \
  --base_model "allenai/OLMo-2-0425-1B" \
  --model_dir "./models/OLMo-2-0425-1B_tulu-3-sft-personas-math_lr1e-05_seed42" \
  --train_dataset loris3/tulu-3-sft-personas-math-sample \
  --test_dataset loris3/tulu-3-sft-personas-math-sample \
  --train_dataset_split=test \
  --test_dataset_split=test
rm -rf /tmp/cache/gradients
python3 estimate_influence.py \
  --base_model "allenai/OLMo-2-0425-1B" \
  --model_dir "./models/OLMo-2-0425-1B_squad_v2_conversational_lr1e-05_seed42" \
  --train_dataset loris3/squad_v2_conversational-sample \
  --test_dataset loris3/squad_v2_conversational-sample \
  --train_dataset_split=test \
  --test_dataset_split=test
rm -rf /tmp/cache/gradients
