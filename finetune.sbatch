#!/bin/bash
#SBATCH --job-name="[EEMC] Finetuning Sweep"
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1  
#SBATCH --account=datamining
#SBATCH --partition=p_datamining

# Print environment info
python3 --version
df -h





export TOKENIZERS_PARALLELISM=true


# first run: 
# python3 -m wandb sweep --project=eemc_finetuning sweep_sft.yaml

#  and set the sweep ID below:
python3 -m  wandb agent loriss/eemc_finetuning/n4li6o48
