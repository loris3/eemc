#!/bin/bash
#SBATCH --job-name="[EEMC] Finetuning Sweep"
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=1-16:00:00
#SBATCH --nodes=1  
#SBATCH --account=datamining
#SBATCH --partition=p_datamining

# Print environment info
python3 --version
df -h





export TOKENIZERS_PARALLELISM=true


# first run: 
# python3 -m wandb sweep --project=eemc_finetuning sweep.yaml
#  and set the sweep ID below:
python3 -m  wandb agent loriss/eemc_finetuning/aubdfouh
