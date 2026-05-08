#!/bin/bash
#SBATCH --job-name="[CEELM] Case Study Olmo3"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:0
#SBATCH --time=0-4:00:00


mkdir -p ./out


source .env
ulimit -n 65535



python3 --version
df -h


python3.10 -m jupyter nbconvert \
  --to notebook \
  --execute "case_study.ipynb" \
  --output "./out/case_study.ipynb"
