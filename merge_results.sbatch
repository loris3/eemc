#!/bin/bash
#SBATCH --job-name="[EEMC] Merging Parquet Files"
#SBATCH --mem=64G
#SBATCH --cpus-per-task=64  
#SBATCH --time=0-01:59:00
#SBATCH --nodes=1
python3 --version
source .env
df -h
ulimit -n 65535

echo "Start"
python3 merge_results.py --source_dir="results/similarity" --target_dir="results/similarity_merged"
echo "Done!"