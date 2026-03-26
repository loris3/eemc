#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --partition=p_low
#SBATCH 
python3 --version
df -h

echo "success"
