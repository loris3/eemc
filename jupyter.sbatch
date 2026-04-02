#!/bin/bash
#SBATCH --job-name="[EEMC] Jupyter Lab Server"
#SBATCH --mem=36G
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:0
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --output=jupyter.out
#SBATCH --container-writable


source .env
ulimit -n 65535



python3 --version

df -h
echo "http://localhost:2203/lab?token=$JUPYTER_TOKEN"
python3 -m jupyterlab --NotebookApp.token="$JUPYTER_TOKEN" --port=2203 --allow-root
