### Building Docker Images
1. Build dockefiles in `./docker` and push them to a Docker registry
3. Set `CONTAINER_IMAGE` and `CONTAINER_IMAGE_EVAL` in the `.env` file accordingly

### Running Experiments
0. Copy `.env_example` to `.env`
1. Fine-tune: `./schedule.sh --job-name="[EMMC finetuning]" finetune.sbatch`
2. Benchmark: `./finetuning_eval.sh`
3. Gradient-based influence estimation: `./schedule estimate_influence.sh`
4. Representations: `python3 schedule_representations.py`
5. Similarity: `python3 schedule_similarity.py`
6. Cross-method predictability: `python3 schedule_predictability.py`
7. On completion: `./schedule.sh merge_results.sbatch`
8. Finally: `./schedule.sh run_notebooks.sbatch` to generate figures