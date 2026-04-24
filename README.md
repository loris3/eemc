
0. Copy `.env_example` to `.env`
1. Fine-tune: `./schedule.sh --job-name="[EMMC finetuning]" finetune.sbatch`
2. Benchmark: `./finetuning_eval.sh`
3. Gradient-based influence estimation: `./schedule estimate_influence.sh`
4. Representations: `python3 schedule_representations.py`
5. Similarity: `python3 schedule_similarity.py`