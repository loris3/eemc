
0. Copy `.env_example` to `.env`
1. Fine-tune: `./schedule.sh --job-name="[EMMC finetuning]" finetune.sbatch`
2. Benchmark: `./finetuning_eval.sh`