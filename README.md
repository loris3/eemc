Repository for the paper *On Representational and Functional Alignment of Training Data Attribution Methods for Language Models*.

See `./out/` for notebooks.

All models and data are made available via the Hugging Face Hub in [this collection](https://hf.co/collections/loris3/on-rep-and-func-alignment-of-tda-methods-for-lms).
### Building Docker Images
1. Either use the image provided in this repository or build dockefiles in `./docker` and push them to a Docker registry
3. Set `CONTAINER_IMAGE` and `CONTAINER_IMAGE_EVAL` in the `.env` file accordingly

### Running Experiments
0. Copy `.env_example` to `.env`
1. Fine-tune: `./schedule.sh --job-name="[EMMC finetuning]" finetune.sbatch`
2. Benchmark: `./finetuning_eval.sh`
3. Gradient-based influence estimation: `./schedule estimate_influence.sh`
4. Representations: `python3 schedule_representations.py`
5. Similarity scoring: `python3 schedule_similarity.py`
6. Cross-method predictability scoring: `python3 schedule_predictability.py`
7. On completion: `./schedule.sh merge_results.sbatch`
8. Finally: `./schedule.sh run_notebooks.sbatch` to generate figures
