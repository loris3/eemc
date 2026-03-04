import glob
import subprocess

# List of paired models and datasets
experiments = [
    {
        "name": "SFT",
        "model": "./models/Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42",
        "train_dataset_name": "loris3/tulu-3-sft-olmo-2-mixture-0225-sample",
        "train_dataset_split": "train",
        "test_dataset_name": "loris3/tulu-3-sft-olmo-2-mixture-0225-sample",
        "test_dataset_split": "test",
    },
    {
        "name": "ftrace",
        "model": "./models/OLMo-2-0425-1B_ftrace_date-lm_conversational_lr1e-05_seed42",
        "train_dataset_name": "loris3/ftrace_date-lm_conversational",
        "train_dataset_split": "train",
        "test_dataset_name": "loris3/ftrace_date-lm_conversational",
        "test_dataset_split": "train",
    },
]

# Find all run_*.py scripts
scripts_no_gpu = [
"representations/run_bm25.py",
"representations/run_datainf.py",
"representations/run_jaccard.py",
"representations/run_less_gradients.py",
"representations/run_tfidf.py"
]

scripts_gpu = [
    "representations/run_sbert.py",
    "representations/run_repsim.py",
    "representations/run_model_embd.py",
    "representations/run_norm_attn.py",
]
scripts = scripts_no_gpu + scripts_gpu


def main():
    for script_path in scripts:
        for exp in experiments:
            args = [
                "--model", exp["model"],
                "--train_dataset_name", exp["train_dataset_name"],
                "--train_dataset_split", exp["train_dataset_split"],
                "--test_dataset_name", exp["test_dataset_name"],
                "--test_dataset_split", exp["test_dataset_split"]
            ]
            cmd = ["./schedule.sh", "representations/run_template.sbatch" if script_path not in scripts_gpu else "representations/run_template_gpu.sbatch", script_path.replace(".py","").replace("/",".")] + args
            print("Scheduling:", " ".join(cmd))
            subprocess.run(cmd)
if __name__ == "__main__":
    main()