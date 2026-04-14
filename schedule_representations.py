import glob
import subprocess

base_models = [
    # "Olmo-3-1025-7B",
    # "OLMo-2-0425-1B",
    # "Llama-3.2-1B",
    # "Qwen2.5-0.5B"
    "Qwen3-0.6B-Base"
]


experiments_ = [
    {
        "name": "SFT",
        "model": "_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42",
        "train_dataset_name": "loris3/tulu-3-sft-olmo-2-mixture-0225-sample",
        "train_dataset_split": "test",
        "test_dataset_name": "loris3/tulu-3-sft-olmo-2-mixture-0225-sample",
        "test_dataset_split": "test",
    },
    {
        "name": "ftrace",
        "model": "_ftrace_date-lm_conversational_lr0.0001_seed42",
        "train_dataset_name": "loris3/ftrace_date-lm_conversational-sample",
        "train_dataset_split": "test",
        "test_dataset_name": "loris3/ftrace_date-lm_conversational-sample",
        "test_dataset_split": "test",
    },
        {
        "name": "math",
        "model": "_tulu-3-sft-personas-math_lr0.0001_seed42",
        "train_dataset_name": "loris3/tulu-3-sft-personas-math-sample",
        "train_dataset_split": "test",
        "test_dataset_name": "loris3/tulu-3-sft-personas-math-sample",
        "test_dataset_split": "test",
    },
    {
        "name": "code",
        "model": "_tulu-3-sft-personas-code_lr0.0001_seed42",
        "train_dataset_name": "loris3/tulu-3-sft-personas-code-sample",
        "train_dataset_split": "test",
        "test_dataset_name": "loris3/tulu-3-sft-personas-code-sample",
        "test_dataset_split": "test",
    },
            {
        "name": "squad_v2",
        "model": "_squad_v2_conversational_lr0.0001_seed42",
        "train_dataset_name": "loris3/squad_v2_conversational-sample",
        "train_dataset_split": "test",
        "test_dataset_name": "loris3/squad_v2_conversational-sample",
        "test_dataset_split": "test",
    },

]
experiments = []
for base in base_models:
    for exp in experiments_:
        new_exp = exp.copy()
        new_exp["model"] = f"./models/{base}{exp['model']}"
        experiments.append(new_exp)
        
print(experiments)
# exit()

# Find all run_*.py scripts
scripts_no_gpu = [
    "representations/run_gradsim_normalized.py",
    "representations/run_gradsim_unnormalized.py",
    "representations/run_gradsim_adam.py",
    "representations/run_datainf.py",
    "representations/run_bm25.py",
    "representations/run_jaccard.py",
    "representations/run_tfidf.py"
]

scripts_gpu = [
    "representations/run_embd_Qwen3-Embedding-8B.py",
    "representations/run_embd_llama-embed-nemotron-8b.py",
    "representations/run_repsim.py",
    "representations/run_full_model.py",
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
           # Determine which sbatch script to use
            sbatch_file = "representations/run_template_gpu.sbatch" if script_path in scripts_gpu else "representations/run_template.sbatch"

            # Convert script path to module-style name
            module_name = script_path.replace(".py", "").replace("/", ".")

            # Build a readable job name
            job_label = script_path.split("/")[1].split(".")[0].replace("run_", "")
            job_name = f"[EEMC] Representation {job_label} for {exp['model']}"

            # Combine all parts into the command
            cmd = ["./schedule.sh", sbatch_file, module_name, "--job-name", job_name] + args
            print("Scheduling:", " ".join(cmd))
            subprocess.run(cmd)
     
if __name__ == "__main__":
    main()