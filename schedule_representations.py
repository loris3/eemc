import glob
import subprocess

import yaml
import glob
import subprocess
import os
with open("experiments.yaml", "r") as f:
    cfg = yaml.safe_load(f)

base_models = [b.split('/')[1] for b in cfg["base_models"]]
experiments_ = cfg["experiments"]

experiments = []
for base in base_models:
    for exp in experiments_:
        new_exp = exp.copy()
        new_exp["model"] = f"./models/{base}{exp['model']}"
        if not os.path.exists(new_exp["model"]):
            print(f"[ERROR] Model {new_exp['model']} not found")
            exit(-1)
        experiments.append(new_exp)



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
                "--train_dataset_name", exp["dataset"],
                "--train_dataset_split", "test",
                "--test_dataset_name", exp["dataset"],
                "--test_dataset_split", "test"
            ]

            sbatch_file = "representations/run_template_gpu.sbatch" if script_path in scripts_gpu else "representations/run_template.sbatch"

            # Convert script path to module-style name
            module_name = script_path.replace(".py", "").replace("/", ".")

            job_label = script_path.split("/")[1].split(".")[0].replace("run_", "")
            job_name = f"[EEMC] Representation {job_label} for {exp['model']}"

            
            cmd = ["./schedule.sh", sbatch_file, module_name, "--job-name", job_name] + args
            print("Scheduling:", " ".join(cmd))
            subprocess.run(cmd)
     
if __name__ == "__main__":
    main()