import glob
import subprocess
import itertools

from schedule_representations import experiments, scripts_gpu, scripts_no_gpu

scripts = scripts_no_gpu + scripts_gpu

# scripts = [s for s in scripts if ("data" not in s) or ("less" in s)]

print(list(itertools.combinations(scripts, 2)))

for exp in experiments:
    for script_a, script_b in itertools.combinations(scripts, 2):
        args = [
            "--model", exp["model"],
            "--train_dataset_name", exp["train_dataset_name"],
            "--train_dataset_split", exp["train_dataset_split"],
            "--test_dataset_name", exp["test_dataset_name"],
            "--test_dataset_split", exp["test_dataset_split"],
            "--top_k", "50 100 200 500", #"10 20",
            "--sim_script_a", script_a,
            "--sim_script_b", script_b,
            "--experiment_name", exp["name"],
        ]
        cmd = ["./schedule.sh", "run_similarity_template.sbatch", "structural_similarity"] + args
        print("\n", " ".join(cmd))
        subprocess.run(cmd)
        
