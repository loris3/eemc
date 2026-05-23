import subprocess
import itertools
from schedule_representations import experiments, scripts_gpu, scripts_no_gpu
import math
import os

def get_dir_size_gb(path: str) -> float:
    """Returns directory size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)


def get_ram(exp, representation_name, cache_root="./cache/representations"):
    """
    Estimate RAM requirement (in GB) based on cached tensor directory size.

    Args:
        exp: experiment dict
        representation_name: e.g. 'datainf', 'gradsim_normalized', etc.
        cache_root: base cache dir
        safety_factor: multiplier to avoid OOM

    Returns:
        int: RAM in GB (rounded up)
    """
    cache_dir = os.path.join(
        cache_root,
        representation_name,
        os.path.basename(exp["dataset"]),
        "test",
        os.path.basename(exp["dataset"]),
        "test",
        exp["model"],
    )



    size_gb = get_dir_size_gb(cache_dir)

  

    return math.ceil(size_gb)   

SCRIPTS = scripts_no_gpu + scripts_gpu
KS = [-1,25]
NORMALIZE = [False]
def main():
    for exp in experiments:
        for script_a, script_b in itertools.combinations_with_replacement(SCRIPTS, 2):
            # Convert script paths to module-style names
            module_a = script_a.replace(".py", "").replace("/", ".")
            module_b = script_b.replace(".py", "").replace("/", ".")

            # Build readable job name
            label_a = script_a.split("/")[1].replace("run_", "").replace(".py", "")
            label_b = script_b.split("/")[1].replace("run_", "").replace(".py", "")
            job_name = f"[EEMC] Structural Similarity {label_a} x {label_b} for {exp['name']}"

            # Combine arguments
            args = [
                "--model", exp["model"],
                "--train_dataset_name", exp["dataset"],
                "--train_dataset_split", "test",
                "--test_dataset_name", exp["dataset"],
                "--test_dataset_split", "test",
                "--sim_script_a", module_a,
                "--sim_script_b", module_b,
                "--experiment_name", exp["name"]
            ]
            

            for normalize in NORMALIZE:
                for k in KS:#, 50, 100]: #25, 
                    ram_a = get_ram(exp, representation_name=label_a)
                    ram_b = get_ram(exp, representation_name=label_b)
                    ram = ((ram_a + ram_b)*3) + 36 # 36G for docker image
                    # print("RAM_a", ram_a, "RAM_b", ram_b)
                    cmd = ["./schedule.sh", "run_similarity_template.sbatch", "structural_similarity", "--job-name", job_name, f"--mem={ram}G"] + args\
                            + (["--normalize"] if normalize else [])\
                            + ["--k", str(k)]
      
                    print(f"Scheduling with {ram}GB:", " ".join(cmd))
                    subprocess.run(cmd) 
    
                    

if __name__ == "__main__":
    main()