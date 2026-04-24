import subprocess
import itertools
from schedule_representations import experiments, scripts_gpu, scripts_no_gpu

scripts = scripts_no_gpu + scripts_gpu

def main():
    for exp in experiments:
        for script_a, script_b in itertools.combinations(scripts, 2):
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
            

            for normalize in [True, False]:
                for k in [10]:#, 50, 100]: #25, 
                    cmd = ["./schedule.sh", "run_similarity_template.sbatch", "structural_similarity", "--job-name", job_name] + args\
                            + (["--normalize"] if normalize else [])\
                            + ["--k", str(k)]
                    print("Scheduling:", " ".join(cmd))
                    subprocess.run(cmd) 
                    # exit()
                    

if __name__ == "__main__":
    main()