
import itertools
from schedule_representations import experiments, scripts_gpu, scripts_no_gpu
import subprocess

SCRIPTS = scripts_no_gpu + scripts_gpu









for exp in experiments:
    for script_a, script_b in itertools.combinations_with_replacement(SCRIPTS, 2):

        module_a = script_a.replace(".py", "").replace("/", ".")
        module_b = script_b.replace(".py", "").replace("/", ".")

        label_a = script_a.split("/")[1].replace("run_", "").replace(".py", "")
        label_b = script_b.split("/")[1].replace("run_", "").replace(".py", "")

        job_name = (
            f"[EEMC] Rank Predictability {exp['model']}"
            f"{label_a} x {label_b} for {exp['name']}"
        )

        print(job_name)

        args = [
        "--model",exp["model"],
        "--train_dataset_name",exp["dataset"],
        "--train_dataset_split","test",
        "--test_dataset_name",exp["dataset"],
        "--test_dataset_split","test",
        "--sim_script_a",module_a,
        "--sim_script_b",module_b,
        "--experiment_name",exp["name"],
        ]
        cmd = ["./schedule.sh", "run_predictability_template.sbatch", "predictability", "--job-name", job_name,] + args
           
      
        print(" ".join(cmd))
        subprocess.run(cmd)
       

        