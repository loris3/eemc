import os
import importlib.util
import argparse
from representations.common_args import add_common_args
import time
from tqdm import tqdm
import itertools
import pandas as pd
import torch.nn.functional as F 
from metrics import AlignmentMetrics
CKNNA_RESULTS_DIR = "results/structural_similarity"

from pathlib import Path




def load_features(script_path, inner_args):
    """Dynamically load a script and run its main() function."""
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    print("Loading module:", module_name)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(inner_args)

def build_inner_args(args):
    inner_args = []
    for key, value in vars(args).items():
        if key in ["experiment_name", "sim_script_a", "sim_script_b", "top_k"]:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                inner_args.append(flag)
        else:
            inner_args.extend([flag, str(value)])
    inner_args.append("--results_only")
    return inner_args

def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--sim_script_a", required=True)
    parser.add_argument("--sim_script_b", required=True)
    parser.add_argument("--top_k", required=True, type=int, nargs='+',
                    help="One or more top_k values, e.g. --top_k 5 10 20")
    
    args = parser.parse_args(args)
    
    print(args)
    results_folder = os.path.join(
        CKNNA_RESULTS_DIR,
        args.experiment_name,
        args.train_dataset_name,
        args.train_dataset_split,
        args.test_dataset_name,
        args.test_dataset_split,
        args.model,
        Path(args.sim_script_a).stem +"_"+  Path(args.sim_script_b).stem,
        )
    os.makedirs(results_folder, exist_ok=True)
    
    
    inner_args = build_inner_args(args)
    
    # Load features for both scripts
    features_a = load_features(args.sim_script_a, inner_args)
    features_b = load_features(args.sim_script_b, inner_args)
  
    features_a_normalized = F.normalize(features_a, dim=-1)
    features_b_normalized = F.normalize(features_b, dim=-1)
    print("Features loaded for both A and B")
    
    
    
    cols = [
        "Experiment",
        "Rep_A",
        "Rep_B",
        "score_name",
        "score",
        "top_k",
        "normalize",
        "model",
        "train_dataset",
        "train_dataset_split",
        "test_dataset",
        "test_dataset_split",
    ]

    experiments = []

    for normalize in [True, False]:
        experiments.append({
            "score_name": "cka",
            "normalize": normalize,
        })
        for k in args.top_k:
            experiments.append({
                "score_name": "cknna",
                "normalize": normalize,
                "k": k,
            })
            experiments.append({
                "score_name": "cycle_knn",
                "normalize": normalize,
                "k": k,
            })
    metric_fns = {
    "cka": lambda a, b, k: AlignmentMetrics.cka(a, b),
    "cknna": lambda a, b, k: AlignmentMetrics.cknna(a, b, topk=k),
    "cycle_knn": lambda a, b, k: AlignmentMetrics.cycle_knn(a, b, topk=k),
    }

    for exp in tqdm(experiments, desc="Running experiments"):
        score_name = exp["score_name"]
        normalize = exp["normalize"]
        k = exp.get("k")
        if k is None:
            out_path = os.path.join(
                results_folder,
                score_name,
                f"normalize_{normalize}",
                "scores.parquet"
         )
        else:
            out_path = os.path.join(
                results_folder,
                score_name,
                f"k_{k}",
                f"normalize_{normalize}",
                "scores.parquet"
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            print("Skipping", out_path)
            continue
        feats_a = features_a_normalized if normalize else features_a
        feats_b = features_b_normalized if normalize else features_b

        start_time = time.time()
        score = metric_fns[score_name](feats_a, feats_b, k)
        end_time = time.time()

        print(
            f"time {score_name} (k={k}, normalize={normalize}): "
            f"{end_time-start_time:.2f}s", flush=True
        )

        df = pd.DataFrame([[
            args.experiment_name,
            os.path.basename(args.sim_script_a),
            os.path.basename(args.sim_script_b),
            score_name,
            float(score),
            k,
            normalize,
            args.model,
            args.train_dataset_name,
            args.train_dataset_split,
            args.test_dataset_name,
            args.test_dataset_split,
        ]], columns=cols)
        df['top_k'] = pd.array(df['top_k'], dtype="Int64")
        df.to_parquet(out_path, index=False)
    print(f"Results written to {results_folder}")
if __name__ == "__main__":
    main()