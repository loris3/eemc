import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
from scipy import stats
from metrics import AlignmentMetrics
from representations.common_args import add_common_args
from importlib.machinery import SourceFileLoader
from tqdm import tqdm
CKNNA_RESULTS_DIR = "results/similarity"

def load_results(module_name, args):
    file_path = os.path.join(*module_name.split(".")) + ".py"
    module = SourceFileLoader(module_name, file_path).load_module()
    return module.main(args)

def build_inner_args(args):
    inner_args = []
    for key, value in vars(args).items():
        if key in ["experiment_name","sim_script_a","sim_script_b","k","normalize"]:
            continue
        flag = f"--{key}"
        if isinstance(value,bool):
            if value:
                inner_args.append(flag)
        else:
            inner_args.extend([flag,str(value)])
    inner_args.append("--results_only")
    return inner_args

def run_experiment(features_a, features_b, score_name, k, args, metric_fn, out_path):
    print("score_name", score_name, flush=True)

    score = metric_fn(features_a, features_b, k)
    print("score",score, flush=True)
    df = pd.DataFrame([[
        args.experiment_name,
        os.path.basename(args.sim_script_a),
        os.path.basename(args.sim_script_b),
        score_name,
        float(score),
        k,
        args.normalize,
        args.model,
        args.train_dataset_name,
        args.train_dataset_split,
        args.test_dataset_name,
        args.test_dataset_split,
    ]], columns=[
        "Experiment","Rep_A","Rep_B","score_name","score","k","normalize",
        "model","train_dataset","train_dataset_split","test_dataset","test_dataset_split"
    ])
    df['k'] = df['k'].astype('Int64')
    df.to_parquet(out_path, index=False)
def results_path(results_folder, score_name, k, args):
    return os.path.join(
            results_folder,
            score_name,
            *(["k_" + str(k)] if k is not None else []),
            f"normalize_{args.normalize}",
            "scores.parquet"
        )
def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--sim_script_a", required=True)
    parser.add_argument("--sim_script_b", required=True)
    parser.add_argument("--k", required=True, type=int)
    # parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    base_folder = os.path.join(
        args.experiment_name,
        args.train_dataset_name,
        args.train_dataset_split,
        args.test_dataset_name,
        args.test_dataset_split,
        args.model,
        Path(args.sim_script_a).suffix[1:] + "_" + Path(args.sim_script_b).suffix[1:]
    )
    results_folder_rep = os.path.join(CKNNA_RESULTS_DIR, "rep", base_folder)
    results_folder_func = os.path.join(CKNNA_RESULTS_DIR, "func", base_folder)
    os.makedirs(results_folder_rep, exist_ok=True)
    os.makedirs(results_folder_func, exist_ok=True)

    inner_args = build_inner_args(args)

    # Lazy loading
    features_a = None
    features_b = None
    influence_estimate_a = None
    influence_estimate_b = None
    # REP experiments
    experiments_rep = [
        ("cka", None, lambda a,b,k: AlignmentMetrics.cka(a,b)),
        ("unbiased_cka", None, lambda a,b,k: AlignmentMetrics.unbiased_cka(a,b)),
        # ("svcca", None, lambda a,b,k: AlignmentMetrics.svcca(a,b)),
        ("cknna", args.k, lambda a,b,k: AlignmentMetrics.cknna(a,b,topk=k)),
        ("cycle_knn", args.k, lambda a,b,k: AlignmentMetrics.cycle_knn(a,b,topk=k)),
    ]
    for score_name, k, fn in tqdm(experiments_rep, desc="experiments_rep"):
        out_path = results_path(results_folder_rep, score_name, k, args)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            print(f"Skipping existing {out_path}", flush=True)
            continue
    
        if features_a is None:
            print("Loading features (REP)...")
            features_a, influence_estimate_a = load_results(args.sim_script_a, inner_args)
            features_b, influence_estimate_b = load_results(args.sim_script_b, inner_args)
            features_a = features_a - features_a.mean(dim=0, keepdim=True)
            features_b = features_b - features_b.mean(dim=0, keepdim=True)
            if args.normalize:
                features_a = F.normalize(features_a, dim=-1)
                features_b = F.normalize(features_b, dim=-1)
        run_experiment(features_a, features_b, score_name, k, args, fn, out_path)
    del features_a, features_b
    

    # FUNC experiments
    experiments_func = [
        ("spearman_rho", None, lambda a,b,k: np.array([stats.spearmanr(a[i], b[i]).correlation for i in range(a.shape[0])]).mean(axis=0)
)
    ]
    for score_name, k, fn in tqdm(experiments_func, desc="experiments_func"):
        out_path = (results_path(results_folder_func, score_name, k, args))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            print(f"Skipping existing {out_path}", flush=True)
            continue
        if influence_estimate_a is None:
            print("Loading features (FUNC)...")
            _, influence_estimate_a = load_results(args.sim_script_a, inner_args)
            _, influence_estimate_b = load_results(args.sim_script_b, inner_args)
            if args.normalize:
                continue
        print(influence_estimate_a.shape)
        run_experiment(influence_estimate_a, influence_estimate_b, score_name, k, args, fn, out_path)

    print(f"Results written to:\nREP: {results_folder_rep}\nFUNC: {results_folder_func}")

if __name__ == "__main__":
    main()