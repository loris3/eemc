import os
import argparse
import numpy as np
import torch
import rbo as rbo_
from scipy.stats import rankdata
import resource
from pathlib import Path
from scipy.stats import ConstantInputWarning
from sklearn.metrics import ndcg_score
from sklearn.metrics import recall_score
import pandas as pd
import warnings
from sklearn.metrics import f1_score
import torch.nn.functional as F
from scipy import stats
from metrics import AlignmentMetrics
from representations.common_args import add_common_args
from importlib.machinery import SourceFileLoader
from tqdm import tqdm




def spearman(a,b,k):
    results = []
    for i in range(a.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            corr = stats.spearmanr(a[i], b[i]).correlation
            if np.isnan(corr):
                print(f"[INFO] document {i} has const attribution")
        results.append(corr)
    return np.nanmean(results,axis=0)
def rbo(a, b, k=None):
    results = []

    for i in range(a.shape[0]):
        a_i = np.argsort(-a[i]).tolist()
        b_i = np.argsort(-b[i]).tolist()

        results.append(
            rbo_.RankingSimilarity(a_i, b_i).rbo(k=k)
        )
    return np.nanmean(results)
def ndcg_k(a,b,k):
    assert k is None or k > 0
    results = []
    
    for i in range(a.shape[0]):
        try:
            a_ = a[i] - a[i].min()
            b_ = b[i] - b[i].min()
            results.append(ndcg_score(a_.reshape(1, -1), b_.reshape(1, -1), k=k))


        except Exception as e:

            print(f"Error at index {i}: {e}", flush=True)
            exit()
    return np.mean(results,axis=0)
def f1_at_k(a, b, k):
    results = []

    for i in range(a.shape[0]):
        k_eff = k if k is not None else a.shape[1]

        # top-k indices
        a_topk = np.argsort(-a[i])[:k_eff]
        b_topk = np.argsort(-b[i])[:k_eff]

        # binary vectors
        y_true = np.zeros(a.shape[1], dtype=int)
        y_pred = np.zeros(a.shape[1], dtype=int)

        y_true[a_topk] = 1
        y_pred[b_topk] = 1

        results.append(
            f1_score(y_true, y_pred, zero_division=0)
        )

    return float(np.mean(results))

experiments_rep = [
    ("cka", lambda a,b,k: AlignmentMetrics.cka(a,b)),
    ("unbiased_cka", lambda a,b,k: AlignmentMetrics.unbiased_cka(a,b)),
    # ("svcca", lambda a,b,k: AlignmentMetrics.svcca(a,b)),
    ("cknna", lambda a,b,k: AlignmentMetrics.cknna(a,b,topk=k)),
    ("cycle_knn", lambda a,b,k: AlignmentMetrics.cycle_knn(a,b,topk=k)),
]
experiments_func = [
        ("spearman_rho", spearman),
        ("ndcg", lambda a, b, _: ndcg_k(a, b, k=None)),
        ("ndcg_k", lambda a, b, k: ndcg_k(a, b, k)),
        ("sign_agreement", lambda a, b, _: torch.mean(torch.stack([(torch.sign(a[i]) == torch.sign(b[i])).float().mean() for i in range(a.shape[0])])).item()),
        ("f1_at_k", lambda a, b, k: f1_at_k(a, b, k)),    
        ("rbo", lambda a, b, k: rbo(a, b, k)),      
        ] 
def print_peak_ram():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_kb = usage.ru_maxrss

    peak_gb = peak_kb / (1024 ** 2)
    print(f"\n[PEAK RAM] {peak_gb:.3f} GB", flush=True)

CKNNA_RESULTS_DIR = "results/similarity"
def to_tensor(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return torch.as_tensor(x).float()
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

def run_experiment(features_a, features_b, score_name, k, args, metric_fn, out_path, selection_strategy_name_a, selection_strategy_name_b):
    torch.manual_seed(42)
    np.random.seed(42)
    score = metric_fn(features_a, features_b, k)
    print("score_name", score_name, "score",score, out_path, flush=True)
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
        selection_strategy_name_a,
        selection_strategy_name_b
    ]], columns=[
        "Experiment","Rep_A","Rep_B","score_name","score","k","normalize",
        "model","train_dataset","train_dataset_split","test_dataset","test_dataset_split",
        "selection_strategy_name_a", "selection_strategy_name_b"
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
    os.makedirs(results_folder_rep, exist_ok=True)
 

    inner_args = build_inner_args(args)

    # Lazy loading
    features_a = None
    features_b = None
    influence_estimate_a = None
    influence_estimate_b = None
    # REP experiments

    for score_name, fn in tqdm(experiments_rep, desc="experiments_rep"):
        out_path = results_path(results_folder_rep, score_name, args.k, args)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            # pass
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
        run_experiment(to_tensor(features_a), to_tensor(features_b), score_name, args.k, args, fn, out_path, "-", "-")
    del features_a, features_b
    


    # FUNC experiments

    
    
    if_sorting_strategies = [
        ("most_harmful", lambda estimate: estimate),
        ("most_helpful", lambda estimate: -estimate),
        ("least_influential", lambda estimate: -torch.abs(estimate)),
        ("most_influential", lambda estimate: torch.abs(estimate)),
    ]
    for score_name, fn in tqdm(experiments_func, desc="experiments_func"):
        is_IF = lambda sim_script : ("gradsim" in sim_script) or ("datainf" in sim_script)
        for (selection_strategy_name_a, selection_strategy_fn_a) in if_sorting_strategies:
            for (selection_strategy_name_b, selection_strategy_fn_b) in if_sorting_strategies:
                results_folder_func = os.path.join(
                        CKNNA_RESULTS_DIR, "func", 
                        args.experiment_name,
                        args.train_dataset_name,
                        args.train_dataset_split,
                        args.test_dataset_name,
                        args.test_dataset_split,
                        args.model,
                        f"{Path(args.sim_script_a).suffix[1:]}_{selection_strategy_name_a if is_IF(args.sim_script_a) else '-'}_{Path(args.sim_script_b).suffix[1:]}_{selection_strategy_name_b if is_IF(args.sim_script_b) else '-'}"
                )
                os.makedirs(results_folder_func, exist_ok=True)
                out_path = (results_path(results_folder_func, score_name, args.k, args))
                # print("out_path", out_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if os.path.exists(out_path):
                    continue
                    # print(f"Skipping existing {out_path}", flush=True)
                    # continue
                if args.normalize:
                        continue
                if influence_estimate_a is None:
                    print("Loading features (FUNC)...")
                    _, influence_estimate_a = load_results(args.sim_script_a, inner_args)
                    _, influence_estimate_b = load_results(args.sim_script_b, inner_args)
                    
                
                
                a = to_tensor(influence_estimate_a)
                b = to_tensor(influence_estimate_b)
                
                
                if is_IF(args.sim_script_a):
                    a = selection_strategy_fn_a(a)
                if is_IF(args.sim_script_b):
                    b = selection_strategy_fn_b(b)   
                # print("is_IF", is_IF(args.sim_script_a), is_IF(args.sim_script_b))
                
                run_experiment(a, b, score_name, args.k, args, fn, out_path, selection_strategy_name_a if is_IF(args.sim_script_a) else "-", selection_strategy_name_b if is_IF(args.sim_script_b) else "-")


if __name__ == "__main__":
    main()
    print_peak_ram()

