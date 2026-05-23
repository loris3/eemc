import matplotlib.pyplot as plt

import numpy as np
from representations.common_args import add_common_args

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from structural_similarity import load_results, build_inner_args
import os
from scipy.stats import spearmanr
import os
import itertools
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
RESULTS_DIR = "results/rank_predictability"



if_sorting_strategies = [
    ("most_harmful", lambda estimate: estimate),
    ("most_helpful", lambda estimate: -estimate),
    ("least_influential", lambda estimate: -estimate.abs()),
    ("most_influential", lambda estimate: estimate.abs()),
]

is_IF = lambda sim_script: (
    ("gradsim" in sim_script) or ("datainf" in sim_script)
)

def align_dfs_by_index(df_i, df_j):
    df_i = df_i.sort_values("z'").reset_index(drop=True)
    df_j = df_j.sort_values("z'").reset_index(drop=True)

    assert np.all(df_i["z'"].values == df_j["z'"].values)
    return df_i, df_j
def results_path(args, selection_strategy_name_a, selection_strategy_name_b):
    is_IF = lambda sim_script: (
        ("gradsim" in sim_script) or ("datainf" in sim_script)
    )

    base_folder = os.path.join(
        RESULTS_DIR,
        args.experiment_name,
        args.train_dataset_name,
        args.train_dataset_split,
        args.test_dataset_name,
        args.test_dataset_split,
        args.model,
        f"{Path(args.sim_script_a).suffix[1:]}_"
        f"{selection_strategy_name_a if is_IF(args.sim_script_a) else '-'}_"
        f"{Path(args.sim_script_b).suffix[1:]}_"
        f"{selection_strategy_name_b if is_IF(args.sim_script_b) else '-'}"
    )

    os.makedirs(base_folder, exist_ok=True)
    return os.path.join(base_folder, "scores.parquet")

def build_pairwise_data_for_instance(
    row_i,
    row_j,
    n_pairs=5000,
    seed=0
):
    """
    Builds pairwise comparisons for a single test instance z'
    """

    rng = np.random.default_rng(seed)

    scores_i = row_i[1:].values.astype(float)
    scores_j = row_j[1:].values.astype(float)

    n = len(scores_i)

    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)

    mask = a != b
    a = a[mask]
    b = b[mask]

    # valid = scores_j[a] != scores_j[b]

    # a = a[valid]
    # b = b[valid]

    x = scores_i[a] - scores_i[b]
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (scores_j[a] > scores_j[b]).astype(int)

    return x.reshape(-1, 1), y


def fit_instance_model(
    row_i,
    row_j,
    n_pairs=5000,
    test_size=0.2,
    seed=0
):
    """
    Fit model for one test instance z'
    """

    X, y = build_pairwise_data_for_instance(
        row_i,
        row_j,
        n_pairs=n_pairs,
        seed=seed
    )

    if len(np.unique(y)) < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    clf = LogisticRegression(
        fit_intercept=False,
        solver="lbfgs"
    )

    clf.fit(X_train, y_train)

    w_hat = clf.coef_[0][0]

    test_acc = clf.score(X_test, y_test)

    return {
        "w_hat": w_hat,
        "test_acc": test_acc,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }


def cross_method_predictability(
    df_i,
    df_j,
    n_pairs=5000,
    seed=0
):
    df_i, df_j = align_dfs_by_index(df_i, df_j)

    results = []

    for z in range(len(df_i)):
        result = fit_instance_model(
            df_i.iloc[z],
            df_j.iloc[z],
            n_pairs=n_pairs,
            seed=seed + z
        )

        if result is None:
            continue

        result["z_prime"] = df_i.iloc[z]["z'"]
        results.append(result)

    results_df = pd.DataFrame(results)

    summary = {
        "mean_w": results_df["w_hat"].mean(),
        "std_w": results_df["w_hat"].std(),
        "mean_accuracy": results_df["test_acc"].mean(),
        "median_accuracy": results_df["test_acc"].median(),
        "n_instances": len(results_df)
    }

    return results_df, summary

def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--sim_script_a", required=True)
    parser.add_argument("--sim_script_b", required=True)

    args = parser.parse_args()

    df_i_raw = None
    df_j_raw = None
    for (
        selection_strategy_name_a, 
        selection_strategy_fn_a,
    ) in if_sorting_strategies if is_IF(args.sim_script_a) else if_sorting_strategies[0:1]:
        for (
            selection_strategy_name_b,
            selection_strategy_fn_b,
        ) in if_sorting_strategies if is_IF(args.sim_script_b) else if_sorting_strategies[0:1]:

            out_path = results_path(
                args,
                selection_strategy_name_a
                if is_IF(args.sim_script_a) else "-",
                selection_strategy_name_b
                if is_IF(args.sim_script_b) else "-"
            )

            # if os.path.exists(out_path):
            #     print(f"Skipping existing result: {out_path}")
            #     continue
            
            if df_i_raw is None:
                _, df_i_raw = load_results(
                    args.sim_script_a,
                    build_inner_args(args, func_only=True)
                )
            if df_j_raw is None:
                _, df_j_raw = load_results(
                    args.sim_script_b,
                    build_inner_args(args, func_only=True)
                )
            df_i = df_i_raw.copy()
            df_j = df_j_raw.copy()

            if is_IF(args.sim_script_a):
                df_i = selection_strategy_fn_a(df_i)

            if is_IF(args.sim_script_b):
                df_j = selection_strategy_fn_b(df_j)

            
            df_i = df_i.reset_index().rename(
                columns={"index": "z'"}
            )
            df_j = df_j.reset_index().rename(
                columns={"index": "z'"}
            ) 
            
            w_df, summary = cross_method_predictability(
                df_i,
                df_j,
                n_pairs=2000 
            )

            
                    

            result_df = pd.DataFrame(
            [[
                args.experiment_name,
                os.path.basename(args.sim_script_a),
                os.path.basename(args.sim_script_b),

                summary["mean_w"],              
                summary["std_w"],               
                summary["mean_accuracy"],      
                summary["n_instances"],         

                args.model,
                args.train_dataset_name,
                args.train_dataset_split,
                args.test_dataset_name,
                args.test_dataset_split,
                selection_strategy_name_a if is_IF(args.sim_script_a) else "-",
                selection_strategy_name_b if is_IF(args.sim_script_b) else "-"
            ]],
            columns=[
                "Experiment",
                "Rep_A",
                "Rep_B",
                "mean_w",
                "std_w",
                "mean_accuracy",
                "n_instances",
                "model",
                "train_dataset",
                "train_dataset_split",
                "test_dataset",
                "test_dataset_split",
                "selection_strategy_name_a",
                "selection_strategy_name_b",
            ]
        )

            result_df.to_parquet(out_path, index=False)

            print(result_df[[ "mean_w",
                "std_w",
                "mean_accuracy",
                "n_instances","selection_strategy_name_a",
                "selection_strategy_name_b",]], flush=True)


if __name__ == "__main__":
    main()