
 

import pandas as pd
import os
import warnings
import yaml
from structural_similarity import experiments_rep, experiments_func, load_results, build_inner_args
warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

abbrev_map = {
    'representations.run_bm25': r"$r_{bm25}$",
    'representations.run_datainf': r"$r_{s_{test}}$",
    'representations.run_embd_KaLM-Embedding-Gemma3-12B-2511': 'kalm',
    'representations.run_embd_Qwen3-Embedding-8B': 'qwen',
    'representations.run_full_model': r"$r_{mp}$",
    'representations.run_gradsim_adam': r"$r_{\nabla f}^{aa}$",
    'representations.run_gradsim_unnormalized': r"$r^{id_{X}}_{\nabla f}$",
    'representations.run_gradsim_normalized': r"$r^{L2}_{\nabla f}$",
    'representations.run_repsim': r"$r_{repsim}$",
    'representations.run_norm_attn': r"$r_{attn}$",
    'representations.run_embd_llama-embed-nemotron-8b': 'nemot',
    'representations.run_jaccard':  r"$r_{jaccard}$",
    'representations.run_tfidf':  r"$r_{tfidf}$",
}

abbrev_map_strategy = {
    'most_influential': r" {most inf}$",
    'least_influential': r" {least inf}$",
    'most_harmful': r" {harm}$",
    'most_helpful': r" {help}$",
}

def abbreviate_row(row):
    rep_a, rep_b = row["Rep_A"], row["Rep_B"]

    if rep_a not in abbrev_map or rep_b not in abbrev_map:
        raise NotImplementedError(
            f"Abbreviation for {rep_a} or {rep_b} not implemented"
        )
    rep_a = abbrev_map[rep_a]
    rep_b = abbrev_map[rep_b]
    
    sel_a = row.get("selection_strategy_name_a")
    sel_b = row.get("selection_strategy_name_b")

    if sel_a != "-":
        rep_a = rep_a[:-1] + abbrev_map_strategy[sel_a]

    if sel_b != "-":
        rep_b = rep_b[:-1] + abbrev_map_strategy[sel_b]
    row["Rep_A"] = rep_a
    row["Rep_B"] = rep_b
    return row


def load():
    df = pd.read_parquet("./results/similarity_merged/")
    df = df[~df["normalize"]]
    return df
def format(df):
    
    df = df.apply(abbreviate_row, axis=1)
    if "k" in df.columns:
        df["k"] = df["k"].fillna(-1)
    df["Rep_A"] = df["Rep_A"].str.replace(".py","")
    df["Rep_A"] = df["Rep_A"].str.replace("run_","")
    df["Rep_B"] = df["Rep_B"].str.replace(".py","")
    df["Rep_B"] = df["Rep_B"].str.replace("run_","")

    df["Rep_A"] = df["Rep_A"].str.replace("representations.","")
    df["Rep_B"] = df["Rep_B"].str.replace("representations.","")

    
    with open("experiments.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    base_models = [b.split('/')[1] for b in cfg["base_models"]]
    df["base_model"] = df["model"].apply(
        lambda x: next(b for b in base_models if b in x)
    )
    if "score_name" in df.columns:
        df["test_type"] = df["score_name"].apply(lambda s : "func" if (s in next(zip(*experiments_func))) else "rep")
    return df