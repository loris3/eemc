import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--train_dataset_split", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--test_dataset_split", required=True)
    parser.add_argument("--embd_model", required=False)
    parser.add_argument("--mode", required=False)
    parser.add_argument("--mem", required=False)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--results_only", action="store_true")
    parser.add_argument("--func_only", action="store_true")
    return parser


