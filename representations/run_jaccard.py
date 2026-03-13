import argparse
from utils import load_lora_model
from representations.main import JaccardRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset
from representations.load_data import load_data_and_estimators

def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    # Load datasets
    train_dataset, test_dataset = load_data_and_estimators(
        args.train_dataset_name,
        args.test_dataset_name,
        args.train_dataset_split,
        args.test_dataset_split
    )

    # Load model and tokenizer if needed
    _, tokenizer = load_lora_model(args.model) if not args.results_only else (None, None)

    # Tokenize test dataset
    tokenized_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        num_proc=10,
        re_index=False,
    ) if not args.results_only else None


    tokenized_texts = [
        [tokenizer.decode([tok_id], clean_up_tokenization_spaces=True).strip()
         for tok_id in doc['input_ids']
         if tok_id not in tokenizer.all_special_ids]  # remove special tokens
        for doc in tokenized_dataset
    ] if tokenized_dataset else None

    rep = JaccardRepresentation(
        tokenized_docs=tokenized_texts,  
        train_dataset=train_dataset,
        train_dataset_name=args.train_dataset_name,
        train_dataset_split=args.train_dataset_split,
        test_dataset=test_dataset,
        test_dataset_name=args.test_dataset_name,
        test_dataset_split=args.test_dataset_split,
        results_only=args.results_only
    )

    # Compute similarity matrix
    similarity_matrix = rep.compute()

    # Cleanup
    del train_dataset, test_dataset, tokenizer
    return similarity_matrix

if __name__ == "__main__":
    main()