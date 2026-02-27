import argparse
from utils import load_lora_model


from representations.main import JaccardRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset

from representations.load_data import (   
    load_data_and_estimators)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    train_dataset, test_dataset = load_data_and_estimators(args.train_dataset_name, 
                                                           args.test_dataset_name, 
                                                           args.train_dataset_split, 
                                                           args.test_dataset_split)
    
    _, tokenizer = load_lora_model(args.model)

    tokenized_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        max_length=512,
        num_proc=10,
        re_index=False,
    )
    rep = JaccardRepresentation(
            tokenized_docs=tokenized_dataset,
            train_dataset=train_dataset,
            train_dataset_name=args.train_dataset_name,
            train_dataset_split=args.train_dataset_split,
            test_dataset=test_dataset,
            test_dataset_name=args.test_dataset_name,
            test_dataset_split=args.test_dataset_split
        )
    similarity_matrix = rep.compute()
    del train_dataset, test_dataset, tokenizer
    return similarity_matrix

   
if __name__ == "__main__":
    main()
    
    
