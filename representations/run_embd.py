import argparse
from utils import load_lora_model


from representations.main import SentenceTransformerRepresentation
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
    

    # we need to apply_chat_template given how the models where (LoRA) fine-tuned
    # args.model is NOT the embedding model here, we just load it to determine max_seq_len!
    _,tokenizer_sample_model = load_lora_model(args.model) if not args.results_only else (None, None)
    tokenized_sample_dataset = tokenize_dataset(
        test_dataset,
        tokenizer_sample_model,
        num_proc=10,
        re_index=False,
        tokenize=False
    ) if not args.results_only else None
    rep = SentenceTransformerRepresentation(args.embd_model, tokenized_sample_dataset,
                    train_dataset, 
                    args.train_dataset_name, 
                    args.train_dataset_split, 
                    test_dataset, 
                    args.test_dataset_name, 
                    args.test_dataset_split, 
                    results_only=args.results_only
                    )
    similarity_matrix = rep.compute()
    del train_dataset, test_dataset
    return similarity_matrix

if __name__ == "__main__":
    main()
    
    
