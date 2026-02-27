import argparse
from utils import load_lora_model


from representations.main import SBERTRepresentation
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
    

    # use a tokenizer that supports apply_chat_template
    _,tokenizer_sample_model = load_lora_model("loris3/Llama-3.2-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42")
    tokenized_sample_dataset = tokenize_dataset(
        test_dataset,
        tokenizer_sample_model,
        max_length=512,
        num_proc=10,
        re_index=False,
    )
    decoded_texts_sample = [tokenizer_sample_model.decode(m["input_ids"]) for m in tokenized_sample_dataset]

    print("sbert")
    rep = SBERTRepresentation("all-MiniLM-L6-v2", decoded_texts_sample,
                    train_dataset, 
                    args.train_dataset_name, 
                    args.train_dataset_split, 
                    test_dataset, 
                    args.test_dataset_name, 
                    args.test_dataset_split, 
                    )
    similarity_matrix = rep.compute()
    del train_dataset, test_dataset
    return similarity_matrix

if __name__ == "__main__":
    main()
    
    
