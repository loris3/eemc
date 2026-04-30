import argparse
from utils import load_lora_model
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset
from representations.load_data import load_data_and_estimators


from rank_bm25 import BM25Okapi
import torch
import pandas as pd
import os
from influence_estimation.estimator import BaseEstimator
from tqdm import tqdm
class BM25Estimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split,
                 tokenized_docs, device="cpu", param_list=[], param_list_gradients=[], eval_mode=False):
        super().__init__(
            model_path, train_dataset, train_dataset_name, train_dataset_split,
            test_dataset, test_dataset_name, test_dataset_split,
            device=device, param_list=param_list, param_list_gradients=param_list_gradients,
            eval_mode=eval_mode
        )


        self.tokenized_docs = tokenized_docs

        

    def get_config_string(self):
        return f"{self.__class__.__name__}: BM25 using pre-tokenized docs"

    def init_model_and_tokenizer(self):
        pass


    def run(self):
        self.bm25 = BM25Okapi(self.tokenized_docs)
        results = [self.bm25.get_scores(doc) for doc in tqdm(self.tokenized_docs, desc="bm25 loop")]
        # print(results)

        self.influence_estimate = pd.DataFrame(
            results,
            index=list(range(len(self.tokenized_docs))),
            columns=list(range(len(self.tokenized_docs)))
        )

        os.makedirs(os.path.dirname(self.influence_estimate_path), exist_ok=True)
        self.save()
from rank_bm25 import BM25Okapi
import torch

class BM25Representation(BaseRepresentation):
    def __init__(self, tokenized_docs, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, name="bm25", results_only=False,func_only=False,):
        super().__init__(
            name, train_dataset, train_dataset_name, train_dataset_split,
            test_dataset, test_dataset_name, test_dataset_split,
            "-", results_only=results_only, func_only=func_only,
        )

        self.tokenized_docs = tokenized_docs if not results_only else None
         
        self.estimator = BM25Estimator(
                model_path="-",
                train_dataset=train_dataset,
                train_dataset_name=train_dataset_name,
                train_dataset_split=train_dataset_split,
                test_dataset=test_dataset,
                test_dataset_name=test_dataset_name,
                test_dataset_split=test_dataset_split,
                tokenized_docs=tokenized_docs,  
                device="cpu",
                param_list=[],
                param_list_gradients=[],
                eval_mode=False
            )
        self.estimator.run_cached()
        if not func_only:
            self.estimator.embeddings = self.compute()
    def _embed_doc(self, idx: int) -> torch.Tensor:
        scores = self.estimator.get_explanation(idx)
        return torch.tensor(scores, dtype=torch.float32).cpu()

def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    train_dataset, test_dataset = load_data_and_estimators(
        args.train_dataset_name,
        args.test_dataset_name,
        args.train_dataset_split,
        args.test_dataset_split
    )

 
    _, tokenizer = load_lora_model(args.model) if not args.results_only else (None, None)


    tokenized_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        num_proc=10,
        re_index=False,
    ) if not args.results_only else None


    tokenized_texts = [
        [tokenizer.decode([tok_id], clean_up_tokenization_spaces=True).strip()
         for tok_id in doc['input_ids']
         if tok_id not in tokenizer.all_special_ids]
        for doc in tokenized_dataset
    ] if tokenized_dataset else None

  
    rep = BM25Representation(
        tokenized_docs=tokenized_texts,  
        train_dataset=train_dataset,
        train_dataset_name=args.train_dataset_name,
        train_dataset_split=args.train_dataset_split,
        test_dataset=test_dataset,
        test_dataset_name=args.test_dataset_name,
        test_dataset_split=args.test_dataset_split,
        results_only=args.results_only,
        func_only=args.func_only,
    )


    # Cleanup
    del train_dataset, test_dataset, tokenizer
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())

if __name__ == "__main__":
    main()