import argparse
from utils import load_lora_model

import torch
import os
import pandas as pd
from influence_estimation.estimator import BaseEstimator
from representations.main import BaseRepresentation

from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset

from representations.load_data import (   
    load_data_and_estimators)


class RepSimEstimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split,
                 device="cpu", param_list=[], param_list_gradients=[], eval_mode=False):
        super().__init__(
            model_path, train_dataset, train_dataset_name, train_dataset_split,
            test_dataset, test_dataset_name, test_dataset_split,
            device=device, param_list=param_list, param_list_gradients=param_list_gradients,
            eval_mode=eval_mode
        )
    def get_config_string(self):
        return f"{self.__class__.__name__}: {os.path.basename(model_path)}"
    def init_model_and_tokenizer(self):
        pass
    def run(self):
        results = self.embeddings @ self.embeddings.T
        self.influence_estimate = pd.DataFrame(
            results,
            index=list(range(self.embeddings.shape[0])),
            columns=list(range(self.embeddings.shape[0]))
        )
        os.makedirs(os.path.dirname(self.influence_estimate_path), exist_ok=True)
        self.save()

class RepSimRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, name="rep_sim", results_only=False):

        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)
        self.estimator = RepSimEstimator(
                model_path=model_path,
                train_dataset=train_dataset,
                train_dataset_name=train_dataset_name,
                train_dataset_split=train_dataset_split,
                test_dataset=test_dataset,
                test_dataset_name=test_dataset_name,
                test_dataset_split=test_dataset_split,
                device="cpu",
                param_list=[],
                param_list_gradients=[],
                eval_mode=False
            )
        if not self.results_only:
            self.tokenized_dataset = tokenized_dataset
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = model.to(self.device).eval()
        self.estimator.embeddings = self.compute()
    def _embed_doc(self, idx: int) -> torch.Tensor:

        if self.results_only:
            raise RuntimeError(f"results_only=True but cache not found for doc {idx}")

        m = self.tokenized_dataset[idx]

        input_ids = torch.tensor(m["input_ids"], device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(m["attention_mask"], device=self.device).unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        last_hidden = outputs.hidden_states[-1]

        # index of last non-padding token
        last_idx = attention_mask[0].nonzero()[-1].item()

        z = last_hidden[0, last_idx]   # shape: (d,)

        return z.cpu()
def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    train_dataset, test_dataset = load_data_and_estimators(args.train_dataset_name, 
                                                           args.test_dataset_name, 
                                                           args.train_dataset_split, 
                                                           args.test_dataset_split)
    
    model, tokenizer = load_lora_model(args.model) if not args.results_only else (None, None)

    tokenized_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        num_proc=10,
        re_index=False,
    ) if not args.results_only else None
    rep = RepSimRepresentation(model,
                tokenized_dataset, 
                train_dataset, 
                args.train_dataset_name, 
                args.train_dataset_split, 
                test_dataset, 
                args.test_dataset_name, 
                args.test_dataset_split, 
                args.model,
                results_only=args.results_only,
                name=f"rep_sim")

    del train_dataset, test_dataset, tokenizer, model
    return rep.estimator.embeddings, rep.get_influence_estimate()

   
if __name__ == "__main__":
    main()
    
    
