import argparse
from utils import load_lora_model


from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset
import torch
import os
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset
from sentence_transformers import SentenceTransformer
from influence_estimation.estimator import BaseEstimator
import pandas as pd
from representations.load_data import (   
    load_data_and_estimators)


class FullModelEstimator(BaseEstimator):
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
        

class FullModelRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, name="full_model", results_only=False,func_only=False,):

        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only, func_only=func_only)
        self.estimator = FullModelEstimator(
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
        if not func_only:
            self.estimator.embeddings = self.compute()

    def _embed_doc(self, idx: int) -> torch.Tensor:

        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")

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

        token_embeddings = outputs.hidden_states[-1]  # (1, n, d)
        mask = attention_mask.unsqueeze(-1).float()   # (1, n, 1)
        masked_embeddings = token_embeddings * mask

        sentence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1)

        return sentence_embedding.squeeze(0).cpu()
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
    rep = FullModelRepresentation(
                 model,
                 tokenized_dataset, 
                 train_dataset, 
                 args.train_dataset_name, 
                 args.train_dataset_split, 
                 test_dataset, 
                 args.test_dataset_name, 
                 args.test_dataset_split, 
                 args.model,
                 results_only=args.results_only,func_only=args.func_only,
      
        )

    del train_dataset, test_dataset, tokenizer, model
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())

   
if __name__ == "__main__":
    main()
    
    
