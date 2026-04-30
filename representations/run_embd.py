import argparse
from utils import load_lora_model
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

class SentenceTransformerEstimator(BaseEstimator):
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

class SentenceTransformerRepresentation(BaseRepresentation):
    def __init__(self, model_path, decoded_texts, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 results_only=False,func_only=False,):

        super().__init__(f"embd_{model_path}", train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only,func_only=func_only)
        self.decoded_texts = decoded_texts

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.estimator = SentenceTransformerEstimator(
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
        print("results_only", results_only)
        if not results_only:
            self.st_model = SentenceTransformer(model_path, trust_remote_code=True)
            self.st_model.to(self.device)
        if not func_only:
            self.estimator.embeddings = self.compute()



    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")

        text = self.decoded_texts[idx]["text"]
        emb = self.st_model.encode([text])[0]
        return torch.tensor(emb, dtype=torch.float32).to("cpu")
    
    
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
                    results_only=args.results_only, func_only=args.func_only,
                    )
    del train_dataset, test_dataset
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())

if __name__ == "__main__":
    main()
    
    
