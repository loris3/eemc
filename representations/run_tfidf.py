import argparse
from utils import load_lora_model
from representations.common_args import add_common_args
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset
from sentence_transformers import SentenceTransformer
from influence_estimation.estimator import BaseEstimator
from influence_estimation.util import tokenize_dataset
from representations.load_data import load_data_and_estimators
import torch
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
from representations.main import BaseRepresentation
class TFIDFEstimator(BaseEstimator):
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
        return f"{self.__class__.__name__}: -"
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

class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, name="tfidf", results_only=False, func_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         "-", results_only=results_only, func_only=func_only)
        
        self.estimator = TFIDFEstimator(
                model_path="-",
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
        if not results_only:
            self.tokenized_docs = tokenized_docs
            # Convert tokenized docs to strings for CountVectorizer's "pretokenized" mode
            self.tokenized_str_docs = [[str(tok) for tok in doc] for doc in tokenized_docs]

            # Build count matrix
            self.count_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,  # no-op tokenizer
                preprocessor=lambda x: x,  # no-op preprocessor
                lowercase=False
            )
            count_matrix = self.count_vectorizer.fit_transform(self.tokenized_str_docs)

            # Convert counts to TF-IDF
            tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            self.tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)
        if not func_only:
            self.estimator.embeddings = self.compute()
    def _embed_doc(self, idx: int) -> torch.Tensor:
        vec = torch.tensor(self.tfidf_matrix[idx].toarray(), dtype=torch.float32)
        return vec.squeeze(0).cpu()
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

    # Extract token IDs for TF-IDF
    tokenized_dataset_ids = [doc['input_ids'] for doc in tokenized_dataset] if tokenized_dataset else None

    rep = TFIDFRepresentation(
        tokenized_docs=tokenized_dataset_ids,  
        train_dataset=train_dataset,
        train_dataset_name=args.train_dataset_name,
        train_dataset_split=args.train_dataset_split,
        test_dataset=test_dataset,
        test_dataset_name=args.test_dataset_name,
        test_dataset_split=args.test_dataset_split,
        results_only=args.results_only,
        func_only=args.func_only
    )


    # Cleanup
    del train_dataset, test_dataset, tokenizer
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())

if __name__ == "__main__":
    main()