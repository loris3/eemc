import argparse
import pickle
from influence_estimation.data_inf import DataInfEstimator
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
import torch
import os
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
from influence_estimation.util import tokenize_dataset

from influence_estimation.estimator import BaseEstimator
import pandas as pd
from representations.load_data import (   
    load_data_and_estimators)
class DataInfRepresentation(BaseRepresentation):
    def __init__(self, estimator, results_only=False,func_only=False,):
        super().__init__("datainf",
                         estimator.test_dataset,
                         estimator.test_dataset_name,
                         estimator.test_dataset_split,
                         estimator.test_dataset,
                         estimator.test_dataset_name,
                         estimator.test_dataset_split,
                         estimator.model_path,
                         results_only=results_only,func_only=func_only)
        self.estimator = estimator
        self._hvp_dict = None
        if not func_only:
            self.estimator.embeddings = self.compute()
        
    @property
    def hvp_dict(self):
        if self._hvp_dict is None:
            self.compute_hvp_approx_with_cache()
        return self._hvp_dict

    def compute_hvp_approx_with_cache(self, lambda_const_param=10, batch_size=32,
                                      val_batch_size=1000, inner_batch_size=1):
        estimator = self.estimator
        hvp_cache_path = os.path.join(estimator.gradient_out_dir, "hvp",
                                      estimator.test_dataset_name, estimator.test_dataset_split,
                                      estimator.test_dataset_name, estimator.test_dataset_split, "hvp.pt")
        os.makedirs(os.path.dirname(hvp_cache_path), exist_ok=True)

        if os.path.exists(hvp_cache_path):
            print(f"Loading cached HVP from {hvp_cache_path}")
            with open(hvp_cache_path, "rb") as f:
                self._hvp_dict = pickle.load(f)
            print(f"Got cached HVP from {hvp_cache_path}")
            return

        if self.results_only:
            raise RuntimeError(f"results_only=True but HVP cache not found at {hvp_cache_path}")

      

    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self._hvp_dict is None and self.results_only:
            raise RuntimeError(f"results_only=True but HVP cache not loaded for doc {idx}")
        gradient = torch.cat([self.hvp_dict[idx][k].flatten() for k in sorted(self.hvp_dict[idx].keys())])
        return gradient.cpu().float()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    test_dataset, _ = load_data_and_estimators(args.test_dataset_name, 
                                                           args.test_dataset_name, 
                                                           args.test_dataset_split, 
                                                           args.test_dataset_split)
    estimator_datainf = DataInfEstimator(
        args.model,
        test_dataset,
        args.test_dataset_name,
        args.test_dataset_split,
        test_dataset,
        args.test_dataset_name,
        args.test_dataset_split,
        eval_mode=True # must be True so that cache/gradients not /tmp/cache... is used      
    )

    rep = DataInfRepresentation(estimator_datainf, results_only=args.results_only, func_only=args.func_only,)


    del test_dataset
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())
if __name__ == "__main__":
    main()
    
    
