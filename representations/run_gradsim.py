import argparse

from influence_estimation.less_inf import LESSEstimator
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
class GradientRepresentation(BaseRepresentation):
    def __init__(self, estimator, name="GradientRepresentation", results_only=False,func_only=False,):
        super().__init__(name,
                         estimator.train_dataset,
                         estimator.train_dataset_name,
                         estimator.train_dataset_split,
                         estimator.test_dataset,
                         estimator.test_dataset_name,
                         estimator.test_dataset_split,
                         estimator.model_path,
                         results_only=results_only,
                         func_only=func_only,)
        self.estimator = estimator
        if not func_only:
            self.estimator.embeddings = self.compute()
        
    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")
        gradient = self.estimator.get_gradient(
            self.estimator.test_dataset_name,
            self.test_dataset_split,
            [idx]
        )[0].float()
       
        return gradient.cpu()



def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)

    train_dataset, test_dataset = load_data_and_estimators(args.train_dataset_name, 
                                                           args.test_dataset_name, 
                                                           args.train_dataset_split, 
                                                           args.test_dataset_split)
    print("MODE", args.mode)
    estimator_less = LESSEstimator(
        args.model,
        train_dataset,
        args.train_dataset_name,
        args.train_dataset_split,
        test_dataset,
        args.test_dataset_name,
        args.test_dataset_split,
        mode=args.mode,
        eval_mode=True # must be True so that cache/gradients not /tmp/cache... is used
      )


    rep = GradientRepresentation(estimator_less, name=f"gradsim_{args.mode}", results_only=args.results_only, func_only=args.func_only,)


    del train_dataset, test_dataset
    return (rep.estimator.embeddings, rep.get_influence_estimate()) if not args.func_only else (None, rep.get_influence_estimate())
if __name__ == "__main__":
    main()
    
    
