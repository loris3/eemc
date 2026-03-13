import argparse

from influence_estimation.less_inf import LESSEstimator
from representations.main import GradientRepresentation
from representations.common_args import add_common_args

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


    rep = GradientRepresentation(estimator_less, name=f"gradsim_{args.mode}", results_only=args.results_only)

    similarity_matrix = rep.compute()
    del train_dataset, test_dataset
    return similarity_matrix
if __name__ == "__main__":
    main()
    
    
