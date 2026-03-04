import argparse

from influence_estimation.data_inf import DataInfEstimator
from representations.main import DataInfRepresentation
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
    estimator_datainf = DataInfEstimator(
        args.model,
        train_dataset,
        args.train_dataset_name,
        args.train_dataset_split,
        test_dataset,
        args.test_dataset_name,
        args.test_dataset_split,
        eval_mode=True # must be True so that cache/gradients not /tmp/cache... is used      
    )

    rep = DataInfRepresentation(estimator_datainf, results_only=args.results_only)

    similarity_matrix = rep.compute()
    del train_dataset, test_dataset
    return similarity_matrix
if __name__ == "__main__":
    main()
    
    
