ssh 'nlp-data.dm.univie.ac.at' python3 eemc/merge_results.py --source_dir="eemc/results/similarity" --target_dir="eemc/results/similarity_merged"\
& ssh 'nlp-data.dm.univie.ac.at' python3 eemc/merge_results.py --source_dir="eemc/results/rank_predictability" --target_dir="eemc/results/rank_predictability_merged"
