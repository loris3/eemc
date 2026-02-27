from datasets import load_dataset

import time






def load_dataset_with_retry(
    name,
    split,
    retries=10,
    delay=5,
    backoff=2,
    **kwargs,
):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(name, split=split, **kwargs)
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            wait = delay * (backoff ** (attempt - 1))
            print(
                f"[load_dataset] attempt {attempt}/{retries} failed: {e}\n"
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)
    raise last_err


def load_data_and_estimators(train_dataset_name, test_dataset_name, train_dataset_split, test_dataset_split, results_only=False):
    train_dataset = (
        load_dataset_with_retry(
            train_dataset_name,
            split=train_dataset_split,
        )
        if not results_only
        else None
    )
    train_dataset = train_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True,
        num_proc=10
    ) if not results_only else None

    test_dataset = (
        load_dataset_with_retry(
            test_dataset_name,
            split=test_dataset_split,
        )
        if not results_only
        else None
    )
    test_dataset = test_dataset.map(
        lambda example, idx: {"indices": idx},
        with_indices=True,
        num_proc=10
    )  if not results_only else None

    return train_dataset, test_dataset