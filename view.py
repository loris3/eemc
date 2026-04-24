from datasets import load_dataset
from influence_estimation.util import tokenize_dataset

# Load the dataset (default split is usually "train")
dataset = load_dataset("loris3/tulu-3-sft-olmo-2-mixture-0225-sample", split="train")

idx = 899
# Get the example at index 899
example = dataset[idx]

# Print it
print(example)
from utils import load_lora_model

_, tokenizer = load_lora_model("./models/Qwen3-0.6B-Base_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42")

td = tokenize_dataset(
        dataset.select([50]),
        tokenizer,
        num_proc=1,
        re_index=False,
    )
print("td[0]",td[0])