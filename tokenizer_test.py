import os
from datasets import load_dataset
from transformers import AutoTokenizer
from influence_estimation.util import tokenize_dataset  

def main():
  
    tokenizer_name = "./models/OLMo-2-0425-1B_tulu-3-sft-olmo-2-mixture-0225_lr0.0001_seed42"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("tokenizer max len", tokenizer.model_max_length)
    dataset_name = "loris3/tulu3-test-small"
    dataset = load_dataset(dataset_name, split="train")

    print(f"Loaded dataset with {len(dataset)} examples")

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        chat_template_path="./chat_template.jinja",  
        assistant_only_loss=True,
        text_column="messages",  
        num_proc=20,
        re_index=True
    )

    # Print a single example to verify
    print("Tokenized example:")
    print(tokenized_dataset[111:113])

if __name__ == "__main__":
    main()
