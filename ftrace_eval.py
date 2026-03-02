#!/usr/bin/env python3

import os
import json
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def load_model_and_tokenizer(path_or_repo):
    if os.path.exists(path_or_repo):
        print(f"Loading model from local folder: {path_or_repo}")
        adapter_path = os.path.join(path_or_repo, "adapter_config.json")
        chat_template_path = os.path.join(path_or_repo, "chat_template.jinja")
        if not os.path.exists(chat_template_path):
            chat_template_path = "chat_template.jinja"
    else:
        print(f"Downloading from Hugging Face repo: {path_or_repo}")
        adapter_path = hf_hub_download(repo_id=path_or_repo, filename="adapter_config.json")
        chat_template_path = hf_hub_download(repo_id=path_or_repo, filename="chat_template.jinja")

    adapter_config = json.load(open(adapter_path))
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    chat_template = open(chat_template_path).read()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.chat_template = chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, path_or_repo, is_trainable=False)
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="finetuning_eval")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    save_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset("loris3/ftrace_date-lm_conversational")["train"]
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Step 1: generate responses
    def get_exact_match(examples):
        torch.manual_seed(42)
        np.random.seed(42)
        print(examples["id"])
        user_prompts = [m[0]["content"] for m in examples["messages"]]
        references = [m[1]["content"].strip() for m in examples["messages"]]

        # generate outputs
        output = generator(
            user_prompts,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            return_full_text=False
        )
        print("op", output)
        generated = [item[0]["generated_text"].strip() for item in output]

        return {
            "id": examples["id"],
            "prompt": user_prompts,
            "reference": references,
            "generated": generated
        }

    print("Generating model outputs...")
    results = dataset.map(get_exact_match, batched=True, batch_size=args.batch_size)

    # Step 2: compute exact matches outside
    # Step 2: compute exact matches outside
    import re

    def clean_text(text):
        text = text.strip()  # remove leading/trailing whitespace
        text = re.sub(r"^[^a-zA-Z0-9]+", "", text)  # remove leading non-alphanum (like ":")
        text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
        return text

    print("Computing exact-match accuracy...")

    all_exact = []
    prediction_file = os.path.join(save_dir, "predictions.jsonl")
    with open(prediction_file, "w") as f:
        for idx, prompt, ref, gen in zip(results["id"], results["prompt"], results["reference"], results["generated"]):
            ref_clean = clean_text(ref)
            gen_clean = clean_text(gen)
            is_exact = ref_clean == gen_clean
            all_exact.append(is_exact)
            print(json.dumps({
                "id": idx,
                "prompt": prompt,
                "reference": ref,
                "generated": gen,
                "clean_reference": ref_clean,
                "clean_generated": gen_clean,
                "exact_match": is_exact
            }))
            f.write(json.dumps({
                "id": idx,
                "prompt": prompt,
                "reference": ref,
                "generated": gen,
                "clean_reference": ref_clean,
                "clean_generated": gen_clean,
                "exact_match": is_exact
            }) + "\n")

    final_accuracy = float(np.mean(all_exact))

    # Save metrics
    metrics = {
        "task": "ftrace_date-lm_conversational",
        "model": args.model_path,
        "exact_match_accuracy": final_accuracy,
        "num_samples": len(dataset)
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nExact-Match Accuracy: {final_accuracy:.4f}")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()