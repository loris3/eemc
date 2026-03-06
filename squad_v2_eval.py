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
from evaluate import load



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
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    save_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset("loris3/squad_v2_conversational")["validation"]
    print("dataset",dataset)
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))


    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = 0 if torch.cuda.is_available() else -1


    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )


    def get_generation(examples):
        batch_len = len(examples["id"])
        torch.manual_seed(42)
        np.random.seed(42)


        user_prompts = [
            tokenizer.apply_chat_template(
                [conv[0]],                     
                tokenize=False,
                add_generation_prompt=True     
            )
            for conv in examples["messages"]
        ]

        # Generate outputs
        output = generator(
            user_prompts,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=128,
            top_p=1.0,
            return_full_text=False
        )
        print(output)
        generated = [item[0]["generated_text"].strip() for item in output]

        return {
            "id": examples["id"],
            "prediction_text": generated,
            "no_answer_probability": [0] * batch_len
        }

    def get_reference(examples):
        batch_len = len(examples["id"])
        return {
            "id": examples["id"],
            "answers": examples["answers"],
        }

    print("Generating model outputs...")
    predictions = dataset.map(get_generation, batched=True, batch_size=args.batch_size)
    references = dataset.map(get_reference, batched=True, batch_size=args.batch_size)

    predictions_list = [
        {"id": i_id, "prediction_text": pred, "no_answer_probability": prob}
        for i_id, pred, prob in zip(
            predictions["id"], predictions["prediction_text"], predictions["no_answer_probability"]
        )
    ]

    references_list = [
        {"id": i_id, "answers": ans,}# "no_answer_threshold": th}
        for i_id, ans, #th
        in zip(
            references["id"], references["answers"],# references["no_answer_threshold"]
        )
    ]

    # Compute metric
    squad_metric = load("squad_v2")
    results = squad_metric.compute(predictions=predictions_list, references=references_list)


    print("Evaluation results:", results)

    # --- Save predictions to file ---
    predictions_file = os.path.join(save_dir, "predictions.jsonl")
    with open(predictions_file, "w") as f:
        for item in predictions_list:
            f.write(json.dumps(item) + "\n")

    # --- Save metrics to file ---
    metrics_file = os.path.join(save_dir, "metrics.json")
    metrics = {
        "task": "squad_v2_conversational",
        "model": args.model_path,
        "results": results,
        "num_samples": len(dataset)
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPredictions saved to: {predictions_file}")
    print(f"Metrics saved to: {metrics_file}")



if __name__ == "__main__":
    main()