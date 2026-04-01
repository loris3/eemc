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



# def load_model_and_tokenizer(path_or_repo):
#     if os.path.exists(path_or_repo):
#         print(f"Loading model from local folder: {path_or_repo}")
#         adapter_path = os.path.join(path_or_repo, "adapter_config.json")
#         chat_template_path = os.path.join(path_or_repo, "chat_template.jinja")
#         if not os.path.exists(chat_template_path):
#             chat_template_path = "chat_template.jinja"
#     else:
#         print(f"Downloading from Hugging Face repo: {path_or_repo}")
#         adapter_path = hf_hub_download(repo_id=path_or_repo, filename="adapter_config.json")
#         chat_template_path = hf_hub_download(repo_id=path_or_repo, filename="chat_template.jinja")

#     adapter_config = json.load(open(adapter_path))
#     base_model_name_or_path = adapter_config["base_model_name_or_path"]
#     chat_template = open(chat_template_path).read()

#     tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
#     tokenizer.chat_template = chat_template
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
#     model.resize_token_embeddings(len(tokenizer))
#     model = PeftModel.from_pretrained(model, path_or_repo, is_trainable=False)
#     model.eval()

#     return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="finetuning_eval")
    parser.add_argument("--dataset", default="finetuning_eval", required=True)
    parser.add_argument("--split", default="finetuning_eval", required=True)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/")).replace("-merged","")
  


    dataset = load_dataset(args.dataset)[args.split]
    
    if args.max_samples:
        dataset = dataset.select(range(args.max_samples))


    model, tokenizer = load_model_and_tokenizer(args.model_path.replace("-merged",""))
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
        references = [m[1]["content"].strip() for m in examples["messages"]]
        # Generate outputs
        output = generator(
            user_prompts,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=512 if "hard-coded" in args.dataset else 128,
            top_p=1.0,
            return_full_text=False
        )
  
        generated = [item[0]["generated_text"].strip() for item in output]
        answers_list = [{"text": [ref], "answer_start": [0]} for ref in references] if "answers" not in examples else examples["answers"]
        return {
            "id": examples["id"],
            "prediction_text": generated,
            
            "answers": answers_list
        }

   

    print("Generating model outputs...")
    results = dataset.map(get_generation, batched=True, batch_size=args.batch_size)


    REFUSAL = "cannot be answered"

    def normalize_prediction(pred):
        if REFUSAL in pred.lower():
            return "", 1
        return pred.strip(), 0

    predictions_list = []
    for i_id, pred in zip(results["id"], results["prediction_text"]):
        text, na_prob = normalize_prediction(pred)
        predictions_list.append({
            "id": i_id,
            "prediction_text": text,
            "no_answer_probability": na_prob
        })

    references_list = [
        {"id": i_id, "answers": ans,}
        for i_id, ans in zip(
            results["id"], results["answers"]
        )
    ]

    squad_metric = load("squad_v2")
    results = squad_metric.compute(predictions=predictions_list, references=references_list)


    print("Evaluation results:", results)

    task_name = None
    if "ftrace" in args.dataset:
        task_name = "qa:ftrace_conversational"
    if "squad_v2" in args.dataset:
        task_name = "qa:squad_v2_conversational"
    if "hard-coded" in args.dataset:
        task_name = "qa:hardcoded"
        
    save_dir = os.path.join(args.output_dir, task_name, model_name)
    os.makedirs(save_dir, exist_ok=True)
    predictions_file = os.path.join(save_dir, "predictions.jsonl")
    with open(predictions_file, "w") as f:
        for item in predictions_list:
            f.write(json.dumps(item) + "\n")

    # --- Save metrics to file ---
    all_primary_scores = [f"{model_name}:{task_name}: {results['f1']}"]
    tasks = [{
        "alias": f"{model_name}:{task_name}",
        "metrics": {
            "exact": results["exact"],
            "f1": results["f1"],
            "HasAns_exact": results.get("HasAns_exact", None),
            "HasAns_f1": results.get("HasAns_f1", None),
            "NoAns_exact": results.get("NoAns_exact", None),
            "NoAns_f1": results.get("NoAns_f1", None),
            "primary_score": results["f1"]  # using f1 as primary
        },
        "num_instances": len(dataset),
        "task_config": {
            "task_name": task_name,
            "dataset": args.dataset,
            "split": args.split,
            "primary_metric": "f1"
        }
    }]
    metrics_file = os.path.join(save_dir, "metrics.json")
    metrics = {
        "all_primary_scores": all_primary_scores,
        "tasks": tasks,
        "model_config": {"model": args.model_path}
    }
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nPredictions saved to: {predictions_file}")
    print(f"Metrics saved to: {metrics_file}")



if __name__ == "__main__":
    main()
    
# python3 qa_eval.py     --model_path ./models/OLMo-2-0425-1B_ftrace_date-lm_conversational_lr0.0001_seed42     --dataset loris3/ftrace_date-lm_conversational --split=train    

# python3 qa_eval.py     --model_path ./models/OLMo-2-0425-1B_squad_v2_conversational_lr1e-05_seed42     --dataset loris3/squad_v2_conversational --split=validation    
# python3 qa_eval.py     --model_path ./models/OLMo-2-0425-1B_tulu-3-hard-coded-10x_lr1e-05_seed42     --dataset allenai/tulu-3-hard-coded-10x --split=train    