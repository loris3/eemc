
        # answers formatted as SQuAD v2 expects: list of text, list of start indices
        
        return {
            "id": examples["id"],
            "prediction_text": generated,
            "answers": answers_list,
        }

    print("Generating model outputs...")
    dataset_results = dataset.map(get_generation, batched=True, batch_size=args.batch_size)

    # Format predictions and references for squad_v2 metric
    predictions_list = [
        {"id": i_id, "prediction_text": pred, "no_answer_probability": 0.0}
        for i_id, pred in zip(dataset_results["id"], dataset_results["prediction_text"])
    ]

    references_list = [
        {"id": i_id, "answers": ans}
        for i_id, ans in zip(dataset_results["id"], dataset_results["answers"])
    ]

    squad_metric = load("squad_v2")
    results = squad_metric.compute(predictions=predictions_list, references=references_list)

    # Save predictions
    predictions_file = os.path.join(save_dir, "predictions.jsonl")
    with open(predictions_file, "w") as f:
        for item in predictions_list:
            f.write(json.dumps(item) + "\n")

    # Save metrics
    metrics_file = os.path.join(save_dir, "metrics.json")
    metrics = {
        "task": "ftrace_date-lm_conversational",
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