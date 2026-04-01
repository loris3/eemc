def tokenize_dataset(
    dataset,
    tokenizer,
    chat_template_path="./chat_template.jinja",
    assistant_only_loss=True,
    text_column="messages",
    num_proc=1,
    re_index=True,
    tokenize=True,
):
    """
    Tokenizes a conversational dataset using a chat template:
    - Uses tokenizer.apply_chat_template with `chat_template.jinja`
    - for assistant_only_loss=True, masks labels so only assistant tokens count in loss
    - adds ordinal 'indices' if re_index=True
    """
    tokenizer.model_max_length = 4096 # TODO remove after retrain
    tokenizer.chat_template = open(chat_template_path).read()
    print("Tokenizer max length:", tokenizer.model_max_length)
    def _tokenize_fn(batch, idx=None, tokenize=True):
        # list of dicts: [{"role": "user", "content": ...}, {"role": "assistant", ...}]
        texts = [
            tokenizer.apply_chat_template(conv, tokenize=False)
            for conv in batch[text_column]
        ]
        if tokenize:
            enc = tokenizer(
                texts,
                truncation=True,
                return_attention_mask=True,
            )

            labels = enc["input_ids"].copy()
            if assistant_only_loss:
                # mask out everything except assistant spans
                for i, conv in enumerate(batch[text_column]):
                    label_mask = [-100] * len(labels[i])
                    rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
                    tokenized_render = tokenizer(rendered, truncation=True)["input_ids"]
                    for turn in conv:
                        if turn["role"] == "assistant":
                            turn_ids = tokenizer(turn["content"], add_special_tokens=False)["input_ids"]
                            for j in range(len(tokenized_render) - len(turn_ids) + 1):
                                if tokenized_render[j : j + len(turn_ids)] == turn_ids:
                                    label_mask[j : j + len(turn_ids)] = turn_ids
                                    break
                    labels[i] = label_mask

            enc["labels"] = labels
        else:
            enc = {"text": texts}
        if re_index and idx is not None:
            enc["indices"] = idx
        return enc

    tokenized = dataset.map(
        lambda batch, idx: _tokenize_fn(batch, idx, tokenize=tokenize),
        with_indices=True,
        batched=True,
        num_proc=num_proc,
        remove_columns=[col for col in dataset.column_names if col != text_column],
    )

    return tokenized
