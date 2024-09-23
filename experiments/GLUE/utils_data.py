from datasets import load_dataset

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_processed_datasets(
    tokenizer,
    args
):
    # Load the raw datasets first
    raw_datasets = load_dataset(
        "nyu-mll/glue", 
        args.task_name,
        trust_remote_code=True,
        cache_dir="./cache"
    )

     # Labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def preprocess_function(examples, tokenizer, padding, max_length):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)

        if "label" in examples:
            result["labels"] = examples["label"]
        return result
    
    processed_datasets = raw_datasets.map(
        preprocess_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": "max_length" if args.pad_to_max_length else False,
            "max_length": args.max_length
        },
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    return processed_datasets, num_labels, label_list
