from datasets import load_dataset
from multiprocessing import cpu_count

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
        trust_remote_code=not args.not_trust_remote_code,
        cache_dir="./cache",
        num_proc=cpu_count() - 1,
    )

     # Labels
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    def preprocess_function(examples, tokenizer, padding, max_length):

        # Get the sentence keys for the task
        sentence1_key, sentence2_key = task_to_keys[args.task_name]

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
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        overwrite_cache=args.overwrite_cache,
    )

    return processed_datasets, num_labels, label_list
