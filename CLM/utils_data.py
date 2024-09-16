"""
This script provides utility functions to preprocess datasets from Hugging Face's datasets hub for causal language modeling tasks.
It includes functions to load, tokenize, and prepare datasets for training, with options
for handling validation splits, tokenization, and text grouping into fixed-size blocks.
"""

import warnings
from itertools import chain
from datasets import load_dataset

def get_processed_datasets(
    tokenizer,
    args
):
    # Load the raw dataset first
    raw_datasets = load_dataset(
        path=args.dataset_name, 
        name=args.dataset_config_name,
        trust_remote_code=not args.not_trust_remote_code,
        cache_dir=args.cache_dir
    )

    # Get the column names of the dataset
    column_names = raw_datasets["train"].column_names

    # get the text column name if it exists, otherwise use the first column, this is used for tokenization
    text_column_name = "text" if "text" in column_names else column_names[0]

    # define a tokenize function for the dataset to map
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    # if validation set does not exist, create it by splitting the training data
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            path=args.dataset_name,
            name=args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",
            trust_remote_code=not args.not_trust_remote_code,
            cache_dir=args.cache_dir
        )
        raw_datasets["train"] = load_dataset(
            path=args.dataset_name,
            name=args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
            trust_remote_code=not args.not_trust_remote_code,
            cache_dir=args.cache_dir
        )

    # tokenize the dataset
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    # determine the block size
    if args.block_size is None:
        warnings.warn(
            f"A block_size was not provided. Using the minimum of 1024 and the model_max_length ({tokenizer.model_max_length}) instead. "
            f"You can change this default value by passing --block_size xxx."
        )
        block_size = min(1024, tokenizer.model_max_length)
    else:
        if args.block_size > tokenizer.model_max_length:
            warnings.warn(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # define a function to group the texts into chunks of block_size
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # group the texts into chunks of block_size
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets