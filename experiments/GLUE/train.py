import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    default_data_collator,
)

import composer
from composer.utils.dist import get_sampler
from composer.utils import reproducibility
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import LanguageCrossEntropy
from composer.loggers import WandBLogger

from .utils_data import get_processed_datasets

import pruners

torch_dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32
}

def parse_args():
    """
    Parses command-line arguments and YAML configuration.

    Returns:
        args: Parsed arguments.
        config: Loaded YAML configuration.
    """

    # YAML config parser
    config_parser = argparse.ArgumentParser(description='YAML Config', add_help=False)
    config_parser.add_argument('-c', '--config', type=str, required=True, metavar='FILE', help='Path to YAML config file.')

    # main parser
    parser = argparse.ArgumentParser(description="Prune and finetune a pretrained model on a GLUE task.")


    args_config, remaining = config_parser.parse_known_args()

    with open(args_config.config, 'r') as f:
        config = yaml.safe_load(f)
        if "Arguments" in config:
            parser.set_defaults(**config["Arguments"])

    args = parser.parse_args(remaining)

    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path is required")
    if args.dataset_name is None:
        raise ValueError("dataset_name is required")

    return args, config

def main():

    # parse the arguments
    args, config = parse_args()

    # set the seed for reproducibility
    reproducibility.seed_all(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=not args.use_slow_tokenizer, 
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed_datasets, num_labels, label_list = get_processed_datasets(args)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
    )
    config.pad_token_id = tokenizer.pad_token_id

    torch_dtype = torch_dtype_map.get(args.torch_dtype, "auto")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype
    )

    if not args.task_name == "stsb":
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
