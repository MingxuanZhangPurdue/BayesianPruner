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

def maiu():

    args, config = parse_args()
