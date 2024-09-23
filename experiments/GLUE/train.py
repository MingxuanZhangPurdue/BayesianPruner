import argparse
import yaml

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)

import composer
from composer.utils.dist import get_sampler
from composer.utils import reproducibility
from composer.models.huggingface import HuggingFaceModel
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

     # misc arguments
    parser.add_argument("--not_trust_remote_code", action="store_true", help="Do not trust remote code when loading the tokenizer.")

    # caching arguments
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Where to store the pretrained models and datasets")
    parser.add_argument("--overwrite_cache", action="store_true",  help="Overwrite the cached processed training and evaluation sets")

    # model arguments
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float16", "bfloat16", "float32"], help="Specify the dtype of the model.")

    # datasets arguments
    parser.add_argument("--task_name", default="mnli", choices=["mrpc", "stsb", "sst2", "qqp", "mnli", "qnli", "rte"], type=str, help="The name of the GLUE task.")

    parser.add_argument("--not_use_fast", action="store_true", help="Do not use fast tokenizer when loading the tokenizer.")
    parser.add_argument("--preprocessing_num_workers", default=None, type=int, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--max_length", default=None, type=int, help="The size of the blocks to group the texts into.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="Pad to the maximum length of the dataset.")

    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for the training dataloader.")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of workers to use for the training dataloader.")

    # training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_train_microbatch_size", type=int, default=None, help="The micro-batch size to use for training (per device). If not specified, will use automatic microbatching.")
    parser.add_argument("--precision", type=str, default="amp_fp16", help="The training precision to use, can be fp32, amp_fp16, or amp_bf16.", choices=[None, "amp_fp16", "amp_bf16"])
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--max_duration", type=str, default="1ep", help="Total number of training epochs/batches/steps to perform.")

    # checkpointing
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the checkpoints.")
    parser.add_argument("--save_interval",  type=str, default="1dur", help="Interval to save the checkpoints.")
    parser.add_argument("--autoresume", action="store_true", help="If passed, will resume the latest checkpoint if any.")
    parser.add_argument("--save_overwrite", action="store_true", help="If passed, will overwrite the checkpoints if any.")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the checkpoint.")

    # evaluation
    parser.add_argument("--eval_interval", type=str, default="1dur", help="Interval to evaluate the model.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")

    # wandb logging
    parser.add_argument("--wandb_project_name", type=str, default="BayesianPruning", help="The wandb project to log to.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="The wandb run name.")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

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

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, 
        use_fast=not args.not_use_fast,
        trust_remote_code=not args.not_trust_remote_code,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # get the processed datasets
    processed_datasets, num_labels, label_list = get_processed_datasets(
        tokenizer,
        args
    )

    # load the config and model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=not args.not_trust_remote_code,
        cache_dir=args.cache_dir,
    )
    config.pad_token_id = tokenizer.pad_token_id

    torch_dtype = torch_dtype_map.get(args.torch_dtype, "auto")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=not args.not_trust_remote_code,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )

    if not args.task_name == "stsb":
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        if args.precision == "amp_fp16" or args.precision == "amp_bf16":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)


    train_sampler = get_sampler(
        train_dataset, 
        shuffle=True
    )
    eval_sampler = get_sampler(
        eval_dataset, 
        shuffle=False
    )

    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False
    )

    # initialize the composer model
    composer_model = HuggingFaceModel(
        model, 
        use_logits=True,
    )

    # initialize the optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        composer_model.parameters(), 
        lr=args.learning_rate, 
        betas=[0.9, 0.95], 
        eps=1e-8,
        weight_decay=0.0,
        fused=True
    )

    # calculate the total number of training steps
    train_time = composer.Time.from_timestring(args.max_duration)
    if train_time.unit == composer.TimeUnit.EPOCH:
        total_train_steps = len(train_dataloader) * train_time.value
    elif train_time.unit == composer.TimeUnit.BATCH:
        total_train_steps = train_time.value
    else:
        raise ValueError(f"Unsupported time unit: {train_time.unit}")
    
    # initialize the lr scheduler
    lr_scheduler = getattr(
        composer.optim, 
        config["LRScheduler"]["name"]
    )(**config["LRScheduler"]["params"])

    # initialize the sparsity scheduler
    sparsity_scheduler = getattr(
        pruners, 
        config["SparsityScheduler"]["name"]
    )(
        total_train_steps=total_train_steps,
        **config["SparsityScheduler"]["params"]
    )

    # initialize the prior scheduler
    prior_scheduler = getattr(
        pruners, 
        config["PriorScheduler"]["name"]
    )(
        total_train_steps=total_train_steps,
        **config["PriorScheduler"]["params"]
    )

    # initialize the pruner algorithm
    pruner =  getattr(
        pruners, 
        config["Pruner"]["name"]
    )(train_size=len(train_dataset),
      sparsity_scheduler=sparsity_scheduler,
      prior_scheduler=prior_scheduler,
      **config["Pruner"]["params"])

    # initialize the wandb logger
    wandb_logger = WandBLogger(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        init_kwargs = {"config": {"args": vars(args), "YAML": config}}
    )

    # initialize the trainer
    trainer = composer.Trainer(
        # training
        model=composer_model,
        train_dataloader=train_dataloader,
        optimizers=optimizer,
        max_duration=args.max_duration,
        device_train_microbatch_size=args.per_device_train_microbatch_size if args.per_device_train_microbatch_size is not None else "auto",
        device='gpu' if torch.cuda.is_available() else 'cpu',
        precision=args.precision,
        schedulers=lr_scheduler,

        # evaluation
        eval_dataloader=eval_dataloader,
        eval_interval=args.eval_interval,

        # logging
        loggers=[wandb_logger],

        # algorithms
        algorithms=[pruner],

        # checkpointing
        run_name=args.run_name,
        save_folder=args.save_folder,
        save_interval=args.save_interval,
        save_overwrite=args.save_overwrite,
        autoresume=args.autoresume,
        load_path=args.load_path,

        # reproducibility
        seed=args.seed,
    )

    # start the training
    trainer.fit()


if __name__ == "__main__":

    main()

