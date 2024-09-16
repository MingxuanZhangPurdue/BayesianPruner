import argparse
import warnings
import yaml

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

import composer
from composer.utils.dist import get_sampler
from composer.utils import reproducibility
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import LanguageCrossEntropy
from composer.loggers import WandBLogger

from CLM.utils_data import get_processed_datasets

import pruners

torch_dtype_map = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": "auto"
}

def parse_args():

    # YAML config parser
    config_parser = argparse.ArgumentParser(description='YAML Config', add_help=False)
    config_parser.add_argument('-c', '--config', type=str, required=True, metavar='FILE', help='Path to YAML config file.')

    # main parser
    parser = argparse.ArgumentParser(description="Prune and finetune a pretrained model on a causal language modeling task.")

    # caching arguments
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Where to store the pretrained models and datasets")
    parser.add_argument("--overwrite_cache", action="store_true",  help="Overwrite the cached processed training and evaluation sets")

    # model arguments
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--attn_implementation", type=str, default="sdpa", choices=["eager", "flash_attention_2", "sdpa"], help="Attention implementation to use.")
    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Specify the dtype of the model.")

    # datasets arguments
    parser.add_argument("--dataset_name", required=True, type=str, help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use (via the datasets library).")
    parser.add_argument("--validation_split_percentage", default=5, help="The percentage of the train set used as validation set in case there's no validation split")

    # datasets preprocessing arguments
    parser.add_argument("--block_size", type=int, default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="The number of processes to use for the preprocessing.")

    # training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_train_microbatch_size", type=int, default=None, help="The micro-batch size to use for training (per device). If not specified, will use automatic microbatching.")
    parser.add_argument("--precision", type=str, default=None, help="The training precision to use, can be fp32, amp_fp16, or amp_bf16.", choices=[None, "fp32", "amp_fp16", "amp_bf16"])
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
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64, help="Batch size (per device) for the evaluation dataloader.")

    # wandb logging
    parser.add_argument("--wandb_project_name", type=str, default="upstream_bert_base", help="The wandb project to log to.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="The wandb run name.")

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    args_config, remaining = config_parser.parse_known_args()

    with open(args_config.config, 'r') as f:
        config = yaml.safe_load(f)
        if "Arguments" in config:
            parser.set_defaults(**config["Arguments"])

    args = parser.parse_args(remaining)

    return args, config



def main():

    # parse the arguments
    args, config = parse_args()

    # set the seed for reproducibility
    reproducibility.seed_all(args.seed)

    # load the model and tokenizer
    torch_dtype = torch_dtype_map.get(args.torch_dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path, 
        use_fast=True,
        trust_remote_code=True, 
        cache_dir=args.cache_dir
    )
    
    lm_datasets = get_processed_datasets(
        tokenizer=tokenizer,
        args=args
    )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        warnings.warn(f"Resizing embedding size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

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
        collate_fn=default_data_collator, 
        batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        collate_fn=default_data_collator, 
        batch_size=args.per_device_eval_batch_size
    )

    # initialize the composer model
    metrics = [
        LanguageCrossEntropy(),
    ]
    composer_model = HuggingFaceModel(
        model, 
        tokenizer=tokenizer, 
        metrics=metrics, 
        use_logits=True
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

    # initialize the lr scheduler
    lr_scheduler = getattr(
        composer.optim, 
        config["LRScheduler"]["name"]
    )(**config["LRScheduler"]["params"])

    # initialize the sparsity scheduler
    sparsity_scheduler = getattr(
        pruners, 
        config["SparsityScheduler"]["name"]
    )(**config["SparsityScheduler"]["params"])

    # initialize the prior scheduler
    prior_scheduler = getattr(
        pruners, 
        config["PriorScheduler"]["name"]
    )(**config["PriorScheduler"]["params"])

    # initialize the pruner algorithm
    block_size = train_dataset[0]["input_ids"].shape[0]
    pruner =  getattr(
        pruners, 
        config["Pruner"]["name"]
    )(train_size=len(train_dataset)*(block_size-1),
      sparsity_scheduler=sparsity_scheduler,
      prior_scheduler=prior_scheduler,
      **config["Pruner"]["params"])

    # initialize the wandb logger
    wandb_logger = WandBLogger(
        project=args.wandb_project_name,
        name=args.wandb_run_name,
        init_kwargs = {"config": {"args": vars(args), "YAML": vars(config)}}
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
