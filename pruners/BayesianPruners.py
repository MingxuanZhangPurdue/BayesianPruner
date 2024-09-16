import torch
import math
from typing import Optional, List, Union, Tuple

from .utils import _compile_target_modules
from .SparsitySchedulers import SparsityScheduler
from .PriorSchedulers import PriorScheduler

from composer.core import Algorithm, Event

class UnstructuredBayesianPruner(Algorithm):

    def __init__(
        self,
        train_size: int,
        sparsity_scheduler: SparsityScheduler,
        prior_scheduler: PriorScheduler,
        target_modules: Optional[Union[List[str], str]] = None,
    ):

        self.train_size = train_size

        self.sparsity_scheduler = sparsity_scheduler

        self.prior_scheduler = prior_scheduler

        self.target_modules = _compile_target_modules(target_modules)

        self.mask = None

        self.total_target_params = None

    def is_target_module(
        self, 
        module_name: str
    ) -> bool:

        if self.target_modules is None:
            return True
        return any(pattern.search(module_name) for pattern in self.target_modules)
    
    def print_pruning_modules(
        self, 
        model: torch.nn.Module
    ) -> None:

        print("List of model modules to be pruned:")
        prunable_modules = []
        self.total_target_params = 0

        for name, param in model.named_parameters():
            if self.is_target_module(name):
                prunable_modules.append(name)
                self.total_target_params += param.numel()
        
        if not prunable_modules:
            print("No modules found for pruning.")
        else:
            for module_name in prunable_modules:
                print(f"- {module_name}")
        
        print(f"Total number of candidate parameters for pruning: {self.total_target_params:,}")

    def apply_prior_grad(
        self,
        model: torch.nn.Module,
        lambda_mix: float,
        sigma0: float,
        sigma1: float
    ) -> float:

        c1 = math.log(lambda_mix) - math.log(1 - lambda_mix) + 0.5 * math.log(sigma0) - 0.5 * math.log(sigma1)
        c2 = 0.5 / sigma0 - 0.5 / sigma1
        
        factor1 = (sigma0 - sigma1) / (self.train_size * sigma0 * sigma1)
        factor2 = -1 / (self.train_size * sigma1)

        prior_threshold = math.sqrt(math.log((1 - lambda_mix) / lambda_mix * math.sqrt(sigma1 / sigma0)) / (
            0.5 / sigma0 - 0.5 / sigma1))
        
        with torch.no_grad():
            for n, p in model.named_parameters():
                if self.is_target_module(n):
                    p.grad.sub_(
                        p.mul(
                            p.pow(2).mul(c2).add(c1).exp().add(1).pow(-1)
                            .mul(factor1).add(factor2)
                        )
                    )

        return prior_threshold
    
    def calculate_pruning_threshold(
        self, 
        model: torch.nn.Module,
        sparsity: float
    ) -> Tuple[float, dict]:

        is_dict = {n: p.detach().abs() for n, p in model.named_parameters() if self.is_target_module(n)}
        all_values = torch.cat([tensor.view(-1) for tensor in is_dict.values()])        
        pruning_threshold = torch.kthvalue(all_values, int(all_values.shape[0] * sparsity))[0].item()
        return pruning_threshold, is_dict   
    
    def create_pruning_mask(
        self, 
        model: torch.nn.Module,
        pruning_threshold: float, 
        is_dict: dict
    ) -> dict:

        mask = {}
        for n, _ in model.named_parameters():
            if self.is_target_module(n):
                mask[n] = (is_dict[n] < pruning_threshold)
        return mask
    
    def prune_with_sparsity(
        self, 
        model: torch.nn.Module,
        sparsity: float
    ) -> Tuple[float, dict]:

        pruning_threshold, is_dict = self.calculate_pruning_threshold(model, sparsity)
        mask = self.create_pruning_mask(model, pruning_threshold, is_dict)
        for n, p in model.named_parameters():
            if self.is_target_module(n):
                p.detach().masked_fill_(mask[n], 0.0)
        return pruning_threshold, mask
    
    def prune_with_mask(
        self, 
        model: torch.nn.Module,
        mask: dict[str, bool]
    ) -> None:
        
        for n, p in model.named_parameters():
            if self.is_target_module(n):
                p.detach().masked_fill_(mask[n], 0.0)

    def prune(
        self, 
        model: torch.nn.Module,
        train_step: int
    ) -> Tuple[float, float, Optional[str]]:

        sparsity, pruning_action = self.sparsity_scheduler.get_sparsity(train_step)
        mask = None
        pruning_threshold = None

        if pruning_action == "sparsity" and sparsity > 0.0:
            pruning_threshold, mask = self.prune_with_sparsity(model, sparsity)
            self.mask = mask
        elif pruning_action == "mask" and self.mask is not None:
            self.prune_with_mask(model, self.mask)
        elif pruning_action == "mask" and self.mask is None:
            raise ValueError("Mask is not set. Please set the mask first.")
        elif pruning_action is None:
            pruning_threshold = None
        else:
            raise ValueError(f"Invalid pruning action: {pruning_action}")

        return sparsity, pruning_threshold, pruning_action, mask
    
    def match(self, event, state):
        
        return event in [Event.FIT_START, Event.AFTER_TRAIN_BATCH, Event.BATCH_END]

    def apply(self, event, state, logger):

        if event == Event.FIT_START:
            self.print_pruning_modules(state.model)

        elif event == Event.AFTER_TRAIN_BATCH:
            train_step = state.timestamp.batch.value
            if train_step <= self.sparsity_scheduler.pruning_end_step:
                lambda_mix, sigma0, sigma1 = self.prior_scheduler.get_prior_parameters(train_step)
                prior_threshold = self.apply_prior_grad(state.model, lambda_mix, sigma0, sigma1)
                if logger is not None:
                    logger.log_metrics({"prior/sigma0": float(sigma0)})
                    logger.log_metrics({"prior/sigma1": float(sigma1)})
                    logger.log_metrics({"prior/lambda_mix": float(lambda_mix)})
                    logger.log_metrics({"prior/prior_threshold": float(prior_threshold)})

        elif event == Event.BATCH_END:
            train_step_index = state.timestamp.batch.value - 1
            # perform pruning
            sparsity, pruning_threshold, pruning_action, mask = self.prune(state.model, train_step_index)
            # log the current sparsity
            if logger is not None:
                logger.log_metrics({"pruning/sparsity": float(sparsity)})
                # if the current pruning threshold is not None, log the current pruning threshold
                if pruning_threshold is not None:
                    logger.log_metrics({"pruning/pruning_threshold": float(pruning_threshold)})

