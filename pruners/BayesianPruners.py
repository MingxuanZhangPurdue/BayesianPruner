import torch
import math
from typing import Optional, List, Union, Tuple

from .utils import _compile_target_modules
from .SparsitySchedulers import SparsityScheduler
from .PriorSchedulers import PriorScheduler


class UnstructuredBayesianPruner:

    def __init__(
        self,
        model: torch.nn.Module,
        train_size: int,
        sparsity_scheduler: SparsityScheduler,
        prior_scheduler: Optional[PriorScheduler] = None,
        sigma0: float = None,
        sigma1: float = None,
        lambda_mix: float = None,
        target_modules: Optional[Union[List[str], str]] = None,
    ):
        
        if prior_scheduler is None:
            if sigma0 is None or sigma1 is None or lambda_mix is None:
                raise ValueError("When the prior_scheduler is not specified, you must specify the values of sigma0, sigma1, and lambda_mix!")

        self.model = model

        self.train_size = train_size

        self.sparsity_scheduler = sparsity_scheduler

        self.prior_scheduler = prior_scheduler

        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.lambda_mix = lambda_mix

        self.target_modules = _compile_target_modules(target_modules)

        self.mask = None

        self.total_target_params = sum(tensor.numel() for tensor in model.named_parameters() if self.is_target_module(tensor))

    def is_target_module(
        self, 
        module_name: str
    ) -> bool:

        if self.target_modules is None:
            return True
        return any(pattern.search(module_name) for pattern in self.target_modules)
    

    def apply_prior_grad(
        self,
        train_step: int
    ) -> float:
        
        model = self.model

        if self.prior_scheduler is not None:
            lambda_mix, sigma0, sigma1 = self.prior_scheduler(train_step)
        else:
            lambda_mix = self.lambda_mix
            sigma0 = self.sigma0
            sigma1 = self.sigma1

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
        mask: dict
    ) -> None:
        
        for n, p in model.named_parameters():
            if self.is_target_module(n):
                p.detach().masked_fill_(mask[n], 0.0)
   
    def prune(
        self, 
        train_step: int
    ) -> Tuple[float, float, str]:
        
        model = self.model

        sparsity, pruning_action = self.sparsity_scheduler.get_sparsity(train_step)

        if pruning_action == "sparsity" and sparsity > 0.0:
            pruning_threshold, mask = self.prune_with_sparsity(model, sparsity)
            self.mask = mask
        elif pruning_action == "mask" and self.mask is not None:
            pruning_threshold = None
            self.prune_with_mask(model, self.mask)
        elif pruning_action == "mask" and self.mask is None:
            raise ValueError("Mask is not set. Please set the mask first.")
        elif pruning_action is None:
            pruning_threshold = None
        else:
            raise ValueError(f"Invalid pruning action: {pruning_action}")

        return sparsity, pruning_threshold, pruning_action

