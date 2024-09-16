from abc import ABC, abstractmethod

class PriorScheduler(ABC):

    @abstractmethod
    def get_prior_parameters(
        self, 
        train_step: int
    ) -> tuple[float, float, float]:
        pass

class ConstantPriorScheduler(PriorScheduler):

    def __init__(
        self, 
        total_train_steps: int,
        lambda_mix: float, 
        sigma0: float, 
        sigma1: float
    ):
        self.lambda_mix = lambda_mix
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.total_train_steps = total_train_steps
        
    def get_prior_parameters(
        self, 
        train_step: int
    ) -> tuple[float, float, float]:
        return self.lambda_mix, self.sigma0, self.sigma1
        