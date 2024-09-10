from abc import ABC, abstractmethod

class PriorScheduler(ABC):

    @abstractmethod
    def get_prior_parameters(self, train_step: int) -> tuple[float, float, float]:
        pass