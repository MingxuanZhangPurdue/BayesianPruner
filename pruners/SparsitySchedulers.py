from abc import ABC, abstractmethod
from typing import Literal, Union

PruningAction = Literal["sparsity", "mask", None]
StepType = Union[int, float]

class SparsityScheduler(ABC):
    @abstractmethod
    def calculate_sparsity(self, train_step: int) -> float:
        pass

    @abstractmethod
    def define_pruning_action(self, train_step: int) -> PruningAction:
        pass

    def get_sparsity_and_action(self, train_step: int) -> tuple[float, PruningAction]:
        sparsity = self.calculate_sparsity(train_step)
        action = self.define_pruning_action(train_step)
        return sparsity, action

class CubicSparsityScheduler(SparsityScheduler):
    def __init__(self, initial_sparsity: float, final_sparsity: float, 
                 total_train_steps: int, pruning_start_step: StepType, 
                 pruning_end_step: StepType, pruning_interval: StepType):

        assert 0 <= initial_sparsity <= final_sparsity < 1, "Sparsity values must be between 0 and 1"
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.total_train_steps = total_train_steps
        self.pruning_start_step = self._convert_to_step(pruning_start_step)
        self.pruning_end_step = self._convert_to_step(pruning_end_step)
        self.pruning_interval = self._convert_to_step(pruning_interval)

    def _convert_to_step(self, value: StepType) -> int:
        if isinstance(value, float) and 0 <= value <= 1:
            return int(value * self.total_train_steps)
        elif isinstance(value, int):
            return value
        else:
            raise ValueError("Step values must be either an integer or a float between 0 and 1")

    def calculate_sparsity(self, train_step: int) -> float:
        if train_step < self.pruning_start_step:
            return 0.0
        elif train_step > self.pruning_end_step:
            return self.final_sparsity
        else:
            frac_of_total = 1 - (train_step - self.pruning_start_step) / (self.pruning_end_step - self.pruning_start_step)
            return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (frac_of_total ** 3)

    def define_pruning_action(self, train_step: int) -> PruningAction:
        if train_step < self.pruning_start_step:
            return None
        elif train_step == self.pruning_end_step:
            return "sparsity"
        elif train_step > self.pruning_end_step:
            return "mask"
        elif (train_step - self.pruning_start_step) % self.pruning_interval == 0:
            return "sparsity"
        else:
            return None

class StepSparsityScheduler(SparsityScheduler):
    def __init__(self, final_sparsity: float, 
                 total_train_steps: int, pruning_start_step: StepType, 
                 pruning_end_step: StepType, sparsity_delta: float):
        
        assert 0 <= sparsity_delta <= final_sparsity < 1, "Sparsity values must be between 0 and 1"
        self.initial_sparsity = sparsity_delta
        self.final_sparsity = final_sparsity
        self.total_train_steps = total_train_steps
        self.pruning_start_step = self._convert_to_step(pruning_start_step)
        self.pruning_end_step = self._convert_to_step(pruning_end_step)
        self.sparsity_delta = sparsity_delta
        self.num_pruning_steps = int((self.final_sparsity - self.initial_sparsity) / self.sparsity_delta)
        self.pruning_interval = (self.pruning_end_step - self.pruning_start_step) // self.num_pruning_steps

    def _convert_to_step(self, value: StepType) -> int:
        if isinstance(value, float) and 0 <= value <= 1:
            return int(value * self.total_train_steps)
        elif isinstance(value, int):
            return value
        else:
            raise ValueError("Step values must be either an integer or a float between 0 and 1")

    def calculate_sparsity(self, train_step: int) -> float:
        if train_step < self.pruning_start_step:
            return 0.0
        elif train_step >= self.pruning_end_step:
            return self.final_sparsity
        else:
            steps_since_start = train_step - self.pruning_start_step
            current_step = steps_since_start // self.pruning_interval
            return min(self.initial_sparsity + current_step * self.sparsity_delta, self.final_sparsity)

    def define_pruning_action(self, train_step: int) -> PruningAction:
        if train_step < self.pruning_start_step:
            return None
        elif train_step == self.pruning_end_step:
            return "sparsity"
        elif train_step > self.pruning_end_step:
            return "mask"
        else:
            if (train_step - self.pruning_start_step) % self.pruning_interval == 0:
                return "sparsity"
            else:
                return None

