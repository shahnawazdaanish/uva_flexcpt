from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def fit_data(self, X, y):
        pass
    
    @abstractmethod
    def plan_multistep_batch(self, current_location, q_steps, w_distance, num_candidates):
        pass