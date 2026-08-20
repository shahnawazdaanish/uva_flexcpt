import torch

from src.soed.agents.constrained_multistep_mimo_agent import ConstrainedMultiStepMIMOAgent
from src.soed.agents.constrained_multistep_mimo_agent_fixed import ConstrainedMultiStepMIMOAgentFixed


class PlannerComparisonHarness:
    """Simple side-by-side runner for the original vs fixed constrained planner prototypes."""

    def __init__(self, bounds, feature_names, scaler_x=None, **kwargs):
        self.bounds = bounds
        self.feature_names = feature_names
        self.scaler_x = scaler_x
        self.kwargs = kwargs

        self.old_agent = ConstrainedMultiStepMIMOAgent(
            bounds=bounds,
            feature_names=feature_names,
            scaler_x=scaler_x,
            **kwargs,
        )
        self.new_agent = ConstrainedMultiStepMIMOAgentFixed(
            bounds=bounds,
            feature_names=feature_names,
            scaler_x=scaler_x,
            **kwargs,
        )

    def fit_same_data(self, X, Y):
        self.old_agent.fit_data(X, Y)
        self.new_agent.fit_data(X, Y)

    def compare(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0):
        old_path = self.old_agent.plan_multistep_batch(
            current_location=current_location,
            q_steps=q_steps,
            num_scenarios=num_scenarios,
            w_dist=w_dist,
            enforce_feasible_sampling=False,
            enforce_feasible_sobol=False,
            include_local_paths=True,
            local_path_fraction=0.5,
        )

        new_path = self.new_agent.plan_multistep_batch(
            current_location=current_location,
            q_steps=q_steps,
            num_scenarios=num_scenarios,
            w_dist=w_dist,
            enforce_feasible_sampling=False,
            feasible_margin_weight=25.0,
        )

        old_feasible = self.old_agent._feasible_mask(old_path.unsqueeze(0))
        new_feasible = self.new_agent._feasible_mask(new_path.unsqueeze(0))

        return {
            "old_path": old_path,
            "new_path": new_path,
            "old_feasible": bool(old_feasible.item()),
            "new_feasible": bool(new_feasible.item()),
        }
