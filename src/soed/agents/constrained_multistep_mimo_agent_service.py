from __future__ import annotations

import torch

from src.soed.constraints.engineering_constraints import EngineeringConstraints
from src.soed.services.planner_service import PlanningConfig, PlannerService
from src.soed.services.solid_eig_planner import SolidEIGPlanner, SolidPlanningConfig, build_solid_planner

# TASKS_SET_A = ['CH4', 'NMHC', 'CO'] 
# TASKS_SET_B = ['Pmax', 'PRR4_max', 'CA50'] 
# TASKS_SET_C = ['IEMP', 'ITE', 'Lambda', 'CO2']
# TASKS_SET_D = ['Nox']

class ConstrainedMultiStepMIMOAgentService:

    def __init__(
        self,
        bounds,
        scaler_x=None,
        constraints:EngineeringConstraints=None,
    ):
        raw_bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if raw_bounds.ndim != 2 or 2 not in raw_bounds.shape:
            raise ValueError("bounds must be a 2D tensor-like of shape [2, d] or [d, 2]")

        self.bounds = raw_bounds if raw_bounds.shape[0] == 2 else raw_bounds.T
        self.scaler_x = scaler_x
        self.constraints = constraints

        self.model_group_keys = ['A', 'B', 'C', 'D']

        self.X = None
        self.Y = None
        self.models = []
        self.likelihoods = []
        self.task_cols = []
        self.input_cols = []
        self.planner_service = None
        self.main_inputs = None
        self.active_planner = None

    def set_trained_models(self, model_group = None, main_inputs=None):
        self.main_inputs = main_inputs
        if model_group is None or not isinstance(model_group, dict):
            raise ValueError("model_group must be a dictionary of trained models.")

        for group_key in self.model_group_keys:
            model_state = model_group.get(group_key)
            if model_state is None:
                raise ValueError(f"Model group '{group_key}' is missing in model_group.")

            model = model_state['model'].to(torch.float64)
            likelihood = model_state['likelihood'].to(torch.float64)
            self.models.append(model)
            self.likelihoods.append(likelihood)
            self.task_cols.append(model_state['task_cols'])
            self.input_cols.append(model_state['input_cols'])

        return self.initialize_planner_service()

    def initialize_planner_service(self):
        if not self.models or not self.likelihoods:
            raise RuntimeError("Models and likelihoods must be set before initializing the planner service.")

        main_input_cols = self.main_inputs if self.main_inputs is not None else None
        indices_to_use = None
        if main_input_cols is not None and len(self.input_cols) > 0:
            model_cols = list(self.input_cols[0])
            indices_to_use = [main_input_cols.index(col) for col in model_cols if col in main_input_cols]
            if not indices_to_use:
                indices_to_use = None
            else:
                print(f"Filtered input columns for planner service: {indices_to_use}")
        else:
            print("No input-column filtering applied; using the full planner input dimension.")

        self.planner_service = PlannerService(
            bounds=self.bounds,
            constraints=self.constraints,
            models=self.models,
            likelihoods=self.likelihoods,
            scaler_x=self.scaler_x,
            filter_input_indices=indices_to_use
        )
        return self

    def _to_original_units(self, x_scaled):
        print(x_scaled)
        if self.scaler_x is None:
            return x_scaled

        scaler_obj = getattr(self.scaler_x, "scaler", self.scaler_x)
        if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
            return x_scaled

        print(x_scaled)
        mean = torch.as_tensor(scaler_obj.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
        scale = torch.as_tensor(scaler_obj.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
        return x_scaled * scale + mean

    def plan_multistep_batch(
        self,
        current_location,
        q_steps=3,
        num_scenarios=256,
        w_dist=1.0,
        enforce_feasible_sampling=False,
        feasible_margin_weight=25.0,
        model_groups = None
    ):
        if self.planner_service is None:
            raise RuntimeError("The planner service is unavailable. Call fit_data or set_trained_models before planning.")

        # cfg = PlanningConfig(
        #     q_steps=q_steps,
        #     num_scenarios=num_scenarios,
        #     w_dist=w_dist,
        #     feasible_margin_weight=feasible_margin_weight,
        #     enforce_feasible_sampling=enforce_feasible_sampling,
        #     include_local_paths=True,
        #     local_path_fraction=0.10,
        # )

        cfg = SolidPlanningConfig(
            num_candidates=512,
            num_steps=q_steps,
            local_scale=0.08,
            include_local_points=True,
            local_fraction=0.25,
            enforce_feasible_sampling=enforce_feasible_sampling,
        )

        planner = SolidEIGPlanner(
            bounds=self.bounds, 
            constraints=self.constraints, 
            scaler_x=self.scaler_x, 
            feature_names=self.main_inputs,
        )
        planner.set_model_groups(model_groups)
        planner.num_steps = cfg.num_steps if cfg is not None else 3
        planner.num_candidates = cfg.num_candidates if cfg is not None else 512
        planner.local_scale = cfg.local_scale if cfg is not None else 0.08
        planner.include_local_points = cfg.include_local_points if cfg is not None else True
        planner.local_fraction = cfg.local_fraction if cfg is not None else 0.25

        current = torch.as_tensor(current_location, dtype=torch.float64, device=self.bounds.device).squeeze()

        next_points = planner.plan(current_location=current_location, num_steps=cfg.num_steps, num_candidates=cfg.num_candidates)
        self.active_planner = planner
        self.planner_service.last_candidate_pool = planner.last_candidate_pool
        self.planner_service.last_sobol_pool = planner.last_sobol_pool
        self.planner_service.last_local_pool = planner.last_local_pool
        self.planner_service.last_selected_path = planner.last_selected_path
        print(next_points)

        # print(f"Original current location: {self._to_original_units(current).cpu().numpy()}")
        # return self.planner_service.plan(current, cfg)
        return next_points
