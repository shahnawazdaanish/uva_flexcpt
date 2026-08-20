from __future__ import annotations

import torch

from src.soed.constraints.engineering_constraints import EngineeringConstraints
from src.soed.services.planner_service import PlanningConfig, PlannerService


def _fit_independent_output_models(train_X, train_Y, bounds):
    """Train one GP per output dimension without wrapping them into a joint multitask model."""
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import gpytorch

    from src.soed.dynamic_gp import DynamicGP

    models = []
    likelihoods = []

    for j in range(train_Y.shape[1]):
        yj = train_Y[:, j : j + 1]
        if yj.std() < 1e-14:
            yj = yj + 1e-8 * torch.randn_like(yj)

        likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0))
        likelihood.noise = torch.tensor(1e-2, dtype=torch.float64)

        model = DynamicGP(train_X, yj, likelihood, bounds)
        model.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(250):
            optimizer.zero_grad()
            loss = -mll(model(train_X), yj.squeeze(-1))
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

        model.eval()
        likelihood.eval()
        models.append(model)
        likelihoods.append(likelihood)

    return models, likelihoods


class ConstrainedMultiStepMIMOAgentService:
    """Modern planner agent built around service objects and a configuration object.

    This version does not define or train a JointMultiOutputGP wrapper. It expects the
    models to be trained once elsewhere and then injected into the planner service.
    """

    def __init__(
        self,
        bounds,
        feature_names,
        scaler_x=None,
        mass1_name="Mass1",
        mass2_name="Mass2",
        boost_name="Boost pressure",
        ivo_name="IVO",
        ivc_name="IVC",
        evo_name="EVO",
        evc_name="EVC",
        load_limit=30.0,
        boost_slope=0.0922,
        boost_intercept=0.8378,
        boost_band=0.5,
        min_load=3.0,
        ambient_pressure=1.0,
        tc_boost_limit=3.8,
    ):
        raw_bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if raw_bounds.ndim != 2 or 2 not in raw_bounds.shape:
            raise ValueError("bounds must be a 2D tensor-like of shape [2, d] or [d, 2]")

        self.bounds = raw_bounds if raw_bounds.shape[0] == 2 else raw_bounds.T
        self.feature_names = list(feature_names)
        self.scaler_x = scaler_x

        self.constraints = EngineeringConstraints(
            feature_names=self.feature_names,
            mass1_name=mass1_name,
            mass2_name=mass2_name,
            boost_name=boost_name,
            ivo_name=ivo_name,
            ivc_name=ivc_name,
            evo_name=evo_name,
            evc_name=evc_name,
            load_limit=load_limit,
            boost_slope=boost_slope,
            boost_intercept=boost_intercept,
            boost_band=boost_band,
            min_load=min_load,
            ambient_pressure=ambient_pressure,
            tc_boost_limit=tc_boost_limit,
        )

        self.X = None
        self.Y = None
        self.models = []
        self.likelihoods = []
        self.planner_service = None

    def set_trained_models(self, models, likelihoods):
        self.models = list(models)
        self.likelihoods = list(likelihoods)
        self.planner_service = PlannerService(self.bounds, self.constraints, self.models, self.likelihoods, scaler_x=self.scaler_x)
        return self

    def fit_data(self, X, Y, models=None, likelihoods=None):
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        valid = torch.isfinite(X).all(dim=1) & torch.isfinite(Y).all(dim=1)
        if not valid.any():
            raise ValueError("All training rows contained NaN/Inf values.")

        X = X[valid]
        Y = Y[valid]
        self.X = X
        self.Y = Y

        if models is not None and likelihoods is not None:
            return self.set_trained_models(models, likelihoods)

        self.models, self.likelihoods = _fit_independent_output_models(X, Y, self.bounds)
        self.planner_service = PlannerService(self.bounds, self.constraints, self.models, self.likelihoods, scaler_x=self.scaler_x)
        return self

    def _to_original_units(self, x_scaled):
        if self.scaler_x is None:
            return x_scaled

        scaler_obj = getattr(self.scaler_x, "scaler", self.scaler_x)
        if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
            return x_scaled

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
    ):
        if self.planner_service is None:
            raise RuntimeError("The planner service is unavailable. Call fit_data or set_trained_models before planning.")

        cfg = PlanningConfig(
            q_steps=q_steps,
            num_scenarios=num_scenarios,
            w_dist=w_dist,
            feasible_margin_weight=feasible_margin_weight,
            enforce_feasible_sampling=enforce_feasible_sampling,
            include_local_paths=True,
            local_path_fraction=0.10,
        )
        current = torch.as_tensor(current_location, dtype=torch.float64, device=self.bounds.device).squeeze()
        return self.planner_service.plan(current, cfg)
