import torch
import gpytorch
from torch.quasirandom import SobolEngine

from src.soed.constraints.engineering_constraints import EngineeringConstraints


class JointMultiOutputGP:
    """Stable wrapper around independent exact GPs for each output dimension.

    This avoids the gpytorch multitask incompatibility while preserving the cleaner
    separation of model fitting and engineering constraints outside the GP layer.

    The constructor intentionally accepts the older ``num_outputs`` keyword so stale
    notebook imports or earlier prototypes do not crash the fixed planner path.
    """

    def __init__(self, train_X, train_Y, bounds, num_outputs=None, **kwargs):
        from src.soed.dynamic_gp import DynamicGP

        self.models, self.likelihoods = [], []
        for j in range(train_Y.shape[1]):
            yj = train_Y[:, j:j + 1]
            if yj.std() < 1e-14:
                yj = yj + 1e-8 * torch.randn_like(yj)

            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0)
            )
            lik.noise = torch.tensor(1e-2, dtype=torch.float64)
            model = DynamicGP(train_X, yj, lik, bounds)
            model.train(); lik.train()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            opt = torch.optim.Adam(model.parameters(), lr=0.03)
            for _ in range(250):
                opt.zero_grad()
                loss = -mll(model(train_X), yj.squeeze(-1))
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    opt.step()

            model.eval(); lik.eval()
            self.models.append(model)
            self.likelihoods.append(lik)


class ConstrainedMultiStepMIMOAgentFixed:
    """Separate implementation for improved constrained planning logic.

    This file intentionally does not touch the existing planner. It provides a clearer,
    separated version with:
      - explicit geometry-based constraints
      - joint-output GP wrapper
      - candidate scoring based on constrained feasibility
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
        enable_br_limit=True,
        enable_vva_limit=True,
        min_load=3.0,
        ambient_pressure=1.0,
        tc_boost_limit=3.8,
    ):
        raw_bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if raw_bounds.ndim != 2 or 2 not in raw_bounds.shape:
            raise ValueError("bounds must be a 2D tensor-like of shape [2, d] or [d, 2]")

        self.bounds = raw_bounds if raw_bounds.shape[0] == 2 else raw_bounds.T
        self.lower, self.upper = self.bounds[0], self.bounds[1]

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
            enable_br_limit=enable_br_limit,
            enable_vva_limit=enable_vva_limit,
        )

        self.X = None
        self.Y = None
        self.models = []
        self.likelihoods = []

    def fit_data(self, X, Y):
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

        wrapper = JointMultiOutputGP(
            train_X=X,
            train_Y=Y,
            bounds=self.bounds,
            num_outputs=Y.shape[1],
        )
        self.models = wrapper.models
        self.likelihoods = wrapper.likelihoods

    def _to_original_units(self, x_scaled):
        if self.scaler_x is None:
            return x_scaled

        if hasattr(self.scaler_x, 'scaler'):
            scaler_obj = self.scaler_x.scaler
        else:
            scaler_obj = self.scaler_x

        if not hasattr(scaler_obj, 'mean_') or not hasattr(scaler_obj, 'scale_'):
            return x_scaled

        mean = torch.as_tensor(scaler_obj.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
        scale = torch.as_tensor(scaler_obj.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
        return x_scaled * scale + mean

    def _feasible_mask(self, paths_scaled):
        x = self._to_original_units(paths_scaled)
        return self.constraints.feasible_mask(x)

    def _constraint_penalty(self, paths_scaled):
        x = self._to_original_units(paths_scaled)
        return self.constraints.constraint_penalty(x)

    def _interior_margin_penalty(self, paths_scaled):
        x = self._to_original_units(paths_scaled)
        return self.constraints.interior_margin_penalty(x)

    def _evaluate_joint_information_gain(self, candidate_paths):
        if not self.models or not self.likelihoods:
            raise RuntimeError("The fixed GP models have not been fit yet.")

        total_ig = torch.zeros(candidate_paths.shape[0], dtype=torch.float64, device=candidate_paths.device)
        for model, likelihood in zip(self.models, self.likelihoods):
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.fast_pred_var(False):
                cov = model.posterior(candidate_paths).distribution.covariance_matrix

            if cov.ndim == 4:
                cov = cov.squeeze(0)
            if not torch.isfinite(cov).all():
                total_ig -= 1e6
                continue

            eye = torch.eye(candidate_paths.shape[1], dtype=cov.dtype, device=cov.device)
            approx = cov + 1e-6 * eye
            try:
                diag = torch.linalg.cholesky(approx).diagonal(dim1=-2, dim2=-1)
                ig = torch.log(diag.clamp_min(1e-12)).sum(dim=-1)
            except RuntimeError:
                sign, logdet = torch.linalg.slogdet(approx)
                ig = sign * logdet

            total_ig += torch.where(torch.isfinite(ig), ig, torch.full_like(ig, -1e6))

        return total_ig

    def plan_multistep_batch(
        self,
        current_location,
        q_steps=3,
        num_scenarios=256,
        w_dist=1.0,
        enforce_feasible_sampling=False,
        feasible_margin_weight=25.0,
    ):
        curr = torch.as_tensor(current_location, dtype=self.bounds.dtype, device=self.bounds.device).squeeze()
        sobol = SobolEngine(dimension=self.bounds.shape[1] * q_steps, scramble=True)
        raw = self.lower + (self.upper - self.lower) * sobol.draw(num_scenarios).view(num_scenarios, q_steps, self.bounds.shape[1])

        candidate_paths = raw
        if enforce_feasible_sampling:
            feasible = self._feasible_mask(candidate_paths)
            candidate_paths = candidate_paths[feasible]
            if candidate_paths.shape[0] == 0:
                candidate_paths = raw

        ig = self._evaluate_joint_information_gain(candidate_paths)
        if candidate_paths.shape[0] == 0:
            raise RuntimeError("No candidate paths available after feasibility filtering.")

        dist = lambda x, y: torch.sqrt(((x - y).pow(2)).sum(dim=-1))
        score = ig - w_dist * (
            dist(candidate_paths[:, 0], curr)
            + dist(candidate_paths[:, 1:], candidate_paths[:, :-1]).sum(dim=-1)
        )

        feasible = self._feasible_mask(candidate_paths)
        feasible_score = score.clone() - float(feasible_margin_weight) * self._interior_margin_penalty(candidate_paths)
        feasible_score[~feasible] = -torch.inf

        if feasible.any():
            idx = torch.argmax(feasible_score)
            return candidate_paths[idx]

        soft_score = score - 50.0 * self._constraint_penalty(candidate_paths) - 10.0 * self._interior_margin_penalty(candidate_paths)
        idx = torch.argmax(soft_score)
        return candidate_paths[idx]
