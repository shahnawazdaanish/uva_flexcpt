from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gpytorch
import torch
from torch.quasirandom import SobolEngine


@dataclass
class SolidPlanningConfig:
    num_candidates: int = 512
    num_steps: int = 3
    local_scale: float = 0.08
    distance_weight: float = 1.0
    enforce_feasible_sampling: bool = True
    include_local_points: bool = True
    local_fraction: float = 0.25


class SolidEIGPlanner:
    """Simplified, solid planner for reusing saved LCM multitask GPs.

    Design rules:
    1. All model evaluation stays in the scaled feature space used during training.
    2. Each saved model is evaluated only on its own feature subset (`input_cols`).
    3. Feasibility checks use engineering-space coordinates after inverse-scaling.
    4. The planner selects a short trajectory of N next points greedily using EIG.
    """

    def __init__(self, bounds, constraints=None, scaler_x=None, feature_names=None):
        raw_bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if raw_bounds.ndim != 2 or raw_bounds.shape[0] not in (2,):
            raise ValueError("bounds must be a tensor shaped as [2, d].")

        self.bounds = raw_bounds if raw_bounds.shape[0] == 2 else raw_bounds.T
        self.lower = self.bounds[0]
        self.upper = self.bounds[1]
        self.span = self.upper - self.lower
        self.constraints = constraints
        self.scaler_x = scaler_x
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model_groups: List[Dict] = []
        self.distance_weight = 1.0
        self.selected_history: List[torch.Tensor] = []
        self.last_candidate_pool = None
        self.last_sobol_pool = None
        self.last_local_pool = None
        self.last_selected_path = None

    def set_model_groups(self, model_groups: Dict[str, Dict]):
        self.model_groups = []
        for key, group in model_groups.items():
            model = group.get("model")
            likelihood = group.get("likelihood")
            if model is None or likelihood is None:
                continue
            self.model_groups.append(
                {
                    "key": key,
                    "model": model.eval(),
                    "likelihood": likelihood.eval(),
                    "input_cols": list(group.get("input_cols", [])),
                    "task_cols": list(group.get("task_cols", [])),
                    "score_mean_r2": group.get("score_mean_r2", None),
                }
            )
        return self

    def _scaled_to_original(self, x_scaled):
        if self.scaler_x is None:
            return x_scaled
        scaler_obj = getattr(self.scaler_x, "scaler", self.scaler_x)
        if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
            return x_scaled

        mean = torch.as_tensor(scaler_obj.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
        scale = torch.as_tensor(scaler_obj.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
        return x_scaled * scale + mean

    def _original_to_scaled(self, x_original):
        if self.scaler_x is None:
            return x_original
        scaler_obj = getattr(self.scaler_x, "scaler", self.scaler_x)
        if not hasattr(scaler_obj, "mean_") or not hasattr(scaler_obj, "scale_"):
            return x_original

        mean = torch.as_tensor(scaler_obj.mean_, dtype=x_original.dtype, device=x_original.device)
        scale = torch.as_tensor(scaler_obj.scale_, dtype=x_original.dtype, device=x_original.device)
        return (x_original - mean) / scale

    def _resolve_model_indices(self, group, full_dim):
        model_cols = list(group.get("input_cols", []))
        if self.feature_names is None:
            if len(model_cols) <= full_dim:
                return list(range(len(model_cols)))
            return list(range(full_dim))

        if not model_cols:
            return list(range(full_dim))

        idxs = [self.feature_names.index(col) for col in model_cols if col in self.feature_names]
        if not idxs:
            return list(range(min(full_dim, len(model_cols))))
        return idxs

    def _candidate_pool(self, current: torch.Tensor, count: int, local_scale: float):
        dim = self.lower.numel()
        sobol = SobolEngine(dimension=dim, scramble=True)
        base = self.lower + self.span * sobol.draw(count)
        base = base.to(dtype=current.dtype, device=current.device)

        if not self.model_groups:
            self.last_sobol_pool = base.clone()
            self.last_local_pool = None
            self.last_candidate_pool = base.clone()
            return base

        if self.constraints is not None and self.constraints.feasible_mask is not None:
            orig = self._scaled_to_original(base)
            feasible = self.constraints.feasible_mask(orig)
            if torch.any(feasible):
                base = base[feasible]
                if base.shape[0] == 0:
                    fallback = self.lower + self.span * torch.rand(count, dim, device=current.device, dtype=current.dtype)
                    self.last_sobol_pool = fallback.clone()
                    self.last_local_pool = None
                    self.last_candidate_pool = fallback.clone()
                    return fallback

        if not self.include_local_points or local_scale <= 0:
            self.last_sobol_pool = base.clone()
            self.last_local_pool = None
            self.last_candidate_pool = base.clone()
            return base

        local_count = max(1, int(round(count * self.local_fraction)))
        noise = torch.randn((local_count, dim), dtype=current.dtype, device=current.device)
        local = (current.view(1, -1) + noise * (self.span * local_scale)).clamp(self.lower, self.upper)

        if self.constraints is not None:
            local_orig = self._scaled_to_original(local)
            feasible = self.constraints.feasible_mask(local_orig)
            if torch.any(feasible):
                local = local[feasible]
            else:
                local = local[: min(local_count, max(1, local_count // 2))]

        if local.numel() == 0:
            combined = base[:count]
            self.last_sobol_pool = base.clone()
            self.last_local_pool = None
            self.last_candidate_pool = combined.clone()
            return combined

        combined = torch.cat([base[: max(1, count - local.shape[0])], local], dim=0)[:count]
        self.last_sobol_pool = base.clone()
        self.last_local_pool = local.clone()
        self.last_candidate_pool = combined.clone()
        return combined

    def _logdet_eig(self, covariance: torch.Tensor, noise: torch.Tensor):
        covariance = covariance.to(dtype=torch.float64)
        noise = noise.to(dtype=torch.float64)
        if covariance.dim() == 3:
            covariance = covariance[0]
        if covariance.dim() == 1:
            covariance = covariance.unsqueeze(0).unsqueeze(0)

        if covariance.shape[-1] == 0:
            return torch.tensor(0.0, dtype=torch.float64, device=covariance.device)

        if noise.dim() == 0:
            noise = torch.full((covariance.shape[-1],), fill_value=float(noise), device=covariance.device, dtype=torch.float64)
        else:
            noise = noise.reshape(-1)
            if noise.numel() == 1:
                noise = torch.full((covariance.shape[-1],), fill_value=float(noise), device=covariance.device, dtype=torch.float64)
            if noise.numel() != covariance.shape[-1]:
                noise = torch.full((covariance.shape[-1],), fill_value=float(noise.mean()), device=covariance.device, dtype=torch.float64)

        matrix = torch.eye(covariance.shape[-1], device=covariance.device, dtype=torch.float64)
        matrix = matrix + covariance / noise.clamp_min(1e-8).view(1, -1)
        sign, logdet = torch.linalg.slogdet(matrix)
        if sign <= 0:
            return torch.tensor(-1e6, dtype=torch.float64, device=covariance.device)
        return logdet

    def _evaluate_point_eig(self, candidate_points: torch.Tensor) -> torch.Tensor:
        if candidate_points.ndim == 1:
            candidate_points = candidate_points.unsqueeze(0)

        scores = torch.zeros(candidate_points.shape[0], dtype=torch.float64, device=candidate_points.device)
        for group in self.model_groups:
            model = group["model"]
            likelihood = group["likelihood"]
            model_cols = group["input_cols"]
            if not model_cols:
                continue

            feature_index = self._resolve_model_indices(group, self.lower.numel())
            if len(feature_index) == 0:
                continue

            x_model = candidate_points[:, feature_index].to(device=candidate_points.device, dtype=next(model.parameters()).dtype)
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.fast_pred_var(False):
                predictive = model.posterior(x_model) if hasattr(model, "posterior") else model(x_model)
                dist = predictive.distribution if hasattr(predictive, "distribution") else predictive
                cov = dist.covariance_matrix

            if cov.ndim == 3:
                cov = cov.squeeze(1)
            if cov.ndim == 2 and cov.shape[0] != cov.shape[1]:
                cov = cov.unsqueeze(0)

            noise = likelihood.noise
            if torch.is_tensor(noise):
                noise = noise.to(device=cov.device, dtype=cov.dtype)
            else:
                noise = torch.tensor(float(noise), device=cov.device, dtype=cov.dtype)

            if cov.dim() == 3:
                cov_list = [cov[i] for i in range(cov.shape[0])]
                eig_list = [self._logdet_eig(cov_i, noise) for cov_i in cov_list]
                scores += torch.tensor(eig_list, device=candidate_points.device, dtype=torch.float64)
            else:
                scores += self._logdet_eig(cov, noise).repeat(candidate_points.shape[0])

        if not self.model_groups:
            return scores
        return scores / float(len(self.model_groups))

    def _feasible_filter(self, candidate_points: torch.Tensor):
        if self.constraints is None:
            return candidate_points
        candidate_points_orig = self._scaled_to_original(candidate_points)
        mask = self.constraints.feasible_mask(candidate_points_orig)
        if torch.any(mask):
            return candidate_points[mask]
        return candidate_points

    def _trajectory_score(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.dim() == 1:
            trajectory = trajectory.unsqueeze(0)
        if trajectory.shape[0] == 0:
            return torch.tensor(-1e12, dtype=torch.float64)

        if self.constraints is not None:
            orig = self._scaled_to_original(trajectory)
            feasible = self.constraints.feasible_mask(orig)
            if not torch.all(feasible):
                return torch.tensor(-1e12, dtype=torch.float64, device=trajectory.device)

        step_scores = self._evaluate_point_eig(trajectory)
        path_gain = step_scores.sum()
        if trajectory.shape[0] > 1:
            path_step_dist = torch.linalg.norm(trajectory[1:] - trajectory[:-1], dim=-1).sum()
            path_gain = path_gain - self.distance_weight * path_step_dist
        return path_gain

    def _generate_candidate_paths(self, current: torch.Tensor, num_steps: int, num_paths: int):
        paths = []
        for _ in range(max(1, num_paths)):
            path = [current.clone()]
            prev = current.clone()
            for _ in range(num_steps):
                pool = self._candidate_pool(
                    prev,
                    count=max(32, min(256, num_paths * 2)),
                    local_scale=self.local_scale if hasattr(self, "local_scale") else 0.08,
                )
                pool = self._feasible_filter(pool)
                if pool.shape[0] == 0:
                    pool = self.lower + self.span * torch.rand(
                        max(32, min(256, num_paths * 2)),
                        self.lower.numel(),
                        dtype=current.dtype,
                        device=current.device,
                    )

                scores = self._evaluate_point_eig(pool)
                if self.constraints is not None:
                    feasible = self.constraints.feasible_mask(self._scaled_to_original(pool))
                    if torch.any(feasible):
                        scores = scores.clone()
                        scores[~feasible] = -torch.inf

                if self.distance_weight > 0:
                    dist = torch.linalg.norm(pool - prev.view(1, -1), dim=-1)
                    scores = scores - self.distance_weight * dist

                best_idx = int(torch.argmax(scores))
                prev = pool[best_idx].clone()
                path.append(prev.clone())

            paths.append(torch.stack(path))

        return paths

    def plan_next_points(self, current_location, num_steps: Optional[int] = None, num_candidates: Optional[int] = None):
        current = torch.as_tensor(current_location, dtype=torch.float64).squeeze().clone()
        if current.dim() == 0:
            current = current.unsqueeze(0)

        if num_steps is None:
            num_steps = self.num_steps if hasattr(self, "num_steps") else 3
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")

        if num_candidates is None:
            num_candidates = self.num_candidates if hasattr(self, "num_candidates") else 512

        candidate_paths = self._generate_candidate_paths(
            current=current,
            num_steps=num_steps,
            num_paths=max(16, min(128, num_candidates // 4)),
        )
        trajectory_scores = torch.stack([self._trajectory_score(path) for path in candidate_paths])
        best_idx = int(torch.argmax(trajectory_scores))
        selected = candidate_paths[best_idx]
        self.selected_history = [p.clone() for p in selected]
        self.last_selected_path = selected.clone()
        return selected

    def plan(self, current_location, num_steps=3, num_candidates=512):
        out = self.plan_next_points(current_location=current_location, num_steps=num_steps, num_candidates=num_candidates)
        return out


def build_solid_planner(bounds, model_groups, constraints=None, scaler_x=None, feature_names=None, config=None):
    planner = SolidEIGPlanner(bounds=bounds, constraints=constraints, scaler_x=scaler_x, feature_names=feature_names)
    planner.set_model_groups(model_groups)
    planner.num_steps = config.num_steps if config is not None else 3
    planner.num_candidates = config.num_candidates if config is not None else 512
    planner.local_scale = config.local_scale if config is not None else 0.08
    planner.include_local_points = config.include_local_points if config is not None else True
    planner.local_fraction = config.local_fraction if config is not None else 0.25
    planner.distance_weight = getattr(config, "distance_weight", 1.0) if config is not None else 1.0
    return planner
