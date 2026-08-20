from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import gpytorch
import torch
from torch.quasirandom import SobolEngine


@dataclass(frozen=True)
class PlanningConfig:
    q_steps: int = 3
    num_scenarios: int = 256
    w_dist: float = 1.0
    feasible_margin_weight: float = 25.0
    enforce_feasible_sampling: bool = False
    include_local_paths: bool = True
    local_path_fraction: float = 0.10
    soft_penalty_strength: float = 50.0
    interior_penalty_strength: float = 10.0
    distance_scale: float = 1.0

    @property
    def normalized_distance_weight(self) -> float:
        return self.w_dist / max(1, self.q_steps)


class CandidateGenerationService:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, constraints=None, scaler_x=None):
        self.lower = lower
        self.upper = upper
        self.span = self.upper - self.lower
        self.constraints = constraints
        self.scaler_x = scaler_x

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

    def generate_sobol(self, num_scenarios: int, q_steps: int) -> torch.Tensor:
        dim = self.lower.numel()
        sobol = SobolEngine(dimension=dim * q_steps, scramble=True)
        samples = []
        max_attempts = max(2000, 20 * num_scenarios)

        for _ in range(max_attempts):
            raw = sobol.draw(num_scenarios - len(samples))
            cand = self.lower + self.span * raw.view(-1, q_steps, dim)
            if self.constraints is not None:
                cand_orig = self._to_original_units(cand)
                feasible = self.constraints.feasible_mask(cand_orig)
                cand = cand[feasible]
                if cand.numel() > 0:
                    samples.append(cand)
                if len(samples) >= num_scenarios:
                    break
            else:
                samples.append(cand)
                if len(samples) >= num_scenarios:
                    break

        if not samples:
            return self.lower + self.span * torch.rand(num_scenarios, q_steps, dim)

        out = torch.cat(samples, dim=0)[:num_scenarios]
        return out

    def _feasible_subset(self, paths: torch.Tensor) -> torch.Tensor:
        if self.constraints is None:
            return paths

        if hasattr(self.constraints, '_to_original_values'):
            paths_orig = self.constraints._to_original_values(paths, self.scaler_x)
            feasible = self.constraints.feasible_mask(paths_orig)
        else:
            feasible = self.constraints.feasible_mask(paths)

        if torch.any(feasible):
            return paths[feasible]
        return paths[:0]

    def generate_mixed_paths(
        self,
        current_location: torch.Tensor,
        q_steps: int,
        num_scenarios: int,
        config: PlanningConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_paths = self.generate_sobol(num_scenarios, q_steps)
        if not config.include_local_paths or config.local_path_fraction <= 0:
            return base_paths, base_paths

        local_count = max(1, int(round(num_scenarios * float(config.local_path_fraction))))
        local_noise = (
            torch.randn((local_count, q_steps, self.lower.numel()), dtype=base_paths.dtype, device=base_paths.device)
            * self.span
            * config.distance_scale
        )
        local_paths = (current_location.view(1, 1, -1) + torch.cumsum(local_noise, dim=1)).clamp(self.lower, self.upper)
        if config.enforce_feasible_sampling:
            local_paths = self._feasible_subset(local_paths)

        combined = torch.cat([base_paths, local_paths], dim=0) if local_paths.numel() > 0 else base_paths
        return combined, base_paths


class PathScoringService:
    def __init__(self, constraints, models, likelihoods):
        self.constraints = constraints
        self.models = models
        self.likelihoods = likelihoods

    def evaluate_information_gain(self, candidate_paths: torch.Tensor) -> torch.Tensor:
        if not self.models or not self.likelihoods:
            raise RuntimeError("No trained GP models attached to the planner service.")

        total_ig = torch.zeros(candidate_paths.shape[0], dtype=torch.float64, device=candidate_paths.device)
        eye = torch.eye(candidate_paths.shape[1], dtype=torch.float64, device=candidate_paths.device)

        for model, likelihood in zip(self.models, self.likelihoods):
            with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.fast_pred_var(False):
                cov = model.posterior(candidate_paths).distribution.covariance_matrix

            if cov.ndim == 4:
                cov = cov.squeeze(0)
            if not torch.isfinite(cov).all():
                total_ig -= 1e6
                continue

            approx = cov + 1e-6 * eye
            try:
                diag = torch.linalg.cholesky(approx).diagonal(dim1=-2, dim2=-1)
                ig = torch.log(diag.clamp_min(1e-12)).sum(dim=-1)
            except RuntimeError:
                sign, logdet = torch.linalg.slogdet(approx)
                ig = sign * logdet

            total_ig += torch.where(torch.isfinite(ig), ig, torch.full_like(ig, -1e6))

        if len(self.models) > 0:
            total_ig = total_ig / float(len(self.models))
        return total_ig

    def score_paths(self, candidate_paths: torch.Tensor, current_location: torch.Tensor, config: PlanningConfig) -> torch.Tensor:
        gamma = self.evaluate_information_gain(candidate_paths)
        dist = lambda x, y: torch.sqrt(((x - y).pow(2)).sum(dim=-1))
        step_cost = dist(candidate_paths[:, 0], current_location)
        if candidate_paths.shape[1] > 1:
            step_cost = step_cost + dist(candidate_paths[:, 1:], candidate_paths[:, :-1]).sum(dim=-1)

        score = gamma - config.normalized_distance_weight * step_cost

        cand_orig = candidate_paths
        if hasattr(self.constraints, '_to_original_values'):
            cand_orig = self.constraints._to_original_values(candidate_paths, getattr(self, 'scaler_x', None))

        feasible = self.constraints.feasible_mask(cand_orig)
        score = score.clone()
        score[feasible] = score[feasible] - float(config.feasible_margin_weight) * self.constraints.interior_margin_penalty(cand_orig[feasible])
        score[~feasible] = score[~feasible] - config.soft_penalty_strength * self.constraints.constraint_penalty(cand_orig[~feasible])
        score[~feasible] = score[~feasible] - config.interior_penalty_strength * self.constraints.interior_margin_penalty(cand_orig[~feasible])
        return score


class PlannerService:
    def __init__(self, bounds: torch.Tensor, constraints, models: List, likelihoods: List, scaler_x=None):
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.constraints = constraints
        self.scaler_x = scaler_x
        self.candidate_generation = CandidateGenerationService(self.lower, self.upper, constraints=constraints, scaler_x=scaler_x)
        self.path_scoring = PathScoringService(constraints, models, likelihoods)
        self.path_scoring.scaler_x = scaler_x
        self.last_candidate_pool = None
        self.last_sobol_pool = None
        self.last_local_pool = None
        self.last_scores = None
        self.last_selected_path = None

    def _feasible_mask_in_planner_space(self, x_scaled):
        if hasattr(self.constraints, '_to_original_values'):
            x_orig = self.constraints._to_original_values(x_scaled, self.scaler_x)
            return self.constraints.feasible_mask(x_orig)
        return self.constraints.feasible_mask(x_scaled)

    def plan(self, current_location: torch.Tensor, config: PlanningConfig) -> torch.Tensor:
        paths, sobol_pool = self.candidate_generation.generate_mixed_paths(
            current_location=current_location,
            q_steps=config.q_steps,
            num_scenarios=config.num_scenarios,
            config=config,
        )
        self.last_candidate_pool = paths.clone()
        self.last_sobol_pool = sobol_pool.clone()
        if paths.shape[0] > sobol_pool.shape[0]:
            self.last_local_pool = paths[sobol_pool.shape[0]:].clone()
        else:
            self.last_local_pool = None

        if config.enforce_feasible_sampling:
            feasible = self._feasible_mask_in_planner_space(paths)
            if torch.any(feasible):
                paths = paths[feasible]
            if paths.shape[0] == 0:
                paths = self.candidate_generation.generate_sobol(config.num_scenarios, config.q_steps)
                self.last_candidate_pool = paths.clone()
                self.last_sobol_pool = paths.clone()
                self.last_local_pool = None

        scores = self.path_scoring.score_paths(paths, current_location, config)
        self.last_scores = scores.clone()
        feasible = self._feasible_mask_in_planner_space(paths)

        if feasible.any():
            feasible_scores = scores.clone()
            feasible_scores[~feasible] = -torch.inf
            best_idx = torch.argmax(feasible_scores)
            selected = paths[best_idx].clone()
            self.last_selected_path = selected.clone()
            return selected

        best_idx = torch.argmax(scores)
        selected = paths[best_idx].clone()
        self.last_selected_path = selected.clone()
        return selected
