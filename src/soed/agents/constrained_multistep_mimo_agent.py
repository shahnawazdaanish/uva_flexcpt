import torch
import gpytorch
from torch.quasirandom import SobolEngine
from pathlib import Path
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.soed.agents.agent import Agent
from src.soed.dynamic_gp import DynamicGP


class ConstrainedMultiStepMIMOAgent(Agent):
    """Multi-output GP planner with hard engineering constraints."""

    def __init__(
        self, bounds, feature_names, scaler_x=None, mass1_name="Mass1", mass2_name="Mass2",
        boost_name="Boost pressure", ivo_name="IVO", ivc_name="IVC", evo_name="EVO",
        evc_name="EVC", load_limit=30.0, boost_slope=0.0922, boost_intercept=0.8378,
        boost_band=0.5, enable_br_limit=True, enable_vva_limit=True,
        min_load=3.0, ambient_pressure=1.0, tc_boost_limit=3.8,
    ):
        raw_bounds = torch.as_tensor(bounds, dtype=torch.float64)
        if raw_bounds.ndim != 2 or 2 not in raw_bounds.shape:
            raise ValueError("bounds must be a 2D tensor-like of shape [2, d] or [d, 2]")
        
        self.bounds = raw_bounds if raw_bounds.shape[0] == 2 else raw_bounds.T
        self.d = self.bounds.shape[1]
        self.lower, self.upper = self.bounds[0], self.bounds[1]

        self.feature_names = list(feature_names)
        self.scaler_x = scaler_x

        # Fetch indices safely
        self.idxs = {
            name: self.feature_names.index(feat) for name, feat in zip(
                ['m1', 'm2', 'bst', 'ivo', 'ivc', 'evo', 'evc'],
                [mass1_name, mass2_name, boost_name, ivo_name, ivc_name, evo_name, evc_name]
            )
        }

        self.load_limit = float(load_limit)
        self.boost_slope = float(boost_slope)
        self.boost_intercept = float(boost_intercept)
        self.boost_band = float(boost_band)
        self.min_load = float(min_load)
        self.ambient_pressure = float(ambient_pressure)
        self.tc_boost_limit = float(tc_boost_limit)
        self.enable_br_limit = bool(enable_br_limit)
        self.enable_vva_limit = bool(enable_vva_limit)

        self.models, self.likelihoods = [], []
        self.X, self.Y = None, None

        self._sanitize_bounds(min_width=1e-6)

    def _sanitize_bounds(self, min_width=1e-6):
        if not torch.isfinite(self.bounds).all():
            raise ValueError("Bounds contain NaN/Inf values.")
        bad = (self.upper - self.lower) <= min_width
        if torch.any(bad):
            print(f"Warning: expanding degenerate bounds for dims {torch.where(bad)[0].tolist()}.")
            self.lower[bad] -= 0.5 * min_width
            self.upper[bad] += 0.5 * min_width
            self.bounds = torch.stack([self.lower, self.upper], dim=0)

    @staticmethod
    def _deduplicate_rows(X, Y, decimals=10):
        scale = 10.0 ** decimals
        X_key = torch.round(X * scale).to(torch.int64)
        unique_keys, inverse = torch.unique(X_key, dim=0, return_inverse=True)
        if unique_keys.shape[0] == X.shape[0]:
            return X, Y
        return torch.stack([X[inverse == i][0] for i in range(len(unique_keys))]), \
               torch.stack([Y[inverse == i].mean(dim=0) for i in range(len(unique_keys))])

    def fit_data(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)
        if Y.ndim == 1: Y = Y.unsqueeze(-1)

        valid = torch.isfinite(X).all(dim=1) & torch.isfinite(Y).all(dim=1)
        if not torch.any(valid): raise ValueError("All training rows contain NaN/Inf.")
        X, Y = self._deduplicate_rows(X[valid], Y[valid], decimals=10)

        # Avoid singular kernels
        near_const = X.std(dim=0) < 1e-12
        if torch.any(near_const):
            X[:, near_const] += 1e-8 * torch.randn((X.shape[0], int(near_const.sum())), dtype=X.dtype)

        self.X, self.Y, self.models, self.likelihoods = X, Y, [], []

        for j in range(Y.shape[1]):
            yj = Y[:, j:j + 1]
            if yj.std() < 1e-14: yj += 1e-8 * torch.randn_like(yj)

            lik = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0))
            lik.noise = torch.tensor(1e-2, dtype=torch.float64)
            model = DynamicGP(X, yj, lik, self.bounds)
            model.train(); lik.train()

            mll = ExactMarginalLogLikelihood(lik, model)
            
            with gpytorch.settings.cholesky_jitter(1e-3), gpytorch.settings.cholesky_max_tries(6):
                try:
                    fit_gpytorch_mll(mll)
                except Exception:
                    # Robust Adam fallback
                    opt = torch.optim.Adam(model.parameters(), lr=0.03)
                    for _ in range(250):
                        opt.zero_grad()
                        loss = -mll(model(X), yj.squeeze(-1))
                        if torch.isfinite(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                            opt.step()

            model.eval(); lik.eval()
            self.models.append(model)
            self.likelihoods.append(lik)

    @staticmethod
    def _state_dict_to_cpu(sd):
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def save_bundle(self, bundle_path, extra_metadata=None):
        if self.X is None or self.Y is None or not self.models or not self.likelihoods:
            raise RuntimeError("No trained models found. Call fit_data before save_bundle.")

        payload = {
            "bundle_version": 1,
            "agent_type": self.__class__.__name__,
            "feature_names": list(self.feature_names),
            "bounds": self.bounds.detach().cpu().clone(),
            "constraint_config": {
                "load_limit": self.load_limit,
                "boost_slope": self.boost_slope,
                "boost_intercept": self.boost_intercept,
                "boost_band": self.boost_band,
                "min_load": self.min_load,
                "ambient_pressure": self.ambient_pressure,
                "tc_boost_limit": self.tc_boost_limit,
                "enable_br_limit": self.enable_br_limit,
                "enable_vva_limit": self.enable_vva_limit,
            },
            "X": self.X.detach().cpu().clone(),
            "Y": self.Y.detach().cpu().clone(),
            "model_state_dicts": [self._state_dict_to_cpu(m.state_dict()) for m in self.models],
            "likelihood_state_dicts": [self._state_dict_to_cpu(l.state_dict()) for l in self.likelihoods],
            "extra_metadata": extra_metadata or {},
        }

        out_path = Path(bundle_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)
        return str(out_path)

    def load_bundle(self, bundle_path, strict=True):
        payload = torch.load(bundle_path, map_location="cpu", weights_only=False)

        if payload.get("agent_type") != self.__class__.__name__:
            raise ValueError(f"Unsupported agent type in bundle: {payload.get('agent_type')}")

        saved_features = payload.get("feature_names", [])
        if list(saved_features) != list(self.feature_names):
            raise ValueError(
                "Feature mismatch between bundle and agent. "
                f"Bundle features: {saved_features}, Agent features: {self.feature_names}"
            )

        X = torch.as_tensor(payload.get("X"), dtype=torch.float64)
        Y = torch.as_tensor(payload.get("Y"), dtype=torch.float64)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        model_state_dicts = payload.get("model_state_dicts", [])
        likelihood_state_dicts = payload.get("likelihood_state_dicts", [])

        if len(model_state_dicts) != Y.shape[1] or len(likelihood_state_dicts) != Y.shape[1]:
            raise ValueError(
                "Bundle output dimension mismatch. "
                f"Expected {Y.shape[1]} states, got models={len(model_state_dicts)}, "
                f"likelihoods={len(likelihood_state_dicts)}"
            )

        self.X = X
        self.Y = Y
        self.models = []
        self.likelihoods = []

        for j in range(Y.shape[1]):
            yj = Y[:, j:j + 1]
            lik = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0)
            )
            model = DynamicGP(X, yj, lik, self.bounds)

            model.load_state_dict(model_state_dicts[j], strict=strict)
            lik.load_state_dict(likelihood_state_dicts[j], strict=strict)

            model.eval()
            lik.eval()
            self.models.append(model)
            self.likelihoods.append(lik)

        return payload

    def _to_original_units(self, x_scaled):
        if self.scaler_x is None: return x_scaled
        mean = torch.as_tensor(self.scaler_x.scaler.mean_, dtype=x_scaled.dtype, device=x_scaled.device)
        scale = torch.as_tensor(self.scaler_x.scaler.scale_, dtype=x_scaled.dtype, device=x_scaled.device)
        return x_scaled * scale + mean

    def _get_vva_bounds(self, bin_idx, dev):
        """Helper to return VVA constraints based on load bin indexing"""
        limits = [
            (self.idxs['ivo'], [350., 330., 345.], [435., 390., 365.]),
            (self.idxs['ivc'], [500., 500., 495.], [540., 570., 535.]),
            (self.idxs['evo'], [128., 128., 128.], [218., 218., 218.]),
            (self.idxs['evc'], [270., 330., 345.], [350., 370., 355.])
        ]
        return [(idx, torch.tensor(L, device=dev)[bin_idx], torch.tensor(H, device=dev)[bin_idx]) for idx, L, H in limits]

    def _feasible_mask(self, paths_scaled):
        x = self._to_original_units(paths_scaled)
        m1, m2, bst = x[..., self.idxs['m1']], x[..., self.idxs['m2']], x[..., self.idxs['bst']]
        m_sum = m1 + m2
        bst_c = self.boost_slope * m_sum + self.boost_intercept

        # Intersection of polygon boundaries:
        # right wall (max load), left wall (min load), roof/floor, and slanted boost band.
        ok = (m1 < self.load_limit) & \
             (m_sum < self.load_limit) & (m_sum >= self.min_load) & \
             (bst <= self.tc_boost_limit) & (bst >= self.ambient_pressure) & \
             (bst >= bst_c - self.boost_band) & (bst <= bst_c + self.boost_band)

        # Vectorized lookup: bin 0 (<10), 1 (10-20), 2 (>20)
        b = torch.clamp((m1 // 10).long(), 0, 2)
        dev = m1.device

        if self.enable_br_limit:
            ok &= (m2 > torch.tensor([0.5, 0.9, 0.0], device=dev)[b]) & \
                  (m2 < torch.tensor([3.5, 3.0, 1.5], device=dev)[b])

        if self.enable_vva_limit:
            for idx, low, high in self._get_vva_bounds(b, dev):
                ok &= (x[..., idx] >= low) & (x[..., idx] <= high)

        return ok.all(dim=1)

    def _constraint_penalty(self, paths_scaled):
        x = self._to_original_units(paths_scaled)
        m1, m2, bst = x[..., self.idxs['m1']], x[..., self.idxs['m2']], x[..., self.idxs['bst']]
        m_sum = m1 + m2
        bst_c = self.boost_slope * m_sum + self.boost_intercept

        # 1) Vertical walls: max-load walls and min-load wall.
        pen = torch.relu(m1 - self.load_limit).pow(2) + \
              torch.relu(m_sum - self.load_limit).pow(2) + \
              torch.relu(self.min_load - m_sum).pow(2)

        # 2) Horizontal roof/floor.
        pen += torch.relu(bst - self.tc_boost_limit).pow(2) + \
               torch.relu(self.ambient_pressure - bst).pow(2)

        # 3) Slanted walls from the boost corridor.
        pen += torch.relu(bst_c - self.boost_band - bst).pow(2) + \
               torch.relu(bst - (bst_c + self.boost_band)).pow(2)

        # 4) Prefer the interior of the feasible polygon rather than the boundary.
        # This prevents the optimizer from clustering points exactly on the load or boost walls.
        boost_margin = 0.15 * self.boost_band
        load_margin = max(0.25, 0.05 * self.load_limit)
        lower_gap = bst - (bst_c - self.boost_band)
        upper_gap = (bst_c + self.boost_band) - bst
        pen += torch.relu(boost_margin - lower_gap).pow(2) + torch.relu(boost_margin - upper_gap).pow(2)
        pen += torch.relu(self.min_load + load_margin - m_sum).pow(2)
        pen += torch.relu(m1 - (self.load_limit - load_margin)).pow(2)
        pen += torch.relu(m_sum - (self.load_limit - load_margin)).pow(2)

        b = torch.clamp((m1 // 10).long(), 0, 2)
        dev = m1.device

        # BR limits (kept consistent with _feasible_mask)
        br_L = torch.tensor([0.5, 0.9, 0.0], device=dev)[b]
        br_H = torch.tensor([3.5, 3.0, 1.5], device=dev)[b]
        pen += torch.relu(br_L - m2).pow(2) + torch.relu(m2 - br_H).pow(2)

        # VVA limits
        for idx, low, high in self._get_vva_bounds(b, dev):
            pen += torch.relu(low - x[..., idx]).pow(2) + torch.relu(x[..., idx] - high).pow(2)

        return pen.sum(dim=-1)

    def _interior_margin_penalty(self, paths_scaled):
        """Small preference for staying off the walls of the feasible polygon."""
        x = self._to_original_units(paths_scaled)
        m1, m2, bst = x[..., self.idxs['m1']], x[..., self.idxs['m2']], x[..., self.idxs['bst']]
        m_sum = m1 + m2
        bst_c = self.boost_slope * m_sum + self.boost_intercept

        boost_margin = 0.15 * self.boost_band
        load_margin = max(0.25, 0.05 * self.load_limit)

        lower_gap = bst - (bst_c - self.boost_band)
        upper_gap = (bst_c + self.boost_band) - bst
        margin_penalty = (
            torch.relu(boost_margin - lower_gap).pow(2)
            + torch.relu(boost_margin - upper_gap).pow(2)
            + torch.relu(self.min_load + load_margin - m_sum).pow(2)
            + torch.relu(m1 - (self.load_limit - load_margin)).pow(2)
            + torch.relu(m_sum - (self.load_limit - load_margin)).pow(2)
        )
        return margin_penalty.sum(dim=-1)

    def _sample_feasible_sobol_paths(self, q_steps, num_scenarios, max_attempts=12):
        """Draw Sobol paths and keep hard-feasible ones via rejection sampling."""
        sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
        kept = []
        total_kept = 0
        for _ in range(max_attempts):
            need = max(num_scenarios - total_kept, 0)
            if need == 0:
                break
            draw_n = max(need * 2, 256)
            cand = self.lower + (self.upper - self.lower) * sobol.draw(draw_n).view(draw_n, q_steps, self.d)
            feasible = self._feasible_mask(cand)
            if torch.any(feasible):
                accepted = cand[feasible]
                kept.append(accepted)
                total_kept += int(accepted.shape[0])

        if total_kept == 0:
            # Fallback to unconstrained Sobol if rejection failed entirely.
            return self.lower + (self.upper - self.lower) * sobol.draw(num_scenarios).view(num_scenarios, q_steps, self.d)

        out = torch.cat(kept, dim=0)
        if out.shape[0] >= num_scenarios:
            return out[:num_scenarios]

        # If feasible rejection produced too few points, top up with standard Sobol.
        # This prevents candidate-set collapse (e.g., a handful of Sobol paths)
        # that can unintentionally let local random-walk paths dominate.
        need = num_scenarios - out.shape[0]
        top_up = self.lower + (self.upper - self.lower) * sobol.draw(need).view(need, q_steps, self.d)
        return torch.cat([out, top_up], dim=0)

    def _sample_paths(
        self,
        q_steps,
        num_scenarios,
        current_location=None,
        local_scale=0.15,
        return_components=False,
        enforce_feasible_sobol=False,
        include_local_paths=True,
        local_path_fraction=1.0,
    ):
        if enforce_feasible_sobol:
            g_paths = self._sample_feasible_sobol_paths(q_steps, num_scenarios)
        else:
            sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
            g_paths = self.lower + (self.upper - self.lower) * sobol.draw(num_scenarios).view(num_scenarios, q_steps, self.d)
        if current_location is None or not include_local_paths or float(local_path_fraction) <= 0.0:
            if return_components:
                return g_paths, {"sobol_paths": g_paths, "local_paths": None}
            return g_paths

        curr = torch.as_tensor(current_location, dtype=g_paths.dtype, device=g_paths.device).squeeze()
        n_local = max(1, int(round(num_scenarios * float(local_path_fraction))))
        l_noise = torch.randn((n_local, q_steps, self.d), dtype=g_paths.dtype, device=g_paths.device) * (self.upper - self.lower) * local_scale
        l_paths = (curr.view(1, 1, -1) + torch.cumsum(l_noise, dim=1)).clamp(self.lower, self.upper)
        all_paths = torch.cat([g_paths, l_paths], dim=0)
        if return_components:
            return all_paths, {"sobol_paths": g_paths, "local_paths": l_paths}
        return all_paths

    def plan_multistep_batch(
        self,
        current_location,
        q_steps=3,
        num_scenarios=512,
        w_dist=1.0,
        enforce_feasible_sampling=False,
        enforce_feasible_sobol=False,
        include_local_paths=True,
        local_path_fraction=1.0,
        feasible_margin_weight=25.0,
    ):
        ls_eff = torch.min(torch.stack([m.covar_module.base_kernel.lengthscale.squeeze().detach() for m in self.models]), dim=0).values
        curr = torch.as_tensor(current_location, dtype=self.bounds.dtype, device=self.bounds.device).squeeze()

        scales = [(1, 0.10), (2, 0.18), (4, 0.28)]
        candidate_sets = []
        candidate_components = []
        for m, s in scales:
            paths, components = self._sample_paths(
                q_steps,
                num_scenarios * m,
                current_location=curr,
                local_scale=s,
                return_components=True,
                enforce_feasible_sobol=enforce_feasible_sobol,
                include_local_paths=include_local_paths,
                local_path_fraction=local_path_fraction,
            )
            candidate_sets.append(paths)
            candidate_components.append(components)

        candidate_sets_for_eval = [p for p in candidate_sets]

        # Optional strict filtering to limit the evaluated candidates to hard-feasible paths.
        # If a scale has no feasible candidates, keep that set so soft fallback can still recover.
        if enforce_feasible_sampling:
            filtered_sets = []
            for paths in candidate_sets_for_eval:
                feasible = self._feasible_mask(paths)
                if torch.any(feasible):
                    filtered_sets.append(paths[feasible])
                else:
                    filtered_sets.append(paths)
            candidate_sets_for_eval = filtered_sets

        best_path, best_score, best_feasible_count = None, None, 0
        feasible_masks_for_eval = []

        for paths in candidate_sets_for_eval:
            total_ig = torch.zeros(paths.shape[0], dtype=torch.float64, device=paths.device)

            for model, lik in zip(self.models, self.likelihoods):
                with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4), gpytorch.settings.fast_pred_var(False):
                    cov = model.posterior(paths).distribution.covariance_matrix
                
                if not torch.isfinite(cov).all():
                    total_ig -= 1e6; continue

                m = torch.eye(q_steps, dtype=cov.dtype, device=cov.device) + cov / lik.noise.clamp_min(1e-8)
                try:
                    ig = torch.linalg.cholesky(m).diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
                except RuntimeError:
                    ig = 0.5 * torch.linalg.slogdet(m)[1]
                total_ig += torch.where(torch.isfinite(ig), ig, torch.full_like(ig, -1e6))

            dist = lambda x, y: torch.sqrt(((x - y) / ls_eff).pow(2).sum(dim=-1))
            scores = total_ig - w_dist * (dist(paths[:, 0], curr) + dist(paths[:, 1:], paths[:, :-1]).sum(dim=-1))

            feasible = self._feasible_mask(paths)
            feasible_masks_for_eval.append(feasible.detach().cpu())
            feasible_count = int(feasible.sum().item())
            best_feasible_count = max(best_feasible_count, feasible_count)

            if feasible_count > 0:
                feasible_scores = scores.clone() - float(feasible_margin_weight) * self._interior_margin_penalty(paths)
                feasible_scores[~feasible] = -torch.inf
                idx = torch.argmax(feasible_scores)
                cand_score = feasible_scores[idx]
                if best_score is None or cand_score > best_score:
                    best_score, best_path = cand_score, paths[idx]
            else:
                # Only consider soft-constrained fallback paths until we find at least one feasible set.
                if best_feasible_count == 0:
                    soft_scores = scores - 50.0 * self._constraint_penalty(paths) - 10.0 * self._interior_margin_penalty(paths)
                    idx = torch.argmax(soft_scores)
                    cand_score = soft_scores[idx]
                    if best_score is None or cand_score > best_score:
                        best_score, best_path = cand_score, paths[idx]
            
            # Continue through all candidate scales so we do not stop at the
            # first feasible set; this improves chance of selecting a higher-IG path.

        if best_path is None: raise RuntimeError("Unable to construct any candidate path.")
        if best_feasible_count == 0: print("Warning: No fully feasible path found; returning best soft-constrained path.")

        # Persist exact planner candidates from this call for post-hoc visualization.
        self.last_plan_debug = {
            "q_steps": q_steps,
            "num_scenarios": num_scenarios,
            "w_dist": w_dist,
            "enforce_feasible_sampling": enforce_feasible_sampling,
            "enforce_feasible_sobol": enforce_feasible_sobol,
            "include_local_paths": include_local_paths,
            "local_path_fraction": float(local_path_fraction),
            "feasible_margin_weight": float(feasible_margin_weight),
            "current_location": curr.detach().cpu(),
            "scales": [{"multiplier": int(m), "local_scale": float(s)} for m, s in scales],
            "candidate_sets_raw": [
                {
                    "all_paths": p.detach().cpu(),
                    "sobol_paths": comp["sobol_paths"].detach().cpu(),
                    "local_paths": None if comp["local_paths"] is None else comp["local_paths"].detach().cpu(),
                }
                for p, comp in zip(candidate_sets, candidate_components)
            ],
            "candidate_sets_eval": [p.detach().cpu() for p in candidate_sets_for_eval],
            "feasible_masks_eval": feasible_masks_for_eval,
            "selected_path": best_path.detach().cpu(),
        }
        return best_path