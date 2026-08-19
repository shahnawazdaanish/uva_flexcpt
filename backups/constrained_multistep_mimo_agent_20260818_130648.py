import torch
import gpytorch
from torch.quasirandom import SobolEngine
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

    def _sample_paths(self, q_steps, num_scenarios, current_location=None, local_scale=0.15):
        sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
        g_paths = self.lower + (self.upper - self.lower) * sobol.draw(num_scenarios).view(num_scenarios, q_steps, self.d)
        if current_location is None: return g_paths

        curr = torch.as_tensor(current_location, dtype=g_paths.dtype, device=g_paths.device).squeeze()
        l_noise = torch.randn_like(g_paths) * (self.upper - self.lower) * local_scale
        l_paths = (curr.view(1, 1, -1) + torch.cumsum(l_noise, dim=1)).clamp(self.lower, self.upper)
        return torch.cat([g_paths, l_paths], dim=0)

    def plan_multistep_batch(self, current_location, q_steps=3, num_scenarios=512, w_dist=1.0):
        ls_eff = torch.min(torch.stack([m.covar_module.base_kernel.lengthscale.squeeze().detach() for m in self.models]), dim=0).values
        curr = torch.as_tensor(current_location, dtype=self.bounds.dtype, device=self.bounds.device).squeeze()

        candidate_sets = [
            self._sample_paths(q_steps, num_scenarios * m, current_location=curr, local_scale=s)
            for m, s in [(1, 0.10), (2, 0.18), (4, 0.28)]
        ]

        best_path, best_score, best_feasible_count = None, None, 0

        for paths in candidate_sets:
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
            feasible_count = int(feasible.sum().item())
            best_feasible_count = max(best_feasible_count, feasible_count)

            if feasible_count > 0:
                feasible_scores = scores.clone()
                feasible_scores[~feasible] = -torch.inf
                idx = torch.argmax(feasible_scores)
                cand_score = feasible_scores[idx]
                if best_score is None or cand_score > best_score:
                    best_score, best_path = cand_score, paths[idx]
            else:
                # Only consider soft-constrained fallback paths until we find at least one feasible set.
                if best_feasible_count == 0:
                    soft_scores = scores - 50.0 * self._constraint_penalty(paths)
                    idx = torch.argmax(soft_scores)
                    cand_score = soft_scores[idx]
                    if best_score is None or cand_score > best_score:
                        best_score, best_path = cand_score, paths[idx]
            
            # Continue through all candidate scales so we do not stop at the
            # first feasible set; this improves chance of selecting a higher-IG path.

        if best_path is None: raise RuntimeError("Unable to construct any candidate path.")
        if best_feasible_count == 0: print("Warning: No fully feasible path found; returning best soft-constrained path.")
        return best_path