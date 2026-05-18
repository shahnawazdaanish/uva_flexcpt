
import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.exceptions import ModelFittingError
from torch.quasirandom import SobolEngine


from src.soed.agents.agent import Agent
from src.soed.dynamic_gp import DynamicGP


class MultiStepMIMOAgent(Agent):
    def __init__(self, bounds):
        self.bounds = bounds
        self.d = bounds.shape[1]
        self.models = []
        self.likelihoods = []
        self.X = None
        self.Y = None
        self.m = None

    def fit_data(self, X, Y):
        X = torch.as_tensor(X, dtype=torch.float64)
        Y = torch.as_tensor(Y, dtype=torch.float64)
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)

        self.X, self.Y = X, Y
        self.m = Y.shape[1]

        self.models = []
        self.likelihoods = []

        for j in range(self.m):
            yj = Y[:, j:j+1]

            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.Interval(1e-4, 1.0)
            )
            model = DynamicGP(X, yj, likelihood, self.bounds)

            model.train()
            likelihood.train()

            mll = ExactMarginalLogLikelihood(likelihood, model)

            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
                for _ in range(150):
                    optimizer.zero_grad()
                    output = model(X)
                    loss = -mll(output, yj.squeeze(-1))
                    loss.backward()
                    optimizer.step()

            model.eval()
            likelihood.eval()

            self.models.append(model)
            self.likelihoods.append(likelihood)
    
    
    def lengthscale_weighted_distance(self, x1, x2, lengthscales):
        """
        Mahalanobis-like distance induced by GP lengthscales.
        """
        return torch.sqrt(((x1 - x2) / lengthscales).pow(2).sum(dim=-1))


    def effective_lengthscales(self, agent, mode="min"):
        """
        Combine per-output lengthscales into a single effective vector.
        """
        lss = torch.stack([
            m.covar_module.base_kernel.lengthscale.squeeze().detach()
            for m in agent.models
        ])

        if mode == "min":
            return torch.min(lss, dim=0).values      # conservative
        elif mode == "mean":
            return lss.mean(dim=0)
        else:
            raise ValueError("mode must be 'min' or 'mean'")



    def plan_multistep_batch(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0, stop_if_small_ig=True, min_ig_ratio=0.01):
        sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
        samples = sobol.draw(num_scenarios)

        paths = samples.view(num_scenarios, q_steps, self.d)
        rng = self.bounds[1] - self.bounds[0]
        paths = self.bounds[0] + rng * paths

        total_ig = torch.zeros(num_scenarios, dtype=torch.float64)
        ls_eff = self.effective_lengthscales(self, mode="min")
        noise_floor = 0.0

        for model, lik in zip(self.models, self.likelihoods):
            noise = lik.noise.item()
            noise_floor += noise

            post = model.posterior(paths)
            cov = post.distribution.covariance_matrix

            I = torch.eye(q_steps, dtype=cov.dtype, device=cov.device)
            # local_noise = lik.noise + post.distribution.variance.mean(dim=-1)
            M = I + cov / noise

            try:
                L = torch.linalg.cholesky(M)
                ig = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            except RuntimeError:
                ig = 0.5 * torch.linalg.slogdet(M)[1]

            total_ig += ig

        curr = current_location.squeeze()

        # ls = model.covar_module.base_kernel.lengthscale.squeeze()
        
        d0 = self.lengthscale_weighted_distance(paths[:, 0], curr, ls_eff)
        d_steps = self.lengthscale_weighted_distance(
            paths[:, 1:], paths[:, :-1], ls_eff
        ).sum(dim=-1)


        # d0 = torch.norm(paths[:, 0] - curr, dim=-1)
        # d_steps = torch.norm(paths[:, 1:] - paths[:, :-1], dim=-1).sum(dim=-1)

        scores = total_ig - w_dist * (d0 + d_steps)

        if stop_if_small_ig:
            if total_ig.max() < min_ig_ratio * noise_floor:
                print("Stopping: expected information gain below noise floor.")
                return None
            
        return paths[scores.argmax()]