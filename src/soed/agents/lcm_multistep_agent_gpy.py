import torch
import gpytorch
from torch.quasirandom import SobolEngine

class LCMMultiStepAgentGpy:
    def __init__(self, bounds, model, likelihood):
        """
        Instead of fitting data inside the agent, pass the pre-trained 
        LCM model and likelihood directly from your surrogate script.
        """
        self.bounds = torch.as_tensor(bounds, dtype=torch.float64)
        self.d = self.bounds.shape[0]
        self.model = model
        self.likelihood = likelihood        
        self.X = None
        self.Y = None

    def fit_data(self):
        pass

    def lengthscale_weighted_distance(self, x1, x2, lengthscales):
        """Mahalanobis-like distance induced by GP lengthscales."""
        
        # 1. Convert the list of lengthscales into a single tensor
        if isinstance(lengthscales, list):
            # Stack them into a matrix and take the mean across the base kernels
            lengthscales = torch.stack(lengthscales).mean(dim=0)
            
        # 2. Ensure the lengthscales are on the exact same device (CPU/GPU) as your data
        lengthscales = lengthscales.to(x1.device)
        
        # 3. Compute the distance safely!
        return torch.sqrt(((x1 - x2) / lengthscales).pow(2).sum(dim=-1))
    
    def effective_lengthscales(self):
        """
        Extract lengthscales from the LCM base kernels. 
        GPyTorch's LCMKernel wraps base kernels in a covar_module_list.
        """
        lss = []
        
        # Iterate through the Multitask kernels stored in the LCM kernel
        for multitask_kernel in self.model.covar_module.covar_module_list:
            
            # Extract the actual spatial/data kernel (e.g., Matern, RBF)
            base_kernel = multitask_kernel.data_covar_module
            
            # If you wrapped your Matern kernel in a ScaleKernel, dig one level deeper:
            if hasattr(base_kernel, 'base_kernel'):
                actual_kernel = base_kernel.base_kernel
            else:
                actual_kernel = base_kernel
                
            # Grab the lengthscale, detach it from the computational graph, and store it
            # Using .squeeze() to ensure it's a flat tensor
            lss.append(actual_kernel.lengthscale.detach().squeeze())
            
        return lss

    # def effective_lengthscales(self):
    #     """
    #     Extract lengthscales from the LCM base kernels. 
    #     LCMKernel wraps multiple base kernels (Matern, RQ).
    #     """
    #     lss = []
    #     for base_kernel in self.model.covar_module.base_kernels:
    #         if hasattr(base_kernel, 'lengthscale') and base_kernel.lengthscale is not None:
    #             lss.append(base_kernel.lengthscale.squeeze().detach())
        
    #     if not lss:
    #         # Fallback if no lengthscales are found
    #         return torch.ones(self.d, dtype=torch.float64, device=self.bounds.device)
            
    #     lss = torch.stack(lss)
    #     return torch.min(lss, dim=0).values  # Conservative minimum lengthscale

    def plan_multistep_batch(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0):
        # Generate candidate paths
        sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
        samples = sobol.draw(num_scenarios).to(self.bounds.device)

        print(f"Bound shape: {self.bounds.shape}, Sample shape: {samples.shape}")

        paths = samples.view(num_scenarios, q_steps, self.d)

        lower_bounds = self.bounds[:, 0] 
        upper_bounds = self.bounds[:, 1]

        rng = upper_bounds - lower_bounds
        paths = lower_bounds + rng * paths

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            # 1. Get the Multitask predictive distribution
            # Shape of paths: (num_scenarios, q_steps, d)
            pred_f = self.model(paths) 
            
            # 2. Pass through likelihood to get the noisy observation covariance
            # The covariance matrix shape will be (num_scenarios, q_steps * T, q_steps * T)
            obs_pred = self.likelihood(pred_f)
            cov_obs = obs_pred.covariance_matrix
            
            # 3. Calculate Information Gain (IG)
            # For GPs with homoscedastic noise, maximizing IG is mathematically 
            # equivalent to maximizing the log-determinant of the predictive observation covariance.
            try:
                # Fast, stable log-determinant using Cholesky
                L = torch.linalg.cholesky(cov_obs)
                # 2 * sum(log(diag(L))) computes the log determinant
                total_ig = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            except RuntimeError:
                # Fallback to slogdet if Cholesky fails due to numerical jitter
                total_ig = torch.linalg.slogdet(cov_obs)[1]

        # 4. Calculate Distance Penalty
        curr = torch.as_tensor(current_location, dtype=paths.dtype, device=paths.device).squeeze()
        ls_eff = self.effective_lengthscales()
        
        d0 = self.lengthscale_weighted_distance(paths[:, 0], curr, ls_eff)
        d_steps = self.lengthscale_weighted_distance(
            paths[:, 1:], paths[:, :-1], ls_eff
        ).sum(dim=-1)

        # 5. Calculate final objective
        scores = total_ig - w_dist * (d0 + d_steps)

        return paths[scores.argmax()]
    

    def plan_multistep_batch_lss(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0):
        """
        Plans steps using a lengthscale-constrained bounded random walk.
        Guarantees that no single step exceeds the Mahalanobis distance of w_dist.
        """
        # 1. Generate raw Sobol samples [0, 1]
        sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
        samples = sobol.draw(num_scenarios).to(self.bounds.device)
        raw_paths = samples.view(num_scenarios, q_steps, self.d)

        # 2. Convert [0, 1] into [-1, 1] directional unit vectors
        offsets = raw_paths * 2.0 - 1.0 
        
        # 3. Constrain offsets to a unit hypersphere (Radius <= 1)
        norms = torch.sqrt(torch.sum(offsets**2, dim=-1, keepdim=True))
        offsets = offsets / torch.clamp(norms, min=1.0)

        # 4. Retrieve and format lengthscales
        lss = self.effective_lengthscales()
        if isinstance(lss, list):
            lss = torch.stack(lss).mean(dim=0).to(self.bounds.device)

        # 5. Scale the unit vectors by the max distance threshold and lengthscales
        step_sizes = offsets * (w_dist * lss)

        # 6. Build physical paths originating from current location
        curr = torch.as_tensor(current_location, dtype=raw_paths.dtype, device=raw_paths.device).squeeze()
        planned_paths = curr + torch.cumsum(step_sizes, dim=1)

        # 7. Enforce global map bounds (don't walk off the map)
        lower_bounds = self.bounds[:, 0] 
        upper_bounds = self.bounds[:, 1]
        planned_paths = torch.max(planned_paths, lower_bounds)
        paths = torch.min(planned_paths, upper_bounds)

        # 8. Evaluate model and calculate Information Gain
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
            pred_f = self.model(paths) 
            obs_pred = self.likelihood(pred_f)
            cov_obs = obs_pred.covariance_matrix
            
            try:
                L = torch.linalg.cholesky(cov_obs)
                total_ig = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            except RuntimeError:
                total_ig = torch.linalg.slogdet(cov_obs)[1]

        # 9. Return the path with the highest Information Gain
        # NOTE: We drop the distance penalty (d0 + d_steps) here because the 
        # w_dist threshold is already mathematically guaranteed by steps 2-5!
        scores = total_ig 

        return paths[scores.argmax()]
    
    # def plan_multistep_batch(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0, lengthscale_weighted=True):
    #     # Generate candidate paths
    #     sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
    #     samples = sobol.draw(num_scenarios).to(self.bounds.device)

    #     # FIX 1: Name this raw_paths so we don't accidentally use it later
    #     raw_paths = samples.view(num_scenarios, q_steps, self.d)
    #     lower_bounds = self.bounds[:, 0] 
    #     upper_bounds = self.bounds[:, 1]

    #     if lengthscale_weighted:
    #         # 1. Convert Sobol [0, 1] into [-1, 1] directional offsets
    #         offsets = raw_paths * 2.0 - 1.0 
    #         norms = torch.sqrt(torch.sum(offsets**2, dim=-1, keepdim=True))
    #         offsets = offsets / torch.clamp(norms, min=1.0)

    #         lss = self.effective_lengthscales()
    #         if isinstance(lss, list):
    #             lss = torch.stack(lss).mean(dim=0).to(self.bounds.device)
            
    #         # Scale the offsets by your max threshold and lengthscales
    #         step_sizes = offsets * (w_dist * lss)
            
    #         planned_paths = current_location + torch.cumsum(step_sizes, dim=1)
            
    #         planned_paths = torch.max(planned_paths, lower_bounds) # Enforce min
    #         planned_paths = torch.min(planned_paths, upper_bounds) # Enforce max
            
    #         # FIX 2: Reassign planned_paths to paths so the rest of the code is unified
    #         paths = planned_paths 
    #     else:
    #         rng = upper_bounds - lower_bounds
    #         paths = lower_bounds + rng * raw_paths

    #     self.model.eval()
    #     self.likelihood.eval()

    #     with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
    #         # 1. Get the Multitask predictive distribution
    #         # We now safely use 'paths' for both conditions
    #         pred_f = self.model(paths) 
            
    #         # 2. Pass through likelihood to get the noisy observation covariance
    #         obs_pred = self.likelihood(pred_f)
    #         cov_obs = obs_pred.covariance_matrix
            
    #         # 3. Calculate Information Gain (IG)
    #         try:
    #             L = torch.linalg.cholesky(cov_obs)
    #             total_ig = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    #         except RuntimeError:
    #             total_ig = torch.linalg.slogdet(cov_obs)[1]

    #     # 4. Calculate Distance Penalty
    #     curr = torch.as_tensor(current_location, dtype=paths.dtype, device=paths.device).squeeze()
    #     ls_eff = self.effective_lengthscales()
        
    #     # This now correctly uses the physical path coordinates
    #     d0 = self.lengthscale_weighted_distance(paths[:, 0], curr, ls_eff)
    #     d_steps = self.lengthscale_weighted_distance(
    #         paths[:, 1:], paths[:, :-1], ls_eff
    #     ).sum(dim=-1)

    #     # 5. Calculate final objective
    #     scores = total_ig - w_dist * (d0 + d_steps)

    #     # This now returns the correct physical coordinates
    #     return paths[scores.argmax()]

    # def plan_multistep_batch(self, current_location, q_steps=3, num_scenarios=256, w_dist=1.0, lengthscale_weighted=True):
    #     # Generate candidate paths
    #     sobol = SobolEngine(dimension=self.d * q_steps, scramble=True)
    #     samples = sobol.draw(num_scenarios).to(self.bounds.device)

    #     # print(f"Bound shape: {self.bounds.shape}, Sample shape: {samples.shape}")

    #     paths = samples.view(num_scenarios, q_steps, self.d)
    #     lower_bounds = self.bounds[:, 0] 
    #     upper_bounds = self.bounds[:, 1]

    #     if lengthscale_weighted:
    #         # 1. Convert Sobol [0, 1] into [-1, 1] directional offsets
    #         offsets = paths * 2.0 - 1.0 
    #         norms = torch.sqrt(torch.sum(offsets**2, dim=-1, keepdim=True))
    #         offsets = offsets / torch.clamp(norms, min=1.0)

    #         lss = self.effective_lengthscales()
    #         if isinstance(lss, list):
    #             lss = torch.stack(lss).mean(dim=0).to(self.bounds.device)
            
    #         # 5. Scale the offsets by your max threshold and lengthscales
    #         step_sizes = offsets * (w_dist * lss)
    #         print(f"Step sizes: {step_sizes}")
        
    #         planned_paths = current_location + torch.cumsum(step_sizes, dim=1)
        
    #         planned_paths = torch.max(planned_paths, lower_bounds) # Enforce min
    #         planned_paths = torch.min(planned_paths, upper_bounds) # Enforce max
    #     else:
    #         rng = upper_bounds - lower_bounds
    #         paths = lower_bounds + rng * paths

    #     self.model.eval()
    #     self.likelihood.eval()

    #     with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
    #         # 1. Get the Multitask predictive distribution
    #         # Shape of paths: (num_scenarios, q_steps, d)
    #         if lengthscale_weighted:
    #             pred_f = self.model(planned_paths)
    #         else:
    #             pred_f = self.model(paths) 
            
    #         # 2. Pass through likelihood to get the noisy observation covariance
    #         # The covariance matrix shape will be (num_scenarios, q_steps * T, q_steps * T)
    #         obs_pred = self.likelihood(pred_f)
    #         cov_obs = obs_pred.covariance_matrix
            
    #         # 3. Calculate Information Gain (IG)
    #         # For GPs with homoscedastic noise, maximizing IG is mathematically 
    #         # equivalent to maximizing the log-determinant of the predictive observation covariance.
    #         try:
    #             # Fast, stable log-determinant using Cholesky
    #             L = torch.linalg.cholesky(cov_obs)
    #             # 2 * sum(log(diag(L))) computes the log determinant
    #             total_ig = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    #         except RuntimeError:
    #             # Fallback to slogdet if Cholesky fails due to numerical jitter
    #             total_ig = torch.linalg.slogdet(cov_obs)[1]

    #     # 4. Calculate Distance Penalty
    #     curr = torch.as_tensor(current_location, dtype=paths.dtype, device=paths.device).squeeze()
    #     ls_eff = self.effective_lengthscales()
        
    #     d0 = self.lengthscale_weighted_distance(paths[:, 0], curr, ls_eff)
    #     d_steps = self.lengthscale_weighted_distance(
    #         paths[:, 1:], paths[:, :-1], ls_eff
    #     ).sum(dim=-1)

    #     # 5. Calculate final objective
    #     scores = total_ig - w_dist * (d0 + d_steps)

    #     return paths[scores.argmax()]