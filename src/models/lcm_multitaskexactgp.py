import gpytorch

class LCM_MultitaskExactGP(gpytorch.models.ExactGP):
    """
    Multi-output Exact GP using LCMKernel (Linear Model of Coregionalization).
    - input kernel = Matern(v=1.5, ARD) + RationalQuadratic(ARD)
    - task coupling via LMC with NUM_LATENTS latent processes
    Expects:
       train_x: (N, D)
       train_y: (N, T)
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks, input_dim, num_latents=2):
        super().__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks

        # Mean: one const mean per task
        # self.mean_module = gpytorch.means.MultitaskMean(
        #     gpytorch.means.ConstantMean(), num_tasks=num_tasks
        # )

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=input_dim), num_tasks=num_tasks
        )

        # Base kernels over inputs
        rbf = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 10.0))
        matern = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dim)
        matern15 = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)
        rq     = gpytorch.kernels.RQKernel(ard_num_dims=input_dim)

        # LCM kernel: combine base kernels into latent processes
        # LCMKernel handles the task coregionalization internally
        self.covar_module = gpytorch.kernels.LCMKernel(
            base_kernels=[matern, rq],  # can add more kernels here
            num_tasks=num_tasks,
            rank=num_latents    # latent rank (num_latents)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)      # (N, T)
        covar_x = self.covar_module(x)    # multitask covariance
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def get_covar_module(self):
        return self.covar_module