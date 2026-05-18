import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

class DynamicGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, likelihood, bounds):
        input_dim = train_X.shape[1] 
        norm_tf = Normalize(d=input_dim, bounds=bounds)
        
        # Define a smooth baseline covariance
        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=input_dim, 
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 5.0)
            )
        )
        
        # Initialize the parent class
        super().__init__(
            train_X=train_X, 
            train_Y=train_Y, 
            likelihood=likelihood, 
            covar_module=covar_module,
            input_transform=norm_tf, 
            outcome_transform=Standardize(m=1) 
        )

        # ENFORCE ZERO MEAN: This prevents the "mountain" effect in empty space
        self.mean_module = gpytorch.means.ZeroMean()