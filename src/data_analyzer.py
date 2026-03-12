import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from src.scaler import Scaler
from src.noise_estimator import NoiseEstimator
from src.constant_manager import ConstantManager
from gp.regressor import Regressors


class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe

    def compute_output_noise(self, output_columns):
        noise_levels = {}
        for col in output_columns:
            y = self.df[col].values
            noise_estimator = NoiseEstimator()
            noise_level = noise_estimator.method_standard(None, y)
            noise_levels[col] = noise_level
        return noise_levels
    
    def compute_lengthscales(self, input_columns, output_columns, n_samples=1000, noises=None):
        # Sample a subset of the data for lengthscale estimation
        if n_samples < len(self.df):
            sampled_df = self.df.sample(n=n_samples, random_state=ConstantManager().RANDOM_STATE)
        else:
            sampled_df = self.df
        
        X = sampled_df[input_columns].values
        y = sampled_df[output_columns].values

        ls_per_of_feature = {}
        for i, col in enumerate(output_columns):
            # Fit a Gaussian Process to estimate lengthscale
            bounds = [
                (1e-2, 1e3),
                (1e-2, 1e20),
                (1e-2, 1e10),
                (1e-2, 1e3),
                (1e-2, 1e10),
                (1e-2, 1e10),
                (1e-2, 1e10),
                (1e-2, 1e10),
                (1e-2, 1e3),
                (1e-2, 1e3)
            ]
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=bounds)
            gp = GaussianProcessRegressor(
                kernel=kernel, 
                n_restarts_optimizer=5, 
                random_state=ConstantManager().RANDOM_STATE, 
                normalize_y=False, 
                alpha=noises[i] if noises is not None else 1e-6
            )
            gp.fit(X, y[:, i:i+1])  # Fit on each output feature separately
            
            # Extract lengthscales from the fitted kernel
            lengthscales = gp.kernel_.k2.length_scale
            ls_per_of_feature[col] = lengthscales
        return ls_per_of_feature
    

    def construct_learned_hyperparameters(self, lengthscales, noises, output_columns):
        learned_hyperparameters = {}
        for i, col in enumerate(output_columns):
            ls = lengthscales[col]
            learned_hyperparameters[col] = {
                'lengthscales': ls,
                'noise': noises[i] if noises is not None else 1e-6,
                'signal_variance': 1.0
            }
        return learned_hyperparameters
    

    def fit_training_data(self, lengthscales, noises, output_features, X, y):
        fitted_gps = {}
        for i, of_feature in enumerate(output_features):
            lengthscale_values = lengthscales[of_feature]
            kernel = C(1.0, constant_value_bounds='fixed') * RBF(length_scale=lengthscale_values, length_scale_bounds='fixed')
            
            gp = GaussianProcessRegressor(
                kernel=kernel, 
                optimizer=None,
                random_state=ConstantManager().RANDOM_STATE, 
                normalize_y=False, 
                alpha=noises[i] if noises is not None else 1e-6
            )

            fitted_gps[of_feature] = gp.fit(X, y[:, i:i+1])

        return fitted_gps
    
    def predict(self, X_test, fitted_gps):
        predictions = {}
        for of_feature, gp in fitted_gps.items():
            pred_mean, pred_std = gp.predict(X_test, return_std=True)
            predictions[of_feature] = {
                'mean': pred_mean,
                'std': pred_std
            }
        return predictions