import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from src.constant_manager import ConstantManager
from src.noise_estimator import NoiseEstimator
from src.predictor import Predictor


class ScaledAnalyzer:
    def __init__(self):
        self.noise_estimator = NoiseEstimator()
        self.predictor = Predictor()
        self.constant_manager = ConstantManager()
        
    def calculate_reference_noise_stats(self, output_data):
        y_rep = np.asarray(output_data)
        m, D = y_rep.shape
        if D != 2:
            raise ValueError("Expected 2 outputs (shape (m,2)).")
        if m <= 1:
            raise ValueError("Need at least 2 repeats to estimate variance.")
        y_mean = y_rep.mean(axis=0)               # shape (2,)
        # per-output sample variances (unbiased)
        var_per_output = y_rep.var(axis=0, ddof=1)  # shape (2,)
        # full sample covariance matrix (unbiased)
        cov_matrix = np.cov(y_rep, rowvar=False, ddof=1)  # shape (2,2)
        return y_mean, var_per_output, cov_matrix

    def learn_noise(self, X, y, bounds=None, params=None):
        return self.noise_estimator.method_mle_ref(
            X, 
            y, 
            bounds=bounds, 
            params=params
        )

    def estimate_noise_from_repeats(self, df, input_cols, output_col):
        """
        Estimate noise variance from repeated inputs for one output.
        
        df : pd.DataFrame
        input_cols : list of str, input column names
        output_col : str, output column name
        """
        variances = []

        # group by unique inputs
        for _, g in df.groupby(input_cols):
            if len(g) > 1:  # only if repeated inputs exist
                variances.append(g[output_col].var(ddof=1))  # sample variance

        if len(variances) > 0:
            return np.mean(variances)  # average noise variance
        else:
            return np.nan  # no repeats found


    def get_predictions(self, X, y, group_dataset, group_params, input_priority, n_test_samples=500):

        sample_data = group_dataset.sample(1, random_state=42)
        min_val = group_dataset[input_priority].min()
        max_val = group_dataset[input_priority].max()

        input_columns = ['if_' + str(i+1) for i in range(7)]

        samples = self.predictor.generate_sorted_samples(
            sample_data[input_columns],
            input_priority,
            min_val,
            max_val,
            n_test_samples
        )[0].values

        # + WhiteKernel(noise_level=alpha * 5, noise_level_bounds="fixed")
        kernel = C(group_params['s'], constant_value_bounds="fixed") * \
            RBF(length_scale=group_params['ls'], length_scale_bounds='fixed') \
            + WhiteKernel(noise_level=group_params['n'], noise_level_bounds="fixed")
        gpr = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, alpha=1e-10, normalize_y=False, random_state=0)

        gpr.fit(X.values, y.values)

        # Make predictions
        y_pred, y_std = gpr.predict(samples, return_std=True)

        return X, y, samples, y_pred, y_std, sample_data
