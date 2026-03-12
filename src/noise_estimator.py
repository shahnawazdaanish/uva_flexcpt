import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from src.constant_manager import ConstantManager
from src.data_loader import DataLoader
from src.scaler import Scaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class NoiseEstimator:
    def __init__(self):
        self.random_seeds = np.array(
            [85, 18, 83, 67, 26, 16, 55, 22, 49, 51], dtype=np.int64)

    def method_standard(self, X, y):
        """
        Estimate the noise level using the standard deviation of the residuals.
        """
        residuals = y - np.mean(y, axis=0)
        return np.std(residuals, ddof=1, axis=0)

    def method_empirical3(self, X, y):
        # Initialize kernel (RBF for smoothness + WhiteKernel for noise)

        # Dictionary to store results
        noise_estimates = {}
        bounds = [
            (1e-5, 1e5),
            (1e-50, 1e500)
        ]

        # Fit GPR for each output dimension
        # For y[:, 0] and y[:, 1]
        for k, output_col in enumerate(['of_1', 'of_2']):
            kernel = RBF(
                length_scale=1.0, length_scale_bounds=bounds[k]) + WhiteKernel(noise_level=1.0)
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gpr.fit(X, y[output_col])

            # Noise variance from WhiteKernel (in scaled space)
            noise_var_gpr = gpr.kernel_.k2.get_params()['noise_level']
            noise_std_gpr = np.sqrt(noise_var_gpr)

            # Noise variance from residuals (in scaled space)
            y_pred = gpr.predict(X)
            residuals = y[output_col] - y_pred
            noise_var_residuals = np.var(residuals, ddof=1)
            noise_std_residuals = np.sqrt(noise_var_residuals)

            # Inverse-transform noise standard deviations to original scale
            noise_std_gpr_orig = noise_std_gpr
            noise_var_gpr_orig = noise_std_gpr_orig ** 2
            noise_std_residuals_orig = noise_std_residuals
            noise_var_residuals_orig = noise_std_residuals_orig ** 2

            # print(gpr.kernel_.k1.k1.constant_value)

            noise_estimates[f'Output_{k+1}'] = {
                # 'GPR_noise_variance_scaled': noise_var_gpr_scaled,
                # 'GPR_noise_std_scaled': noise_std_gpr_scaled,
                # 'Residual_noise_variance_scaled': noise_var_residuals_scaled,
                # 'Residual_noise_std_scaled': noise_std_residuals_scaled,
                'GPR_noise_variance_orig': noise_var_gpr_orig,
                # 'GPR_noise_std_orig': noise_std_gpr_orig,
                'Residual_noise_variance_orig': noise_var_residuals_orig,
                # 'Residual_noise_std_orig': noise_std_residuals_orig
            }

        return noise_estimates

    def scale_data(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler

    def method_empirical(self, X, y):
        scale = False
        normalize_y = False
        scaler_X, scaler_y = None, None

        if scale:
            X, scaler_X = self.scale_data(X)
            # y, scaler_y = self.scale_data(y)
            y = y.values
        else:
            X = X.values
            y = y.values

        # Dictionary to store results
        noise_estimates = {}
        ls_bounds = [
            [(1e-2, 1e5)] * X.shape[1],
            [(1e-2, 1e2)] * X.shape[1]
        ]

        # Fit GPR for each output dimension
        plt.figure(figsize=(11, 4))
        for k in range(0, 2):  # For y[:, 0] and y[:, 1]
            plt.subplot(1, 2, k + 1)

            # Initialize kernel (RBF for smoothness + WhiteKernel for noise) #constant_value_bounds=(1e-10, 1e10)
            kernel = C(1.0, constant_value_bounds=(1e-10, 1e10)) * RBF(length_scale=[
                1.0] * X.shape[1], length_scale_bounds=ls_bounds[k]) + WhiteKernel(noise_level=1e2)
            kernel_no_sigma = RBF(length_scale=[
                                  1.0] * X.shape[1], length_scale_bounds=ls_bounds[k]) + WhiteKernel(noise_level=1.0)
            kernel_no_wk = C(1.0, constant_value_bounds=(
                1e-10, 1e10)) * RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=ls_bounds[k])

            gpr = GaussianProcessRegressor(
                kernel=kernel, random_state=42, normalize_y=normalize_y)
            gpr.fit(X, y[:, k])

            lengthscale = gpr.kernel_.k1.k2.get_params()['length_scale']

            noise_var = gpr.kernel_.k2.get_params()['noise_level']
            sigma2_f = gpr.kernel_.k1.k1.get_params()['constant_value']

            if normalize_y:
                y_std_scaler = gpr._y_train_std
                noise_var = noise_var * y_std_scaler ** 2
                sigma2_f = sigma2_f * y_std_scaler ** 2

            y_pred_all = gpr.predict(X)
            residual = y[:, k] - y_pred_all
            noise_var_residuals = np.var(residual, ddof=1)

            if scale:
                # y_std = scaler_y.scale_[k]
                x_scale = scaler_X.scale_

                # noise_var = noise_var * y_std ** 2
                # noise_var_residuals = noise_var_residuals * y_std ** 2
                lengthscale = lengthscale * x_scale
                # sigma2_f = sigma2_f * y_std ** 2

            noise_estimates[f'of_{k+1}'] = {
                'gpr_noise_var': noise_var,
                'residual_noise_var': noise_var_residuals,
                'length_scale': lengthscale,
                'sigma2_f': sigma2_f
            }

            # take one sample from X_scaled and by the same index take from y_scaled
            np.random.seed(42)
            test_random_index = np.random.randint(0, X.shape[0], size=5)

            # sort test_random_index based on the values in X_test[:, 0]
            test_random_index = test_random_index[np.argsort(
                X[test_random_index][:, 0])]

            X_test = X[test_random_index]
            y_test = y[test_random_index][:, k]

            if scale:
                X_test = scaler_X.inverse_transform(X_test)
                # y_test = scaler_y.scale_[k] * y_test + scaler_y.mean_[k]

            y_pred, y_std = gpr.predict(X_test, return_std=True)
            y_pred_no_sigma, y_std_no_sigma = GaussianProcessRegressor(
                kernel=kernel_no_sigma, random_state=42).fit(X, y[:, k]).predict(X_test, return_std=True)
            y_pred_no_wk, y_std_no_wk = GaussianProcessRegressor(
                kernel=kernel_no_wk, random_state=42).fit(X, y[:, k]).predict(X_test, return_std=True)
            # if scale:
            # y_pred = scaler_y.scale_[k] * y_pred + scaler_y.mean_[k]
            # y_std = scaler_y.scale_[k] * y_std

            # y_pred_no_sigma = scaler_y.scale_[k] * y_pred_no_sigma + scaler_y.mean_[k]
            # y_std_no_sigma = scaler_y.scale_[k] * y_std_no_sigma
            # y_pred_no_wk = scaler_y.scale_[k] * y_pred_no_wk + scaler_y.mean_[k]
            # y_std_no_wk = scaler_y.scale_[k] * y_std_no_wk

            plt.scatter(X_test[:, 0], y_pred,
                        label='Prediction', color='red', alpha=0.5)
            plt.fill_between(X_test[:, 0], y_pred - 1.96 * y_std,
                             y_pred + 1.96 * y_std, alpha=0.2, color='red')

            # plt.scatter(X_test[:, 0], y_pred_no_sigma, label='Prediction (no sigma)', color='orange', alpha=0.5)
            # plt.fill_between(X_test[:, 0], y_pred_no_sigma - 1.96 * y_std_no_sigma, y_pred_no_sigma + 1.96 * y_std_no_sigma, alpha=0.5, color='orange')

            # plt.scatter(X_test[:, 0], y_pred_no_wk, label='Prediction (no wk)', color='green', alpha=0.5)
            # plt.fill_between(X_test[:, 0], y_pred_no_wk - 1.96 * y_std_no_wk, y_pred_no_wk + 1.96 * y_std_no_wk, alpha=0.5, color='green')

            plt.scatter(X_test[:, 0], y_test,
                        label='Selected Sample', color='blue', alpha=0.2)

            # plt.xlim(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2)
            plt.title(
                f'GPR Prediction with 95% Confidence Interval (Output {k + 1})')
            plt.xlabel('Input Feature')
            plt.ylabel('Output')
            plt.legend()
        plt.tight_layout()
        plt.show()

        return noise_estimates
        # kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

        # # Dictionary to store results
        # noise_estimates = {}

        # # Fit GPR for each output dimension
        # for k, output_col in enumerate(['of_1', 'of_2']):  # For y[:, 0] and y[:, 1]
        #     gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
        #     gpr.fit(X, y[output_col])

        #     # Noise variance from WhiteKernel
        #     noise_var_gpr = gpr.kernel_.k2.get_params()['noise_level']
        #     noise_std_gpr = np.sqrt(noise_var_gpr)

        #     # Noise variance from residuals
        #     y_pred = gpr.predict(X)
        #     residuals = y[output_col] - y_pred
        #     noise_var_residuals = np.var(residuals, ddof=1)
        #     noise_std_residuals = np.sqrt(noise_var_residuals)

        #     noise_estimates[f'Output_{k+1}'] = {
        #         'GPR_noise_variance': noise_var_gpr,
        #         'GPR_noise_std': noise_std_gpr,
        #         'Residual_noise_variance': noise_var_residuals,
        #         'Residual_noise_std': noise_std_residuals
        #     }

        # print(noise_estimates)
        # return noise_estimates
        # kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))

        # y = y['of_1']
        # # Initialize and fit the GPR model
        # gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        # gpr.fit(X, y)

        # # Predict the mean
        # y_pred, y_std = gpr.predict(X, return_std=True)

        # # Compute residuals
        # residuals = y - y_pred

        # # Estimate noise level from residuals
        # noise_std_residuals = np.std(residuals, axis=0)
        # noise_variance_residuals = np.var(residuals, axis=0)

        # # Extract the optimized noise level from the WhiteKernel
        # noise_variance_gpr = gpr.kernel_.k2.get_params()['noise_level']
        # noise_std_gpr = np.sqrt(noise_variance_gpr)

        # # Print results
        # # print(f"Residual standard deviation: {noise_std_residuals}")
        # print(f"Residual variance: {noise_variance_residuals}")
        # # print(f"GPR noise standard deviation (from WhiteKernel): {noise_std_gpr}")
        # print(f"GPR noise variance (from WhiteKernel): {noise_variance_gpr}")

        # print(f"\n---------------------------\n")

        # return noise_variance_gpr

    def method_empirical2(self, X, y):
        """
        Estimate the noise level using the empirical method based on the noisy and clean training data.
        """
        X_orig = X.iloc[0, :].values.reshape(1, -1)

        empirical_vars = []
        for xu in X_orig:
            mask = (np.isclose(X, xu[0]))
            ys = y[mask]
            empirical_vars.append(
                np.var(ys, ddof=1, axis=0))
        # average noise variance across inputs
        empirical_sigma2 = np.mean(empirical_vars)
        empirical_sigma = np.sqrt(empirical_sigma2)

        # standard deviation of the noise estimate
        return np.std(y - empirical_sigma, axis=0)

    def best_fit_mle(self, X, y, bounds, params):
        best_lml = None
        nv = None
        sv = None
        ls = None
        
        print(bounds)

        for seed in self.random_seeds:
            np.random.seed(seed)

            kernel = C(params['constant'], constant_value_bounds=bounds['constant_bounds']) \
                * RBF(length_scale=params['lengthscale'], length_scale_bounds=bounds['ls_bounds']) \
                + WhiteKernel(noise_level=params['init_noise_level'],
                              noise_level_bounds=bounds['noise_bounds'])

            # print(f"Initial kernel: {kernel}")

            gpr_mle = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, normalize_y=False, random_state=seed)
            gpr_mle.fit(X, y)

            # print(f"Fitted kernel: {gpr_mle.kernel_}")

            if best_lml is None or gpr_mle.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr_mle.log_marginal_likelihood_value_
                sv = gpr_mle.kernel_.k1.k1.constant_value
                nv = gpr_mle.kernel_.k2.noise_level
                ls = gpr_mle.kernel_.k1.k2.length_scale
        return {
            'noise_level': nv,
            'signal_variance': sv,
            'length_scale': ls
        }

    def method_mle_ref(self, X, y, bounds=None, params=None):
        print(f"Output 1: -----")
        learned_params1 = self.best_fit_mle(
            X,
            y['of_1'],
            bounds=bounds['of_1'],
            params=params['of_1'],
        )

        print(f"Output 2: -----")
        learned_params2 = self.best_fit_mle(
            X,
            y['of_2'],
            bounds=bounds['of_2'],
            params=params['of_2'],
        )

        learned_params = {
            'of_1': {
                'noise_level': learned_params1['noise_level'],
                'signal_variance': learned_params1['signal_variance'],
                'length_scale': learned_params1['length_scale']
            },
            'of_2': {
                'noise_level': learned_params2['noise_level'],
                'signal_variance': learned_params2['signal_variance'],
                'length_scale': learned_params2['length_scale']
            }
        }
        return learned_params

    def method_mle(self, X, y):
        """
        Estimate the noise level using the Maximum Likelihood Estimation (MLE) method based on the noisy and clean training data.
        """
        best_lml1, best_lml2 = None, None
        nv_1, nv_2 = None, None
        sv_1, sv_2 = None, None
        ls_ref_bounds = [
            (1e-5, 1e30),
            (1e-20, 1e100)
        ]
        ls_main_bounds = [
            (1e-2, 1e10),
            (1e-2, 1e20)
        ]

        for seed in self.random_seeds:
            np.random.seed(seed)

            kernel_of1 = C(1.0, constant_value_bounds=(1e-20, 1e2)) * RBF(length_scale=[
                1.0] * X.shape[1], length_scale_bounds=ls_main_bounds[0]) + WhiteKernel(noise_level=1.0)
            kernel_of2 = C(1.0, constant_value_bounds=(1e-10, 1e10)) * RBF(length_scale=[
                1.0] * X.shape[1], length_scale_bounds=ls_main_bounds[1]) + WhiteKernel(noise_level=1.0)

            gpr_mle_of_1 = GaussianProcessRegressor(
                kernel=kernel_of1, n_restarts_optimizer=15, normalize_y=True, random_state=seed)
            gpr_mle_of_1.fit(X, y['of_1'])

            gpr_mle_of_2 = GaussianProcessRegressor(
                kernel=kernel_of2, n_restarts_optimizer=15, normalize_y=True, random_state=seed)
            gpr_mle_of_2.fit(X, y['of_2'])

            if best_lml1 is None or gpr_mle_of_1.log_marginal_likelihood_value_ > best_lml1:
                best_lml1 = gpr_mle_of_1.log_marginal_likelihood_value_
                sv_1 = gpr_mle_of_1.kernel_.k1.k1.constant_value
                nv_1 = gpr_mle_of_1.kernel_.k2.noise_level

                y_std_scaler = gpr_mle_of_1._y_train_std
                nv_1 = nv_1 * y_std_scaler ** 2
                sv_1 = sv_1 * y_std_scaler ** 2

            if best_lml2 is None or gpr_mle_of_2.log_marginal_likelihood_value_ > best_lml2:
                best_lml2 = gpr_mle_of_2.log_marginal_likelihood_value_
                sv_2 = gpr_mle_of_2.kernel_.k1.k1.constant_value
                nv_2 = gpr_mle_of_2.kernel_.k2.noise_level

                y_std_scaler_1 = gpr_mle_of_2._y_train_std
                nv_2 = nv_2 * y_std_scaler_1 ** 2
                sv_2 = sv_2 * y_std_scaler_1 ** 2

        return nv_1, sv_1, nv_2, sv_2

    def method_mle_basic(self, X, y):
        """
        Estimate the noise level using the Maximum Likelihood Estimation (MLE) method based on the noisy and clean training data.
        """
        # dists = pairwise_distances(X)
        # ls0 = np.median(dists)
        # print(ls0/100, ls0 * 100)
        # kernel = RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=(1e-20, 1e10)) + WhiteKernel(noise_level=1.0)
        kernel = RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=(
            1e-4, 1e5)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-9, 1e2))
        gpr_mle = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
        gpr_mle.fit(X, y)

        return gpr_mle.kernel_.k2.noise_level

    def method_in_data_residual(self, y, y_pred, dof=None):
        r = y - y_pred
        sse = np.sum(r**2)
        N = y.shape[0]
        denom = N if dof is None else max(N - dof, 1)
        return sse / denom

    def method_residual(self, X, y):
        """
        Estimate the noise level using the Maximum Likelihood Estimation (MLE) method based on the noisy and clean training data.
        """
        noises = {}
        for idx in range(2):
            # get first feature from y
            feature = y.columns[idx]

            kernel = RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=(
                1e-10, 1e10)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
            gpr_mle = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, normalize_y=True, random_state=42)
            gpr_mle.fit(X, y[feature])

            # create some samples randomly choosen from X which is 7D
            X_sample = X.sample(n=8, random_state=42)
            y_sample = y.loc[X_sample.index]

            y_pred = gpr_mle.predict(X_sample)
            noises[idx] = self.method_in_data_residual(
                y_sample[feature].values, y_pred)

        return noises

    def estimate(self, X, y, method=ConstantManager().EMPIRICAL_METHOD):
        """
        Estimate the noise level based on the noisy and clean training data.
        """
        if method == ConstantManager().EMPIRICAL_METHOD:
            return self.method_empirical(X, y)
        elif method == ConstantManager().STANDARD_NOISE_METHOD:
            return self.method_standard(X, y)
        elif method == ConstantManager().MLE_METHOD:
            return self.method_mle_ref(X, y)
        elif method == ConstantManager().MLE_BASIC_METHOD:
            return self.method_mle_basic(X, y)
        elif method == ConstantManager().RESIDUAL_METHOD:
            return self.method_residual(X, y)
        else:
            raise ValueError(f"Unknown method: {method}")

    def estimate_all_groups(self, df_group1, df_group2, df_group3, method=ConstantManager().EMPIRICAL_METHOD, output_column=None):
        """
        Estimate the noise level for all groups.
        """
        g1_X = DataLoader().load_X(df_group1, abstracted=True)
        g1_y = DataLoader().load_y(df_group1, abstracted=True)
        if output_column is not None:
            g1_y = g1_y[[output_column]]

        g2_X = DataLoader().load_X(df_group2, abstracted=True)
        g2_y = DataLoader().load_y(df_group2, abstracted=True)
        if output_column is not None:
            g2_y = g2_y[[output_column]]

        g3_X = DataLoader().load_X(df_group3, abstracted=True)
        g3_y = DataLoader().load_y(df_group3, abstracted=True)
        if output_column is not None:
            g3_y = g3_y[[output_column]]

        stdev_g1 = self.estimate(g1_X, g1_y, method=method)
        stdev_g2 = self.estimate(g2_X, g2_y, method=method)
        stdev_g3 = self.estimate(g3_X, g3_y, method=method)

        return stdev_g1, stdev_g2, stdev_g3
