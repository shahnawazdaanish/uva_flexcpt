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


class Analyzer:
    def __init__(self, scaled=False):
        self.input_columns = ConstantManager().NEW_INPUT_COLUMNS
        self.categorical_columns = ['cat_1_1', 'cat_1_2']
        self.output_columns = ConstantManager().NEW_OUTPUT_COLUMNS
        self.random_seeds = np.array(
            [85, 18, 83, 67, 26, 16, 55, 22, 49, 51], dtype=np.int64)
        self.scaled = scaled
        self.scaler = Scaler(enable_scaling=scaled)

    def find_top_features_ard(self, data, input_features, output_column, n_features=5):
        """
        Find the top features based on ARD (Automatic Relevance Determination) from the data.

        :param data: DataFrame containing the data
        :param output_column: The output column for which to find the top features
        :param n_features: Number of top features to return
        :return: List of top feature names
        """
        inputs = data[input_features]

        X = inputs.values
        y = data[output_column].values

        # select top n features
        selected = SelectKBest(mutual_info_regression, k=n_features).fit(X, y)
        # print top n features name with comma separated sorted by score
        sorted_indices = np.argsort(selected.scores_)[-n_features:][::-1]
        selected_features = np.array(input_features)[sorted_indices]
        print(f"Top {n_features} features based on ARD:",
              ', '.join(selected_features))
        return selected_features.tolist()

    def learn_from_data(self, data, output_column, bounds=(1e-2, 1e2), add_categorical_data=False):
        best_lml = None
        learned_length_scale = None

        for seed in self.random_seeds:
            np.random.seed(seed)

            if add_categorical_data:
                X = data[self.input_columns + self.categorical_columns]
            else:
                X = data[self.input_columns]
            y = data[output_column]

            sigma2_f = 0.1
            noise_var = 0.1 ** 2
            scale_ratio = [1.0] * X.shape[1]
            if self.scaled:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # x_scale = 1.0 / (scaler.scale_[0] ** 2)
                # sigma2_f *= x_scale

                # y = scaler.fit_transform(
                #     data[output_column].values.reshape(-1, 1)).ravel()
                # sigma_y = scaler.scale_[0]
                # noise_var *= sigma_y ** 2

            # print("scale_ratio:", scale_ratio)
            # print("is scaled:", self.scaled)
            # print("is_scaling_enabled:", self.scaler.is_scaling_enabled())

            kernel = C(sigma2_f, (1e-10, 1e10)) * RBF(length_scale=[1.0] * X.shape[1],
                                                      length_scale_bounds=bounds) + WhiteKernel(noise_level=noise_var)
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, random_state=seed)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                learned_length_scale = gpr.kernel_.k1.k2.length_scale
                sigma2_f = gpr.kernel_.k1.k1.constant_value
                noise_var = gpr.kernel_.k2.noise_level

        # return {
        #     'lengths': lengths_scaled,
        #     'sigma_f2': sigma_f2_scaled,
        #     'noise_var': noise_var_scaled
        # }
        return {
            'ls': learned_length_scale,
            'sigma_f': np.sqrt(sigma2_f),
            'sigma_n': np.sqrt(noise_var)
        }

    def learn_feature_lengthscale_custom(self, data, output_column, bounds=(1e-2, 1e2), constant_bounds=(1e-3, 1e3), alpha=1e-6, add_categorical_data=False):
        best_lml = None
        lengths_orig, sigma_f2_orig, noise_var_orig, y_mean, y_std = None, None, None, None, None

        for seed in self.random_seeds:
            np.random.seed(seed)

            if add_categorical_data:
                X = data[self.input_columns + self.categorical_columns]
            else:
                X = data[self.input_columns]
            y = data[output_column]

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_std = scaler.scale_
            # X -= X.mean(axis=0)
            # X_std = X.std(axis=0)
            # X /= X_std

            kernel = C(1.0, constant_bounds) * RBF(length_scale=[100.0] * X.shape[1],
                                                   length_scale_bounds=bounds) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e5))

            regressor = Regressors(X, y, kernel=kernel, alpha=alpha)
            gpr = regressor.conditional_lml(num_of_restart=10)
            # gpr = GaussianProcessRegressor(
            #     kernel=kernel, n_restarts_optimizer=10, random_state=seed, normalize_y=True)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                lengths_orig = np.asarray(
                    gpr.kernel_.k1.k2.length_scale) * X_std  # .values
                sigma_f2_orig = gpr.kernel_.k1.k1.constant_value
                noise_var_orig = gpr.kernel_.k2.noise_level
                y_mean = gpr._y_train_mean
                y_std = gpr._y_train_std

        return {
            'ls': lengths_orig,
            'sigma_f': np.sqrt(sigma_f2_orig),
            'sigma_n': np.sqrt(noise_var_orig),
            'y_mean': y_mean,
            'y_std': y_std
        }

    def learn_hyperparameters(
        self, data, output_column, bounds=(1e-2, 1e2), constant_bounds=(1e-3, 1e3), alpha=1e-6, scaleX=False, scaleY=False, normalizeY=False, return_original_scale=False
    ):
        best_lml = None

        for seed in self.random_seeds:
            np.random.seed(seed)
            
            X = data[self.input_columns]

            if scaleX:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                X_std = scaler.scale_

            y = data[output_column]
            if scaleY:
                scaler_y = StandardScaler()
                y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
                y_std = scaler_y.scale_[0]

            kernel = C(1.0, constant_bounds) * RBF(length_scale=[1.0] * X.shape[1],
                                                   length_scale_bounds=bounds) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e5))
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, random_state=seed, normalize_y=normalizeY)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                y_mean = gpr._y_train_mean
                y_std_scaler = gpr._y_train_std

                ells = gpr.kernel_.k1.k2.length_scale
                signal_var = gpr.kernel_.k1.k1.constant_value
                noise_var = gpr.kernel_.k2.noise_level
                
                if return_original_scale:
                    if scaleX and X_std is not None:
                        ells = np.asarray(gpr.kernel_.k1.k2.length_scale) * X_std
                    if scaleY and y_std_scaler is not None:
                        signal_var = gpr.kernel_.k1.k1.constant_value * y_std_scaler**2
                        noise_var = gpr.kernel_.k2.noise_level * y_std_scaler**2

        return {
            'ls': ells,
            'signal_var': signal_var,
            'noise_var': noise_var,
            'y_mean': y_mean,
            'y_std': y_std_scaler
        }

    def learn_feature_lengthscale_all_scaled(self, X, y, bounds=(1e-2, 1e2), constant_bounds=(1e-3, 1e3), noise_bounds = (1e-5, 1e5), noise_level=1e-2):
        best_lml = None
        optimized_kernel = None

        for seed in self.random_seeds:
            np.random.seed(seed)

            kernel = C(1.0, constant_bounds) * RBF(length_scale=[1.0] * X.shape[1],
                                                   length_scale_bounds=bounds) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_bounds)
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, random_state=seed, normalize_y=False)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                optimized_kernel = gpr.kernel_

        return optimized_kernel

    def learn_feature_lengthscale_new3(self, data, output_column, bounds=(1e-2, 1e2), constant_bounds=(1e-3, 1e3), alpha=1e-6, add_categorical_data=False):
        best_lml = None
        lengths_orig, sigma_f2_orig, noise_var_orig, y_mean, y_std_scaler = None, None, None, None, None

        for seed in self.random_seeds:
            np.random.seed(seed)

            scaler = StandardScaler()
            X = scaler.fit_transform(data[self.input_columns])
            X_std = scaler.scale_

            y = data[output_column]

            kernel = C(1.0, constant_bounds) * RBF(length_scale=[1.0] * X.shape[1],
                                                   length_scale_bounds=bounds) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e5))
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, random_state=seed, normalize_y=True)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                y_mean = gpr._y_train_mean
                y_std_scaler = gpr._y_train_std

                ells = np.asarray(gpr.kernel_.k1.k2.length_scale) * X_std
                signal_var = gpr.kernel_.k1.k1.constant_value * y_std_scaler**2
                noise_var = gpr.kernel_.k2.noise_level * y_std_scaler**2

        return {
            'ls': ells,
            'signal_var': signal_var,
            'noise_var': noise_var,
            'y_mean': y_mean,
            'y_std': y_std_scaler
        }

    def learn_feature_lengthscale_new2(self, data, output_column, bounds=(1e-2, 1e2), constant_bounds=(1e-3, 1e3), alpha=1e-6, add_categorical_data=False):
        best_lml = None
        lengths_orig, sigma_f2_orig, noise_var_orig, y_mean, y_std = None, None, None, None, None

        for seed in self.random_seeds:
            np.random.seed(seed)

            if add_categorical_data:
                X = data[self.input_columns + self.categorical_columns]
            else:
                X = data[self.input_columns]
            y = data[output_column]

            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_std = scaler.scale_

            kernel = C(1.0, constant_bounds) * RBF(length_scale=[1.0] * X.shape[1],
                                                   length_scale_bounds=bounds) + WhiteKernel(noise_level=alpha, noise_level_bounds=(1e-5, 1e5))
            # kernel = RBF(length_scale=[1.0] * X.shape[1],
            #         length_scale_bounds=bounds) #+ WhiteKernel(noise_level = alpha, noise_level_bounds=(1e-5, 1e5))
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=15, random_state=seed, normalize_y=True)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                lengths_orig = np.asarray(
                    gpr.kernel_.k1.k2.length_scale)  # * X_std #.values
                sigma_f2_orig = gpr.kernel_.k1.k1.constant_value
                noise_var_orig = gpr.kernel_.k2.noise_level
                y_mean = gpr._y_train_mean
                y_std = gpr._y_train_std

        return {
            'ls': lengths_orig,
            'sigma_f': np.sqrt(sigma_f2_orig),
            'sigma_n': np.sqrt(noise_var_orig),
            'y_mean': y_mean,
            'y_std': y_std
        }

    def learn_feature_lengthscale_new(self, data, output_column, bounds=(1e-2, 1e2), alpha=1e-6, add_categorical_data=False):
        best_lml = None
        learned_length_scale = None

        for seed in self.random_seeds:
            np.random.seed(seed)

            if add_categorical_data:
                X = data[self.input_columns + self.categorical_columns]
            else:
                X = data[self.input_columns]
            y = data[output_column]

            scale_ratio = [1.0] * X.shape[1]
            sigma_y = 0.0
            if self.scaled:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                x_scale = 1.0 / (scaler.scale_[0] ** 2)

                scale_ratio = scaler.scale_
                y = scaler.fit_transform(
                    data[output_column].values.reshape(-1, 1)).ravel()
                sigma_y = scaler.scale_[0]

            # print("scale_ratio:", scale_ratio)
            # print("is scaled:", self.scaled)
            # print("is_scaling_enabled:", self.scaler.is_scaling_enabled())

            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * X.shape[1],
                                               length_scale_bounds=bounds)
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, alpha=alpha, random_state=seed)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                learned_length_scale = gpr.kernel_.k2.length_scale

                sigma_f2_scaled = gpr.kernel_.k1.constant_value
                lengths_scaled = gpr.kernel_.k2.length_scale

                # Convert to original units
                lengths_orig = lengths_scaled * scale_ratio
                sigma_f2_orig = sigma_f2_scaled * sigma_y**2

        # return {
        #     'lengths': lengths_scaled,
        #     'sigma_f2': sigma_f2_scaled,
        #     'noise_var': noise_var_scaled
        # }
        return {
            'lengths': lengths_orig,
            'sigma_f2': sigma_f2_orig
        }

    def learn_feature_lengthscale(self, data, output_column, bounds=(1e-2, 1e2), alpha=1e-6, add_categorical_data=False):
        best_lml = None
        learned_length_scale = None

        for seed in self.random_seeds:
            np.random.seed(seed)

            if add_categorical_data:
                X = data[self.input_columns + self.categorical_columns]
            else:
                X = data[self.input_columns]
            y = data[output_column]

            scale_ratio = 1.0
            if self.scaled:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                scale_ratio = scaler.scale_[0]

                print("X scale_ratio:", scale_ratio)
                y = scaler.fit_transform(
                    data[output_column].values.reshape(-1, 1)).ravel()
                print("y scale_ratio:", scaler.scale_[0])

            # print("scale_ratio:", scale_ratio)
            # print("is scaled:", self.scaled)
            # print("is_scaling_enabled:", self.scaler.is_scaling_enabled())

            kernel = C(1.0, constant_value_bounds="fixed") * \
                RBF(length_scale=[0.1] * X.shape[1],
                    length_scale_bounds=bounds)
            gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, alpha=alpha, random_state=seed, normalize_y=True)

            # Fit the model
            gpr.fit(X, y)

            # keep the LML value and compare it with the previous LML value and update the lml if it is better
            if best_lml is None or gpr.log_marginal_likelihood_value_ > best_lml:
                best_lml = gpr.log_marginal_likelihood_value_
                learned_length_scale = gpr.kernel_.k2.length_scale

        return learned_length_scale.reshape(-1, 1)

    def learn_lengthscale_of_all(self, groups=[], bounds=[], alphas=[]):
        ls = {}
        for i, output_column in enumerate(self.output_columns):
            print(f"Learning lengthscale for output column: {output_column}")
            ls[output_column] = {}
            ls[output_column][0] = self.learn_feature_lengthscale_new(
                groups[0], output_column, bounds=bounds[i][0], alpha=alphas[i][0] ** 2)
            ls[output_column][1] = self.learn_feature_lengthscale_new(
                groups[1], output_column, bounds=bounds[i][1], alpha=alphas[i][1] ** 2)
            ls[output_column][2] = self.learn_feature_lengthscale_new(
                groups[2], output_column, bounds=bounds[i][2], alpha=alphas[i][2] ** 2)

        return ls

    def learn_hyperparameters_for_all_features(self, groups=[], bounds=[], constant_bounds=[], alphas=[]):
        hyperparams = {}
        for i, output_column in enumerate(self.output_columns):
            print(
                f"Learning hyperparameters for output column: {output_column}")
            hyperparams[output_column] = {}
            # hyperparams[output_column][0] = self.learn_from_data(
            #     groups[0], output_column, bounds=bounds[i][0])
            # hyperparams[output_column][1] = self.learn_from_data(
            #     groups[1], output_column, bounds=bounds[i][1])
            # hyperparams[output_column][2] = self.learn_from_data(
            #     groups[2], output_column, bounds=bounds[i][2])

            hyperparams[output_column][0] = self.learn_feature_lengthscale_new3(
                groups[0], output_column, bounds=bounds[i][0], constant_bounds=constant_bounds[i][0], alpha=alphas[i][0])
            hyperparams[output_column][1] = self.learn_feature_lengthscale_new3(
                groups[1], output_column, bounds=bounds[i][1], constant_bounds=constant_bounds[i][1], alpha=alphas[i][1])
            hyperparams[output_column][2] = self.learn_feature_lengthscale_new3(
                groups[2], output_column, bounds=bounds[i][2], constant_bounds=constant_bounds[i][2], alpha=alphas[i][2])

            # hyperparams[output_column][0] = self.learn_feature_lengthscale(
            #     groups[0], output_column, bounds=bounds[i][0], alpha=alphas[i][0])
            # hyperparams[output_column][1] = self.learn_feature_lengthscale(
            #     groups[1], output_column, bounds=bounds[i][1], alpha=alphas[i][1])
            # hyperparams[output_column][2] = self.learn_feature_lengthscale(
            #     groups[2], output_column, bounds=bounds[i][2], alpha=alphas[i][2])

        return hyperparams

    def find_outliers(self, dataset, column=None, threshold=3.0):
        outliers = {}
        if column is not None:
            df = dataset.copy()
            # Compute the IQR
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = df[(df[column] < lower_bound)
                                  | (df[column] > upper_bound)]

        return outliers

        # # Compute the z-scores of the dataset
        # z_scores = np.abs(stats.zscore(dataset))
        # # Identify outliers based on the z-score threshold
        # outliers = np.where(z_scores > threshold)
        # return outliers
