import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.constant_manager import ConstantManager
from src.scaler import Scaler


class Predictor:
    def __init__(self):
        self.input_columns = ConstantManager().NEW_INPUT_COLUMNS
        self.output_columns = ConstantManager().NEW_OUTPUT_COLUMNS
        
    def generate_sorted_samples(self, one_sample, input_priority, min_val, max_val, n_samples=50):
        """
        Generate n_samples from one sample where only the input_priority column varies
        linearly between min_val and max_val. Other features remain constant.

        Parameters
        ----------
        one_sample : pd.Series or pd.DataFrame (1 row)
            The base sample (row) to replicate.
        input_priority : str
            Column name to vary.
        min_val, max_val : float
            Range of values for the priority input.
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        pd.DataFrame
            DataFrame of shape (n_samples, n_features) with sorted varying column.
        np.ndarray
            Sorted values used for the varying column (for plotting).
        """
        if isinstance(one_sample, pd.DataFrame):
            one_sample = one_sample.iloc[0]  # ensure it's a Series
            sample_index = one_sample.index
        elif isinstance(one_sample, np.ndarray):
            one_sample = one_sample
            sample_index = list(range(0, len(self.input_columns)))


        # Create n_samples copies of the row
        df = pd.DataFrame([one_sample] * n_samples, columns=sample_index)
        
        # Replace the priority column with sorted values
        sorted_values = np.linspace(min_val, max_val, n_samples)
        df[input_priority] = sorted_values

        return df, sorted_values

    def create_n_sample_from_one(self, sample_data, input_priority, min_val, max_val, n_test_samples=10):
        samples = []
        # convert to numpy row vector
        sample = sample_data.values.flatten()  
        col_idx = list(sample_data.columns).index(input_priority)
        
        for _ in range(n_test_samples):
            # make sorted samples
            new_sample = sample.copy()
            
            new_sample[col_idx] = np.random.uniform(min_val, max_val)
            samples.append(new_sample)

        return np.array(samples)

    def fit_predict_only_one_point(self, dataset, ls, alpha, input_priority='if_1', output_column=None, n_test_samples=50, sigma_f2=[], samples=None):
        X = dataset[self.input_columns].values
        if output_column is None:
            y = dataset[self.output_columns].values
        else:
            y = dataset[output_column].values.reshape(-1, 1)

        if samples is None:
            sample_data = dataset.sample(n_test_samples)[self.input_columns].values
        else:
            sample_data = samples[self.input_columns].values
            
            
        # create X_train and y_train avoiding samples in it, sample_data has not attribute called index
        X_train = X[~np.isin(X, sample_data).all(axis=1)]
        y_train = y[~np.isin(X, sample_data).all(axis=1)]

        # sample indices from ndarray
        sample_ids = [np.where((X == sample).all(axis=1))[0][0] for sample in sample_data]

        # sample_idx = np.where((X == sample_data).all(axis=1))[0][0]

        # # create X_train removing the sample
        # X_train = np.delete(X, sample_idx, axis=0)
        # y_train = np.delete(y, sample_idx, axis=0)
        kernel = C(sigma_f2, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds='fixed') + WhiteKernel(noise_level=alpha, noise_level_bounds="fixed")
        gpr = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, alpha=alpha, normalize_y=True, random_state=0)

        # Fit the model
        gpr.fit(X_train,y_train)

        # Make predictions
        y_pred, y_std = gpr.predict(sample_data, return_std=True)

        # calculate rmse
        rmse = np.sqrt(np.mean((y[sample_ids] - y_pred) ** 2))
        r2 = r2_score(y[sample_ids], y_pred)

        return X, y, sample_data, y_pred, y_std, sample_ids, rmse, r2

    def fit_predict_single_point_custom(self, X, y, sample, min_val, max_val, kernel, input_priority=0, n_test_samples=50, seed=0):
        df_samples = pd.DataFrame(data = np.tile(sample, (n_test_samples, 1)))
        
        # Replace the priority column with sorted values
        sorted_values = np.linspace(min_val, max_val, n_test_samples)
        df_samples[input_priority] = sorted_values
        samples = df_samples.values
        

        kernel_ = C(kernel.k1.k1.constant_value, constant_value_bounds="fixed") \
            * RBF(length_scale=kernel.k1.k2.length_scale, length_scale_bounds='fixed') #+ WhiteKernel(noise_level=alpha * 5, noise_level_bounds="fixed")

        gpr = GaussianProcessRegressor(
            kernel=kernel_, optimizer=None, normalize_y=False, alpha=kernel.k2.noise_level)

        # Fit the model
        gpr.fit(X, y)

        # Make predictions
        y_pred, y_std = gpr.predict(samples, return_std=True)

        return y_pred, y_std, samples

    def fit_predict_single_point(self, dataset, ls, alpha, input_priority='if_1', output_column=None, n_test_samples=50, sigma_f2=[]):
        X = dataset[self.input_columns].values
        if output_column is None:
            y = dataset[self.output_columns].values
        else:
            y = dataset[output_column].values.reshape(-1, 1)

        sample_data = dataset.sample(1, random_state=50)
        sample_idx = sample_data.index[0]
        min_val = dataset[input_priority].min()
        max_val = dataset[input_priority].max()
        
        # keep other features same but create 50 samples variying input priority by its low and high value
        # samples = self.create_n_sample_from_one(sample_data, input_priority, min_val, max_val, n_test_samples)
        samples = self.generate_sorted_samples(sample_data[self.input_columns], input_priority, min_val, max_val, n_test_samples)[0].values

        # print(samples)
        # return
        
        kernel = C(sigma_f2, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds='fixed') #+ WhiteKernel(noise_level=alpha * 5, noise_level_bounds="fixed")
        gpr = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, alpha=alpha, normalize_y=True, random_state=0)

        # Fit the model
        gpr.fit(X, y)

        # Make predictions
        y_pred, y_std = gpr.predict(samples, return_std=True)

        return X, y, samples, y_pred, y_std, sample_data

    def fit_predict_test_train_split(self, dataset, ls, alpha, output_column=None, test_size=0.2, sigma_f2=[]):
        X = dataset[self.input_columns].values

        if output_column is None:
            y = dataset[self.output_columns].values
        else:
            y = dataset[output_column].values.reshape(-1, 1)
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


        kernel = C(sigma_f2, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds='fixed') # + WhiteKernel(noise_level=alpha, noise_level_bounds="fixed")
        gpr = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, alpha=alpha, normalize_y=True, random_state=0)

        gpr.fit(X_train, y_train)
        y_pred, y_std = gpr.predict(X_test, return_std=True)

        return X, y, X_test, y_pred, y_std


    def predict(self, dataset, ls, alpha, input_priority='if_1', output_column=None, n_test_samples=500, samples=None, test_size=0.2, sigma_f2=[], method=None):
        """
        Make predictions using the provided model.

        Parameters:
        dataset (pd.DataFrame): Input features for prediction.
        output_column (str): The name of the output column to predict.

        Returns:
        array: Predicted values.
        """
        
        if method is ConstantManager().METHOD_SINGLE_SAMPLE:
            return self.fit_predict_single_point(dataset, ls, alpha, input_priority=input_priority, output_column=output_column, n_test_samples=n_test_samples, sigma_f2=sigma_f2)
        elif method is ConstantManager().METHOD_TEST_TRAIN_SPLIT:
            return self.fit_predict_test_train_split(dataset, ls, alpha, output_column=output_column, test_size=test_size, sigma_f2=sigma_f2)

        elif method is ConstantManager().METHOD_ONLY_ONE_SAMPLE:
            return self.fit_predict_only_one_point(dataset, ls, alpha, input_priority=input_priority, output_column=output_column, n_test_samples=n_test_samples, sigma_f2=sigma_f2, samples=samples)
        # input_cols = self.input_columns
        # output_cols = self.output_columns
        
        # if output_column not in dataset.columns:
        #     raise ValueError(
        #         f"Output column '{output_column}' not found in the dataset.")

        # # Step 1: Prepare data
        # X = dataset[input_cols].values
        # # y = dataset[output_column].values.reshape(-1, 1)
        # # X -= X.mean(axis=0)  # Center the data
        # # X /= X.std(axis=0)  # Scale the data

        # if output_column is None:
        #     y = dataset[self.output_columns].values
        # else:
        #     y = dataset[output_column].values.reshape(-1, 1)

        #     # if type(alpha) is not int and alpha.ndim == 0:
        #     #     alpha = alpha
        #     # else:
        #     #     alpha = alpha[output_cols.index(output_column)].reshape(-1, 1)
        
        # scale_ratio = 1.0
        # if scaled:
        #     X = scaler.fit_transform(X)
        #     x_scale_ratio = scaler.ratio()
        #     ls = ls / x_scale_ratio
            
        #     y = scaler.fit_transform(y)
        #     y_scale_ratio = scaler.ratio()
            
        #     if output_column is not None:
        #         # alpha_column = self.output_columns.index(output_column)
        #         alpha = alpha / (y_scale_ratio ** 2)
        #         sigma_f2 = sigma_f2 / (y_scale_ratio ** 2)
        #     else:
        #         alpha = alpha / (y_scale_ratio ** 2)
        #         # take mean of alpha and make it same shape as y
        #         alpha_mean = np.mean(alpha)
        #         alpha = np.array([alpha_mean] * y.shape[0])
            
            
        # # update alpha based on scaling ratio
        # # alpha = alpha / (y_scale_ratio ** 2)
        # # ls_priority = ls[input_cols.index(input_priority)] # x_scale_ratio[input_cols.index(input_priority)]
        # # ls_priorities = np.ones(len(input_cols))
        # # ls_priorities[input_cols.index(input_priority)] = ls_priority
        
        # # print(f"Predicting for input priority: {input_priority}, length scale: {ls_priority}, alpha: {alpha}")

        # # take min and max values of input priority column and create n_test_samples samples with other columns as zeros
        # # create n_test_samples samples for prediction
        # # input_priority_values = np.linspace(X[:, input_cols.index(input_priority)].min(),
        # #                                     X[:, input_cols.index(
        # #                                         input_priority)].max(),
        # #                                     n_test_samples).reshape(-1, 1)
        
        # # create a X_sample filled with mean value of each column of n_test_samples
                
        # # col_means = []
        # # for i, col in enumerate(input_cols):
        # #     mean = np.mean(X[:, i])
        # #     col_means.append(mean)
            
        # # X_sample = np.tile(col_means, (n_test_samples, 1))
        # # # fill the input_priority column with input_priority_values
        # # X_sample[:, input_cols.index(input_priority)] = input_priority_values.flatten()

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # # take one sample from X of ndarray
        # # X_sample = X[np.random.choice(X.shape[0], 40)]
        
        # # sort the sample based on input priority column
        # # X_sample = X_sample[np.argsort(X_sample[:, input_cols.index(input_priority)])]

        # # get index of the sample
        # # sample_idx = np.where(X == X_sample)[0][0]
        

        # # X_sample = np.zeros((n_test_samples, len(input_cols)))
        # # X_sample[:, input_cols.index(
        #     # input_priority)] = input_priority_values.flatten()

        # # # keep values of input_priority only and replace others with zeros
        # # if input_priority in input_cols:
        # #     # fill all rows with mean value of input_priority column
        # #     X_sample = np.zeros((n_test_samples, len(input_cols)))
        # #     X_sample[:, input_cols.index(
        # #         input_priority)] = input_priority_values.flatten()
        # # else:
        # #     raise ValueError(
        # #         f"Input priority '{input_priority}' not found in the dataset.")

        # # kernel = C(sigma_f2, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds='fixed')
        # # gpr = GaussianProcessRegressor(
        # #     kernel=kernel, optimizer=None, alpha=alpha, normalize_y=True, random_state=42)

        # kernel = C(sigma_f2, constant_value_bounds="fixed") * RBF(length_scale=ls, length_scale_bounds='fixed') # + WhiteKernel(noise_level=alpha, noise_level_bounds="fixed")
        # gpr = GaussianProcessRegressor(
        #     kernel=kernel, optimizer=None, alpha=alpha, normalize_y=True, random_state=0)

        # # Fit the model
        # gpr.fit(X_train, y_train)

        # # Make predictions
        # y_pred, y_std = gpr.predict(X_test, return_std=True)

        # return X, y, X_test, y_pred, y_std

    def predict_all_input_features(self, dataset, ls, alpha, output_column='of_1', n_test_samples=500, samples=None, sigma_f2=[], test_size=None, method=ConstantManager().METHOD_TEST_TRAIN_SPLIT):
        """
        Make predictions for all input features using the provided model.

        Parameters:
        dataset (pd.DataFrame): Input features for prediction.
        output_column (str): The name of the output column to predict.

        Returns:
        array: Predicted values.
        """
        predictions = []
        for i in range(len(self.input_columns)):
            input_priority = self.input_columns[i]
            predictions.append(self.predict(dataset, ls, alpha, input_priority=input_priority,
                               output_column=output_column, n_test_samples=n_test_samples, samples=samples, sigma_f2=sigma_f2, test_size=test_size, method=method))
        return predictions
