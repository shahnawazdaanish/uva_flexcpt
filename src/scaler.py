
from sklearn.preprocessing import StandardScaler


class Scaler:
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, data, columns=None):
        if columns is not None:
            data = data[columns]
        self.scaler.fit(data)

    def transform(self, data, columns=None):
        if columns is not None:
            data = data[columns]
        return self.scaler.transform(data)

    def fit_transform(self, data, columns=None):
        if columns is not None:
            data = data[columns]
        return self.scaler.fit_transform(data)
    
    def get_scale_factors(self):
        if hasattr(self.scaler, 'scale_'):
            return self.scaler.scale_
        else:
            raise AttributeError("The underlying scaler does not have scale_ attribute.")
        
    def get_scale_means(self):
        if(hasattr(self.scaler, 'mean_')):
            return self.scaler.mean_
        else:
            raise AttributeError("The underlying scaler does not have mean_ attribute")

    def inverse_transform(self, data, dim_idx=None):
        """
        data can be a single value (if dim_idx is provided) or a full array (if dim_idx is None).
        """
        if dim_idx is not None:
            sigma = self.scaler.scale_[dim_idx]
            mu = self.scaler.mean_[dim_idx]
            return (data * sigma) + mu
        else:
            return self.scaler.inverse_transform(data)
