
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
