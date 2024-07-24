from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
from icecream import ic


class Scaler:
    def __init__(self, strategy=None):
        if strategy is None:
            self.scaler = None
        elif strategy == "standard":
            self.scaler = StandardScaler()
        elif strategy == "minmax":
            self.scaler = MinMaxScaler()
        elif strategy == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling: {strategy}")

    def fit_transform(self, X):
        if self.scaler is None:
            return X
        # もしXがseriesの場合
        if isinstance(X, pd.Series):
            series_name = X.name
            X_np = X.values
            scaled_X_np = self.scaler.fit_transform(X_np.reshape(-1, 1)).flatten()
            scaled_X = pd.Series(scaled_X_np, name=series_name)
            return scaled_X
        # もしXがndarrayの場合で、1次元の場合
        elif isinstance(X, np.ndarray) and len(X.shape) == 1:
            scaled_X = self.scaler.fit_transform(X.reshape(-1, 1)).flatten()
            return scaled_X
        # もしXがdataframeの場合や、ndarrayの場合で、2次元以上の場合
        elif isinstance(X, pd.DataFrame) or (isinstance(X, np.ndarray) and len(X.shape) > 1):
            scaled_X = self.scaler.fit_transform(X)
            return scaled_X
        else:
            raise ValueError(f"Unknown type: {type(X)}")

    def inverse_transform(self, X):
        if self.scaler is None:
            return X
        # もしXがseriesの場合
        if isinstance(X, pd.Series):
            series_name = X.name
            X_np = X.values
            scaled_X_np = self.scaler.inverse_transform(X_np.reshape(-1, 1)).flatten()
            scaled_X = pd.Series(scaled_X_np, name=series_name)
            return scaled_X
        # もしXがndarrayの場合で、1次元の場合
        elif isinstance(X, np.ndarray) and len(X.shape) == 1:
            scaled_X = self.scaler.inverse_transform(X.reshape(-1, 1)).flatten()
            return scaled_X
        # もしXがdataframeの場合や、ndarrayの場合で、2次元以上の場合
        elif isinstance(X, pd.DataFrame) or (isinstance(X, np.ndarray) and len(X.shape) > 1):
            scaled_X = self.scaler.inverse_transform(X)
            return scaled_X
        else:
            raise ValueError(f"Unknown type: {type(X)}")

    def transform(self, X):
        if self.scaler is None:
            return X
        scaled_X = self.scaler.transform(X)
        return scaled_X
