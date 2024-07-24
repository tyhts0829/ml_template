from model import Model
import optuna
from sklearn.neighbors import KNeighborsRegressor
from scaler import Scaler
from icecream import ic


class ModelKNN(Model):
    """
    KNeighborsRegressorをsklearn APIで構築
    """

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # ハイパーパラメータの設定
        params = dict(self.params)
        # スケーリング
        self.scaler_X = Scaler(strategy=self.scaler_strategy)
        self.scaler_y = Scaler(strategy=self.scaler_strategy)
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = self.scaler_y.fit_transform(y_train)
        # 学習
        self.model = KNeighborsRegressor(**params)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.scaler_X.transform(X_test)
        y_pred = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred)
        return y_pred

    def param_tuning(self, trial: optuna.Trial):
        tuned_params = dict(
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            algorithm=trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
            leaf_size=trial.suggest_int("leaf_size", 10, 100),
            p=trial.suggest_int("p", 1, 2),
            metric=trial.suggest_categorical("metric", ["minkowski", "manhattan", "euclidean"]),
        )
        params = {**self.params, **tuned_params}
        self.params = params
        self.scaler_strategy = trial.suggest_categorical("scaler_strategy", ["standard", "minmax", "robust"])
