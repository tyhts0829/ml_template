from model import Model
import optuna
from xgboost import XGBRegressor


class ModelXGB(Model):
    """
    XGBRegressorをsklearn APIで構築
    """

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # ハイパーパラメータの設定
        params = dict(self.params)
        # 学習
        validation = X_val is not None
        if validation:
            self.model = XGBRegressor(**params)
            eval_set = [(X_val, y_val)]
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model = XGBRegressor(**params)
            self.model.fit(X_train, y_train, verbose=False)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def param_tuning(self, trial: optuna.Trial):
        tuned_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            early_stopping_rounds=trial.suggest_int("early_stopping_rounds", 10, 50),
        )
        params = {**self.params, **tuned_params}
        self.params = params
