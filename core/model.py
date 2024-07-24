import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from typing import Optional
from pathlib import Path
from util import Util


class Model(ABC):

    def __init__(self, model_fold_name: str, params: dict) -> None:
        """コンストラクタ

        :param model_fold_name: モデルの名前とfoldの番号を組み合わせた名前
        :param params: ハイパーパラメータ
        """
        self.model_fold_name = model_fold_name
        self.params = params
        self.scaler_strategy = None
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """モデルの学習を行い、学習済のモデルを保存する

        :param X_train: 学習データの特徴量
        :param y_train: 学習データの目的変数
        :param X_val: バリデーションデータの特徴量
        :param y_val: バリデーションデータの目的変数
        """
        pass

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """学習済のモデルでの予測値を返す

        :param X_test: バリデーションデータやテストデータの特徴量
        :return: 予測値
        """
        pass

    @abstractmethod
    def param_tuning(self, trial) -> None:
        """ハイパーパラメータのチューニングを行う

        :param trial: optuna.Trial
        """
        pass

    def save_model(self) -> None:
        """モデルの保存を行う"""
        model_path = Path("output/model") / f"{self.model_fold_name}.model"
        Util.dump(self.model, model_path)

    def load_model(self) -> None:
        """モデルの読み込みを行う"""
        model_path = Path("output/model") / f"{self.model_fold_name}.model"
        self.model = Util.load(model_path)
