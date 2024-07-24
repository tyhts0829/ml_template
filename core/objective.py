import optuna
from feature_engineering import FeatureEngineering
from pathlib import Path
import pandas as pd
from model import Model
from runner import Runner
from icecream import ic
import numpy as np
from omegaconf import DictConfig
from util import Util
from typing import List, Dict


class Objective:
    def __init__(self, cfg: DictConfig, fold_models: List[Dict[int, Model]]):
        self.cfg = cfg
        self.fold_models = fold_models
        self.model_name = fold_models[0].model_fold_name.split("-")[0]
        self.train = self.load_train()
        self.test = self.load_test()
        self.train["dataset"] = "train"
        self.test["dataset"] = "test"
        self.preprocess()
        self.Xy = pd.concat([self.train, self.test], axis=0).reset_index()
        self.Xy = self.Xy.convert_dtypes()
        self.best_value = None
        self.result = {}
        self.best_runner = None

    def __call__(self, trial: optuna.Trial):
        # 特徴量エンジニアリングをtrialを用いて行う
        engineered_Xy = FeatureEngineering(trial, self.Xy, self.cfg.run.target_colname).engineered_Xy
        # モデルごとにハイパーパラメータチューニングを行う
        [fold_model.param_tuning(trial) for fold_model in self.fold_models.values()]
        runner = Runner(
            self.cfg,
            engineered_Xy,
            self.fold_models,
        )
        trained_fold_models, val_scores, pred_val = runner.run_train_cv()
        val_score = np.mean(val_scores)
        # best_valueの場合のみ結果を保存
        if self.is_best_value(val_score):
            for trained_fold_model, val_fold_score in zip(trained_fold_models.values(), val_scores):
                trained_fold_model.save_model()
                Util.dump(val_fold_score, f"output/val_scores/{trained_fold_model.model_fold_name}-val_fold_score.pkl")
            Util.dump(pred_val, f"output/y_train_pred/{self.model_name}-y_train_pred.pkl")
            Util.dump(engineered_Xy, f"output/engineered_Xy/{self.model_name}-engineered_Xy.pkl")
            self.best_runner = runner

        return val_score

    def preprocess(self):
        """
        ここはデータ全体に対して初回の前処理を行う
        コンペに応じて書き換え
        """
        target_colname = self.cfg.run.target_colname
        # TARGET_COLNAMEにnanが含まれる行を削除
        self.train = self.train.dropna(subset=[target_colname])
        # TARGET_COLNAMEに数字以外の文字が含まれる行を削除
        self.train = self.train[~self.train[target_colname].str.contains("[^0-9]")]
        # TARGET_COLNAMEをintに変換
        self.train[target_colname] = self.train[target_colname].astype("Int64")
        # TARGET_COLNAMEをlog1p変換
        self.train[target_colname] = self.train[target_colname].apply(lambda x: np.log1p(x))

    def load_train(self):
        return pd.read_csv(self.cfg.run.train_path)

    def load_test(self):
        return pd.read_csv(self.cfg.run.test_path)

    def is_best_value(self, current_value):
        direction = self.cfg.run.optimize_direction
        if self.best_value is None:
            self.best_value = current_value
            return True
        if direction == "maximize":
            if current_value > self.best_value:
                self.best_value = current_value
                return True
        elif direction == "minimize":
            if current_value < self.best_value:
                self.best_value = current_value
                return True
        return False
