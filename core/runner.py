import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from typing import Tuple
from model import Model
from icecream import ic
from omegaconf import DictConfig
from pathlib import Path
import warnings


class Runner:
    def __init__(self, cfg: DictConfig, Xy: pd.DataFrame, fold_models: dict):
        """
        cfg: 設定ファイル
        Xy: 全データ
        """
        self.cfg = cfg
        self.fold_models = fold_models
        target_colname = cfg.run.target_colname
        self.X_train = Xy.query("dataset == 'train'").drop(columns=[target_colname]).reset_index(drop=True).drop(columns=["dataset"])
        self.y_train = Xy.query("dataset == 'train'")[target_colname].reset_index(drop=True).drop(columns=["dataset"])
        self.X_test = Xy.query("dataset == 'test'").drop(columns=[target_colname]).reset_index(drop=True).drop(columns=["dataset"])
        self.n_folds = cfg.run.n_folds
        self.eval_metric = cfg.run.eval_metric
        self.fold_index = None
        self._compute_stratified_kfold_indices()

    def update_fold_models(self, fold_models: dict):
        """
        fold_modelsを更新する
        """
        self.fold_models = fold_models

    def run_predict_cv(self):
        """
        クロスバリデーションで学習した各fold_modelでテストデータを予測し、
        それらの平均を最終の予測とする
        """
        pred_test_list = []
        for i_fold, fold_model in self.fold_models.items():
            pred_test = fold_model.predict(self.X_test)
            pred_test_list.append(pred_test)
        pred_test = np.mean(pred_test_list, axis=0)
        return pred_test

    def run_train_cv(self):
        """
        クロスバリデーションで学習・予測を行う
        fold_modelごとに学習・予測を行う
        """
        idx_val_list = []
        pred_val_list = []
        val_scores = []
        trained_fold_models = {}
        for i_fold, fold_model in self.fold_models.items():
            trained_fold_model, val_idx, pred_val, val_score = self.train_fold(i_fold, fold_model)
            trained_fold_models[i_fold] = trained_fold_model
            idx_val_list.append(val_idx)
            pred_val_list.append(pred_val)
            val_scores.append(val_score)

        # 各foldの結果をまとめる
        val_idxes = np.concatenate(idx_val_list)
        order = np.argsort(val_idxes)
        pred_val = np.concatenate(pred_val_list)[order]
        return trained_fold_models, val_scores, pred_val

    def _compute_stratified_kfold_indices(self):
        """
        StratifiedKFoldでfoldのインデックスを作成
        戻り値の構造:
        [[[fold1のtrain_idx], [fold1のval_idx]], [[fold2のtrain_idx], [fold2のval_idx]], ...]
        """
        warnings.filterwarnings("ignore")
        # targetがfloatなのでbinに分割
        y_binned = pd.cut(self.y_train, bins=10, labels=False)
        x_dummy = np.zeros(len(self.y_train))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=False)
        self.fold_index = list(skf.split(x_dummy, y_binned))
        warnings.resetwarnings()

    def _load_index_fold(self, i_fold):
        """
        foldの番号を指定して学習データとバリデーションデータのインデックスを返す
        """
        return self.fold_index[i_fold]

    def eval_score(self, y_true, y_pred):
        """
        評価関数
        """
        if self.eval_metric == "rmse":
            return np.sqrt(mean_squared_error(y_true, y_pred))

    def train_fold(self, i_fold, fold_model: Model) -> Tuple[Model, np.array, np.array, float]:
        """
        クロスバリデーションでのfoldを指定して学習・評価を行う

        他のメソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        i_fold: foldの番号
        return: モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア
        """
        # 学習データ・バリデーションデータをセットする
        idx_train, idx_val = self._load_index_fold(i_fold)
        X_train, y_train = self.X_train.iloc[idx_train], self.y_train.iloc[idx_train]
        X_val, y_val = self.X_train.iloc[idx_val], self.y_train.iloc[idx_val]

        # 学習を行う
        fold_model.train(X_train, y_train, X_val, y_val)

        # バリデーションデータへの予測・評価を行う
        pred_val = fold_model.predict(X_val)
        val_score = self.eval_score(y_val, pred_val)
        return fold_model, idx_val, pred_val, val_score
