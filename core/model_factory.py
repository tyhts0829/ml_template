from omegaconf import DictConfig
from model_xgb import ModelXGB
from icecream import ic
from model_knn import ModelKNN

# from model_nn import ModelNN
from model_nnt import ModelNN


class ModelFactory:
    def __init__(self, cfg: DictConfig):
        self.cfg_models = cfg.models

    def build_fold_models(self, n_fold: int):
        """
        フォールドごとのモデルを取得。
        models: cfgのmodel1, model2, ...
        fold_models: modelごとにfold数分のインスタンス化した
        """
        models = {}
        for name, cfg_model in self.cfg_models.items():
            model_cls = self.build_model_cls(cfg_model)
            fold_models = {}
            for i_fold in range(n_fold):
                model_fold_name = f"{name}-{i_fold}"
                hyper_params = cfg_model.hyper_params
                model = model_cls(model_fold_name, hyper_params)
                fold_models[i_fold] = model
            models[name] = fold_models
        return models

    def build_model_cls(self, cfg_model: DictConfig):
        """
        モデルのクラスを取得。
        """
        if cfg_model.cls == "ModelXGB":
            return ModelXGB
        elif cfg_model.cls == "ModelKNN":
            return ModelKNN
        elif cfg_model.cls == "ModelNN":
            return ModelNN
        else:
            raise ValueError(f"model {cfg_model.cls} is not supported")
