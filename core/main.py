from objective import Objective
import optuna
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from icecream import ic
from model_factory import ModelFactory
from util import Util
import os


# TODO NNモデル追加
# TODO stack 2nd layer追加


def main():
    cfg = load_config()
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    os.makedirs("output/optuna", exist_ok=True)

    models = ModelFactory(cfg).build_fold_models(cfg.run.n_folds)
    for name, fold_models in models.items():  # modelごとにハイパーパラメータチューニングと特徴量チューニングを行う
        objective = Objective(cfg, fold_models)
        study = optuna.create_study(direction=cfg.run.optimize_direction, storage=cfg.run.optuna_db_path, study_name=f"{name}-{Util.get_unique_id()}")
        study.optimize(objective, n_trials=cfg.run.n_trials, show_progress_bar=True)
        pred_test = objective.best_runner.run_predict_cv()
        Util.dump(pred_test, f"output/y_test_pred/{name}-y_test_pred.pkl")


def load_config() -> DictConfig:
    yaml_path = Path("conf/base.yaml")
    return OmegaConf.load(yaml_path)


if __name__ == "__main__":
    main()
