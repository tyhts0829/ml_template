import datetime
import logging
import os

import numpy as np
import pandas as pd
import pickle


class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(value, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def get_unique_id(cls):
        return datetime.datetime.now().strftime("%y%m%d_%H%M%S")


class Submission:
    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv("input/sampleSubmission.csv")
        pred = Util.load(f"model/pred/{run_name}-test.pkl")
        for i in range(pred.shape[1]):
            submission[f"Class_{i + 1}"] = pred[:, i]
        submission.to_csv(f"submission/{run_name}.csv", index=False)


def generate_train_test_csv():
    df = pd.read_csv(r"input\spotify-2023.csv", encoding="latin-1")
    print(df.info())
    prediction_target = "streams"
    # trainデータを作成しcsvに保存
    len_train = int(len(df) * 0.8)
    train = df[:len_train]
    train.to_csv(r"input\train.csv", index=False)
    # testデータを作成しcsvに保存
    test = df[len_train:]
    test = test.drop(columns=[prediction_target])
    test.to_csv(r"input\test.csv", index=False)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
