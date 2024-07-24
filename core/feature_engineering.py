import optuna
import pandas as pd
from icecream import ic


class FeatureEngineering:
    """
    FeatureEngineeringのアイデアを気軽に試せるようにするためのクラス
    欠損値の処理もここで行う
    trial: 特徴量の取捨選択やパラメータ調整を担う
    Xy: 特徴量を生成するための元データ。予測ターゲット列、testデータも含む全データ。
    engineered_Xy_pool: 特徴量を生成した後のデータ

    #   Column                Non-Null Count  Dtype
    ---  ------                --------------  -----
    0   index                 952 non-null    Int64
    1   track_name            952 non-null    string
    2   artist(s)_name        952 non-null    string
    3   artist_count          952 non-null    Int64
    4   released_year         952 non-null    Int64
    5   released_month        952 non-null    Int64
    6   released_day          952 non-null    Int64
    7   in_spotify_playlists  952 non-null    Int64
    8   in_spotify_charts     952 non-null    Int64
    9   streams               761 non-null    Float64
    10  in_apple_playlists    952 non-null    Int64
    11  in_apple_charts       952 non-null    Int64
    12  in_deezer_playlists   952 non-null    string
    13  in_deezer_charts      952 non-null    Int64
    14  in_shazam_charts      902 non-null    object
    15  bpm                   952 non-null    Int64
    16  key                   857 non-null    string
    17  mode                  952 non-null    string
    18  danceability_%        952 non-null    Int64
    19  valence_%             952 non-null    Int64
    20  energy_%              952 non-null    Int64
    21  acousticness_%        952 non-null    Int64
    22  instrumentalness_%    952 non-null    Int64
    23  liveness_%            952 non-null    Int64
    24  speechiness_%         952 non-null    Int64
    25  dataset               952 non-null    string

    """

    def __init__(self, trial: optuna.Trial, Xy: pd.DataFrame, target_colname):
        self.trial = trial
        self.Xy = Xy
        self.engineered_Xy_pool = []
        self.target_colname = target_colname
        self._apply_all_features()
        self._add_dataset_label()
        self._add_target()

    # ------------ ユーティリティ関数 ------------
    def _add_dataset_label(self):
        """
        trainとtestを区別するためのラベルを追加する。
        この列は特徴量として使わない。
        学習直前に除外することを想定。
        """
        dataset = self.Xy["dataset"]
        self._add_engineered_feature("dataset", dataset)

    @property
    def engineered_Xy(self):
        return pd.DataFrame(dict(self.engineered_Xy_pool))

    def _apply_all_features(self):
        # クラス内のすべてのメソッド名を取得し、特定の形式（AA__BB）でフィルタリング
        feature_methods = [
            getattr(self, method_name)
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and "__" in method_name and not method_name.startswith("__")
        ]

        for method in feature_methods:
            method()

    def _add_target(self):
        self.engineered_Xy_pool.append((self.target_colname, self.Xy[self.target_colname]))

    def _add_engineered_feature(self, name, value):
        self.engineered_Xy_pool.append((name, value))

    def _toggleFeatureSelection(self, feature_name):
        choices = ["off", "on"]
        choice = self.trial.suggest_categorical(feature_name, choices)
        return choice

    def to_categorical_if_string(self, s):
        if isinstance(s, str):
            print("to_categorical_if_string")
            return s.astype("category")
        return s

    # ------------ artist_count関連の特徴量 ------------
    def artist_count__original(self):
        choice = self._toggleFeatureSelection("artist_count")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["artist_count"])

    # ------------ released_year関連の特徴量 ------------
    def released_year__years_since_release(self):
        choice = self._toggleFeatureSelection("years_since_release")
        if choice == "off":
            return
        elif choice == "on":
            original = self.Xy["released_year"]
            max_year = original.max()
            v = max_year - original
            self._add_engineered_feature(choice, v)

    def released_year__original(self):
        choice = self._toggleFeatureSelection("released_year")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["released_year"])

    # ------------ released_month関連の特徴量 ------------

    def released_month__original(self):
        choice = self._toggleFeatureSelection("released_month")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["released_month"])

    # ------------ released_day関連の特徴量 ------------
    def released_day__original(self):
        choice = self._toggleFeatureSelection("released_day")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["released_day"])

    # ------------ in_spotify_playlists関連の特徴量 ------------
    def in_spotify_playlists__original(self):
        choice = self._toggleFeatureSelection("in_spotify_playlists")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["in_spotify_playlists"])

    # ------------ in_spotify_charts関連の特徴量 ------------
    def in_spotify_charts__original(self):
        choice = self._toggleFeatureSelection("in_spotify_charts")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["in_spotify_charts"])

    # ------------ in_apple_playlists関連の特徴量 ------------
    def in_apple_playlists__original(self):
        choice = self._toggleFeatureSelection("in_apple_playlists")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["in_apple_playlists"])

    # ------------ in_apple_charts関連の特徴量 ------------
    def in_apple_charts__original(self):
        choice = self._toggleFeatureSelection("in_apple_charts")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["in_apple_charts"])

    # ------------ in_deezer_playlists関連の特徴量 ------------
    # TODO

    # ------------ in_deezer_charts関連の特徴量 ------------
    def in_deezer_charts__original(self):
        choice = self._toggleFeatureSelection("in_deezer_charts")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["in_deezer_charts"])

    # ------------ in_shazam_charts関連の特徴量 ------------
    # TODO

    # ------------ bpm関連の特徴量 ------------
    def bpm__original(self):
        choice = self._toggleFeatureSelection("bpm")
        if choice == "off":
            return
        elif choice == "on":
            self._add_engineered_feature(choice, self.Xy["bpm"])
