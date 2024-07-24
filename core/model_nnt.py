from model import Model
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scaler import Scaler


class ModelNN(Model):
    """
    pytorchのモデルを構築する
    """

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # gpuが使えるかどうか
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # modelのインスタンス化
        self.model = Net(X_train.shape[1], dropout=self.params["dropout"]).to(device)
        # optimizerの設定
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])
        # lossの設定 RMSEとする
        self.criterion = nn.MSELoss()
        # X_trainのnanを埋める
        X_train = X_train.fillna(0)
        # スケーリング
        self.scaler_X = Scaler(strategy="standard")
        self.scaler_y = Scaler(strategy="standard")
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = self.scaler_y.fit_transform(y_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        # データセットの作成
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=763, shuffle=True, num_workers=23)
        # 学習
        self.model.train()
        N_EPOCHS = 100
        for epoch in range(N_EPOCHS):
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                self.optimizer.zero_grad()
                y_hat = self.model(X_batch).view(-1)
                loss = torch.sqrt(self.criterion(y_hat, y_batch))
                loss.backward()
                self.optimizer.step()
            print("train_loss: ", loss.item())

    def predict(self, X_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test = X_test.fillna(0)
        X_test = self.scaler_X.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(device)
            y_pred = self.model(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred.detach().numpy())
        return y_pred

    def param_tuning(self, trial: optuna.Trial):
        tuned_params = dict(
            lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
        )
        params = {**self.params, **tuned_params}
        self.params = params


class Net(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
