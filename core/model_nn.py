from model import Model
import optuna
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scaler import Scaler


# Modelとpl.LightningModuleを継承
class ModelNN(Model, pl.LightningModule):
    def __init__(self, model_fold_name, params):
        Model.__init__(self, model_fold_name, params)
        pl.LightningModule.__init__(self)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        # sizeをそろえる
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        loss = torch.sqrt(loss)
        print("------------------train_loss-----------------", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        loss = torch.sqrt(loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        loss = torch.sqrt(loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # X_trainのnanを埋める
        X_train = X_train.fillna(0)
        # スケーリング
        self.scaler_X = Scaler(strategy="standard")
        self.scaler_y = Scaler(strategy="standard")
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = self.scaler_y.fit_transform(y_train)

        input_dim = X_train.shape[1]
        self.build_model(input_dim)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=763, shuffle=True, num_workers=23, persistent_workers=True, pin_memory=True)

        trainer = pl.Trainer(max_epochs=10000, log_every_n_steps=1)
        trainer.fit(self, train_dataloaders=train_loader)

    def build_model(self, input_dim):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        self.loss = torch.nn.MSELoss()

    def predict(self, X_test):
        # X_testのnanを埋める
        X_test = X_test.fillna(0)
        X_test = self.scaler_X.transform(X_test)
        dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=763, shuffle=False)
        preds = []
        for batch in loader:
            x = batch[0]
            y_hat = self.model(x)
            preds.append(y_hat.detach().numpy())
        y_pred = np.concatenate(preds)
        y_pred = self.scaler_y.inverse_transform(y_pred)
        return y_pred

    def param_tuning(self, trial: optuna.Trial):
        pass
