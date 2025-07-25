import os
import sys
from itertools import product

import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import torch
import torch.nn as nn
from backend.database import SessionLocal, Transaction
from sqlalchemy.orm import Session

load_dotenv()
mlflow.set_experiment("CreditCardFraudDetection")

def get_data_from_db():
    session: Session = SessionLocal()
    transactions = session.query(Transaction).all()
    session.close()

    data = [{
        "Time": t.time,
        **{f"V{i}": getattr(t, f"v{i}") for i in range(1, 29)},
        "Amount": t.amount,
        "Class": t.class_label
    } for t in transactions]

    return pd.DataFrame(data)

class NNet(torch.nn.Module):
    def __init__(self, n_inputs, n_hiddens_per_layer, n_outputs, act_func='relu', dropout=0.0):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        for nh in n_hiddens_per_layer:
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(n_inputs, nh),
                nn.ReLU() if act_func == 'relu' else nn.Tanh(),
                nn.Dropout(dropout)
            ))
            n_inputs = nh
        self.output_layer = nn.Linear(n_inputs, n_outputs)
        self.Xmeans = None
        self.Xstds = None
        self.error_trace = []

    def forward(self, X):
        for layer in self.hidden_layers:
            X = layer(X)
        return self.output_layer(X)

    def train_model(self, X, T, n_epochs=50, learning_rate=0.001,
                    batch_size=512, loss_type='bce', optimizer_type='adam', pos_weight=599, verbose=False):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        if not isinstance(T, torch.Tensor):
            T = torch.from_numpy(T).float()

        if self.Xmeans is None:
            self.Xmeans = X.mean(0)
            self.Xstds = X.std(0)
            self.Xstds[self.Xstds == 0] = 1
        X = (X - self.Xmeans) / self.Xstds

        if loss_type == 'mse':
            loss_fn = nn.MSELoss()
        elif loss_type == 'bce':
            loss_fn = nn.BCELoss()
        elif loss_type == 'weighted_bce':
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) if optimizer_type == 'adam' else torch.optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            permutation = torch.randperm(X.size(0))
            epoch_loss = 0.0
            for i in range(0, X.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_T = X[indices], T[indices]
                Y = self.forward(batch_X)
                if loss_type == 'bce':
                    Y = torch.sigmoid(Y)
                loss = loss_fn(Y, batch_T)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
            self.error_trace.append(epoch_loss)

    def use(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        X = (X - self.Xmeans) / self.Xstds
        logits = self.forward(X)
        return torch.sigmoid(logits).detach().numpy()

def perform_grid_search(Xtrain, Ttrain, Xval, Tval):
    results = []
    hidden_layer_options = [[32], [64, 32], [128, 64, 32]]
    dropout_options = [0.0, 0.2]
    learning_rate_options = [0.001, 0.0005]
    loss_type_options = ['mse', 'bce', 'weighted_bce']
    batch_size_options = [1024]
    optimizer_type_options = ['adam']
    activation_options = ['relu', 'tanh']
    epoch_options = [50]

    for hl, do, lr, lt, bs, opt, af, ep in product(
        hidden_layer_options, dropout_options,
        learning_rate_options, loss_type_options,
        batch_size_options, optimizer_type_options,
        activation_options, epoch_options):

        print(f"Training → Layers: {hl}, Dropout: {do}, LR: {lr}, Loss: {lt}, Batch: {bs}, Optimizer: {opt}, ActFunc: {af}, Epochs: {ep}")

        model = NNet(n_inputs=Xtrain.shape[1], n_hiddens_per_layer=hl, n_outputs=1, act_func=af, dropout=do)
        n_pos = Ttrain.sum()
        n_neg = len(Ttrain) - n_pos
        pw = (n_neg / n_pos).item() if lt == 'weighted_bce' else None

        model.train_model(Xtrain, Ttrain, n_epochs=ep, learning_rate=lr,
                          batch_size=bs, loss_type=lt, optimizer_type=opt, pos_weight=pw)

        Yval = model.use(Xval)
        Yval_binary = (Yval >= 0.5).astype(int)

        acc = accuracy_score(Tval, Yval_binary)
        auprc = average_precision_score(Tval, Yval)

        print(f"→ Accuracy: {acc:.4f}, AUPRC: {auprc:.4f}\n")

        results.append({
            'model': model,
            'layers': hl, 'dropout': do, 'lr': lr,
            'loss_type': lt, 'batch_size': bs,
            'optimizer': opt, 'activation': af,
            'epochs': ep, 'accuracy': acc, 'auprc': auprc
        })

    return sorted(results, key=lambda r: r['auprc'], reverse=True)

def train_model(seed: int = 42):
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run():
        df = get_data_from_db()
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = df.drop(columns=["Class"])
        y = df["Class"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict_proba(X_val)[:, 1]
        rf_auprc = average_precision_score(y_val, rf_preds)
        mlflow.log_metric("RF_AUPRC", rf_auprc)
        mlflow.sklearn.log_model(rf, "random_forest_model")
        print(f"Random Forest AUPRC: {rf_auprc:.4f}")

        # --- Neural Net Grid Search ---
        top_results = perform_grid_search(X_train.values, y_train.values.reshape(-1, 1),
                                          X_val.values, y_val.values.reshape(-1, 1))
        best = top_results[0]
        print("\nTop 5 NN Results:")
        for i, r in enumerate(top_results[:5]):
            print(f"Top {i+1}:\n  Layers: {r['layers']}\n  Dropout: {r['dropout']}\n  LR: {r['lr']}\n  Loss: {r['loss_type']}\n  Batch Size: {r['batch_size']}\n  Optimizer: {r['optimizer']}\n  Activation: {r['activation']}\n  Epochs: {r['epochs']}\n  Accuracy: {r['accuracy']:.4f}\n  AUPRC: {r['auprc']:.4f}\n")

        # Log best model
        mlflow.log_metric("NN_AUPRC", best['auprc'])
        mlflow.log_metric("NN_Accuracy", best['accuracy'])
        mlflow.pytorch.log_model(best['model'], "best_neural_net_model")

if __name__ == "__main__":
    train_model()
