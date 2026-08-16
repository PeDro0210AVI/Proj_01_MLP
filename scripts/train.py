import sys
import pickle
import tomllib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class MLP(nn.Module):
    def __init__(self, in_features, hidden_layers, activation, dropout):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers = []
        prev = in_features
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_preprocessor(df, target_column):
    features = df.drop(columns=[target_column])
    num_cols = features.select_dtypes(include=np.number).columns.tolist()
    cat_cols = features.select_dtypes(exclude=np.number).columns.tolist()

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre, num_cols, cat_cols


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def main(config_path):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    torch.manual_seed(cfg["data"]["seed"])
    np.random.seed(cfg["data"]["seed"])

    df = pd.read_csv(cfg["data"]["csv_path"])
    target_column = cfg["data"]["target_column"]

    pre, num_cols, cat_cols = build_preprocessor(df, target_column)
    X = df.drop(columns=[target_column])
    y = df[target_column].values.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["seed"]
    )

    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_val_t = X_val_t.toarray()

    X_train_t = torch.tensor(X_train_t, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_t, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    model = MLP(
        in_features=X_train_t.shape[1],
        hidden_layers=cfg["model"]["hidden_layers"],
        activation=cfg["model"]["activation"],
        dropout=cfg["model"]["dropout"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = cfg["training"]["early_stopping_patience"]
    bad_epochs = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_rmse = rmse(val_pred, y_val_t)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or bad_epochs == 0:
            print(f"epoch {epoch:4d}  val_rmse={val_rmse:.4f}  best={best_val:.4f}")

        if bad_epochs >= patience:
            print(f"early stopping en epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"mejor val_rmse: {best_val:.4f}")

    artifact = {
        "state_dict": best_state,
        "in_features": X_train_t.shape[1],
        "hidden_layers": cfg["model"]["hidden_layers"],
        "activation": cfg["model"]["activation"],
        "dropout": cfg["model"]["dropout"],
        "preprocessor": pre,
        "feature_columns": num_cols + cat_cols,
        "target_column": target_column,
    }

    with open(cfg["output"]["model_path"], "wb") as f:
        pickle.dump(artifact, f)

    print(f"modelo guardado en {cfg['output']['model_path']}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    main(config_path)
