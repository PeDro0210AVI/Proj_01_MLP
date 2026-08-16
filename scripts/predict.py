import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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


def compare_with_expected(pred_df, id_col, expected_csv):
    expected = pd.read_csv(expected_csv)
    expected_id_col = "Id" if "Id" in expected.columns else expected.columns[0]
    expected_target_col = "Prediction" if "Prediction" in expected.columns else expected.columns[1]

    merged = pred_df.merge(
        expected[[expected_id_col, expected_target_col]],
        left_on=id_col, right_on=expected_id_col, how="inner",
        suffixes=("", "_true"),
    )
    if merged.empty:
        print("no hubo coincidencias de Id entre las predicciones y el archivo esperado")
        return

    true_col = expected_target_col if expected_target_col != "Prediction" else "Prediction_true"
    y_pred = merged["Prediction"].values.astype(np.float64)
    y_true = merged[true_col].values.astype(np.float64)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print(f"comparado contra {expected_csv}: {len(merged)} filas, RMSE = {rmse:.4f}")


def main(model_path, input_csv, output_csv, expected_csv=None):
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = MLP(
        in_features=artifact["in_features"],
        hidden_layers=artifact["hidden_layers"],
        activation=artifact["activation"],
        dropout=artifact["dropout"],
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    df = pd.read_csv(input_csv)
    id_col = artifact.get("id_column")
    drop_cols = [c for c in [artifact["target_column"], id_col] if c]
    X = df.drop(columns=drop_cols, errors="ignore")

    X_t = artifact["preprocessor"].transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()
    X_t = torch.tensor(X_t, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_t).numpy()

    out = pd.DataFrame({
        "Id": df[id_col] if id_col else df.index,
        "Prediction": preds,
    })
    out.to_csv(output_csv, index=False)
    print(f"predicciones guardadas en {output_csv}")

    if expected_csv:
        compare_with_expected(out, "Id", expected_csv)


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.pkl"
    input_csv = sys.argv[2] if len(sys.argv) > 2 else "../data/test.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "../data/predictions.csv"
    expected_csv = sys.argv[4] if len(sys.argv) > 4 else None
    main(model_path, input_csv, output_csv, expected_csv)
