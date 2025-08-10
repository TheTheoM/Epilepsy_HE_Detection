# Optional path fix if helper.py lives one folder up from this file
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from helper import read_data_file, graph_seizure_labels
# Optional import. If missing, a fallback is defined below.
try:
    from preprocessor import is_missing_data
except Exception:
    def is_missing_data(df):
        import numpy as _np
        import pandas as _pd  # noqa: F401
        # If df is a pandas DataFrame or Series
        if hasattr(df, "isna"):
            if df.isna().values.any():
                return True
            # Check infinities only on numeric columns
            try:
                num = df.select_dtypes(include=[_np.number])
            except Exception:
                try:
                    num = _np.asarray(df, dtype=float)
                    return _np.isinf(num).any()
                except Exception:
                    return False
            return _np.isinf(num.to_numpy()).any()
        # Otherwise treat as numpy-like
        arr = _np.asarray(df)
        # For non numeric dtypes, skip isnan to avoid TypeError
        if arr.dtype.kind in ("U", "S", "O"):
            return False
        return _np.isnan(arr).any() or _np.isinf(arr).any()

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "he_eeg_mlp.pt")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
CALIB_NPZ = os.path.join(ART_DIR, "calib.npz")


def train_HE_adapted_plaintext(manual_seed: int = 0) -> float:
    os.makedirs(ART_DIR, exist_ok=True)

    # 1) Load
    ESR = read_data_file('Epileptic Seizure Recognition.csv')
    if is_missing_data(ESR):
        raise ValueError("Missing data detected. Please clean it before proceeding.")

    # Multi-class to binary: 1 = seizure, all others = non-seizure
    target_labels = ESR['y'].copy()
    seizure_labels = target_labels.copy()
    seizure_labels[seizure_labels > 1] = 0

    # 2) Preprocess
    X = ESR.drop(columns=['y', 'Unnamed'], errors='ignore')
    y = seizure_labels.values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 3) Model
    torch.manual_seed(manual_seed)

    class HE_EEG_MLP(nn.Module):
        def __init__(self, input_size=178):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def square_activation(self, x):
            return x ** 2  # HE-friendly polynomial activation

        def forward(self, x):
            x = self.square_activation(self.fc1(x))
            x = self.square_activation(self.fc2(x))
            return self.fc3(x)  # raw score trained toward 0 or 1

    model = HE_EEG_MLP(input_size=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4) Train
    epochs = 200
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 5) Eval
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        preds = (preds > 0.5).float()
        acc = (preds == y_test_t).float().mean().item()
        print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # 6) Save artifacts for FHE compilation and inference
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Save a small calibration set required by Concrete-ML quantization
    # Keep it modest in size to reduce compile memory and time
    calib = X_train[:512].astype(np.float32)
    np.savez(CALIB_NPZ, calib=calib)

    print(f"Saved: {MODEL_PATH}, {SCALER_PATH}, {CALIB_NPZ}")
    return float(acc * 100)


if __name__ == "__main__":
    print(train_HE_adapted_plaintext())