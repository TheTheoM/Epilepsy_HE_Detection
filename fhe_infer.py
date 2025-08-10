# Optional path fix if helper.py lives one folder up from this file
import os, sys, time
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from helper import read_data_file

# Optional import. If missing, use a safe fallback.
try:
    from preprocessor import is_missing_data
except Exception:
    def is_missing_data(df):
        import numpy as _np
        if hasattr(df, "isna"):
            if df.isna().values.any():
                return True
            try:
                num = df.select_dtypes(include=[_np.number])
            except Exception:
                try:
                    num = _np.asarray(df, dtype=float)
                    return _np.isinf(num).any()
                except Exception:
                    return False
            return _np.isinf(num.to_numpy()).any()
        arr = _np.asarray(df)
        if arr.dtype.kind in ("U", "S", "O"):
            return False
        return _np.isnan(arr).any() or _np.isinf(arr).any()

# Concrete-ML
from concrete import fhe
from concrete.ml.torch.compile import compile_torch_model

# Key cache for faster repeat runs
configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keycache",
)

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "he_eeg_mlp.pt")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
CALIB_NPZ = os.path.join(ART_DIR, "calib.npz")

# FHE knobs
FHE_LIMIT = int(os.getenv("FHE_LIMIT", "1"))   # keep tiny for first run
N_BITS = int(os.getenv("N_BITS", "4"))         # fewer bits -> faster

class HE_EEG_MLP(nn.Module):
    def __init__(self, input_size=178):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def square_activation(self, x):
        return x ** 2

    def forward(self, x):
        x = self.square_activation(self.fc1(x))
        x = self.square_activation(self.fc2(x))
        return self.fc3(x)

def load_data_and_preprocess():
    ESR = read_data_file('Epileptic Seizure Recognition.csv')
    if is_missing_data(ESR):
        raise ValueError("Missing data detected. Please clean it before proceeding.")

    target_labels = ESR['y'].copy()
    seizure_labels = target_labels.copy()
    seizure_labels[seizure_labels > 1] = 0

    X = ESR.drop(columns=['y', 'Unnamed'], errors='ignore').values.astype(np.float32)
    y = seizure_labels.values.astype(np.float32)

    scaler: StandardScaler = joblib.load(SCALER_PATH)
    X = scaler.transform(X).astype(np.float32)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def build_and_load_model(input_size: int):
    model = HE_EEG_MLP(input_size=input_size)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    # Load data and artifacts
    X_test, y_test = load_data_and_preprocess()
    model = build_and_load_model(input_size=X_test.shape[1])

    # Load calibration set saved during training
    calib = np.load(CALIB_NPZ)["calib"].astype(np.float32)

    # Compile Torch -> QuantizedModule
    print("Compiling to QuantizedModule with Concrete-ML... this can take a while")
    qmodule = compile_torch_model(
        model,
        calib,
        n_bits=N_BITS,
        configuration=configuration,
        p_error=1e-3,              # looser single-PBS error target
        global_p_error=None,       # let p_error drive it
        rounding_threshold_bits=4, # round accumulators
        verbose=True,
    )

    # Quantized clear run
    y_hat_clear = qmodule.forward(X_test.astype(np.float32), fhe="disable").reshape(-1)
    y_pred_clear = (y_hat_clear > 0.5).astype(np.float32)
    acc_clear = (y_pred_clear == y_test.astype(np.float32)).mean()
    print(f"Quantized clear run accuracy: {acc_clear * 100:.2f}%")

    # Encrypted run - keep tiny at first
    X_fhe = X_test[:FHE_LIMIT].astype(np.float32)
    y_true_fhe = y_test[:FHE_LIMIT].astype(np.float32)

    print(f"Running keygen for cache...")
    t0 = time.time()
    qmodule.fhe_circuit.keygen(force=True)   # fills .keycache
    print(f"Keygen done in {time.time()-t0:.2f}s")

    print(f"Running FHE on {len(X_fhe)} samples...")
    t1 = time.time()
    y_hat_fhe = qmodule.forward(X_fhe, fhe="execute").reshape(-1)
    enc_time = time.time()-t1
    y_pred_fhe = (y_hat_fhe > 0.5).astype(np.float32)
    acc_fhe = (y_pred_fhe == y_true_fhe).mean()
    print(f"FHE accuracy: {acc_fhe * 100:.2f}%  |  FHE time: {enc_time:.2f}s")

if __name__ == "__main__":
    main()
