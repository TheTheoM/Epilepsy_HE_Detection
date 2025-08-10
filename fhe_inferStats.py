# Optional path fix if helper.py lives one folder up from this file
import os, sys, time, json, argparse
from datetime import datetime
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import joblib
import matplotlib.pyplot as plt

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

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "he_eeg_mlp.pt")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
CALIB_NPZ = os.path.join(ART_DIR, "calib.npz")

# Default configuration with key cache for faster repeats
CONFIGURATION = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keycache",
)


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


def choose_threshold_from_clear(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true.astype(np.float32), y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def compute_metrics(y_true, y_pred, y_score=None, labels=(0, 1)):
    # Force a 2x2 confusion matrix even if a class is missing (e.g., when fhe-limit=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    # cm is guaranteed 2x2 now
    tn, fp, fn, tp = cm.ravel().tolist()
    report = classification_report(
        y_true, y_pred,
        labels=list(labels),
        target_names=["non-seizure", "seizure"],
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    out = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision_pos": float(report["seizure"]["precision"]),
        "recall_pos": float(report["seizure"]["recall"]),   # sensitivity
        "f1_pos": float(report["seizure"]["f1-score"]),
        "precision_neg": float(report["non-seizure"]["precision"]),
        "recall_neg": float(report["non-seizure"]["recall"]),  # specificity
        "f1_neg": float(report["non-seizure"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["avg_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            pass
    return out, cm


def plot_confusion_matrix(cm, labels, title, save_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label', title=title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_pr(y_true, y_score, title_prefix, out_dir):
    # Need positive and negative examples to make ROC/PR meaningful
    if len(np.unique(y_true)) < 2:
        return
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(fpr, tpr, label=f'AUC={auc:.3f}')
        ax1.plot([0, 1], [0, 1], linestyle=':')
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.set_title(f'{title_prefix} ROC')
        ax1.legend(loc='lower right')
        fig1.tight_layout()
        fig1.savefig(os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_roc.png"), dpi=150)
        plt.close(fig1)

        pr, rc, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(rc, pr, label=f'AP={ap:.3f}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{title_prefix} PR')
        ax2.legend(loc='lower left')
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"{title_prefix.lower().replace(' ', '_')}_pr.png"), dpi=150)
        plt.close(fig2)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description='FHE inference for EEG MLP with Concrete ML')
    p.add_argument('--fhe-limit', type=int, default=int(os.getenv('FHE_LIMIT', '1')), help='How many samples to run under FHE')
    p.add_argument('--n-bits', type=int, default=int(os.getenv('N_BITS', '4')), help='Quantization bit width')
    p.add_argument('--p-error', type=float, default=1e-3, help='Per PBS error target. Ignored if global-p-error is set')
    p.add_argument('--global-p-error', type=float, default=None, help='Global error target for whole circuit')
    p.add_argument('--rounding-bits', type=int, default=4, help='Accumulator rounding threshold bits')
    p.add_argument('--threshold', type=str, default='auto', help='Decision threshold. Use "auto" to learn from clear run, or a float like 0.5')
    p.add_argument('--results-dir', type=str, default='results', help='Base folder to save figures and metrics')
    p.add_argument('--verbose', action='store_true', help='Print Concrete ML verbose compile info')
    return p.parse_args()


def main():
    args = parse_args()

    # Prepare results output path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.results_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    # Load data and artifacts
    X_test, y_test = load_data_and_preprocess()
    model = build_and_load_model(input_size=X_test.shape[1])
    calib = np.load(CALIB_NPZ)["calib"].astype(np.float32)

    # Compile Torch -> QuantizedModule
    print("Compiling to QuantizedModule with Concrete ML...")
    compile_kwargs = {
        "n_bits": args.n_bits,
        "configuration": CONFIGURATION,
        "rounding_threshold_bits": args.rounding_bits,
        "verbose": args.verbose,
    }
    if args.global_p_error is not None:
        compile_kwargs["global_p_error"] = args.global_p_error
    else:
        compile_kwargs["p_error"] = args.p_error

    t_compile0 = time.time()
    # Pass model and calibration set POSITIONALLY for version compatibility
    qmodule = compile_torch_model(model, calib, **compile_kwargs)
    t_compile = time.time() - t_compile0

    # Quantized clear scores and threshold
    y_hat_clear = qmodule.forward(X_test.astype(np.float32), fhe="disable").reshape(-1)
    if args.threshold == 'auto':
        best_thr = choose_threshold_from_clear(y_test, y_hat_clear)
    else:
        try:
            best_thr = float(args.threshold)
        except Exception:
            best_thr = 0.5
    y_pred_clear = (y_hat_clear > best_thr).astype(np.float32)
    clear_metrics, clear_cm = compute_metrics(y_test, y_pred_clear, y_hat_clear)

    print(f"Quantized clear run accuracy: {clear_metrics['accuracy'] * 100:.2f}% (thr={best_thr:.4f})")

    # FHE execution on subset
    limit = max(1, min(args.fhe_limit, len(X_test)))
    X_fhe = X_test[:limit].astype(np.float32)
    y_true_fhe = y_test[:limit].astype(np.float32)

    print("Running keygen for cache...")
    t_keygen0 = time.time()
    qmodule.fhe_circuit.keygen(force=True)
    t_keygen = time.time() - t_keygen0
    print(f"Keygen done in {t_keygen:.2f}s")

    print(f"Running FHE on {len(X_fhe)} samples...")
    t_exec0 = time.time()
    y_hat_fhe = qmodule.forward(X_fhe, fhe="execute").reshape(-1)
    t_exec = time.time() - t_exec0

    y_pred_fhe = (y_hat_fhe > best_thr).astype(np.float32)
    fhe_metrics, fhe_cm = compute_metrics(y_true_fhe, y_pred_fhe, y_hat_fhe)

    per_sample = t_exec / float(limit)

    print("\n=== Summary ===")
    print(f"compile_time_s: {t_compile:.2f}")
    print(f"keygen_time_s: {t_keygen:.2f}")
    print(f"fhe_time_s: {t_exec:.2f}")
    print(f"fhe_time_per_sample_s: {per_sample:.3f}")
    print(f"threshold_used: {best_thr:.4f}")
    print(f"clear_acc: {clear_metrics['accuracy']:.4f}  |  fhe_acc: {fhe_metrics['accuracy']:.4f}")
    print(f"clear_sensitivity: {clear_metrics['recall_pos']:.4f}  |  clear_specificity: {clear_metrics['recall_neg']:.4f}")
    print(f"fhe_sensitivity: {fhe_metrics['recall_pos']:.4f}    |  fhe_specificity: {fhe_metrics['recall_neg']:.4f}")

    # Save figures
    plot_confusion_matrix(clear_cm, ["non-seizure", "seizure"], "Clear quantized CM", os.path.join(run_dir, "clear_confusion.png"))
    plot_confusion_matrix(fhe_cm, ["non-seizure", "seizure"], "FHE CM", os.path.join(run_dir, "fhe_confusion.png"))
    plot_roc_pr(y_test, y_hat_clear, "Clear", run_dir)
    plot_roc_pr(y_true_fhe, y_hat_fhe, "FHE", run_dir)

    # Save JSON summary
    summary = {
        "args": vars(args),
        "threshold": best_thr,
        "compile_time_s": t_compile,
        "keygen_time_s": t_keygen,
        "fhe_time_s": t_exec,
        "fhe_time_per_sample_s": per_sample,
        "clear_metrics": clear_metrics,
        "fhe_metrics": fhe_metrics,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {run_dir}")


if __name__ == "__main__":
    main()