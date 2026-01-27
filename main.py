# BitTorch - Training Script (walk-forward + proper baselines + production artifacts)
# pip install pandas numpy matplotlib yfinance torch scikit-learn

import os
import json
import time
import random
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# -------------------------
# Indicators / Features
# -------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds feature table using returns (stationary) instead of raw price.
    Target: next-day log return.
    """
    out = df.copy()

    # Flatten MultiIndex columns if present (yfinance quirk)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    out["log_close"] = np.log(out["Close"])
    out["ret1"] = out["log_close"].diff()

    # Target = tomorrow's return
    out["target_ret_next"] = out["ret1"].shift(-1)

    # Lagged returns
    out["ret_1"] = out["ret1"]
    out["ret_2"] = out["ret1"].shift(1)
    out["ret_3"] = out["ret1"].shift(2)

    # Volatility
    out["vol_7"] = out["ret1"].rolling(7).std()
    out["vol_14"] = out["ret1"].rolling(14).std()

    # Momentum
    out["mom_7"] = out["log_close"].diff(7)
    out["mom_14"] = out["log_close"].diff(14)

    # RSI
    out["rsi_14"] = rsi(out["Close"], 14)

    # MA distance
    ma_10 = out["Close"].rolling(10).mean()
    ma_30 = out["Close"].rolling(30).mean()
    out["ma10_dist"] = (out["Close"] - ma_10) / (ma_10 + 1e-12)
    out["ma30_dist"] = (out["Close"] - ma_30) / (ma_30 + 1e-12)

    # Range proxy
    if {"High", "Low", "Close"}.issubset(out.columns):
        out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-12)
    else:
        out["hl_range"] = np.nan

    # Volume change (may be spotty)
    if "Volume" in out.columns:
        out["vol_chg"] = np.log((out["Volume"] + 1.0) / (out["Volume"].shift(1) + 1.0))
    else:
        out["vol_chg"] = np.nan

    return out


def select_features(
    df: pd.DataFrame,
    candidates: List[str],
    max_nan_frac: float = 0.05,
    segments: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """
    Select features not too NaN-heavy or constant.
    If segments provided, check NaN fraction in each segment separately.
    """
    kept = []
    n = len(df)

    for col in candidates:
        if col not in df.columns:
            continue

        global_nan = df[col].isna().sum() / max(1, n)
        if global_nan > max_nan_frac:
            continue

        if segments:
            bad_segment = False
            for start, end in segments:
                seg = df[col].iloc[start:end]
                seg_nan = seg.isna().sum() / max(1, len(seg))
                if seg_nan > max_nan_frac:
                    bad_segment = True
                    break
            if bad_segment:
                continue

        if df[col].std(skipna=True) < 1e-10:
            continue

        kept.append(col)

    return kept


# -------------------------
# Sequences
# -------------------------
def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(xs), np.asarray(ys)


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        else:
            self.y = torch.zeros(len(X), 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# Model
# -------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


# -------------------------
# Metrics
# -------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def compute_baselines(y_true: np.ndarray, lastret: np.ndarray) -> Dict[str, float]:
    """
    Two baselines for next-day return:
      - zero: predict 0
      - persistence: predict last observed return
    """
    zero = np.zeros_like(y_true)
    return {
        "zero_mae": mae(y_true, zero),
        "zero_diracc": directional_accuracy(y_true, zero),
        "lastret_mae": mae(y_true, lastret),
        "lastret_diracc": directional_accuracy(y_true, lastret),
    }


# -------------------------
# Walk-forward folds
# -------------------------
@dataclass
class Fold:
    train_end: int
    val_end: int
    test_end: int


def make_folds(
    n: int,
    seq_len: int,
    min_train: int = 365,
    val_days: int = 90,
    test_days: int = 90,
    step: int = 90,
) -> List[Fold]:
    """
    Expanding-window walk-forward folds.
    """
    min_window = seq_len + 1
    if val_days < min_window:
        raise ValueError(f"val_days ({val_days}) must be >= seq_len + 1 ({min_window})")
    if test_days < min_window:
        raise ValueError(f"test_days ({test_days}) must be >= seq_len + 1 ({min_window})")

    folds = []
    min_size = min_train + val_days + test_days

    test_end = min_size
    while test_end <= n:
        val_end = test_end - test_days
        train_end = val_end - val_days
        if train_end >= min_train:
            folds.append(Fold(train_end=train_end, val_end=val_end, test_end=test_end))
        test_end += step

    return folds


# -------------------------
# Training config
# -------------------------
@dataclass
class TrainConfig:
    seq_len: int = 30
    batch: int = 64
    epochs: int = 80
    patience: int = 12
    lr: float = 1e-3
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    grad_clip: float = 1.0


# -------------------------
# Data prep for a fold
# -------------------------
@dataclass
class FoldData:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    lastret_test: np.ndarray
    scaler: StandardScaler


def prepare_fold_data(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    fold: Fold,
    seq_len: int,
) -> FoldData:
    """
    Prepares train/val/test arrays with proper context handling.
    Scaler fit on train only.
    """
    X_all = df_feat[feature_cols].values.astype(np.float32)
    y_all = df_feat["target_ret_next"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_all[: fold.train_end])

    # Train
    Xtr_scaled = scaler.transform(X_all[: fold.train_end])
    ytr_raw = y_all[: fold.train_end]
    Xtr, ytr = create_sequences(Xtr_scaled, ytr_raw, seq_len)

    # Val: prepend seq_len rows for context
    val_start = max(0, fold.train_end - seq_len)
    Xva_scaled = scaler.transform(X_all[val_start : fold.val_end])
    yva_raw = y_all[val_start : fold.val_end]
    Xva, yva = create_sequences(Xva_scaled, yva_raw, seq_len)

    # Test: prepend seq_len rows for context
    test_start = max(0, fold.val_end - seq_len)
    Xte_scaled = scaler.transform(X_all[test_start : fold.test_end])
    yte_raw = y_all[test_start : fold.test_end]
    Xte, yte = create_sequences(Xte_scaled, yte_raw, seq_len)

    # Baseline: "persistence" = predict last observed return (UNSCALED)
    # For test sample i, target is y_all[test_start + seq_len + i]
    # Last observed return is y_all[test_start + seq_len + i - 1]
    if len(yte) > 0:
        baseline_indices = np.arange(test_start + seq_len - 1, test_start + seq_len - 1 + len(yte))
        # Clamp indices to valid range
        baseline_indices = np.clip(baseline_indices, 0, len(y_all) - 1)
        lastret_test = y_all[baseline_indices].copy()
    else:
        lastret_test = np.array([])

    return FoldData(
        Xtr=Xtr, ytr=ytr,
        Xva=Xva, yva=yva,
        Xte=Xte, yte=yte,
        lastret_test=lastret_test,
        scaler=scaler,
    )


# -------------------------
# Training loop
# -------------------------
def train_model(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[LSTMRegressor, Dict[str, List[float]]]:
    train_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=cfg.batch, shuffle=True)
    val_loader = DataLoader(SeqDataset(Xva, yva), batch_size=cfg.batch, shuffle=False)

    model = LSTMRegressor(
        n_features=Xtr.shape[-1],
        hidden=cfg.hidden,
        layers=cfg.layers,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=4, factor=0.5)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train": [], "val": []}

    for _ in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += loss_fn(model(xb), yb).item()
        va_loss /= max(1, len(val_loader))

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        sched.step(va_loss)

        if va_loss < best_val - 1e-8:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, history


def predict(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 256) -> np.ndarray:
    loader = DataLoader(SeqDataset(X), batch_size=batch, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(preds)


# -------------------------
# Production model
# -------------------------
def train_production(
    df_feat: pd.DataFrame,
    feature_cols: List[str],
    cfg: TrainConfig,
    device: torch.device,
    out_dir: str = "artifacts",
) -> None:
    """
    Retrain on all data with small holdout for early stopping.
    Save model, scaler, metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    n = len(df_feat)
    holdout = max(90, cfg.seq_len * 3)
    train_end = n - holdout

    fold = Fold(train_end=train_end, val_end=n, test_end=n)
    data = prepare_fold_data(df_feat, feature_cols, fold, cfg.seq_len)

    model, _ = train_model(data.Xtr, data.ytr, data.Xva, data.yva, cfg, device)

    torch.save(model.state_dict(), os.path.join(out_dir, "production_model.pth"))
    with open(os.path.join(out_dir, "feature_scaler.pkl"), "wb") as f:
        pickle.dump(data.scaler, f)

    meta = {
        "symbol": "BTC-USD",
        "target": "next_day_log_return",
        "feature_cols": feature_cols,
        "seq_len": cfg.seq_len,
        "n_rows": n,
        "model": {"hidden": cfg.hidden, "layers": cfg.layers, "dropout": cfg.dropout},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


# -------------------------
# Main
# -------------------------
def main():
    set_seed(42, deterministic=False)

    print("=" * 70)
    print("BitTorch: Return Forecasting (Walk-Forward + Production Artifacts)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nDownloading BTC-USD...")
    t0 = time.time()
    df = yf.download("BTC-USD", period="5y", interval="1d", auto_adjust=True, progress=False)
    df = flatten_yf_columns(df)
    df = df.reset_index()
    print(f"Downloaded {len(df)} rows in {time.time() - t0:.1f}s")

    df_feat = make_features(df)

    candidates = [
        "ret_1", "ret_2", "ret_3",
        "vol_7", "vol_14",
        "mom_7", "mom_14",
        "rsi_14",
        "ma10_dist", "ma30_dist",
        "hl_range",
        "vol_chg",
    ]

    n_raw = len(df_feat)
    segments = [
        (0, int(n_raw * 0.6)),
        (int(n_raw * 0.6), int(n_raw * 0.8)),
        (int(n_raw * 0.8), n_raw),
    ]

    feature_cols = select_features(df_feat, candidates, max_nan_frac=0.05, segments=segments)
    df_feat = df_feat.dropna(subset=feature_cols + ["target_ret_next"]).reset_index(drop=True)

    n = len(df_feat)
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Rows after cleanup: {n}")

    cfg = TrainConfig()
    folds = make_folds(n, seq_len=cfg.seq_len, min_train=365, val_days=90, test_days=90, step=90)

    if not folds:
        print("Not enough data for walk-forward. Reduce min_train or get more data.")
        return

    print(f"\nRunning {len(folds)} walk-forward folds...")
    os.makedirs("checkpoints", exist_ok=True)

    results = []
    for i, fold in enumerate(folds, 1):
        print(f"\nFold {i}: train [0,{fold.train_end}) val [{fold.train_end},{fold.val_end}) test [{fold.val_end},{fold.test_end})")

        data = prepare_fold_data(df_feat, feature_cols, fold, cfg.seq_len)
        model, _ = train_model(data.Xtr, data.ytr, data.Xva, data.yva, cfg, device)

        torch.save(model.state_dict(), f"checkpoints/fold_{i}.pth")

        y_pred = predict(model, data.Xte, device)
        y_true = data.yte.reshape(-1)
        base = compute_baselines(y_true, data.lastret_test)

        row = {
            "fold": i,
            "n_test": len(y_true),
            "mae": mae(y_true, y_pred),
            "diracc": directional_accuracy(y_true, y_pred),
            "base_mae": base["lastret_mae"],
            "base_diracc": base["lastret_diracc"],
        }
        results.append(row)

        print(f"  MAE: {row['mae']:.6f} (baseline: {row['base_mae']:.6f})")
        print(f"  DirAcc: {row['diracc']:.3f} (baseline: {row['base_diracc']:.3f})")

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Summary")
    print(res_df.to_string(index=False))
    print(f"\nAvg MAE: {res_df['mae'].mean():.6f} (baseline: {res_df['base_mae'].mean():.6f})")
    print(f"Avg DirAcc: {res_df['diracc'].mean():.3f} (baseline: {res_df['base_diracc'].mean():.3f})")

    print("\n" + "=" * 70)
    print("Training production model...")
    train_production(df_feat, feature_cols, cfg, device)
    print("Saved: artifacts/production_model.pth, feature_scaler.pkl, metadata.json")

    plt.figure(figsize=(10, 4))
    plt.plot(df_feat["target_ret_next"].values[:300])
    plt.title("Next-day log returns (sample)")
    plt.xlabel("Time")
    plt.ylabel("Log return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("artifacts/returns_sample.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()