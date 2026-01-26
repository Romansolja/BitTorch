# pip install pandas numpy matplotlib yfinance torch scikit-learn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1) Get device
print("=" * 60)
print("BTC Price Prediction With PyTorch")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("-" * 60)

# 2) Download & plot BTC data (REWROTE)
print("Downloading Bitcoin data from Yahoo Finance...")
print("This may take 5-15 seconds depending on connection speed")
start_time = time.time()
btc = yf.download("BTC-USD", period="2y", interval="1d", auto_adjust=True, progress=False)
btc.reset_index(inplace=True)
download_time = time.time() - start_time
print(f"Download complete! ({download_time:.1f} seconds)")
print(f"Downloaded {len(btc)} days of Bitcoin price data")
print("-" * 60)

# Non-blocking plot
print("Generating price history plot (window will open in background)...")
plt.figure(figsize=(12, 6))
plt.plot(btc["Date"], btc["Close"])
plt.title("Bitcoin Price History")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True, alpha=0.3)
plt.show(block=False)
plt.pause(0.1)  # Ensures plots render properly

# 3) Create sequences BEFORE splitting to maintain temporal order
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i + seq_len])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)


SEQ_LEN = 7
prices = btc["Close"].values.reshape(-1, 1)

# 4) Split data BEFORE scaling to avoid leakage
# Using 70% train, 20% val, 10% test
train_size = int(0.7 * len(prices))
val_size = int(0.2 * len(prices))

prices_train = prices[:train_size]
prices_val = prices[train_size:train_size + val_size]
prices_test = prices[train_size + val_size:]

# 5) Fit scaler ONLY on training data
scaler = MinMaxScaler()
scaler.fit(prices_train)  # Fit only on train!

# Transform all sets
prices_train_scaled = scaler.transform(prices_train)
prices_val_scaled = scaler.transform(prices_val)
prices_test_scaled = scaler.transform(prices_test)

# 6) Create sequences for each set
X_train, y_train = create_sequences(prices_train_scaled, SEQ_LEN)
X_val, y_val = create_sequences(prices_val_scaled, SEQ_LEN)
X_test, y_test = create_sequences(prices_test_scaled, SEQ_LEN)

print(f"Train samples: {len(X_train)}")
print(f"Val samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# 7) Calculate baseline performance (tomorrow = today)
# For scaled data
baseline_val_mse = np.mean((X_val[:, -1, 0] - y_val[:, 0]) ** 2)
baseline_test_mse = np.mean((X_test[:, -1, 0] - y_test[:, 0]) ** 2)

# For actual prices (more interpretable)
prices_val_unscaled = scaler.inverse_transform(prices_val_scaled)
prices_test_unscaled = scaler.inverse_transform(prices_test_scaled)

baseline_val_mae = np.mean(np.abs(prices_val_unscaled[:-SEQ_LEN] - prices_val_unscaled[SEQ_LEN:]))
baseline_test_mae = np.mean(np.abs(prices_test_unscaled[:-SEQ_LEN] - prices_test_unscaled[SEQ_LEN:]))

print(f"\nBaseline (tomorrow = today):")
print(f"Val MSE (scaled): {baseline_val_mse:.6f}")
print(f"Test MSE (scaled): {baseline_test_mse:.6f}")
print(f"Val MAE ($): ${baseline_val_mae:.2f}")
print(f"Test MAE ($): ${baseline_test_mae:.2f}")


# 8) Dataset + DataLoader
class SeqDataset(Dataset):
    def __init__(self, X, y):
        super(SeqDataset, self).__init__()
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


BATCH = 32  # Increased batch size
train_ds = SeqDataset(X_train, y_train)
val_ds = SeqDataset(X_val, y_val)
test_ds = SeqDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
test_loader = DataLoader(test_ds, batch_size=BATCH)


# 9) Enhanced model with more capacity
class BitcoinLSTM(nn.Module):
    def __init__(self, in_size=1, hid=128, layers=2, out_size=1, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hid, num_layers=layers,
                            batch_first=True, dropout=drop if layers > 1 else 0)
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(hid, out_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


model = BitcoinLSTM().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 10) Training with better monitoring
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
best_val_loss = float('inf')
patience = 15
epochs_without_improvement = 0

EPOCHS = 150
train_losses = []
val_losses = []

print("\n Starting training...")
for ep in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        opt.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += loss_fn(model(xb), yb).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Print progress
    if ep % 10 == 0 or ep == 1:
        improvement = "POSITIVE" if val_loss < baseline_val_mse else "NEGATIVE"
        print(f"Epoch {ep:03d} ▶ train: {train_loss:.6f}, val: {val_loss:.6f} "
              f"(vs baseline: {baseline_val_mse:.6f}) {improvement}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"⏹ Early stopping at epoch {ep}")
            break

    sched.step(val_loss)

# 11) Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.axhline(y=baseline_val_mse, color='r', linestyle='--', label='Baseline Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 12) Evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_preds = []
test_actuals = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        test_preds.extend(preds)
        test_actuals.extend(yb.numpy())

test_preds = np.array(test_preds)
test_actuals = np.array(test_actuals)

# Calculate metrics
test_mse = np.mean((test_preds - test_actuals) ** 2)
test_mae_scaled = np.mean(np.abs(test_preds - test_actuals))

# Unscale for dollar metrics
test_preds_dollars = scaler.inverse_transform(test_preds)
test_actuals_dollars = scaler.inverse_transform(test_actuals)
test_mae_dollars = np.mean(np.abs(test_preds_dollars - test_actuals_dollars))
test_mape = np.mean(np.abs((test_actuals_dollars - test_preds_dollars) / test_actuals_dollars)) * 100

print(f"\nFinal Results:")
print(f"Test MSE (scaled): {test_mse:.6f} (baseline: {baseline_test_mse:.6f})")
print(f"Test MAE ($): ${test_mae_dollars:.2f} (baseline: ${baseline_test_mae:.2f})")
print(f"Test MAPE: {test_mape:.2f}%")

improvement = ((baseline_test_mae - test_mae_dollars) / baseline_test_mae) * 100
print(f"\nModel improvement over baseline: {improvement:.1f}%")

# 13) Visualize predictions vs actuals
plt.figure(figsize=(12, 6))
sample_size = min(100, len(test_actuals_dollars))
plt.plot(test_actuals_dollars[:sample_size], label='Actual', alpha=0.7)
plt.plot(test_preds_dollars[:sample_size], label='Predicted', alpha=0.7)
plt.xlabel('Time Steps')
plt.ylabel('Price ($)')
plt.title('Bitcoin Price: Actual vs Predicted (Test Set Sample)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 14) Make next-day prediction
# Use the last SEQ_LEN days from the original data
last_seq = prices[-SEQ_LEN:]
last_seq_scaled = scaler.transform(last_seq).reshape(1, SEQ_LEN, 1)
inp = torch.tensor(last_seq_scaled).float().to(device)

with torch.no_grad():
    next_day_scaled = model(inp).cpu().numpy()

next_day_price = scaler.inverse_transform(next_day_scaled)[0][0]
current_price = prices[-1][0]

print(f"\nCurrent BTC price: ${current_price:,.2f}")
print(f"Next-day prediction: ${next_day_price:,.2f}")
print(
    f"Expected change: ${next_day_price - current_price:,.2f} ({((next_day_price / current_price - 1) * 100):.2f}%)")