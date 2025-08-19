import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import uuid
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from app.models.ml_models import BitcoinLSTM
from app.database import SessionLocal, ModelMetrics
from app.config import MODEL_PATH, DATA_DIR


class TrainingService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_jobs: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X).float()
            self.y = torch.tensor(y).float()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    def create_sequences(self, data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i + seq_len])
            ys.append(data[i + seq_len])
        return np.array(xs), np.array(ys)

    async def train_model_async(self, params: Dict[str, Any]) -> str:
        """Async wrapper for training"""
        training_id = str(uuid.uuid4())

        # Store job info
        self.training_jobs[training_id] = {
            "status": "running",
            "started_at": datetime.now(),
            "parameters": params,
            "progress": 0
        }

        # Run training in background
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            self.executor,
            self._train_model,
            training_id,
            params
        )

        return training_id

    def _train_model(self, training_id: str, params: Dict[str, Any]):
        """Actual training logic"""
        try:
            # Update status
            self.training_jobs[training_id]["status"] = "downloading_data"

            # Download data
            btc = yf.download("BTC-USD", period="2y", interval="1d",
                              auto_adjust=True, progress=False)
            prices = btc["Close"].values.reshape(-1, 1)

            # Split data
            train_size = int(0.7 * len(prices))
            val_size = int(0.2 * len(prices))

            prices_train = prices[:train_size]
            prices_val = prices[train_size:train_size + val_size]
            prices_test = prices[train_size + val_size:]

            # Scale data
            scaler = MinMaxScaler()
            scaler.fit(prices_train)

            prices_train_scaled = scaler.transform(prices_train)
            prices_val_scaled = scaler.transform(prices_val)
            prices_test_scaled = scaler.transform(prices_test)

            # Create sequences
            seq_len = params.get("sequence_length", 7)
            X_train, y_train = self.create_sequences(prices_train_scaled, seq_len)
            X_val, y_val = self.create_sequences(prices_val_scaled, seq_len)
            X_test, y_test = self.create_sequences(prices_test_scaled, seq_len)

            # Create datasets
            batch_size = params.get("batch_size", 32)
            train_ds = self.SeqDataset(X_train, y_train)
            val_ds = self.SeqDataset(X_val, y_val)
            test_ds = self.SeqDataset(X_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size)
            test_loader = DataLoader(test_ds, batch_size=batch_size)

            # Initialize model
            model = BitcoinLSTM().to(self.device)
            loss_fn = nn.MSELoss()
            lr = params.get("learning_rate", 0.001)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=5, factor=0.5
            )

            # Training loop
            epochs = params.get("epochs", 150)
            best_val_loss = float('inf')
            patience = 15
            epochs_without_improvement = 0

            self.training_jobs[training_id]["status"] = "training"

            for epoch in range(1, epochs + 1):
                # Training
                model.train()
                train_loss = 0
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        val_loss += loss_fn(model(xb), yb).item()
                val_loss /= len(val_loader)

                # Update progress
                self.training_jobs[training_id]["progress"] = epoch / epochs * 100
                self.training_jobs[training_id]["current_epoch"] = epoch
                self.training_jobs[training_id]["train_loss"] = train_loss
                self.training_jobs[training_id]["val_loss"] = val_loss

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save checkpoint
                    checkpoint_path = DATA_DIR / "models" / f"checkpoint_{training_id}.pth"
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

                scheduler.step(val_loss)

            # Load best model
            model.load_state_dict(torch.load(checkpoint_path))

            # Test evaluation
            model.eval()
            test_mse = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    test_mse += loss_fn(model(xb), yb).item()
            test_mse /= len(test_loader)

            # Calculate baseline
            baseline_test_mse = np.mean((X_test[:, -1, 0] - y_test[:, 0]) ** 2)
            improvement = ((baseline_test_mse - test_mse) / baseline_test_mse) * 100

            # Save final model
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            torch.save(model.state_dict(), MODEL_PATH)

            # Save metrics to database
            db = SessionLocal()
            try:
                metrics = ModelMetrics(
                    model_version=model_version,
                    train_date=datetime.now(),
                    mse=test_mse,
                    mae=0,  # Calculate if needed
                    mape=0,  # Calculate if needed
                    baseline_improvement=improvement
                )
                db.add(metrics)
                db.commit()
            finally:
                db.close()

            # Update job status
            self.training_jobs[training_id]["status"] = "completed"
            self.training_jobs[training_id]["model_version"] = model_version
            self.training_jobs[training_id]["test_mse"] = test_mse
            self.training_jobs[training_id]["improvement"] = improvement
            self.training_jobs[training_id]["completed_at"] = datetime.now()

            # Cleanup checkpoint
            os.remove(checkpoint_path)

        except Exception as e:
            self.training_jobs[training_id]["status"] = "failed"
            self.training_jobs[training_id]["error"] = str(e)

    def get_training_status(self, training_id: str) -> Dict[str, Any]:
        """Get status of a training job"""
        return self.training_jobs.get(training_id, {"status": "not_found"})


training_service = TrainingService()