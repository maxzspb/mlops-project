# train.py
import os
import argparse
import time
import socket
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--worker", action="store_true", help="flag for worker (optional)")
    return p.parse_args()

def init_distributed():
    """
    Try to init torch.distributed using environment (common pattern: env:// with proper RANK/WORLD_SIZE)
    If not configured, return False.
    """
    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        try:
            torch.distributed.init_process_group(backend='gloo', init_method='env://')
            return True
        except Exception as e:
            print("Failed to init distributed:", e)
            return False
    return False

class SimpleNet(nn.Module):
    def __init__(self, in_dim=4, hid=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def load_data():
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    return train_ds, test_ds

def train_loop(device, model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    return total_loss / total

def eval_loop(device, model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()

    # MLflow / MinIO env (tracking uri should be set by env MLFLOW_TRACKING_URI or passed externally)
    mlflow_tracking = os.environ.get("MLFLOW_TRACKING_URI", os.environ.get("MLFLOW_URI", "http://mlflow.kubeflow.svc.cluster.local:5000"))
    mlflow.set_tracking_uri(mlflow_tracking)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "kserve-experiment")
    mlflow.set_experiment(experiment_name)

    # Try distributed init if requested via env
    distributed = init_distributed()
    local_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"Distributed: {distributed}, RANK={local_rank}, WORLD_SIZE={world_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds, test_ds = load_data()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SimpleNet()
    if distributed:
        # wrap with DDP if cuda available and using nccl; for CPU use DistributedDataParallel with gloo
        try:
            model = model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
        except Exception as e:
            print("DDP wrap failed (continuing single-process):", e)
    else:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run() as run:
        for epoch in range(1, args.epochs + 1):
            loss = train_loop(device, model, train_loader, optimizer, loss_fn)
            acc = eval_loop(device, model, test_loader)
            mlflow.log_metric("epoch_loss", loss, step=epoch)
            mlflow.log_metric("accuracy", acc, step=epoch)
            print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}, acc={acc:.4f}")
        # Log model artifact
        mlflow.pytorch.log_model(model.module if hasattr(model, "module") else model, artifact_path="model")
        artifact_uri = mlflow.get_artifact_uri("model")
        print("Model logged to MLflow at:", artifact_uri)

if __name__ == "__main__":
    main()
