import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse

# setting
CSV_PATH    = "./seoul_vol_B-32_20250501_20250531.csv"
EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 1e-3
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
class TrafficDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.Y_mean = df['y'].mean()
        self.Y_std = df['y'].std()

        df['y_norm'] = (df['y'] - self.Y_mean) / self.Y_std

        self.X = torch.tensor(df[['x1', 'x2', 'x3', 'x4']].values, dtype=torch.float32)
        self.Y = torch.tensor(df['y_norm'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# Model
class TrafficMLP(nn.Module):
    def __init__(self, hidden_size=128, num_layers=3):
        super().__init__()
        layers = [nn.Linear(4, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# Training
def train(
    lr: float,
    num_epochs: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    data_path: str,
):
    dataset = TrafficDataset(data_path)
    loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=2, 
                pin_memory=True
            )
    model = TrafficMLP(hidden_size=hidden_size, num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            Y_batch = Y_batch.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(X_batch).squeeze(-1)
                    loss = loss_fn(pred, Y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(X_batch).squeeze(-1)
                loss = loss_fn(pred, Y_batch)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                all_X = dataset.X.to(DEVICE)
                all_Y = dataset.Y.to(DEVICE)

                if scaler:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        pred_norm = model(all_X).squeeze(-1)
                else:
                    pred_norm = model(all_X).squeeze(-1)
                pred_vol = pred_norm * dataset.Y_std + dataset.Y_mean
                real_vol = all_Y * dataset.Y_std + dataset.Y_mean
                mae = (pred_vol - real_vol).abs().mean().item()
 
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {total_loss/len(loader):.4f} | MAE: {mae:.1f}")

    torch.save(model.state_dict(), "model.pt")
    print(f"val-loss={mae:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', type=str, default=CSV_PATH)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(
        lr = args.lr,
        num_epochs = args.num_epochs,
        hidden_size = args.hidden_size,
        num_layers = args.num_layers,
        batch_size = args.batch_size,
        data_path = args.data_path,
    )
