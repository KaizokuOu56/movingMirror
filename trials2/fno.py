# ============================================================
# fno.py
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from neuralop.models import FNO

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load dataset
# -----------------------------
data = np.load("dataset.npz")

X = torch.tensor(data["X"], dtype=torch.float32)
Y = torch.tensor(data["Y"], dtype=torch.float32)

# -----------------------------
# Train / Validation split  (80:20 split)
# -----------------------------
dataset = TensorDataset(X, Y)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

# -----------------------------
# Model
# -----------------------------
in_channels = X.shape[1]

model = FNO(
    n_modes=(12,12),
    hidden_channels=64,
    in_channels=in_channels,
    out_channels=2,
    n_layers=4
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------------
#       Training 
#TODO   Increase epochs later
# -----------------------------
EPOCHS = 20

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch:02d} | Train {train_loss:.4e} | Val {val_loss:.4e}")