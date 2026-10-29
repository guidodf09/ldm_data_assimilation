"""Train the CNN-LSTM surrogate (geomodel -> well rates).

    python scripts/train_surrogate.py configs/surrogate_train.yaml

Data: all geomodels (h5) and simulated rates (all_data_all_times_part{1,2,3}.npy), shuffled with
seed 42, z-scored (StandardScaler on all models / all rates), split 80/10/10 with seed 42.
Saves model_trained.pth (best validation) and scalers.joblib to the paths in base.yaml,
plus losses and split indices in the same folder.
"""
import argparse
import os
import sys

import h5py
import joblib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldmda.surrogate import GeoToRatesModel  # noqa: E402
from ldmda.utils import model2tricat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/surrogate_train.yaml")
ap.add_argument("--base", default="configs/base.yaml")
args = ap.parse_args()
with open(args.base) as f:
    base = yaml.safe_load(f)
with open(args.config) as f:
    tc = yaml.safe_load(f)
arch, g, paths = base["surrogate"], base["grid"], base["paths"]
device = torch.device(base["device"] if torch.cuda.is_available() else "cpu")
torch.manual_seed(tc["seed"])
out_dir = os.path.dirname(paths["surrogate_weights"]) or "."
os.makedirs(out_dir, exist_ok=True)

# ---- data ----
with h5py.File(paths["geomodels_h5"], "r") as f:
    models = model2tricat(f["data"][:].astype(np.float32) / 255.0, *base["ldm"]["thresholds"])  # (N, 1, nx, ny, nz)
rates = np.concatenate([np.load(os.path.join(paths["rates_dir"], f"all_data_all_times_part{k}.npy"))
                        for k in (1, 2, 3)])                                                   # (N, 30, 15)
assert len(models) == len(rates), f"{len(models)} geomodels vs {len(rates)} rate samples"
order = np.random.RandomState(tc["seed"]).permutation(len(rates))
models, rates = models[order], rates[order]

model_scaler = StandardScaler().fit(models.reshape(len(models), -1))
rate_scaler = StandardScaler().fit(rates.reshape(-1, rates.shape[-1]))
joblib.dump((model_scaler, rate_scaler), paths["surrogate_scalers"])

x = torch.tensor(model_scaler.transform(models.reshape(len(models), -1)).reshape(models.shape),
                 dtype=torch.float32)
y = torch.tensor(rate_scaler.transform(rates.reshape(-1, rates.shape[-1])).reshape(rates.shape),
                 dtype=torch.float32)
del models

n = len(x)
n_train, n_val = int(tc["split"][0] * n), int(tc["split"][1] * n)
train_ds, val_ds, test_ds = random_split(TensorDataset(x, y), [n_train, n_val, n - n_train - n_val],
                                         generator=torch.Generator().manual_seed(tc["seed"]))
np.savez(os.path.join(out_dir, "split_indices.npz"), order=order,
         train=train_ds.indices, val=val_ds.indices, test=test_ds.indices)
train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False)

# ---- model ----
model = GeoToRatesModel(
    input_shape=(1, g["nx"], g["ny"], g["nz"]), conv_channels=arch["conv_channels"],
    latent_dim=arch["latent_dim"], time_steps=arch["time_steps"],
    lstm_hidden_size=arch["lstm_hidden_size"], lstm_layers=arch["lstm_layers"],
    bidirectional=arch["bidirectional"], output_dim=arch["n_qoi"], dropout_rate=arch["dropout"],
).to(device)
print(model)
print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

criterion = nn.HuberLoss(delta=tc["huber_delta"]) if tc["huber_delta"] else nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=tc["lr_scheduler"]["factor"],
                                                       patience=tc["lr_scheduler"]["patience"])

# ---- training with early stopping ----
best, wait, train_losses, val_losses = float("inf"), 0, [], []
for epoch in range(tc["num_epochs"]):
    model.train()
    tl = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        if tc["grad_clip"]:
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        optimizer.step()
        tl += loss.item() * len(xb)
    train_losses.append(tl / len(train_ds))

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            vl += criterion(model(xb), yb).item() * len(xb)
    val_losses.append(vl / len(val_ds))
    scheduler.step(val_losses[-1])
    print(f"epoch {epoch + 1}/{tc['num_epochs']} | train {train_losses[-1]:.6f} | val {val_losses[-1]:.6f}")

    if val_losses[-1] < best:
        best, wait = val_losses[-1], 0
        torch.save(model.state_dict(), paths["surrogate_weights"])
    else:
        wait += 1
        if wait >= tc["patience"]:
            print(f"early stopping at epoch {epoch + 1}")
            break

np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))
print(f"best val loss {best:.6f} -> {paths['surrogate_weights']}")
