import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Subset
from models import GeoRatesDataset, GeoToRatesModel
from train import train_model
import time
import optuna
import json
from hyperopt import create_objective
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(mode="train",
         dataset_dir=".",
         dataset_file="paired_dataset_combined_chunk000.h5",
         rates_file="all_data_all_times.npy",
         case_name="case0",
         pics_name="pics"):
    # -----------------------------
    # Directory Setup
    # -----------------------------
    case_dir = os.path.join(os.getcwd(), case_name)
    pics_dir = os.path.join(case_dir, pics_name)
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(pics_dir, exist_ok=True)

    # -----------------------------
    # Load Data  (X has 4 channels: [facies, poro, perm, ar])
    # -----------------------------
    with h5py.File(os.path.join(dataset_dir, dataset_file), 'r') as f:
        X_loaded = np.array(f["X"])                 # (N, 128, 128, 32, 4)

    rates_loaded = np.load(os.path.join(dataset_dir, rates_file))  # (N, T, F)

    # channels-last -> channels-first, drop facies (channel 0)
    # keep [poro, perm, ar] -> (N, 3, 128, 128, 32)
    X_loaded = np.transpose(X_loaded, (0, 4, 1, 2, 3))[:, 1:].astype(np.float32)
    X_loaded[:, 1] = np.log(X_loaded[:, 1])   # log-transform perm channel
    X_loaded[:, 2] = np.log(X_loaded[:, 2])   # log-transform ar channel

    # -----------------------------
    # Shuffle (same permutation across both)
    # -----------------------------
    np.random.seed(42)
    random_order = np.random.permutation(rates_loaded.shape[0])
    rates_loaded = rates_loaded[random_order]
    X_loaded     = X_loaded[random_order]

    # -----------------------------
    # Split indices first, then normalize using train stats only
    # -----------------------------
    N, T, F = rates_loaded.shape
    total_samples = N
    train_size    = int(0.8 * total_samples)
    val_size      = int(0.1 * total_samples)
    test_size     = total_samples - train_size - val_size

    # Deterministic index split (matches random_split with same seed)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    train_idx = idx[:train_size]
    val_idx   = idx[train_size:train_size + val_size]
    test_idx  = idx[train_size + val_size:]

    # Fit scalers on train only
    rate_scaler = StandardScaler()
    rates_normalized = np.zeros_like(rates_loaded, dtype=np.float32)
    rates_normalized[train_idx] = rate_scaler.fit_transform(
        rates_loaded[train_idx].reshape(-1, F)).reshape(-1, T, F)
    rates_normalized[val_idx]  = rate_scaler.transform(
        rates_loaded[val_idx].reshape(-1, F)).reshape(-1, T, F)
    rates_normalized[test_idx] = rate_scaler.transform(
        rates_loaded[test_idx].reshape(-1, F)).reshape(-1, T, F)
    rates_normalized = torch.tensor(rates_normalized, dtype=torch.float32)
    torch.save(
        {'mean': torch.tensor(rate_scaler.mean_), 'std': torch.tensor(rate_scaler.scale_)},
        os.path.join(case_dir, 'rate_scaler_stats.pt')
    )

    geo = torch.tensor(X_loaded, dtype=torch.float32)
    geo_train = geo[train_idx]
    ch_mean = geo_train.mean(dim=(0, 2, 3, 4), keepdim=True)
    ch_std  = geo_train.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-6
    geo_normalized = (geo - ch_mean) / ch_std
    geo_stats = {'mean': ch_mean, 'std': ch_std}
    torch.save(geo_stats, os.path.join(case_dir, 'geo_channel_stats.pt'))

    # -----------------------------
    # Dataset and DataLoaders
    # -----------------------------
    full_dataset  = GeoRatesDataset(geo_normalized, rates_normalized)
    train_dataset = Subset(full_dataset, train_idx.tolist())
    val_dataset   = Subset(full_dataset, val_idx.tolist())
    test_dataset  = Subset(full_dataset, test_idx.tolist())

    batch_size   = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Modes
    # -----------------------------
    if mode == 'hyperopt':
        objective = create_objective(train_loader, val_loader, time_steps=T, output_dim=F)
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=100)
        print("Best trial:")
        print(study.best_trial)

    elif mode == 'train':
        best_params = {
            'conv_channels':    (16, 32, 64),
            'latent_dim':       256,
            'lstm_hidden_size': 128,
            'lstm_layers':      2,
            'bidirectional':    False,
            'dropout':          0.3,
            'lr':               1e-4,
            'weight_decay':     1e-4,
            'use_huber':        True,
            'grad_clip':        1.0,
            'pool_output':      (4, 4, 2),
            'use_coords':       True,
        }

        model = GeoToRatesModel(
            conv_channels=best_params["conv_channels"],
            latent_dim=best_params["latent_dim"],
            time_steps=T,
            lstm_hidden_size=best_params["lstm_hidden_size"],
            lstm_layers=best_params["lstm_layers"],
            bidirectional=best_params["bidirectional"],
            output_dim=F,
            dropout_rate=best_params['dropout'],
            pool_output=best_params['pool_output'],
            use_coords=best_params['use_coords'],
        ).to(device)

        criterion = nn.HuberLoss(delta=1.0) if best_params['use_huber'] else nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        start_time = time.time()
        model, train_losses, val_losses = train_model(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader, device,
            model_save_path=os.path.join(case_dir, "model_trained.pth"),
            num_epochs=1000, patience=30,
            grad_clip=best_params['grad_clip'],
        )
        total_time = time.time() - start_time

        print(f"Training complete.")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average time per epoch: {total_time / len(train_losses):.2f} seconds")

        np.save(os.path.join(case_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(case_dir, 'val_losses.npy'),   np.array(val_losses))
        pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
            os.path.join(case_dir, 'loss_history.csv'), index_label='epoch'
        )
        with open(os.path.join(case_dir, "model_config.json"), "w") as f:
            json.dump(best_params, f, indent=4)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'train' or 'hyperopt'.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    DATASET_DIR  = "../dataset/dataset_chunks/"
    DATASET_FILE = "paired_dataset_combined_chunk000_old.h5"
    RATES_FILE   = "all_data_all_times_chunk000_old.npy"
    CASE_NAME    = "case0"
    PICS_NAME    = "pics"

    main(mode=mode,
         dataset_dir=DATASET_DIR,
         dataset_file=DATASET_FILE,
         rates_file=RATES_FILE,
         case_name=CASE_NAME,
         pics_name=PICS_NAME)