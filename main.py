import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from utils import model2tricat 
from models_impr import GeoRatesDataset, GeoToRatesModel
from train_impr import train_model
import time
import optuna
from hyperopt_impr import create_objective

from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(mode="train"):
    # -----------------------------
    # Directory Setup
    # -----------------------------
    h5_file_path = '/oak/stanford/groups/lou/gdifede/3d_datasets/geomodels_128_paper.h5'
    case_dir = os.path.join(os.getcwd(), 'case4')
    pics_dir = os.path.join(case_dir, 'pics')

    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(pics_dir, exist_ok=True)

    # -----------------------------
    # Load Data
    # -----------------------------
    with h5py.File(h5_file_path, 'r') as f:
        raw_models = np.array(f["data"])[:] / 255.0  # keep original 0/1/2 or 0/0.5/1

    models_loaded = model2tricat(raw_models, thresh1=0.25, thresh2=0.8)

    rates_loaded_1 = np.load('/oak/stanford/groups/lou/gdifede/3d_surrogate_data/all_data_all_times_ok_1.npy')
    rates_loaded_2 = np.load('/oak/stanford/groups/lou/gdifede/3d_surrogate_data/all_data_all_times_ok_2.npy')
    rates_loaded_3 = np.load('/oak/stanford/groups/lou/gdifede/3d_surrogate_data/all_data_all_times_ok_3.npy')
    rates_loaded = np.concatenate((rates_loaded_1, rates_loaded_2, rates_loaded_3), axis=0)

    # -----------------------------
    # Shuffle
    # -----------------------------
    np.random.seed(42)
    random_order = np.random.permutation(rates_loaded.shape[0])
    rates_loaded = rates_loaded[random_order]
    models_loaded = models_loaded[random_order]

    # -----------------------------
    # Normalize data using sklearn scalers
    # -----------------------------
    normalization_type = 'standard'  # or 'minmax'
    Scaler = StandardScaler if normalization_type == 'standard' else MinMaxScaler

    # Normalize rates: shape [N, T, F]
    rate_scaler = Scaler()
    N, T, F = rates_loaded.shape
    rates_flat = rates_loaded.reshape(-1, F)
    rates_normalized = rate_scaler.fit_transform(rates_flat)
    rates_normalized = torch.tensor(rates_normalized.reshape(N, T, F), dtype=torch.float32)

    # Normalize models: shape [N, H, W, D]
    model_scaler = Scaler()
    models_shape = models_loaded.shape
    models_flat = models_loaded.reshape(models_shape[0], -1)
    models_normalized = model_scaler.fit_transform(models_flat)
    models_normalized = torch.tensor(models_normalized.reshape(models_shape), dtype=torch.float32)

    # Store scalers for inverse transform later
    rate_stats = {"scaler": rate_scaler, "method": normalization_type}
    model_stats = {"scaler": model_scaler, "method": normalization_type}

    # -----------------------------
    # Dataset and DataLoader
    # -----------------------------
    total_samples = len(models_loaded)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset, test_dataset = random_split(
        GeoRatesDataset(models_normalized, rates_normalized),
        [train_size, val_size, test_size],
        generator=generator
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # -----------------------------
    # Modes
    # -----------------------------
    if mode == 'hyperopt':
        objective = create_objective(train_loader, val_loader)
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=100)

        print("Best trial:")
        print(study.best_trial)
        best_params = study.best_trial.params

    elif mode == 'train':

        best_params = {'conv_channels': (16, 32, 64, 128),
                      'latent_dim': 512,
                      'lstm_hidden_size': 64,
                      'lstm_layers': 1,
                      'bidirectional': False,
                      'dropout': 0.1155930716554599,
                      'lr': 0.002038540481683897,
                      'weight_decay': 2.1669376391175675e-06,
                      'use_huber': True } #True}

        model = GeoToRatesModel(
            conv_channels=best_params["conv_channels"],
            latent_dim=best_params["latent_dim"],
            time_steps=rates_loaded.shape[1],
            lstm_hidden_size=best_params["lstm_hidden_size"],
            lstm_layers=best_params["lstm_layers"],
            bidirectional=best_params["bidirectional"],
            output_dim=rates_loaded.shape[2],
            dropout_rate=best_params['dropout']
        ).to(device)

        criterion = nn.HuberLoss(delta=1.0) if best_params['use_huber'] else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # -----------------------------
        # Train
        # -----------------------------
        start_time = time.time()

        model, train_losses, val_losses = train_model(
            model, criterion, optimizer, scheduler,
            train_loader, val_loader, device,
            model_save_path=os.path.join(case_dir, "model_trained.pth"),
            num_epochs=1000, patience=30
        )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_epoch = total_time / len(train_losses)

        print(f"Training complete. Best model weights loaded.")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average time per epoch: {avg_time_per_epoch:.2f} seconds")

        # -----------------------------
        # Save loss histories
        # -----------------------------
        np.save(os.path.join(case_dir, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(case_dir, 'val_losses.npy'), np.array(val_losses))
        pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses}).to_csv(
            os.path.join(case_dir, 'loss_history.csv'), index_label='epoch'
        )

        # -----------------------------
        # Save model hyperparameters
        # -----------------------------
        import json
        with open(os.path.join(case_dir, "model_config.json"), "w") as f:
            json.dump(best_params, f, indent=4)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'train' or 'hyperopt'.")

if __name__ == "__main__":
    main(mode="train")  # or main(mode="hyperopt")
