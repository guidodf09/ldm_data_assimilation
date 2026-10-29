import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from models import GeoToRatesModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_objective(train_loader, val_loader, time_steps, output_dim):
    def objective(trial):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sample hyperparameters
        conv_channels = trial.suggest_categorical(
            "conv_channels",
            [(16,32,64,128), (32,64,128), (32,64,128,256)]
        )
        latent_dim       = trial.suggest_categorical("latent_dim", [256, 512, 1024])
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
        lstm_layers      = trial.suggest_int("lstm_layers", 1, 2)
        bidirectional    = trial.suggest_categorical("bidirectional", [True, False])
        dropout          = trial.suggest_float("dropout", 0.05, 0.3)
        lr               = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay     = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        use_huber        = trial.suggest_categorical("use_huber", [True, False])
        grad_clip        = trial.suggest_float("grad_clip", 0.5, 5.0)

        model = GeoToRatesModel(
            conv_channels=conv_channels,
            latent_dim=latent_dim,
            time_steps=time_steps,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            output_dim=output_dim,
            dropout_rate=dropout,
        ).to(device)

        criterion = nn.HuberLoss(delta=1.0) if use_huber else nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val_loss = float('inf')
        patience, wait = 8, 0

        for epoch in range(30):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb), yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        return best_val_loss
    return objective