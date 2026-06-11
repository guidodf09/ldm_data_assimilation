import torch
import torch.nn as nn
import os

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader,
                device, model_save_path, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # -----------------------------
        # Training
        # -----------------------------
        model.train()
        train_loss = 0.0

        for geo_batch, rate_batch in train_loader:
            geo_batch = geo_batch.to(device)
            rate_batch = rate_batch.to(device)

            optimizer.zero_grad()
            outputs = model(geo_batch)
            loss = criterion(outputs, rate_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * geo_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for geo_batch, rate_batch in val_loader:
                geo_batch = geo_batch.to(device)
                rate_batch = rate_batch.to(device)

                outputs = model(geo_batch)
                loss = criterion(outputs, rate_batch)
                val_loss += loss.item() * geo_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Scheduler step (based on val_loss)
        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # -----------------------------
        # Early Stopping
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # -----------------------------
    # Load best weights and save model
    # -----------------------------
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    return model, train_losses, val_losses
