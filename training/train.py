"""Training loop for the beam-parameter regression CNN."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import config
from data.dataset import BeamDataset
from models.cnn import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_regression_loss(params_pred, params_true, presence_true, loss_fn):
    mask = presence_true.unsqueeze(1)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=params_pred.device)
    diff = loss_fn(params_pred, params_true) * mask
    return diff.sum() / (mask.sum() * params_pred.shape[1])


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    bce = torch.nn.BCEWithLogitsLoss()
    smooth_l1 = torch.nn.SmoothL1Loss(reduction="none")

    total_loss, n_batches = 0.0, 0
    with torch.set_grad_enabled(is_train):
        for images, targets, presence in loader:
            images, targets, presence = images.to(device), targets.to(device), presence.to(device)
            presence_logit, params_pred = model(images)

            loss_presence = bce(presence_logit, presence)
            loss_reg = masked_regression_loss(params_pred, targets, presence, smooth_l1)
            loss = loss_presence + loss_reg

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def train_model(labels_csv, images_dir, epochs, batch_size, lr, model_name, output):
    """Train `model_name` and checkpoint the best-val-loss weights to `output`.

    Returns a dict with the best validation loss and the checkpoint path.
    """
    device = get_device()
    print(f"Using device: {device}")

    train_ds = BeamDataset(labels_csv, images_dir, split="train")
    val_ds = BeamDataset(labels_csv, images_dir, split="val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "model_name": model_name}, output)
            print(f"  saved new best model -> {output}")

    return {"best_val_loss": best_val, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Train the beam-parameter regression CNN.")
    parser.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--model", default=config.MODEL_NAME, choices=list(config.MODEL_CHOICES))
    parser.add_argument("--output", default=None, help="Defaults to checkpoints/<model>_best.pt")
    args = parser.parse_args()

    output = args.output or str(config.checkpoint_path(args.model))
    train_model(args.labels_csv, args.images_dir, args.epochs, args.batch_size, args.lr, args.model, output)


if __name__ == "__main__":
    main()
