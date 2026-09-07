"""Quick evaluation: pixel-space MAE and beam-presence accuracy on a split."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.dataset import BeamDataset
from models.cnn import build_model
from .train import get_device


def denormalize(params):
    x = params[:, 0] * config.IMG_WIDTH
    y = params[:, 1] * config.IMG_HEIGHT
    sx = params[:, 2] * config.MAX_SIGMA_NORM
    sy = params[:, 3] * config.MAX_SIGMA_NORM
    return np.stack([x, y, sx, sy], axis=1)


def evaluate_model(labels_csv, images_dir, model_path, split="test"):
    """Returns {model_name, presence_accuracy, mae} for the given split.

    `mae` is a length-4 array [x, y, sigma_x, sigma_y] in pixels, or None if
    the split has no beam-present samples.
    """
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", ckpt.get("backbone", "medium"))
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = BeamDataset(labels_csv, images_dir, split=split)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    all_errors = []
    presence_correct, total = 0, 0
    with torch.no_grad():
        for images, targets, presence in loader:
            images = images.to(device)
            presence_logit, params_pred = model(images)
            pred_presence = (torch.sigmoid(presence_logit) > 0.5).float().cpu()
            presence_correct += (pred_presence == presence).sum().item()
            total += presence.shape[0]

            mask = presence.numpy() == 1
            if mask.any():
                pred_px = denormalize(params_pred.cpu().numpy())
                true_px = denormalize(targets.numpy())
                all_errors.append(np.abs(pred_px[mask] - true_px[mask]))

    mae = np.concatenate(all_errors, axis=0).mean(axis=0) if all_errors else None
    return {
        "model_name": model_name,
        "presence_accuracy": presence_correct / total if total else 0.0,
        "mae": mae,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained regression CNN.")
    parser.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    parser.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    parser.add_argument("--model-path", default=str(config.checkpoint_path(config.MODEL_NAME)))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    metrics = evaluate_model(args.labels_csv, args.images_dir, args.model_path, args.split)
    print(f"Model: {metrics['model_name']}")
    print(f"Beam-presence accuracy: {metrics['presence_accuracy']:.3f}")
    if metrics["mae"] is not None:
        x, y, sx, sy = metrics["mae"]
        print(f"MAE (pixels) -> x: {x:.2f}  y: {y:.2f}  sigma_x: {sx:.2f}  sigma_y: {sy:.2f}")
    else:
        print("No beam-present samples found in this split.")


if __name__ == "__main__":
    main()
