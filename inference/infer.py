"""Run the trained CNN on a real beam-camera image."""
import argparse

import numpy as np
import torch
from PIL import Image

import config
from models.cnn import build_model
from .preprocess import preprocess_for_inference


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt.get("model_name", ckpt.get("backbone", "medium"))
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_inference(image_path, model_path=None, device=None, resolution=config.TRAIN_RESOLUTION):
    device = device or torch.device("cpu")
    model_path = model_path or config.checkpoint_path(config.MODEL_NAME)
    model = load_model(model_path, device)

    _, subtracted, roi, bbox = preprocess_for_inference(image_path)
    roi_resized = Image.fromarray(roi).resize((resolution, resolution), Image.BILINEAR)
    roi_arr = np.asarray(roi_resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(roi_arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        presence_logit, params = model(tensor)
        has_beam = torch.sigmoid(presence_logit).item() > 0.5
        params = params.squeeze(0).cpu().numpy()

    x0, y0, x1, y1 = bbox
    roi_w, roi_h = (x1 - x0), (y1 - y0)

    # x, y are fractions of the input frame -> scale by the actual ROI size.
    x_roi = params[0] * roi_w
    y_roi = params[1] * roi_h
    # sigma was normalized against a full config.IMG_WIDTH/HEIGHT canvas during
    # training, so rescale by how much smaller/larger the ROI is than that.
    sigma_x = params[2] * config.MAX_SIGMA_NORM * (roi_w / config.IMG_WIDTH)
    sigma_y = params[3] * config.MAX_SIGMA_NORM * (roi_h / config.IMG_HEIGHT)

    return {
        "has_beam": bool(has_beam),
        "x": float(x0 + x_roi),
        "y": float(y0 + y_roi),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "roi_bbox": bbox,
    }


def main():
    parser = argparse.ArgumentParser(description="Run beam-parameter inference on a real image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", default=str(config.checkpoint_path(config.MODEL_NAME)))
    args = parser.parse_args()

    result = run_inference(args.image, args.model_path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
