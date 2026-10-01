"""Run the trained CNN on a real beam-camera image."""
import argparse

import torch

import config
from models.cnn import build_model
from .preprocess import preprocess_for_inference


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", ckpt.get("backbone", "medium"))
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_frame(model, model_input, device, resolution=None):
    """Run the model on one preprocessed frame.

    `model_input` is the (1, 1, R, R) array from
    inference.preprocess.to_model_input. Returns (has_beam, presence, x, y,
    sigma_x, sigma_y) with the parameters already denormalized to pixels.
    """
    resolution = resolution or config.TRAIN_RESOLUTION
    tensor = torch.from_numpy(model_input).to(device)

    with torch.no_grad():
        presence_logit, params = model(tensor)
        has_beam = torch.sigmoid(presence_logit).item() > 0.5
        params = params.squeeze(0).cpu().numpy()

    # The network sees the whole frame, so its outputs are already in
    # full-frame pixels -- no ROI rescaling.
    return {
        "has_beam": bool(has_beam),
        "presence_confidence": float(torch.sigmoid(presence_logit).item()),
        "x": float(params[0] * config.IMG_WIDTH),
        "y": float(params[1] * config.IMG_HEIGHT),
        "sigma_x": float(params[2] * config.MAX_SIGMA_NORM),
        "sigma_y": float(params[3] * config.MAX_SIGMA_NORM),
        "model_input_size": resolution,
    }


def run_inference(image_path, model_path=None, device=None):
    device = device or torch.device("cpu")
    model_path = model_path or config.checkpoint_path(config.MODEL_NAME)
    model = load_model(model_path, device)

    _raw, _subtracted, model_input, bbox = preprocess_for_inference(image_path)
    result = predict_frame(model, model_input, device)
    result["blob_bbox"] = tuple(int(v) for v in bbox)
    result["image_path"] = str(image_path)
    return result


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