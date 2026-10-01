#!/usr/bin/env python3
"""Single entry point for the Gaussian-beam CNN regression pipeline.

The dataset is the real BPM capture set under `new_dataset/`. Per-image labels
are derived from each frame by data/real_dataset.py, then used to train the
regression CNNs (tiny/medium/resnet18).

    python main.py                 # == python main.py all

Subcommands (see README.md for the full reference):

    python main.py all         # build labels if missing, then train all models
    python main.py labels      # extract labels.csv from the captured frames
    python main.py generate    # build the synthetic dataset (legacy)
    python main.py validate    # check image/label quality
    python main.py preview     # save a per-category sample grid
    python main.py train       # train a single regression CNN
    python main.py evaluate    # pixel-space MAE + presence accuracy on a split
    python main.py infer       # run a trained model on a real image
"""
import argparse
import sys
from pathlib import Path

import config
from data.preview import save_preview_grid
from data.real_dataset import build_label_index, iter_capture_images
from data.validate import print_validation_report, validate_dataset
from inference.infer import run_inference
from synthetic.dataset_builder import build_dataset
from training.evaluate import evaluate_model
from training.train import train_model


def dataset_exists(labels_csv=None, images_dir=None):
    """True if labels.csv exists and matches the captures currently on disk."""
    labels_csv = Path(labels_csv or config.LABELS_CSV)
    images_dir = Path(images_dir or config.IMAGES_DIR)
    if not labels_csv.exists() or not images_dir.exists():
        return False

    import pandas as pd
    on_disk = sum(1 for _ in iter_capture_images(images_dir))
    labeled = len(pd.read_csv(labels_csv))
    if on_disk != labeled:
        print(f"Index is stale: {labeled} label rows for {on_disk} captures on disk.")
        return False
    return True


def cmd_labels(args):
    build_label_index(args.images_dir, args.labels_csv, args.val_split, args.seed)


def cmd_generate(args):
    counts = {
        "normal": args.normal,
        "tilted": args.tilted,
        "noisy": args.noisy,
        "background": args.background,
    }
    build_dataset(output_dir=args.output_dir, counts=counts, seed=args.seed)


def cmd_validate(args):
    summary, issues = validate_dataset(args.labels_csv, args.images_dir)
    print_validation_report(summary, issues)
    if issues:
        sys.exit(1)


def cmd_preview(args):
    save_preview_grid(args.labels_csv, args.images_dir, args.per_category, args.output)


def cmd_train(args):
    output = args.output or str(config.checkpoint_path(args.model))
    train_model(args.labels_csv, args.images_dir, args.epochs, args.batch_size, args.lr, args.model, output)


def cmd_evaluate(args):
    metrics = evaluate_model(args.labels_csv, args.images_dir, args.model_path, args.split, args.by_category)
    print(f"Model: {metrics['model_name']}")
    print(f"Beam-presence accuracy: {metrics['presence_accuracy']:.3f}")
    if metrics["mae"] is not None:
        x, y, sx, sy = metrics["mae"]
        print(f"MAE (pixels) -> x: {x:.2f}  y: {y:.2f}  sigma_x: {sx:.2f}  sigma_y: {sy:.2f}")
    else:
        print("No beam-present samples found in this split.")
    for row in metrics.get("mae_by_category", []):
        print(
            f"  {row['category']:<6} n={row['count']:<4}"
            f" x: {row['mae_x']:>7.2f}  y: {row['mae_y']:>7.2f}"
            f"  sigma_x: {row['mae_sigma_x']:>7.2f}  sigma_y: {row['mae_sigma_y']:>7.2f}"
        )


def cmd_infer(args):
    result = run_inference(args.image, args.model_path)
    for key, value in result.items():
        print(f"{key}: {value}")


def _print_footer(results):
    print("\n" + "=" * 64)
    print("Training complete.")
    print("=" * 64)
    header = f"{'model':<10}{'presence_acc':>14}{'mae_x':>10}{'mae_y':>10}{'mae_sx':>10}{'mae_sy':>10}"
    print(header)
    for name, metrics in results.items():
        if metrics["mae"] is not None:
            x, y, sx, sy = metrics["mae"]
            print(f"{name:<10}{metrics['presence_accuracy']:>14.3f}{x:>10.2f}{y:>10.2f}{sx:>10.2f}{sy:>10.2f}")
        else:
            print(f"{name:<10}{metrics['presence_accuracy']:>14.3f}{'--':>10}{'--':>10}{'--':>10}{'--':>10}")
    print(
        "\nCheckpoints saved under checkpoints/<model>_best.pt\n\n"
        "Next steps:\n"
        "  python main.py evaluate --model-path checkpoints/<model>_best.pt --split test\n"
        "  python main.py infer --image <path_to_image> --model-path checkpoints/<model>_best.pt\n\n"
        "-------------------------------------------------------------------\n"
        "Gaussian Beam CNN Regression Pipeline -- see README.md for full usage.\n"
        "-------------------------------------------------------------------"
    )


def cmd_all(args):
    if dataset_exists(args.labels_csv, args.images_dir):
        print(f"Labels found at {args.labels_csv} -- skipping extraction.")
    else:
        print("Extracting labels from the captured frames...")
        build_label_index(args.images_dir, args.labels_csv)

    results = {}
    for model_name in args.models:
        print(f"\n=== Training '{model_name}' ({args.epochs} epochs) ===")
        output = config.checkpoint_path(model_name)
        train_model(args.labels_csv, args.images_dir, args.epochs, args.batch_size, args.lr, model_name, str(output))
        results[model_name] = evaluate_model(args.labels_csv, args.images_dir, str(output), split="test")

    _print_footer(results)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Gaussian-beam CNN regression pipeline (labels -> train -> infer).",
    )
    sub = parser.add_subparsers(dest="command")

    p_all = sub.add_parser("all", help="Extract labels if missing, then train all models (default)")
    p_all.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_all.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_all.add_argument("--models", nargs="+", default=list(config.MODEL_CHOICES), choices=list(config.MODEL_CHOICES))
    p_all.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_all.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_all.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p_all.set_defaults(func=cmd_all)

    p_labels = sub.add_parser("labels", help="Extract labels.csv from the captured frames")
    p_labels.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_labels.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_labels.add_argument("--val-split", type=float, default=config.VAL_SPLIT)
    p_labels.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_labels.set_defaults(func=cmd_labels)

    p_gen = sub.add_parser("generate", help="Generate the synthetic dataset (legacy)")
    p_gen.add_argument("--output-dir", default=str(config.SYNTHETIC_DIR))
    p_gen.add_argument("--normal", type=int, default=config.NUM_NORMAL)
    p_gen.add_argument("--tilted", type=int, default=config.NUM_TILTED)
    p_gen.add_argument("--noisy", type=int, default=config.NUM_NOISY)
    p_gen.add_argument("--background", type=int, default=config.NUM_BACKGROUND)
    p_gen.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_gen.set_defaults(func=cmd_generate)

    p_val = sub.add_parser("validate", help="Validate image/label quality")
    p_val.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_val.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_val.set_defaults(func=cmd_validate)

    p_prev = sub.add_parser("preview", help="Save a preview grid of sample images per category")
    p_prev.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_prev.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_prev.add_argument("--per-category", type=int, default=4)
    p_prev.add_argument("--output", default=str(config.DATASET_DIR / "preview.png"))
    p_prev.set_defaults(func=cmd_preview)

    p_train = sub.add_parser("train", help="Train a regression CNN")
    p_train.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_train.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_train.add_argument("--model", default=config.MODEL_NAME, choices=list(config.MODEL_CHOICES))
    p_train.add_argument("--epochs", type=int, default=config.EPOCHS)
    p_train.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p_train.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p_train.add_argument("--output", default=None, help="Defaults to checkpoints/<model>_best.pt")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained checkpoint")
    p_eval.add_argument("--labels-csv", default=str(config.LABELS_CSV))
    p_eval.add_argument("--images-dir", default=str(config.IMAGES_DIR))
    p_eval.add_argument("--model-path", default=str(config.checkpoint_path(config.MODEL_NAME)))
    p_eval.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--by-category", action="store_true", help="Also report MAE per gs class")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = sub.add_parser("infer", help="Run inference on a real beam-camera image")
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--model-path", default=str(config.checkpoint_path(config.MODEL_NAME)))
    p_infer.set_defaults(func=cmd_infer)

    return parser


def main():
    argv = sys.argv[1:] or ["all"]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
