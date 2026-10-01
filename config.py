"""Central configuration for the Gaussian beam profiling pipeline."""
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Real BPM captures. Images are 640x480 grayscale PNGs laid out as
# <split>/<gs class>/<BPM>_<n>.png, so the dataset root doubles as the image
# root and every label row's `filename` is relative to it.
DATASET_DIR = PROJECT_ROOT / "new_dataset"
IMAGES_DIR = DATASET_DIR
LABELS_CSV = DATASET_DIR / "labels.csv"

# The supplied split. train1 is carved into train/val; test2 is held out
# wholesale because its gs classes do not overlap train1's.
TRAIN_SUBDIR = "train1"
TEST_SUBDIR = "test2"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"

# Legacy synthetic dataset (kept for the `generate` subcommand only). Kept
# separate from DATASET_DIR so synthetic images can never be mixed into the
# real capture set.
SYNTHETIC_DIR = PROJECT_ROOT / "dataset"


def checkpoint_path(model_name):
    """Standard checkpoint location for a given model variant."""
    return CHECKPOINT_DIR / f"{model_name}_best.pt"

# ---- Image geometry --------------------------------------------------------
IMG_WIDTH = 640
IMG_HEIGHT = 480
TRAIN_RESOLUTION = 128  # downsampled square fed to the CNN (keeps laptop load light)

# ---- Synthetic dataset breakdown (2,000 images total) ----------------------
NUM_NORMAL = 1500
NUM_TILTED = 75
NUM_NOISY = 250
NUM_BACKGROUND = 175
TOTAL_IMAGES = NUM_NORMAL + NUM_TILTED + NUM_NOISY + NUM_BACKGROUND

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

RANDOM_SEED = 42

# ---- Gaussian parameter ranges ----------------------------------------------
MIN_SIGMA = 15.0
MAX_SIGMA = 90.0
MAX_SIGMA_NORM = 120.0  # divisor used to keep normalized sigma within ~[0, 1]
CENTER_MARGIN = 40  # keep beam centers this many px away from the frame border

# ---- Label extraction (data/real_dataset.py) --------------------------------
# These frames are real BPM captures, not clean synthetic Gaussians, so labels
# are derived per image: Otsu-threshold, keep the dominant blob, and take
# background-subtracted intensity moments. The frame border is excluded to
# suppress the fixed edge bars present in most captures.
EXTRACT_BLUR_SIGMA = 3.0
EXTRACT_BORDER_X = 40
EXTRACT_BORDER_Y = 30
EXTRACT_MIN_BLOB_FRAC = 0.0004  # ignore specks below this share of the frame
EXTRACT_ELLIPSE_FILL = 0.785    # pi/4: an ideal thresholded Gaussian blob

# ---- Training ---------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_NAME = "medium"  # "tiny", "medium", or "resnet18" -- see models/cnn.py
MODEL_CHOICES = ("tiny", "medium", "resnet18")
