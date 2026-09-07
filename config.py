"""Central configuration for the Gaussian beam profiling pipeline."""
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"


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

# ---- Training ---------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_NAME = "medium"  # "tiny", "medium", or "resnet18" -- see models/cnn.py
MODEL_CHOICES = ("tiny", "medium", "resnet18")
