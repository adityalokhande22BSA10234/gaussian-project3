"""Streamlit dashboard for the Gaussian Beam CNN Regression pipeline.

Run from anywhere (paths are resolved from this file):

    streamlit run gaussian-project/app.py

Wraps every CLI stage (generate / validate / preview / train / evaluate /
infer) in a web UI. Dataset generation and training are long-running, so
they run on a background thread and stream their printed output into the
page while the UI stays responsive.
"""
import io
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    sys.exit("Streamlit is not installed. Run: pip install streamlit")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image

import config
from data.preview import save_preview_grid
from data.validate import validate_dataset
from inference.infer import run_inference
from models.cnn import build_model
from synthetic.dataset_builder import build_dataset
from training.evaluate import evaluate_model
from training.train import train_model

# --------------------------------------------------------------------------
# Session / background jobs
# --------------------------------------------------------------------------

BG_JOB_KEY = "background_job"

_STDOUT_LOCK = threading.Lock()
_BASE_STDOUT = sys.stdout


class _LogTee(io.TextIOBase):
    """Mirrors stdout while appending each written chunk to a deque."""

    def __init__(self, buffer):
        self._buffer = buffer

    def write(self, text):
        self._buffer.append(text)
        try:
            _BASE_STDOUT.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            _BASE_STDOUT.flush()
        except Exception:
            pass


class BackgroundJob:
    """Runs a long stage (dataset generation / training) on a worker thread."""

    def __init__(self, label):
        self.label = label
        self.status = "running"  # running | done | error
        self.started = time.time()
        self.lines = deque(maxlen=6000)
        self.result = None
        self.error = None

    @property
    def running(self):
        return self.status == "running"

    def start(self, fn, *args, **kwargs):
        def worker():
            try:
                with _STDOUT_LOCK:
                    previous = sys.stdout
                    sys.stdout = _LogTee(self.lines)
                    try:
                        self.result = fn(*args, **kwargs)
                        self.status = "done"
                    finally:
                        sys.stdout = previous
            except Exception as exc:  # noqa: BLE001
                self.status = "error"
                self.error = exc
                self.lines.append(f"\n!!! {type(exc).__name__}: {exc}\n")
                self.lines.append(traceback.format_exc())

        threading.Thread(target=worker, daemon=True).start()
        st.session_state[BG_JOB_KEY] = self

    def text(self):
        return "".join(self.lines)


def get_active_job():
    return st.session_state.get(BG_JOB_KEY)


def _render_active_job():
    """Poll a running background job, or show its finished/error state."""
    job = get_active_job()
    if job is None:
        return

    if job.status == "running":
        st.info(f"**{job.label}** is running...")
        st.markdown(f"Elapsed: {time.time() - job.started:.0f}s")
        with st.expander("Console output", expanded=True):
            st.code(job.text(), language="text")
        time.sleep(0.8)
        st.rerun()
        return

    if job.status == "done":
        st.success(f"**{job.label}** finished in {time.time() - job.started:.0f}s.")
    else:
        st.error(f"**{job.label}** failed: {job.error}")

    with st.expander("Console output", expanded=True):
        st.code(job.text(), language="text")

    if st.button("Dismiss", key="dismiss_job"):
        del st.session_state[BG_JOB_KEY]
        st.rerun()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def dataset_exists(labels_csv=config.LABELS_CSV, images_dir=config.IMAGES_DIR):
    if not labels_csv.exists() or not images_dir.exists():
        return False
    return next(images_dir.rglob("*.png"), None) is not None


@st.cache_data(show_spinner=False)
def load_labels_df():
    if not config.LABELS_CSV.exists():
        return None
    return pd.read_csv(config.LABELS_CSV)


def checkpoint_files():
    if not config.CHECKPOINT_DIR.exists():
        return []
    files = sorted(config.CHECKPOINT_DIR.glob("*.pt"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _mb(size):
    return size / (1024 * 1024)


def _dataset_table(labels):
    rows = []
    for cat, group in labels.groupby("category"):
        counts = group["split"].value_counts()
        rows.append({
            "category": cat,
            "count": len(group),
            "train": int(counts.get("train", 0)),
            "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)),
            "beam present": bool(group["has_beam"].iloc[0]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def home_page():
    st.title("Gaussian Beam CNN Regression")
    st.markdown(
        "Synthetic-data pipeline that trains a CNN to regress laser beam "
        "parameters -- center `(x, y)` and width `(sigma_x, sigma_y)` -- from "
        "grayscale beam-profile images, then runs the model on real camera "
        "captures. Generate the dataset, train a model, and run inference, all "
        "from the sidebar."
    )
    st.caption(
        "CLI equivalent: `python main.py all` | dashboard entry: `streamlit run app.py`"
    )

    col_ds, col_ck = st.columns(2)

    with col_ds:
        st.subheader("Dataset")
        labels = load_labels_df()
        if labels is None:
            st.warning("No dataset found on disk yet.")
            st.markdown("Go to **Dataset** to generate the synthetic data.")
        else:
            st.dataframe(_dataset_table(labels), width="stretch")
            st.caption(
                f"Total: {len(labels)} images | "
                f"split = {labels['split'].value_counts().to_dict()}"
            )

    with col_ck:
        st.subheader("Checkpoints")
        files = checkpoint_files()
        if not files:
            st.info("No trained checkpoints found yet.")
            st.markdown("Go to **Training** to train a model.")
        else:
            info = pd.DataFrame(
                [
                    {
                        "model": p.stem.replace("_best", ""),
                        "file": p.name,
                        "size (MB)": round(_mb(p.stat().st_size), 2),
                        "modified": time.strftime(
                            "%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)
                        ),
                    }
                    for p in files
                ]
            )
            st.dataframe(info, width="stretch")

    st.divider()
    with st.expander("How the models work"):
        st.markdown(
            "All three architectures (`tiny`, `medium`, `resnet18`) share one "
            "interface: they take a 128x128 grayscale image and predict a "
            "beam-presence logit plus 4 normalized sigmoid-bounded regression "
            "values `[x, y, sigma_x, sigma_y]`. The presence head matters "
            "because the dataset includes beam-free background frames that have "
            "no meaningful coordinates. Training uses masked Smooth L1 on "
            "beam-present samples plus BCE on the presence logit."
        )


# --------------------------------------------------------------------------


def dataset_page():
    st.title("Dataset")

    labels = load_labels_df()
    if labels is not None:
        st.dataframe(_dataset_table(labels), width="stretch")
    else:
        st.info("No dataset generated yet.")

    st.divider()

    st.subheader("Generate")
    active = get_active_job()
    with st.form("generate_form"):
        c1, c2, c3, c4, c5 = st.columns(5)
        normal = c1.number_input("Normal", 0, 20000, config.NUM_NORMAL, step=100)
        tilted = c2.number_input("Tilted", 0, 10000, config.NUM_TILTED, step=25)
        noisy = c3.number_input("Noisy", 0, 10000, config.NUM_NOISY, step=25)
        background = c4.number_input("Background", 0, 10000, config.NUM_BACKGROUND, step=25)
        seed = c5.number_input("Seed", 0, 1_000_000, config.RANDOM_SEED, step=1)
        replace = st.checkbox(
            "Replace existing dataset (delete current images + labels first)",
            value=False,
        )
        submitted = st.form_submit_button(
            "Generate dataset", disabled=active is not None and active.running
        )

    if submitted:
        counts = {
            "normal": int(normal),
            "tilted": int(tilted),
            "noisy": int(noisy),
            "background": int(background),
        }
        if sum(counts.values()) == 0:
            st.error("Generate at least one image.")
        else:
            if replace and dataset_exists():
                import shutil

                for target in (config.IMAGES_DIR, config.LABELS_CSV):
                    if target.exists():
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                load_labels_df.clear()
            st.session_state.pop("bg_job_handled", None)
            job = BackgroundJob("Dataset generation")
            job.start(build_dataset, counts=counts, seed=int(seed))
            st.rerun()

    _render_active_job()

    st.divider()

    st.subheader("Validate")
    if st.button("Run validation", disabled=not dataset_exists()):
        with st.spinner("Checking every image and label..."):
            summary, issues = validate_dataset(config.LABELS_CSV, config.IMAGES_DIR)
        st.dataframe(summary, width="stretch")
        if issues:
            st.error(f"{len(issues)} issue(s) found")
            with st.expander("Show issues"):
                for issue in issues[:100]:
                    st.write(f"- {issue}")
        else:
            st.success("All images present, correctly sized/grayscale with sane labels. PASS.")

    st.subheader("Preview grid")
    if st.button("Build preview grid", disabled=not dataset_exists()):
        with st.spinner("Rendering preview grid..."):
            output = save_preview_grid(per_category=3)
        st.image(output, width="stretch")
    elif (config.DATASET_DIR / "preview.png").exists():
        st.image(str(config.DATASET_DIR / "preview.png"), width="stretch")

    st.subheader("Sample gallery")
    _sample_gallery()


def _sample_gallery():
    labels = load_labels_df()
    if labels is None:
        st.caption("No dataset to browse.")
        return
    categories = sorted(labels["category"].unique())
    per = st.slider("Samples per category", 1, 6, 3)
    tabs = st.tabs(categories)
    for tab, cat in zip(tabs, categories):
        subset = labels[labels["category"] == cat]
        size = min(per, len(subset))
        if size == 0:
            tab.caption("empty category")
            continue
        picks = subset.sample(size, random_state=0)
        cols = tab.columns(size)
        for col, (_, row) in zip(cols, picks.iterrows()):
            caption = cat
            if row["has_beam"]:
                caption = (
                    f"x={row['x0']:.0f} y={row['y0']:.0f}\n"
                    f"sigma=({row['sigma_x']:.1f}, {row['sigma_y']:.1f})"
                )
            col.image(str(config.IMAGES_DIR / row["filename"]), caption=caption)


# --------------------------------------------------------------------------


def training_page():
    st.title("Training")

    if not dataset_exists():
        st.warning("No dataset found. Generate one on the **Dataset** page first.")

    active = get_active_job()
    with st.form("train_form"):
        c1, c2, c3, c4 = st.columns(4)
        model = c1.selectbox("Model", list(config.MODEL_CHOICES))
        epochs = c2.number_input("Epochs", 1, 500, config.EPOCHS, step=5)
        batch = c3.number_input("Batch size", 1, 256, config.BATCH_SIZE, step=1)
        lr = c4.number_input("Learning rate", 1e-7, 1.0, config.LEARNING_RATE, format="%.5f")
        submitted = st.form_submit_button(
            "Start training", disabled=active is not None and active.running
        )

    if submitted and dataset_exists():
        job = BackgroundJob(f"Training {model}")
        job.start(
            train_model,
            str(config.LABELS_CSV),
            str(config.IMAGES_DIR),
            int(epochs),
            int(batch),
            float(lr),
            model,
            str(config.checkpoint_path(model)),
        )
        st.rerun()

    _render_active_job()

    st.divider()

    st.subheader("Saved checkpoints")
    files = checkpoint_files()
    if not files:
        st.info("Nothing trained yet.")
    else:
        rows = []
        for p in files:
            try:
                ckpt = torch.load(p, map_location="cpu")
                name = ckpt.get("model_name", p.stem)
            except Exception:
                name = p.stem
            rows.append({
                "file": p.name,
                "model": name,
                "size (MB)": round(_mb(p.stat().st_size), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")


# --------------------------------------------------------------------------


def evaluation_page():
    st.title("Evaluation")

    files = checkpoint_files()
    if not files:
        st.info("No checkpoints found. Train a model first, or enter a path below.")

    c1, c2 = st.columns([2, 1])
    with c1:
        options = files if files else [config.checkpoint_path(config.MODEL_NAME)]
        selected = st.selectbox(
            "Checkpoint",
            options,
            format_func=lambda p: f"{p.name}  ({_mb(p.stat().st_size):.1f} MB)" if p.exists() else str(p),
        )
    with c2:
        split = st.selectbox("Split", ["test", "val", "train"])

    if st.button("Run evaluation", disabled=not selected.exists()):
        with st.spinner("Scoring the checkpoint over the split..."):
            metrics = evaluate_model(
                str(config.LABELS_CSV), str(config.IMAGES_DIR), str(selected), split
            )
        st.success(f"Model architecture: **{metrics['model_name']}**")
        m1, m2 = st.columns(2)
        m1.metric("Beam-presence accuracy", f"{metrics['presence_accuracy']:.3f}")
        if metrics["mae"] is not None:
            x, y, sx, sy = metrics["mae"]
            m2.metric("Overall MAE (px)", f"{np.mean([x, y, sx, sy]):.2f}")
            st.markdown("**Mean absolute error, per parameter (pixels)**")
            cols = st.columns(4)
            for col, (label, value) in zip(
                cols,
                [("x center", x), ("y center", y), ("sigma_x", sx), ("sigma_y", sy)],
            ):
                col.metric(label, f"{value:.2f}")
        else:
            st.info("No beam-present samples in this split, so MAE is unavailable.")
        st.caption(
            "MAE is reported on the original 640x480 pixel scale. Presence "
            "accuracy = fraction of frames where predicted beam presence matches "
            "the label (background frames included)."
        )


# --------------------------------------------------------------------------


def _plot_inference(img, result):
    """Two-panel figure: original frame + ROI crop, both annotated."""
    x, y = result["x"], result["y"]
    sx, sy = result["sigma_x"], result["sigma_y"]
    x0, y0, x1, y1 = result["roi_bbox"]
    has_beam = result["has_beam"]

    fig, (ax_orig, ax_roi) = plt.subplots(1, 2, figsize=(12, 5))
    ax_orig.imshow(img, cmap="gray")
    ax_orig.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=1.2, label="ROI")
    )
    if has_beam:
        ax_orig.add_patch(
            Ellipse((x, y), 2 * sx, 2 * sy, fill=False, edgecolor="red", lw=1.6, label="predicted beam")
        )
        ax_orig.plot(x, y, marker="+", color="red", ms=12, mew=1.8)
    ax_orig.set_title("Original frame + prediction")
    ax_orig.legend(loc="lower right", fontsize=8)
    ax_orig.axis("off")

    ax_roi.imshow(img[y0:y1, x0:x1], cmap="gray")
    if has_beam:
        ax_roi.add_patch(
            Ellipse((x - x0, y - y0), 2 * sx, 2 * sy, fill=False, edgecolor="red", lw=1.6)
        )
        ax_roi.plot(x - x0, y - y0, marker="+", color="red", ms=12, mew=1.8)
        ax_roi.set_title(f"ROI crop - 1sigma ellipse ({sx:.1f} x {sy:.1f} px)")
    else:
        ax_roi.set_title("ROI crop - no beam detected")
    ax_roi.axis("off")

    plt.tight_layout()
    return fig


def inference_page():
    st.title("Inference")

    files = checkpoint_files()
    default_path = config.checkpoint_path(config.MODEL_NAME)
    options = files if files else [default_path]
    model_path = st.selectbox(
        "Model checkpoint",
        options,
        format_func=lambda p: p.name,
        help="Checkpoints are trained via the Training page or the CLI `python main.py train`.",
    )

    uploaded = st.file_uploader(
        "Beam image (grayscale or color; color is converted automatically)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )
    if uploaded is None:
        st.caption(
            "If you have no camera capture handy, generate a synthetic image with "
            "`python main.py infer` or any file under `dataset/images/normal/`."
        )
        return

    if st.button("Run inference"):
        import tempfile

        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp_path = f.name
            f.write(uploaded.getbuffer())
        try:
            with st.spinner("Preprocessing (background subtract + ROI crop) and running the CNN..."):
                result = run_inference(tmp_path, str(model_path))
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

        img = np.asarray(Image.open(uploaded).convert("L"))

        st.subheader("Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Beam present", "yes" if result["has_beam"] else "no")
        m2.metric("Center x", f"{result['x']:.1f} px")
        m3.metric("Center y", f"{result['y']:.1f} px")
        m4.metric("1sigma size", f"{result['sigma_x']:.1f} x {result['sigma_y']:.1f} px")

        st.pyplot(_plot_inference(img, result))
        st.caption(
            "The cyan box is the ROI found by thresholding (background subtracted, 30px "
            "padding); the red cross/ellipse is the model's predicted center and "
            f"1sigma width. Dust/ambient noise is handled by a median filter when no "
            "background frame is supplied."
        )


# --------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Gaussian Beam Pipeline", layout="wide")

    with st.sidebar:
        st.title("Gaussian Beam")
        page = st.radio(
            "Navigation",
            ["Home", "Dataset", "Training", "Evaluation", "Inference"],
        )

    if page == "Home":
        home_page()
    elif page == "Dataset":
        dataset_page()
    elif page == "Training":
        training_page()
    elif page == "Evaluation":
        evaluation_page()
    else:
        inference_page()


if __name__ == "__main__":
    main()