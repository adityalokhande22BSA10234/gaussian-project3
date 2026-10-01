"""Streamlit dashboard for the Gaussian Beam CNN Regression pipeline.

Run from anywhere (paths are resolved from this file):

    streamlit run gaussian-project/app.py

Wraps every CLI stage (labels / validate / preview / train / evaluate /
infer) in a web UI. Label extraction and training are long-running, so they
run on a background thread and stream their printed output into the page
while the UI stays responsive.
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


def _running_under_streamlit():
    """True when Streamlit's runtime is already serving this file.

    Both `streamlit run app.py` and Streamlit's own AppTest harness have a
    live runtime; a bare `python app.py` does not, and there is nothing to
    render into a browser in that case.
    """
    try:
        from streamlit.runtime import exists as runtime_exists
    except Exception:  # noqa: BLE001
        return False
    return bool(runtime_exists())


if __name__ == "__main__" and not _running_under_streamlit():
    # Allow `python app.py` as shorthand for `streamlit run app.py`.
    import subprocess

    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

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
from data.real_dataset import build_label_index, iter_capture_images
from data.validate import validate_dataset
from inference.infer import load_model, predict_frame
from inference.preprocess import preprocess_for_inference
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
    """Runs a long stage (label extraction / training) on a worker thread."""

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


def image_count():
    """Count captures only -- generated artifacts in the dataset root are not data."""
    return sum(1 for _ in iter_capture_images(config.IMAGES_DIR))


def dataset_exists():
    return config.LABELS_CSV.exists() and image_count() > 0


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


def _category_table(labels):
    """Per-gs-class counts, split breakdown and the mean derived parameters.

    The mean sigmas are the quickest sanity check on label extraction: they
    should step up smoothly across the gs classes, and any class pinned at the
    clamp bounds means the extractor is mis-picking a blob.
    """
    rows = []
    for category, group in labels.groupby("category"):
        counts = group["split"].value_counts()
        rows.append({
            "category": category,
            "count": len(group),
            "train": int(counts.get("train", 0)),
            "val": int(counts.get("val", 0)),
            "test": int(counts.get("test", 0)),
            "mean x": round(float(group["x0"].mean()), 1),
            "mean y": round(float(group["y0"].mean()), 1),
            "mean sigma_x": round(float(group["sigma_x"].mean()), 1),
            "mean sigma_y": round(float(group["sigma_y"].mean()), 1),
        })
    return pd.DataFrame(rows)


def _clamp_warnings(labels):
    """Flag labels pinned to the representable bounds, which mean bad extraction."""
    lo, hi = 1e-6, config.MAX_SIGMA_NORM
    pinned = (
        (labels["sigma_x"] <= lo) | (labels["sigma_x"] >= hi) |
        (labels["sigma_y"] <= lo) | (labels["sigma_y"] >= hi)
    )
    return int(pinned.sum())


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------


def home_page():
    st.title("Gaussian Beam CNN Regression")
    st.markdown(
        "Trains a CNN on real BPM camera captures so a single grayscale "
        "frame yields the beam's centre `(x, y)` and its 1-sigma widths "
        "`(sigma_x, sigma_y)`. Labels are derived per frame from the image "
        "itself, so there is no manual annotation step."
    )
    st.caption(
        "CLI equivalent: `python main.py all` | dashboard entry: `streamlit run app.py`"
    )

    col_ds, col_ck = st.columns(2)

    with col_ds:
        st.subheader("Capture set")
        labels = load_labels_df()
        on_disk = image_count()
        if not config.IMAGES_DIR.exists():
            st.error(f"Missing dataset directory: `{config.IMAGES_DIR}`")
        elif labels is None:
            st.warning(f"**{on_disk} PNGs found but `labels.csv` has not been extracted yet.**")
            st.markdown("Go to **Dataset** -> *Extract labels*.")
        else:
            st.dataframe(_category_table(labels), width="stretch")
            st.caption(
                f"Indexed {len(labels)} rows for {on_disk} PNGs on disk | "
                f"splits = {labels['split'].value_counts().to_dict()}"
            )
            if len(labels) != on_disk:
                st.error(
                    f"Label index is stale: {len(labels)} rows for {on_disk} images. "
                    "Re-run **Extract labels**."
                )
            pinned = _clamp_warnings(labels)
            if pinned:
                st.warning(
                    f"{pinned} rows have a sigma pinned at the representable bounds "
                    f"(0 or {config.MAX_SIGMA_NORM}) -- extraction likely mis-picked a blob."
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

    with st.expander("How the pipeline works", expanded=True):
        st.markdown(
            "**1. Label extraction.** Each capture is blurred, its fixed frame "
            "border is masked off, Otsu thresholding splits it into bright blobs, "
            "and the best-scoring blob (roundness x ellipse fill x brightness) is "
            "kept. Intensity-weighted moments of that blob give `x0`, `y0`, the "
            "principal-axis `sigma_x` / `sigma_y`, and `theta`."
        )
        st.markdown(
            "**2. Splits.** `train1/` is carved into train/val per gs class; "
            "`test2/` is held out whole because its classes never appear in "
            "`train1`. Test metrics therefore measure generalisation to unseen "
            "beam sizes, not memorisation -- expect a visible gap between the val "
            "and test numbers."
        )
        st.markdown(
            "**3. Model.** All three architectures (`tiny`, `medium`, `resnet18`) "
            "take a 128x128 grayscale frame and emit a beam-presence logit plus "
            "four sigmoid-bounded normalized values `[x, y, sigma_x, sigma_y]`. "
            "Training uses masked Smooth L1 on the regression head plus BCE on the "
            "presence logit."
        )
        st.markdown(
            "**4. Inference.** The whole frame is resized to 128x128 exactly as in "
            "training -- no cropping -- and the outputs are denormalized straight "
            "back to pixels."
        )


# --------------------------------------------------------------------------


def dataset_page():
    st.title("Dataset")

    labels = load_labels_df()
    on_disk = image_count()
    m1, m2 = st.columns(2)
    m1.metric("PNGs on disk", f"{on_disk}")
    m2.metric("Label rows", f"{0 if labels is None else len(labels)}")

    if labels is not None:
        st.dataframe(_category_table(labels), width="stretch")
        pinned = _clamp_warnings(labels)
        if pinned:
            st.warning(
                f"{pinned} rows have a sigma pinned at a bound "
                f"(0 or {config.MAX_SIGMA_NORM}) -- extraction likely mis-picked a blob."
            )
    else:
        st.info("No `labels.csv` yet -- extract it below.")

    st.divider()

    st.subheader("Extract labels")
    st.caption(
        "Runs `data/real_dataset.py` over every PNG and rewrites "
        f"`{config.LABELS_CSV.name}`. Only the CSV is written; the captures are "
        "never modified."
    )
    active = get_active_job()
    with st.form("labels_form"):
        c1, c2 = st.columns(2)
        val_split = c1.slider(
            "Validation fraction of train1", 0.05, 0.30, config.VAL_SPLIT, 0.01,
            help="Applied per gs class inside train1/. test2/ is always held out whole.",
        )
        seed = c2.number_input("Seed", 0, 1_000_000, config.RANDOM_SEED, step=1)
        submitted = st.form_submit_button(
            "Extract labels",
            disabled=on_disk == 0 or (active is not None and active.running),
        )

    if submitted:
        load_labels_df.clear()
        job = BackgroundJob("Label extraction")
        job.start(
            build_label_index,
            str(config.IMAGES_DIR),
            str(config.LABELS_CSV),
            float(val_split),
            int(seed),
        )
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
            output = save_preview_grid(
                str(config.LABELS_CSV), str(config.IMAGES_DIR), 3
            )
        st.image(str(output), width="stretch")
    elif (config.DATASET_DIR / "preview.png").exists():
        st.image(str(config.DATASET_DIR / "preview.png"), width="stretch")

    st.subheader("Sample gallery")
    _sample_gallery()


def _sample_gallery():
    labels = load_labels_df()
    if labels is None:
        st.caption("No label index to browse.")
        return

    categories = sorted(labels["category"].unique())
    per = st.slider("Samples per category", 1, 6, 3)
    tabs = st.tabs(categories)
    for tab, category in zip(tabs, categories):
        subset = labels[labels["category"] == category]
        size = min(per, len(subset))
        if size == 0:
            tab.caption("empty category")
            continue
        picks = subset.sample(size, random_state=0)
        for col, (_, row) in zip(tab.columns(size), picks.iterrows()):
            caption = (
                f"{row['split']}\n"
                f"x={row['x0']:.0f} y={row['y0']:.0f}\n"
                f"sigma=({row['sigma_x']:.1f}, {row['sigma_y']:.1f})"
            )
            col.image(str(config.IMAGES_DIR / row["filename"]), caption=caption)


# --------------------------------------------------------------------------


def training_page():
    st.title("Training")

    if not dataset_exists():
        st.warning("No label index found. Extract labels on the **Dataset** page first.")
        return

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

    if submitted:
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
        return

    rows = []
    for p in files:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
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
                str(config.LABELS_CSV),
                str(config.IMAGES_DIR),
                str(selected),
                split,
                by_category=True,
            )

        st.success(f"Model architecture: **{metrics['model_name']}**")
        m1, m2 = st.columns(2)
        m1.metric("Beam-presence accuracy", f"{metrics['presence_accuracy']:.3f}")
        if metrics["mae"] is not None:
            x, y, sx, sy = metrics["mae"]
            m2.metric("Overall MAE (px)", f"{np.mean([x, y, sx, sy]):.2f}")
            st.markdown("**Mean absolute error, per parameter (pixels)**")
            for col, (label, value) in zip(
                st.columns(4),
                [("x center", x), ("y center", y), ("sigma_x", sx), ("sigma_y", sy)],
            ):
                col.metric(label, f"{value:.2f}")
        else:
            st.info("No beam-present samples in this split, so MAE is unavailable.")

        rows = metrics.get("mae_by_category", [])
        if rows:
            st.markdown("**Per-gs-class MAE (pixels)**")
            st.dataframe(pd.DataFrame(rows), width="stretch")

        labels = load_labels_df()
        if labels is not None:
            split_labels = labels[labels["split"] == split]
            if len(split_labels) and (split_labels["has_beam"] == 1).all():
                st.warning(
                    "Every frame in this split contains a beam, so "
                    "**beam-presence accuracy is trivially 1.000** and says nothing "
                    "about the presence head. The regression MAE below is the "
                    "metric to watch."
                )
            if split == "test":
                trained_on = set(labels[labels["split"] != "test"]["category"])
                unseen = sorted(set(labels[labels["split"] == "test"]["category"]) - trained_on)
                if unseen:
                    st.caption(
                        "Every test class here "
                        f"({', '.join(unseen)}) is unseen during training, so these "
                        "numbers measure generalisation to new beam sizes rather than "
                        "memorisation. Compare against the **val** split to see the gap."
                    )


# --------------------------------------------------------------------------


def _plot_inference(img, result):
    """Full frame with the prediction drawn, plus a zoom on the beam centre."""
    x, y = result["x"], result["y"]
    sx, sy = result["sigma_x"], result["sigma_y"]
    x0, y0, x1, y1 = result["blob_bbox"]
    has_beam = result["has_beam"]

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))

    ax_full.imshow(img, cmap="gray")
    ax_full.add_patch(
        Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", lw=1.2,
                  label="brightest blob")
    )
    if has_beam:
        ax_full.add_patch(
            Ellipse((x, y), 2 * sx, 2 * sy, fill=False, edgecolor="red", lw=1.6,
                    label="predicted 1-sigma")
        )
        ax_full.plot(x, y, marker="+", color="red", ms=12, mew=1.8)
        ax_full.set_title("Full frame + prediction")
    else:
        ax_full.set_title("Full frame - no beam predicted")
    ax_full.legend(loc="lower right", fontsize=8)
    ax_full.axis("off")

    # Zoom on the predicted centre, clamped to the frame.
    if has_beam:
        half = max(3 * sx, 3 * sy, 60)
        zx0, zx1 = int(max(x - half, 0)), int(min(x + half, img.shape[1]))
        zy0, zy1 = int(max(y - half, 0)), int(min(y + half, img.shape[0]))
    else:
        zx0, zx1, zy0, zy1 = x0, x1, y0, y1
    ax_zoom.imshow(img[zy0:zy1, zx0:zx1], cmap="gray")
    if has_beam:
        ax_zoom.add_patch(
            Ellipse((x - zx0, y - zy0), 2 * sx, 2 * sy, fill=False, edgecolor="red", lw=1.6)
        )
        ax_zoom.plot(x - zx0, y - zy0, marker="+", color="red", ms=12, mew=1.8)
        ax_zoom.set_title(f"Zoom - 1-sigma {sx:.1f} x {sy:.1f} px")
    else:
        ax_zoom.set_title("Zoom - brightest blob")
    ax_zoom.axis("off")

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
        help="Trained via the Training page or `python main.py train`.",
    )

    labels = load_labels_df()
    sample_options = ["(none)"]
    if labels is not None:
        sample_options = ["(none)"] + labels["filename"].tolist()

    source = st.radio(
        "Image source",
        ["Upload a capture", "Pick from the dataset"],
        horizontal=True,
    )
    tmp_path = None

    if source == "Upload a capture":
        uploaded = st.file_uploader(
            "Beam image (grayscale or color; color is converted automatically)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )
        if uploaded is None:
            st.caption("Upload a beam capture to run the model on it.")
            return
        import tempfile

        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp_path = f.name
            f.write(uploaded.getbuffer())
        image_path = tmp_path
        img = np.asarray(Image.open(uploaded).convert("L"))
    else:
        if labels is None:
            st.info("No label index yet -- extract labels on the **Dataset** page.")
            return
        picked = st.selectbox(
            "Dataset frame",
            sample_options,
            format_func=lambda f: "(none)" if f == "(none)" else f,
        )
        if picked == "(none)":
            st.caption("Choose a frame to run the model on it.")
            return
        image_path = str(config.IMAGES_DIR / picked)
        img = np.asarray(Image.open(image_path).convert("L"))
        if labels is not None:
            row = labels[labels["filename"] == picked]
            if len(row):
                r = row.iloc[0]
                st.caption(
                    f"Derived label: x={r['x0']:.1f}  y={r['y0']:.1f}  "
                    f"sigma=({r['sigma_x']:.1f}, {r['sigma_y']:.1f})  "
                    f"[{r['category']} / {r['split']}]"
                )

    if st.button("Run inference", disabled=not model_path.exists()):
        try:
            with st.spinner("Resizing the full frame and running the CNN..."):
                _, _, model_input, bbox = preprocess_for_inference(image_path)
                device = torch.device("cpu")
                model = load_model(str(model_path), device)
                result = predict_frame(model, model_input, device)
                result["blob_bbox"] = tuple(int(v) for v in bbox)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        st.subheader("Result")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Beam present", "yes" if result["has_beam"] else "no")
        m2.metric("Center x", f"{result['x']:.1f} px")
        m3.metric("Center y", f"{result['y']:.1f} px")
        m4.metric("1-sigma size", f"{result['sigma_x']:.1f} x {result['sigma_y']:.1f} px")

        st.pyplot(_plot_inference(img, result))
        st.caption(
            "The cyan box is the brightest blob found by thresholding (used only "
            "for the overlay); the red cross and ellipse are the model's predicted "
            "center and 1-sigma width. The network is fed the whole 640x480 frame "
            "resized to 128x128, exactly as during training."
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