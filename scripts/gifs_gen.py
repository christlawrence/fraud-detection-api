# gifs_gen.py
"""
Generate a portfolio-ready suite of animated explainability GIFs for the fraud model.

What it does:
- Loads and preprocesses the Paysim dataset (filters to TRANSFER/CASH_OUT).
- Trains (or loads) an XGBoost model and saves its feature list for consistency.
- Produces 10+ animated visualizations into reports/gifs/.
- Uses a dedicated temp frames directory reports/gifs/_frames (cleaned after each GIF).

Run:
    python gifs_gen.py
    python gifs_gen.py --force-retrain
    python gifs_gen.py --sample-size 4000 --num-frames 32

Notes for reviewers:
- Outputs are written to reports/gifs/
- Temporary frames: reports/gifs/_frames
- This script is self-contained and safe to re-run.
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
import umap

# ------------------------------
# 1) Config
# ------------------------------
class Config:
    # Paths
    ROOT = Path.cwd()
    DATA_PATH = ROOT / "data" / "PS_20174392719_1491204439457_log.csv"
    MODEL_DIR = ROOT / "models"
    MODEL_NAME = "fraud_detection_model.json" #ubjson error avoidance
    OUTPUT_DIR = ROOT / "reports" / "gifs"
    FRAME_DIR = OUTPUT_DIR / "_frames"

    # Model & Data
    RANDOM_STATE = 42
    TEST_SIZE = 0.30
    SAMPLE_SIZE = 4500

    # GIF Config
    NUM_FRAMES = 24
    NUM_3D_ROTATION_FRAMES = 90
    GIF_DURATION = 3.0
    FIG_SIZE = (16, 10)
    DPI = 120

    # Per-GIF pacing
    DURATION_NETWORK = 1.0
    DURATION_GEO = 0.6
    DURATION_3D = 0.5
    DURATION_RISK = 1.5
    DURATION_DRIFT = GIF_DURATION
    DURATION_WIPEOUT = GIF_DURATION
    DURATION_TEMPORAL = GIF_DURATION
    DURATION_LEARNING = GIF_DURATION

    # Holds
    HOLD_FIRST = 7
    HOLD_LAST = 8


# ------------------------------
# 2) Utilities
# ------------------------------
def setup_directories():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.FRAME_DIR.mkdir(parents=True, exist_ok=True)


def load_and_preprocess_data(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)

    # Focus on types where fraud occurs
    df = df[(df["type"] == "TRANSFER") | (df["type"] == "CASH_OUT")].copy()

    # Engineered features consistent with training scripts
    df["errorBalanceOrig"] = (df["newbalanceOrig"] + df["amount"]) - df["oldbalanceOrg"]
    df["errorBalanceDest"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["isBlackHoleTransaction"] = (
        (df["oldbalanceDest"] == df["newbalanceDest"]) & (df["amount"] > 0)
    ).astype(int)
    df["hourOfDay"] = df["step"] % 24

    # One-hot encode; keep drop_first to align with training
    df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=True)

    y = df["isFraud"]
    X = df.drop(["isFraud", "isFlaggedFraud"], axis=1)
    return X, y


def get_or_train_model(X_train: pd.DataFrame, y_train: pd.Series, force_retrain: bool) -> xgb.XGBClassifier:
    model_path = Config.MODEL_DIR / Config.MODEL_NAME
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    features_path = Path(str(model_path) + ".features")

    # Try to load the existing model
    if model_path.exists() and not force_retrain:
        print(f"Loading existing model from: {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        if features_path.exists():
            features = [ln.strip() for ln in features_path.read_text().splitlines() if ln.strip()]
            model.trained_feature_names = features
        return model

    # Train a small but robust model for viz
    print("Training a new XGBoost model for GIF generation...")
    pos = y_train.value_counts().get(1, 1)
    neg = y_train.value_counts().get(0, 1)
    scale_pos_weight = max(1.0, neg / max(1, pos))

    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "seed": Config.RANDOM_STATE,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )
    booster.save_model(str(model_path))

    # Load into sklearn wrapper for predict_proba API
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    model.trained_feature_names = list(X_train.columns)

    #  Feature list.
    features_path.write_text("\n".join(model.trained_feature_names))
    print(f"Model and feature list saved to: {model_path}")
    return model


def create_gif(image_files: List[Path], gif_path: Path, duration: float, hold_first: int = 0, hold_last: int = 0):
    if not image_files:
        return
    try:
        frames = [imageio.imread(str(file)) for file in image_files]
        if hold_first > 0:
            frames = [frames[0]] * hold_first + frames
        if hold_last > 0:
            frames = frames + [frames[-1]] * hold_last

        # Normalize sizes to the first frame (WxH)
        base = frames[0]
        base_h, base_w = base.shape[:2]

        # import
        from PIL import Image # Import PIL to avoid global dependency
        norm_frames = []
        for arr in frames:
            h, w = arr.shape[:2]
            if (h, w) != (base_h, base_w):
                img = Image.fromarray(arr)
                img = img.resize((base_w, base_h), Image.BILINEAR)
                arr = np.asarray(img)
            norm_frames.append(arr)

        imageio.mimsave(str(gif_path), norm_frames, duration=duration)
        print(f"Successfully created GIF: {gif_path}")
    except Exception as e:
        name = gif_path.name if isinstance(gif_path, Path) else os.path.basename(str(gif_path))
        print(f"Failed to create GIF {name}. Error: {e}")
    finally:
        for file in image_files:
            try:
                os.remove(file)
            except OSError:
                pass



# ------------------------------
# 3) GIFs
# ------------------------------
def generate_fraud_network_gif(sample_data: pd.DataFrame, sample_labels: pd.Series):
    print("\nGenerating 'Fraud Contagion' Network GIF...")
    full_G = nx.Graph()
    # Build graph edges where both endpoints exist
    for _, row in sample_data.iterrows():
        if "nameOrig" in row and "nameDest" in row:
            full_G.add_edge(row["nameOrig"], row["nameDest"])

    pos = nx.spring_layout(full_G, seed=Config.RANDOM_STATE, k=0.5, iterations=150)

    image_files: List[Path] = []
    fraud_nodes = set()

    sorted_sample = sample_data.sort_values("step")
    steps = np.linspace(sorted_sample["step"].min(), sorted_sample["step"].max(), Config.NUM_FRAMES, dtype=int)

    frame_G = nx.Graph()
    for i, current_step in enumerate(steps):
        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        ax = plt.gca()

        # append edges up to the current step
        new_edges = sorted_sample[sorted_sample["step"] <= current_step]
        for idx, row in new_edges.iterrows():
            if "nameOrig" in row and "nameDest" in row:
                frame_G.add_edge(row["nameOrig"], row["nameDest"])
                if sample_labels.loc[idx] == 1:
                    fraud_nodes.add(row["nameOrig"])
                    fraud_nodes.add(row["nameDest"])

        nodes_in_frame = list(frame_G.nodes())
        node_colors = ["crimson" if node in fraud_nodes else "lightgray" for node in nodes_in_frame]
        node_sizes = [220 if node in fraud_nodes else 60 for node in nodes_in_frame]

        nx.draw_networkx_nodes(
            frame_G, pos, nodelist=nodes_in_frame, node_color=node_colors, node_size=node_sizes,
            alpha=0.85, ax=ax, linewidths=0.2, edgecolors="white"
        )
        nx.draw_networkx_edges(frame_G, pos, edgelist=list(frame_G.edges()), width=0.6, alpha=0.5, ax=ax)

        ax.set_title(f"Fraud Network Contagion (Time Step: {current_step})", fontsize=20)
        ax.axis("off")

        frame_path = Config.FRAME_DIR / f"network_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches="tight", facecolor="white")
        plt.close()
        image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "1_fraud_contagion.gif",
        duration=Config.DURATION_NETWORK,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST,
    )


def generate_geospatial_hotspot_gif(sample_labels: pd.Series):
    print("\nGenerating 'Geospatial Hotspots' GIF...")
    np.random.seed(Config.RANDOM_STATE)
    n = len(sample_labels)

    # sim lat/long
    leg = np.random.rand(n, 2) * np.array([360, 180]) - np.array([180, 90])
    fraud_hotspot1 = np.random.randn(n, 2) * 5 + np.array([25, 50])
    fraud_hotspot2 = np.random.randn(n, 2) * 5 + np.array([-50, 20])

    labels = np.asarray(sample_labels, dtype=int)
    is_fraud = labels == 1

    coords = np.where(is_fraud[:, None], fraud_hotspot1, leg)
    second_mask = is_fraud & (np.random.rand(n) > 0.5)
    coords[second_mask] = fraud_hotspot2[second_mask]

    image_files: List[Path] = []
    with plt.style.context("dark_background"):
        frames = Config.NUM_FRAMES + 10
        for i in range(frames):
            plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            ax = plt.gca()
            ax.set_facecolor("#000014")
            ax.set_title("Simulated Geospatial Fraud Hotspots", fontsize=20)

            ax.scatter(leg[:, 0], leg[:, 1], s=12, alpha=0.25, c="#5cd1ff", linewidths=0)
            t = i / max(1, frames - 1)
            reveal_ratio = t * t * t
            reveal_count = max(1, int(reveal_ratio * is_fraud.sum())) if is_fraud.sum() > 0 else 0
            if reveal_count > 0:
                fraud_subset = coords[is_fraud][:reveal_count]
                ax.scatter(
                    fraud_subset[:, 0], fraud_subset[:, 1],
                    s=70, c="#ff5cf7", alpha=0.95, edgecolors="white", linewidths=0.6
                )

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xticks([])
            ax.set_yticks([])

            frame_path = Config.FRAME_DIR / f"geo_{i:02d}.png"
            plt.savefig(frame_path, facecolor="#000014", bbox_inches="tight")
            plt.close()
            image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "2_geospatial_hotspots.gif",
        duration=Config.DURATION_GEO,
        hold_first=Config.HOLD_FIRST,
        hold_last=max(Config.HOLD_LAST, 8),
    )


def _style_3d_axes_dark(ax):
    ax.set_facecolor("black")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if hasattr(axis, "pane"):
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
            axis.pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    ax.grid(True, color=(0.7, 0.7, 0.7, 0.08))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def _ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)


def _estimate_local_density(points: np.ndarray, k: int = 16) -> np.ndarray:
    if len(points) == 0:
        return np.array([])
    from sklearn.neighbors import NearestNeighbors

    k = min(k, max(1, len(points) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(points)
    dists, _ = nn.kneighbors(points, return_distance=True)
    mean_dist = dists[:, 1:].mean(axis=1)
    inv = 1.0 / (mean_dist + 1e-6)

    rng = np.ptp(inv)
    if not np.isfinite(rng) or rng < 1e-12:
        return np.zeros_like(inv)
    inv = (inv - inv.min()) / (rng + 1e-12)
    return inv


def normalize_to_unit_interval(values, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros_like(arr, dtype=float)

    value_range = np.nanstd(arr)  # max - min, NaN-safe
    if not np.isfinite(value_range) or value_range < eps:
        return np.zeros_like(arr, dtype=float)

    min_val = np.nanmin(arr)
    return (arr - min_val) / (value_range + eps)


def generate_3d_cluster_gif(shap_values, sample_labels: pd.Series):
    print("\nGenerating enhanced rotating 3D Fraud Cluster GIF...")

    sv_values = shap_values.values if hasattr(shap_values, "values") else np.asarray(shap_values)
    base_vals = None
    if hasattr(shap_values, "base_values"):
        base_vals = np.array(shap_values.base_values).reshape(-1)

    labels = np.asarray(sample_labels).reshape(-1)
    n_samples = min(len(sv_values), len(labels))
    if len(sv_values) != n_samples:
        sv_values = sv_values[:n_samples]
    if len(labels) != n_samples:
        labels = labels[:n_samples]
    is_fraud = labels == 1

    # 2) Risk from SHAP (sigmoid(base + sum(shap))) with guards
    linear_sum = sv_values.sum(axis=1) if n_samples > 0 else np.array([], dtype=float)
    if base_vals is not None and len(base_vals) >= n_samples:
        linear_sum = base_vals[:n_samples] + linear_sum
    risk = 1.0 / (1.0 + np.exp(-linear_sum)) if linear_sum.size else np.array([], dtype=float)
    risk = np.asarray(risk, dtype=float)

    # 3) UMAP embedding in 3D without guards
    if n_samples == 0:
        print("No samples available for 3D embedding; skipping.")
        return
    reducer = umap.UMAP(n_components=3, random_state=Config.RANDOM_STATE)
    embedding = reducer.fit_transform(sv_values)  # shape (n_samples, 3)

    # Align lengths
    m = min(embedding.shape[0], len(is_fraud), len(risk))
    embedding = embedding[:m]
    is_fraud = is_fraud[:m]
    risk = risk[:m]

    if m == 0:
        print("Empty embedding after alignment; skipping 3D GIF.")
        return

    # 4) Density-based sizing for legitimate transactions (visual depth)
    legit_points = embedding[~is_fraud]
    legit_density = _estimate_local_density(legit_points) if len(legit_points) else np.array([])
    legit_sizes = (10 + 40 * legit_density).astype(float) if len(legit_density) else np.full(len(embedding), 20.0)
    # Fraud points get a constant but larger size
    fraud_sizes_const = 80.0

    # 5) Risk-aware colormap for fraud points (use helper for safe normalization)
    import matplotlib as mpl
    cmap = mpl.colormaps.get_cmap("plasma")
    risk_norm = normalize_to_unit_interval(risk)
    fraud_colors = cmap(risk_norm)  # RGBA for all points; we’ll mask to fraud only

    # 6) Smooth camera path
    angles, elevs = [], []
    for i in range(Config.NUM_3D_ROTATION_FRAMES):
        t = i / max(1, Config.NUM_3D_ROTATION_FRAMES - 1)
        t_eased = t * t * (3 - 2 * t)
        azim = 360.0 * t_eased
        elev = 22.0 + 8.0 * np.sin(2 * np.pi * t)
        angles.append(azim)
        elevs.append(elev)

    # 7) Render frames
    image_files = []
    with plt.style.context("dark_background"):
        for i, (azim, elev) in enumerate(zip(angles, elevs)):
            fig = plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("black")
            fig.patch.set_facecolor("black")

            # Plot legit first
            if len(legit_points):
                ax.scatter(
                    legit_points[:, 0], legit_points[:, 1], legit_points[:, 2],
                    c="#9aa4b2",
                    s=legit_sizes[:len(legit_points)],
                    alpha=0.23, depthshade=True, linewidths=0
                )

            # ...then fraud
            fraud_points = embedding[is_fraud]
            if len(fraud_points):
                fraud_colors_masked = fraud_colors[is_fraud]
                ax.scatter(
                    fraud_points[:, 0], fraud_points[:, 1], fraud_points[:, 2],
                    c=fraud_colors_masked,
                    s=np.full(len(fraud_points), fraud_sizes_const),
                    alpha=0.95, depthshade=True, edgecolors="white", linewidths=0.5, marker="o"
                )

            ax.view_init(elev=elev, azim=azim)
            _style_3d_axes_dark(ax)
            ax.set_title("3D Risk Landscape: SHAP-based Separation", fontsize=20)

            # ff3b5c
            # 9aa4b2
            legit_count = int((~is_fraud).sum())
            fraud_count = int(is_fraud.sum())
            mean_risk_legit = float(risk[~is_fraud].mean()) if legit_count > 0 else 0.0
            mean_risk_fraud = float(risk[is_fraud].mean()) if fraud_count > 0 else 0.0
            ax.text2D(0.02, 0.96, f"Legitimate: {legit_count}", transform=ax.transAxes, color="#9aa4b2", fontsize=12)
            ax.text2D(0.02, 0.92, f"Fraud: {fraud_count}", transform=ax.transAxes, color="#ff3b5c", fontsize=12)
            ax.text2D(0.02, 0.88, f"Mean risk (L): {mean_risk_legit:.2f}", transform=ax.transAxes, color="#9aa4b2", fontsize=10)
            ax.text2D(0.02, 0.84, f"Mean risk (F): {mean_risk_fraud:.2f}", transform=ax.transAxes, color="#ff3b5c", fontsize=10)

            # Static colorbar for risk scale
            cax = fig.add_axes([0.90, 0.20, 0.015, 0.60])
            norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cb.set_label("Fraud risk (SHAP)", color="white")
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

            frame_path = Config.FRAME_DIR / f"3d_cluster_{i:03d}.png"
            plt.savefig(frame_path, facecolor="black")
            plt.close()
            image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "3_3D_fraud_clusters.gif",
        duration=Config.DURATION_3D,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST
    )




def generate_risk_escalation_gif(model, explainer, sample_data, sample_labels):
    print("\nGenerating enhanced 'Risk Escalation' GIF (slower + threshold annotated)...")

    # 1) Numeric data only
    numeric_data = sample_data.select_dtypes(include=np.number)

    # 2) Align to the model's trained feature set (order + missing columns -> 0)
    model_features = getattr(model, "trained_feature_names", list(numeric_data.columns))
    numeric_data = numeric_data.reindex(columns=model_features, fill_value=0.0)

    # 3) Predictions (aligned)
    preds = model.predict_proba(numeric_data)[:, 1]

    # 4) Build a risk ladder:
    #    - fraud indices ordered from low -> high probability (borderline -> certain fraud)
    fraud_idx = sample_labels[sample_labels == 1].index
    if len(fraud_idx) == 0:
        print("No fraud examples in sample; skipping risk escalation.")
        return

    ranked_fraud = pd.Series(preds, index=numeric_data.index).loc[fraud_idx].sort_values()
    K = max(10, min(Config.NUM_FRAMES, len(ranked_fraud))) # even coverage
    plot_order = ranked_fraud.index.to_list()
    if len(plot_order) > K:
        sel = np.linspace(0, len(plot_order) - 1, K, dtype=int)
        plot_order = [plot_order[i] for i in sel]

    # 5) Load decision threshold from final model metadata
    decision_threshold = 0.5
    try:
        meta_path = Config.ROOT / "models" / "fraud_detection_model_final.json.metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            decision_threshold = float(meta.get("decision_threshold", decision_threshold))
    except Exception:
        pass
    # Convert threshold to logit for vertical line on decision plot (which is on logit scale)
    eps = 1e-9
    p = min(max(decision_threshold, eps), 1.0 - eps)
    logit_threshold = np.log(p / (1.0 - p))

    # 6) Compute SHAP values once on the aligned matrix
    shap_values = explainer(numeric_data)

    # 7) Render frames (larger figure, generous margins; slower pacing)
    image_files = []
    local_duration = max(Config.DURATION_RISK, 10.0)   # slower per-frame (>= 2.0s)
    local_hold_last = max(Config.HOLD_LAST, 30)       # longer end hold for readability

    for i, idx in enumerate(plot_order):
        plt.figure(figsize=(max(18, Config.FIG_SIZE[0]), max(12, Config.FIG_SIZE[1])), dpi=Config.DPI)
        iloc = numeric_data.index.get_loc(idx)

        # Decision plot (logit space)
        shap.decision_plot(
            explainer.expected_value,
            shap_values.values[iloc],
            numeric_data.iloc[iloc],
            link='logit',
            show=False
        )

        ax = plt.gca()
        # Vertical line at decision threshold (in logit space)
        ax.axvline(logit_threshold, color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.9)
        ax.text(
            logit_threshold, ax.get_ylim()[1],
            f"  threshold (p={decision_threshold:.2f})",
            color="#ff7f0e", fontsize=8, va="top", ha="left", rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff8ed", edgecolor="#ff7f0e", alpha=0.85)
        )

        # Annotation block: model probability and explanation
        model_prob = float(preds[iloc])
        plt.title(
            f"Risk Escalation ({i + 1}/{len(plot_order)}): Borderline → High-Risk",
            fontsize=18, pad=30
        )
        # Place annotation in a stable location inside axes
        ax.text(
            0.02, 0.98,
            f"Model probability (fraud): {model_prob:.3f}\n"
            f"Interpretation: features push the logit (x-axis) rightward past the threshold line → fraud",
            transform=ax.transAxes,
            fontsize=9,
            va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cfcfcf", alpha=0.95)
        )

        ax.set_xlabel("Cumulative contribution to model score (logit)", fontsize=8, labelpad=10)
        plt.subplots_adjust(left=0.25, right=0.98, top=0.86, bottom=0.12)

        frame_path = Config.FRAME_DIR / f"decision_{i:02d}.png"
        plt.savefig(frame_path, facecolor='white')
        plt.close()
        image_files.append(frame_path)

    # Create GIF with slower pacing, longer end hold
    create_gif(
        image_files,
        Config.OUTPUT_DIR / "4_risk_escalation.gif",
        duration=local_duration,
        hold_first=Config.HOLD_FIRST,
        hold_last=local_hold_last
    )


def generate_feature_drift_gif(sample_data: pd.DataFrame, sample_labels: pd.Series, feature: str, title: str, filename: str):
    print(f"\nGenerating '{title}' GIF...")
    image_files: List[Path] = []
    sorted_sample = sample_data.sort_values("step")
    steps = np.linspace(sorted_sample["step"].min(), sorted_sample["step"].max(), Config.NUM_FRAMES, dtype=int)

    for i, current_step in enumerate(steps):
        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        subset = sorted_sample[sorted_sample["step"] <= current_step]
        subset_labels = sample_labels.loc[subset.index]
        sns.kdeplot(subset.loc[subset_labels == 0, feature], color="cornflowerblue", label="Legitimate", fill=True, bw_adjust=0.6)
        if subset_labels.sum() > 1:
            sns.kdeplot(subset.loc[subset_labels == 1, feature], color="crimson", label="Fraud", fill=True, bw_adjust=0.6)
        plt.title(f"{title} (Up to Step {current_step})", fontsize=20)
        plt.legend()
        plt.xlabel(feature)
        plt.ylabel("Density")
        frame_path = Config.FRAME_DIR / f"drift_{feature}_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches="tight", facecolor="white")
        plt.close()
        image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / filename,
        duration=Config.DURATION_DRIFT,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST,
    )


def generate_balance_wipeout_gif(sample_data: pd.DataFrame, sample_labels: pd.Series):
    print("\nGenerating 'Balance Wipeout' Visualizer GIF...")
    fraud_data = sample_data[sample_labels == 1]
    wipeouts = fraud_data[fraud_data["newbalanceOrig"] == 0].head(Config.NUM_FRAMES)
    if wipeouts.empty:
        return

    image_files: List[Path] = []
    for i, (_, row) in enumerate(wipeouts.iterrows()):
        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        bars = plt.bar(["Old Balance", "New Balance"], [row["oldbalanceOrg"], row["newbalanceOrig"]], color=["cornflowerblue", "crimson"])
        plt.bar_label(bars, fmt="${:,.0f}")
        plt.title(f"Balance Wipeout Visualizer (Transaction {i + 1})", fontsize=20)
        plt.ylabel("Account Balance")
        plt.ylim(top=max(1.0, row["oldbalanceOrg"]) * 1.1)
        frame_path = Config.FRAME_DIR / f"wipeout_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches="tight", facecolor="white")
        plt.close()
        image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "7_balance_wipeout.gif",
        duration=Config.DURATION_WIPEOUT,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST,
    )


def generate_fraud_ring_heatmap(sample_data, sample_labels, explainer):
    print("\nGenerating 'Fraud Ring Fingerprint' Heatmap...")

    fraud_orig_counts = sample_data.loc[sample_labels == 1, 'nameOrig'].value_counts()
    if fraud_orig_counts.empty:
        print("No fraudulent originators in sample; skipping fraud ring heatmap.")
        return
    top_fraudster = fraud_orig_counts.index[0]

    ring_data = sample_data[sample_data['nameOrig'] == top_fraudster]
    ring_numeric = ring_data.select_dtypes(include=np.number)

    #  Alignment to booster features
    booster_feature_names = None
    try:
        booster_feature_names = explainer.model.get_booster().feature_names
    except Exception:
        pass
    if not booster_feature_names:
        booster_feature_names = getattr(explainer.model, "trained_feature_names", list(ring_numeric.columns))

    # Exact columns (order and count)
    ring_aligned = ring_numeric.reindex(columns=booster_feature_names, fill_value=0.0).astype(np.float32)

    if ring_aligned.shape[1] != len(booster_feature_names):
        print(f"Ring alignment mismatch: got {ring_aligned.shape[1]} vs expected {len(booster_feature_names)}. Skipping.")
        return
    if ring_aligned.empty:
        print("Ring data after alignment is empty; skipping.")
        return

    try:
        ring_shap = explainer(ring_aligned)
    except Exception as e:
        print(f"Skipping fraud ring heatmap due to SHAP error: {e}")
        return

    plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
    sns.heatmap(ring_shap.values, xticklabels=ring_aligned.columns, yticklabels=False, cmap='viridis', robust=True)
    plt.title(f"SHAP Fingerprint of a Single Fraud Ring (Originator: {top_fraudster})", fontsize=20, pad=14)
    plt.xlabel("Features")
    plt.ylabel("Transactions in Ring")

    out_path = Config.OUTPUT_DIR / "8_static_fraud_ring_fingerprint.png"
    plt.savefig(out_path, facecolor='white')
    plt.close()
    print(f"Successfully created static image: {out_path}")



def generate_temporal_anomaly_gif(sample_data: pd.DataFrame, sample_labels: pd.Series):
    print("\nGenerating 'Temporal Anomaly' GIF...")
    image_files: List[Path] = []
    sorted_sample = sample_data.sort_values("step")
    steps = np.linspace(sorted_sample["step"].min(), sorted_sample["step"].max(), Config.NUM_FRAMES, dtype=int)

    for i, current_step in enumerate(steps):
        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        subset = sorted_sample[sorted_sample["step"] <= current_step]
        subset_labels = sample_labels.loc[subset.index]
        plt.scatter(
            subset.loc[subset_labels == 0, "hourOfDay"], subset.loc[subset_labels == 0, "amount"],
            c="cornflowerblue", alpha=0.1, label="Legitimate", s=12, linewidths=0
        )
        plt.scatter(
            subset.loc[subset_labels == 1, "hourOfDay"], subset.loc[subset_labels == 1, "amount"],
            c="crimson", alpha=0.85, s=50, label="Fraud", edgecolors="white", linewidths=0.4
        )
        plt.title(f"Temporal Anomaly Detection (Up to Step {current_step})", fontsize=20)
        plt.xlabel("Hour of Day")
        plt.ylabel("Transaction Amount")
        plt.legend()
        plt.yscale("log")
        frame_path = Config.FRAME_DIR / f"temporal_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches="tight", facecolor="white")
        plt.close()
        image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "9_temporal_anomalies.gif",
        duration=Config.DURATION_TEMPORAL,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST,
    )


def generate_learning_curve_gif(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    print("\nGenerating 'Model Learning Curve' GIF...")
    image_files: List[Path] = []
    train_sizes = np.linspace(0.1, 1.0, Config.NUM_FRAMES, endpoint=True)
    auc_scores: List[float] = []

    for i, train_size in enumerate(train_sizes):
        plt.figure(figsize=Config.FIG_SIZE, dpi=Config.DPI)
        subset_idx = int(train_size * len(X_train))
        if subset_idx < 2:
            plt.close()
            continue

        X_sub, y_sub = X_train.iloc[:subset_idx], y_train.iloc[:subset_idx]
        if len(pd.Series(y_sub).unique()) < 2:
            plt.close()
            continue

        model.fit(X_sub, y_sub)
        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        auc_scores.append(auc)

        plt.plot(train_sizes[: len(auc_scores)], auc_scores, marker="o", linestyle="-", color="darkviolet")
        plt.title("Model Learning Curve: Performance vs. Training Data Size", fontsize=20)
        plt.xlabel("Proportion of Training Data Used")
        plt.ylabel("Test Set AUC Score")
        plt.xlim(0, 1.1)
        plt.ylim(min(0.8, min(auc_scores) - 0.05 if auc_scores else 0.8), 1.0)
        frame_path = Config.FRAME_DIR / f"learning_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches="tight", facecolor="white")
        plt.close()
        image_files.append(frame_path)

    create_gif(
        image_files,
        Config.OUTPUT_DIR / "10_model_learning_curve.gif",
        duration=Config.DURATION_LEARNING,
        hold_first=Config.HOLD_FIRST,
        hold_last=Config.HOLD_LAST,
    )


# ------------------------------
# 4) Main
# ------------------------------
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Generate a suite of fraud explainability GIFs.")
    parser.add_argument("--force-retrain", action="store_true", help="Force model retraining.")
    parser.add_argument("--sample-size", type=int, default=Config.SAMPLE_SIZE, help="Sample size for SHAP and visuals.")
    parser.add_argument("--num-frames", type=int, default=Config.NUM_FRAMES, help="Frames per animated sequence.")
    args = parser.parse_args()

    # Apply overrides
    Config.SAMPLE_SIZE = max(100, int(args.sample_size))
    Config.NUM_FRAMES = max(10, int(args.num_frames))

    setup_directories()

    # Data prep
    X, y = load_and_preprocess_data(Config.DATA_PATH)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )

    # Model
    model = get_or_train_model(X_train_full.select_dtypes(include=np.number), y_train, args.force_retrain)
    if not hasattr(model, "trained_feature_names"):
        raise ValueError("Model is missing the saved feature list. Retrain or ensure .features is present.")

    # SHAP sampling
    sample_size = min(Config.SAMPLE_SIZE, len(X_test_full))
    X_test_sample = X_test_full.sample(n=sample_size, random_state=Config.RANDOM_STATE)
    y_test_sample = y_test.loc[X_test_sample.index]

    # Explainer on numeric subset aligned with model features
    model_features = model.trained_feature_names
    X_test_sample_numeric = X_test_sample[model_features]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_sample_numeric)

    # Generate GIFs
    generate_fraud_network_gif(X_test_sample, y_test_sample)
    generate_geospatial_hotspot_gif(y_test_sample)
    generate_3d_cluster_gif(shap_values, y_test_sample)
    generate_risk_escalation_gif(model, explainer, X_test_sample, y_test_sample)

    generate_feature_drift_gif(
        X_test_sample, y_test_sample, "amount", "Adversarial Drift (Amount)", "5_animated_amount_drift.gif"
    )
    generate_feature_drift_gif(
        X_test_sample, y_test_sample, "oldbalanceOrg", "Adversarial Drift (Balance)", "6_animated_balance_drift.gif"
    )

    generate_balance_wipeout_gif(X_test_sample, y_test_sample)
    generate_fraud_ring_heatmap(X_test_sample, y_test_sample, explainer)
    generate_temporal_anomaly_gif(X_test_sample, y_test_sample)
    generate_learning_curve_gif(
        xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=150,
            seed=Config.RANDOM_STATE,
            verbosity=0,
        ),
        X_train_full.select_dtypes(include=np.number),
        y_train,
        X_test_full.select_dtypes(include=np.number),
        y_test,
    )

    # Cleanup empty temp frames dir (if all frames were removed)
    try:
        if Config.FRAME_DIR.exists() and not any(Config.FRAME_DIR.iterdir()):
            Config.FRAME_DIR.rmdir()
    except Exception:
        pass

    print("\n--- GIF Generation Complete ---")
    print(f"Output directory: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
