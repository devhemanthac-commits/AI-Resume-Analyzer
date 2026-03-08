"""
clustering/visualizer.py
------------------------
Generates PCA, t-SNE, word clouds, and cluster bar charts for resume clusters.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Nice color palette
PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())


def _to_2d(X, method: str = "pca", n_components: int = 50) -> np.ndarray:
    """Reduce high-dimensional matrix to 2D for plotting."""
    if hasattr(X, "toarray"):
        n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_reduced = svd.fit_transform(X)
    else:
        X_reduced = np.array(X)

    if method == "tsne":
        perplexity = min(30, X_reduced.shape[0] // 3)
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
        return tsne.fit_transform(X_reduced)
    else:
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X_reduced)


# ---------------------------------------------------------------------------
# Core Plot Functions
# ---------------------------------------------------------------------------

def plot_clusters_2d(X, labels: np.ndarray, method: str = "pca",
                     title: str = "Resume Clusters", cluster_names: dict = None):
    """PCA or t-SNE 2D scatter plot coloured by cluster."""
    coords = _to_2d(X, method=method)
    unique_labels = sorted(set(labels))
    colors = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(14, 9))
    for lbl in unique_labels:
        mask = labels == lbl
        name = cluster_names.get(lbl, f"Cluster {lbl}") if cluster_names else f"Cluster {lbl}"
        if lbl == -1:
            name = "Outliers"
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=colors[lbl],
            label=f"{name} ({mask.sum()})",
            alpha=0.7, s=50,
            edgecolors="white", linewidths=0.4,
        )

    ax.set_title(f"{title} ({method.upper()})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    filename = f"clusters_{method}.png"
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Cluster scatter plot ({method}) saved to {path}")
    return path


def plot_domain_distribution(features_list: list, labels: np.ndarray):
    """Bar chart: count of each domain per cluster."""
    import pandas as pd
    domains = [f.get("domain", "Unknown") for f in features_list]
    df = pd.DataFrame({"cluster": labels, "domain": domains})
    df = df[df["cluster"] != -1]

    pivot = df.groupby(["cluster", "domain"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(max(14, len(pivot) * 1.2), 7))
    pivot.plot(kind="bar", ax=ax, colormap="tab20", width=0.8)
    ax.set_title("Domain Distribution per Cluster", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Resumes")
    ax.legend(title="Domain", bbox_to_anchor=(1.01, 1), fontsize=7)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    path = PLOTS_DIR / "domain_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Domain distribution plot saved to {path}")
    return path


def plot_hackathon_distribution(features_list: list, labels: np.ndarray):
    """Stacked bar: hackathon participants vs non-participants per cluster."""
    import pandas as pd
    participated = [int(f.get("hackathon_participated", False)) for f in features_list]
    df = pd.DataFrame({"cluster": labels, "hackathon": participated})
    df = df[df["cluster"] != -1]

    pivot = df.groupby(["cluster", "hackathon"]).size().unstack(fill_value=0)
    pivot.columns = ["No Hackathon", "Hackathon Participant"]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.9), 6))
    pivot.plot(kind="bar", ax=ax, stacked=True,
               color=["#e74c3c", "#2ecc71"], width=0.7)
    ax.set_title("Hackathon Participation per Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.legend(title="Status")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    path = PLOTS_DIR / "hackathon_per_cluster.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Hackathon distribution plot saved to {path}")
    return path


def plot_wordclouds(features_list: list, labels: np.ndarray, max_clusters: int = 8):
    """Generate a word cloud of top skills per cluster."""
    cluster_skills: dict = {}
    for feat, lbl in zip(features_list, labels):
        if lbl == -1:
            continue
        if lbl not in cluster_skills:
            cluster_skills[lbl] = []
        cluster_skills[lbl].extend(feat.get("skills", []))

    unique_clusters = sorted(cluster_skills.keys())[:max_clusters]
    n = len(unique_clusters)
    if n == 0:
        return

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, lbl in enumerate(unique_clusters):
        words = cluster_skills[lbl]
        if not words:
            axes[i].axis("off")
            continue
        freq = Counter(words)
        wc = WordCloud(
            width=400, height=300,
            background_color="white",
            colormap="viridis",
            max_words=40,
        ).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Cluster {lbl}", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top Skills Word Clouds by Cluster", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = PLOTS_DIR / "wordclouds_per_cluster.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Word cloud plot saved to {path}")
    return path


def plot_score_distribution(scores: list, labels: np.ndarray):
    """Box plot of resume scores per cluster."""
    import pandas as pd
    df = pd.DataFrame({"cluster": labels, "score": scores})
    df = df[df["cluster"] != -1]

    fig, ax = plt.subplots(figsize=(max(10, len(df["cluster"].unique()) * 0.8), 6))
    order = sorted(df["cluster"].unique())
    sns.boxplot(data=df, x="cluster", y="score", order=order,
                palette="coolwarm", ax=ax)
    ax.set_title("Resume Score Distribution per Cluster", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Score (0–100)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = PLOTS_DIR / "score_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Score distribution plot saved to {path}")
    return path


def generate_all_plots(X, labels: np.ndarray, features_list: list, scores: list):
    """Convenience wrapper: generate all standard plots."""
    plot_clusters_2d(X, labels, method="pca", title="Resume Clusters (KMeans/DBSCAN)")
    plot_clusters_2d(X, labels, method="tsne", title="Resume Clusters (t-SNE)")
    plot_domain_distribution(features_list, labels)
    plot_hackathon_distribution(features_list, labels)
    plot_wordclouds(features_list, labels)
    plot_score_distribution(scores, labels)
    logger.info("All plots generated successfully.")
