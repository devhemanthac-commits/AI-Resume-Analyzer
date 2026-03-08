"""
clustering/dbscan_cluster.py
-----------------------------
DBSCAN clustering for organic, density-based grouping of resumes.
Also identifies outlier/noise resumes (label == -1).
"""

import logging
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class DBSCANClusterer:
    """
    DBSCAN clusterer with automatic eps estimation using k-distance graph.
    Identifies core clusters and outlier resumes.
    """

    def __init__(self, eps: float = None, min_samples: int = 3,
                 n_components: int = 30, random_state: int = 42):
        self.eps = eps
        self.min_samples = min_samples
        self.n_components = n_components
        self.random_state = random_state
        self.model: DBSCAN = None
        self.labels_: np.ndarray = None
        self.X_reduced_: np.ndarray = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X):
        """Reduce dimensions then fit DBSCAN."""
        self.X_reduced_ = self._reduce(X)
        eps = self.eps if self.eps else self._estimate_eps(self.X_reduced_)
        logger.info(f"Fitting DBSCAN (eps={eps:.4f}, min_samples={self.min_samples}) …")
        self.model = DBSCAN(eps=eps, min_samples=self.min_samples, metric="euclidean")
        self.model.fit(self.X_reduced_)
        self.labels_ = self.model.labels_

        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = np.sum(self.labels_ == -1)
        logger.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise/outlier points")
        return self

    def get_cluster_summary(self) -> dict:
        """Return cluster sizes including outlier count."""
        from collections import Counter
        counts = Counter(self.labels_.tolist())
        return {
            "n_clusters": len([k for k in counts if k != -1]),
            "outliers": counts.get(-1, 0),
            "cluster_sizes": {k: v for k, v in sorted(counts.items()) if k != -1},
        }

    def get_outlier_indices(self) -> list:
        """Return indices of resumes classified as noise."""
        return np.where(self.labels_ == -1)[0].tolist()

    def plot_k_distance(self, X=None):
        """Save the k-distance graph used for eps estimation."""
        data = X if X is not None else self.X_reduced_
        nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(data)
        distances, _ = nbrs.kneighbors(data)
        distances = np.sort(distances[:, -1])[::-1]

        plt.figure(figsize=(10, 5))
        plt.plot(distances, color="steelblue", linewidth=2)
        plt.title("K-Distance Graph (for DBSCAN eps estimation)",
                  fontsize=13, fontweight="bold")
        plt.xlabel("Points (sorted by distance)")
        plt.ylabel(f"{self.min_samples}-NN Distance")
        plt.grid(True, alpha=0.3)
        path = PLOTS_DIR / "dbscan_k_distance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"K-distance plot saved to {path}")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _reduce(self, X) -> np.ndarray:
        if hasattr(X, "toarray"):
            n_comp = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
            if n_comp < 2:
                return X.toarray()
            svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            return svd.fit_transform(X)
        return np.array(X)

    def _estimate_eps(self, X_reduced: np.ndarray) -> float:
        """Estimate eps using the elbow of the k-distance graph."""
        try:
            nbrs = NearestNeighbors(n_neighbors=self.min_samples).fit(X_reduced)
            distances, _ = nbrs.kneighbors(X_reduced)
            kth_distances = np.sort(distances[:, -1])[::-1]
            # Elbow ≈ max second derivative
            diff2 = np.diff(kth_distances, n=2)
            elbow_idx = np.argmin(diff2) + 2
            eps = float(kth_distances[elbow_idx])
            logger.info(f"Auto-estimated eps={eps:.4f}")
            return max(eps, 0.1)  # floor
        except Exception:
            return 0.5
