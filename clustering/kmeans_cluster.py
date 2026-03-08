"""
clustering/kmeans_cluster.py
-----------------------------
KMeans clustering with Elbow Method + Silhouette Score for optimal K selection.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class KMeansClusterer:
    """
    Fits KMeans after determining the optimal number of clusters via
    Elbow Method and Silhouette Score analysis.
    """

    def __init__(self, k_range: range = range(2, 21), random_state: int = 42):
        self.k_range = k_range
        self.random_state = random_state
        self.model: KMeans = None
        self.optimal_k: int = None
        self.labels_: np.ndarray = None
        self.inertias_: list = []
        self.silhouette_scores_: list = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def find_optimal_k(self, X) -> int:
        """Run elbow + silhouette analysis and return optimal K."""
        logger.info(f"Searching for optimal K in range {list(self.k_range)} …")

        # Reduce dimensions for speed if very sparse
        X_dense = self._to_dense(X, n_components=50)

        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state,
                        n_init=10, max_iter=300)
            km.fit(X_dense)
            self.inertias_.append(km.inertia_)
            if k > 1:
                sil = silhouette_score(X_dense, km.labels_, sample_size=min(500, X_dense.shape[0]))
                self.silhouette_scores_.append(sil)
                logger.info(f"  K={k:2d} | Inertia={km.inertia_:.0f} | Silhouette={sil:.4f}")
            else:
                self.silhouette_scores_.append(0)

        # Pick K maximising silhouette score
        best_idx = int(np.argmax(self.silhouette_scores_))
        self.optimal_k = list(self.k_range)[best_idx]
        logger.info(f"Optimal K selected: {self.optimal_k} (silhouette={self.silhouette_scores_[best_idx]:.4f})")
        return self.optimal_k

    def fit(self, X, n_clusters: int = None):
        """Fit KMeans. Uses optimal_k if n_clusters not specified."""
        if n_clusters is None:
            if self.optimal_k is None:
                self.find_optimal_k(X)
            n_clusters = self.optimal_k

        X_dense = self._to_dense(X, n_components=50)
        logger.info(f"Fitting KMeans with K={n_clusters} …")
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=500,
        )
        self.model.fit(X_dense)
        self.labels_ = self.model.labels_
        logger.info(f"KMeans fitted. Cluster distribution: {np.bincount(self.labels_).tolist()}")
        return self

    def predict(self, X) -> np.ndarray:
        X_dense = self._to_dense(X, n_components=50)
        return self.model.predict(X_dense)

    def plot_elbow(self):
        """Save elbow + silhouette plots. Skipped if no search data was collected."""
        if not self.inertias_:
            logger.info("Skipping elbow plot - find_optimal_k was not run.")
            return

        ks = list(self.k_range)[:len(self.inertias_)]
        sil_ks = list(self.k_range)[:len(self.silhouette_scores_)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow
        axes[0].plot(ks, self.inertias_, "bo-", linewidth=2, markersize=6)
        if self.optimal_k:
            axes[0].axvline(x=self.optimal_k, color="red", linestyle="--",
                            label=f"Optimal K={self.optimal_k}")
        axes[0].set_title("Elbow Method - Inertia vs K", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Number of Clusters (K)")
        axes[0].set_ylabel("Inertia (WCSS)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Silhouette
        axes[1].plot(sil_ks, self.silhouette_scores_, "gs-", linewidth=2, markersize=6)
        if self.optimal_k:
            axes[1].axvline(x=self.optimal_k, color="red", linestyle="--",
                            label=f"Best K={self.optimal_k}")
        axes[1].set_title("Silhouette Score vs K", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Number of Clusters (K)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = PLOTS_DIR / "kmeans_elbow_silhouette.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Elbow/Silhouette plot saved to {path}")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_dense(X, n_components: int = 50) -> np.ndarray:
        """Reduce sparse matrix to dense via TruncatedSVD (LSA)."""
        try:
            if hasattr(X, "toarray"):
                n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
                if n_components < 2:
                    return X.toarray()
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                return svd.fit_transform(X)
            return np.array(X)
        except Exception:
            return X.toarray() if hasattr(X, "toarray") else np.array(X)
