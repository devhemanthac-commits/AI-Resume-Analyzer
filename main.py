"""
main.py
-------
CLI entry point for the AI Resume Analyzer.

Usage:
    python main.py --input data/raw/UpdatedResumeDataSet.csv --clusters 10 --algorithm kmeans --visualize --report
    python main.py --algorithm both --visualize --report
"""

import argparse
import logging
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Project modules
from data.download_dataset import main as download_data
from data.preprocess import ResumePreprocessor
from features.extractor import ResumeFeatureExtractor
from features.vectorizer import ResumeFeatureMatrix
from clustering.kmeans_cluster import KMeansClusterer
from clustering.dbscan_cluster import DBSCANClusterer
from clustering.visualizer import generate_all_plots, plot_clusters_2d
from analyzer.scorer import score_all
from analyzer.reporter import ClusterReporter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/analyzer.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

Path("outputs").mkdir(exist_ok=True)
Path("outputs/plots").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------

def load_data(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        logger.warning(f"Input file not found at {path}. Running dataset download …")
        download_data()
        path = Path("data/raw/UpdatedResumeDataSet.csv")

    logger.info(f"Loading dataset from {path} …")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    logger.info(f"Loaded {len(df)} records. Columns: {df.columns.tolist()}")

    # Normalise column name
    text_col = None
    for col in ["Resume_str", "resume", "Resume", "text", "Text", "content"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No recognized text column found in {df.columns.tolist()}")
    if text_col != "Resume_str":
        df = df.rename(columns={text_col: "Resume_str"})

    df = df.dropna(subset=["Resume_str"])
    df["Resume_str"] = df["Resume_str"].astype(str)
    logger.info(f"After cleaning: {len(df)} valid records.")
    return df


def preprocess_resumes(df: pd.DataFrame) -> tuple:
    preprocessor = ResumePreprocessor()
    extractor = ResumeFeatureExtractor()

    logger.info("Preprocessing resumes with NLTK …")
    preprocessed_list = []
    features_list = []
    clean_texts = []

    for i, row in df.iterrows():
        text = str(row["Resume_str"])
        try:
            preprocessed = preprocessor.preprocess(text)
            features = extractor.extract(text, preprocessed)
        except Exception as e:
            logger.warning(f"  Row {i} failed preprocessing: {e}")
            preprocessed = {"clean_text": text, "tokens": []}
            features = extractor.extract(text)

        preprocessed_list.append(preprocessed)
        features_list.append(features)
        clean_texts.append(preprocessed.get("clean_text", text))

        if (i + 1) % 100 == 0:
            logger.info(f"  Preprocessed {i+1}/{len(df)} …")

    logger.info("Preprocessing complete.")
    return clean_texts, features_list


def build_feature_matrix(clean_texts: list, features_list: list):
    logger.info("Building feature matrix (TF-IDF + structured) …")
    builder = ResumeFeatureMatrix(tfidf_weight=0.7, structured_weight=0.3)
    X = builder.fit_transform(clean_texts, features_list)
    return X, builder


def run_kmeans(X, n_clusters: int = None) -> np.ndarray:
    k_range = range(2, min(21, X.shape[0] // 5 + 1))
    clusterer = KMeansClusterer(k_range=k_range)
    if n_clusters:
        clusterer.fit(X, n_clusters=n_clusters)
    else:
        clusterer.find_optimal_k(X)
        clusterer.fit(X)
    clusterer.plot_elbow()
    return clusterer.labels_


def run_dbscan(X) -> np.ndarray:
    clusterer = DBSCANClusterer()
    clusterer.fit(X)
    clusterer.plot_k_distance()
    summary = clusterer.get_cluster_summary()
    logger.info(f"DBSCAN summary: {summary}")
    return clusterer.labels_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Resume Analyzer – Cluster resumes by domain, projects, hackathons, and more.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="data/raw/UpdatedResumeDataSet.csv",
                        help="Path to resume CSV file (default: data/raw/UpdatedResumeDataSet.csv)")
    parser.add_argument("--clusters", type=int, default=None,
                        help="Number of KMeans clusters (auto-detected if not specified)")
    parser.add_argument("--algorithm", choices=["kmeans", "dbscan", "both"], default="kmeans",
                        help="Clustering algorithm to use (default: kmeans)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate all visualisation plots")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML + CSV cluster report")
    parser.add_argument("--download", action="store_true",
                        help="Force re-download the Kaggle dataset")

    args = parser.parse_args()

    start = time.time()
    logger.info("=" * 60)
    logger.info("  AI RESUME ANALYZER – Starting Pipeline")
    logger.info("=" * 60)

    # 1. Download data if needed
    if args.download:
        from data.download_dataset import main as dl
        dl()

    # 2. Load
    df = load_data(args.input)

    # 3. Preprocess + Feature Extract
    clean_texts, features_list = preprocess_resumes(df)

    # 4. Score
    scores_list = score_all(features_list)
    composite_scores = [s["composite_score"] for s in scores_list]

    # 5. Build Feature Matrix
    X, _ = build_feature_matrix(clean_texts, features_list)

    # 6. Cluster
    labels = None
    algorithm_used = args.algorithm

    if args.algorithm in ("kmeans", "both"):
        logger.info("Running KMeans clustering …")
        km_labels = run_kmeans(X, args.clusters)
        labels = km_labels

    if args.algorithm in ("dbscan", "both"):
        logger.info("Running DBSCAN clustering …")
        db_labels = run_dbscan(X)
        if labels is None:
            labels = db_labels
        else:
            # Save DBSCAN results too
            df_dbscan = df.copy()
            df_dbscan["dbscan_cluster"] = db_labels
            df_dbscan.to_csv("outputs/dbscan_clusters.csv", index=False)
            logger.info("DBSCAN clusters saved to outputs/dbscan_clusters.csv")
            if args.visualize:
                plot_clusters_2d(X, db_labels, method="pca",
                                 title="DBSCAN Resume Clusters")

    # 7. Visualise
    if args.visualize:
        logger.info("Generating visualisations …")
        generate_all_plots(X, labels, features_list, composite_scores)

    # 8. Report
    if args.report:
        logger.info("Generating cluster report …")
        reporter = ClusterReporter(algorithm=algorithm_used)
        reporter.build_report(df, labels, features_list, scores_list)

    elapsed = round(time.time() - start, 2)
    logger.info("=" * 60)
    logger.info(f"  Pipeline complete in {elapsed}s")
    logger.info(f"  Outputs saved to: outputs/")
    logger.info("=" * 60)

    print(f"\n[DONE] Pipeline complete! Check the 'outputs/' folder for results.")
    print(f"   [PLOTS]  : outputs/plots/")
    print(f"   [CSV]    : outputs/cluster_report.csv")
    print(f"   [HTML]   : outputs/summary.html")


if __name__ == "__main__":
    main()
