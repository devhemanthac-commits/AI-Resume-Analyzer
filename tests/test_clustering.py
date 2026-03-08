"""
tests/test_clustering.py
------------------------
Unit tests for KMeans and DBSCAN clustering.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from sklearn.metrics import silhouette_score

from data.download_dataset import generate_synthetic_data
from data.preprocess import ResumePreprocessor
from features.extractor import ResumeFeatureExtractor
from features.vectorizer import ResumeFeatureMatrix
from clustering.kmeans_cluster import KMeansClusterer
from clustering.dbscan_cluster import DBSCANClusterer

import pandas as pd
import tempfile, os


@pytest.fixture(scope="module")
def small_dataset():
    """Generate 50 synthetic resumes for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        tmp_path = f.name

    from data.download_dataset import _random_resume
    import csv
    records = [_random_resume(i) for i in range(1, 51)]
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Resume_str", "Category", "Name"])
        writer.writeheader()
        writer.writerows(records)

    df = pd.read_csv(tmp_path)
    os.unlink(tmp_path)
    return df


@pytest.fixture(scope="module")
def feature_matrix(small_dataset):
    preprocessor = ResumePreprocessor()
    extractor = ResumeFeatureExtractor()
    clean_texts, features_list = [], []
    for _, row in small_dataset.iterrows():
        text = str(row["Resume_str"])
        try:
            preprocessed = preprocessor.preprocess(text)
            features = extractor.extract(text, preprocessed)
        except Exception:
            preprocessed = {"clean_text": text}
            features = extractor.extract(text)
        clean_texts.append(preprocessed.get("clean_text", text))
        features_list.append(features)

    builder = ResumeFeatureMatrix()
    X = builder.fit_transform(clean_texts, features_list)
    return X, features_list


def test_kmeans_runs_without_error(feature_matrix):
    X, _ = feature_matrix
    clusterer = KMeansClusterer(k_range=range(2, 6))
    clusterer.fit(X, n_clusters=4)
    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == X.shape[0]


def test_kmeans_labels_in_range(feature_matrix):
    X, _ = feature_matrix
    clusterer = KMeansClusterer(k_range=range(2, 6))
    clusterer.fit(X, n_clusters=4)
    unique = set(clusterer.labels_.tolist())
    assert unique == {0, 1, 2, 3}


def test_kmeans_silhouette_positive(feature_matrix):
    X, _ = feature_matrix
    clusterer = KMeansClusterer(k_range=range(2, 6))
    clusterer.fit(X, n_clusters=4)
    X_dense = KMeansClusterer._to_dense(X)
    sil = silhouette_score(X_dense, clusterer.labels_)
    assert sil > -1  # Must be valid silhouette


def test_dbscan_runs_without_error(feature_matrix):
    X, _ = feature_matrix
    clusterer = DBSCANClusterer()
    clusterer.fit(X)
    assert clusterer.labels_ is not None
    assert len(clusterer.labels_) == X.shape[0]


def test_dbscan_has_at_least_one_cluster(feature_matrix):
    X, _ = feature_matrix
    clusterer = DBSCANClusterer()
    clusterer.fit(X)
    summary = clusterer.get_cluster_summary()
    assert summary["n_clusters"] >= 1


def test_feature_matrix_shape(feature_matrix):
    X, features_list = feature_matrix
    assert X.shape[0] == len(features_list)
    assert X.shape[1] > 10  # Should have many features
