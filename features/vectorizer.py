"""
features/vectorizer.py
----------------------
Converts cleaned resume text (from preprocess.py) into numerical feature matrices
using TF-IDF and structured feature arrays for clustering.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

class ResumeTfidfVectorizer:
    """
    Wraps sklearn TfidfVectorizer with sensible defaults for resume text.
    Uses unigrams + bigrams, max 5000 features.
    """

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,         # log(1+tf) scaling
            min_df=2,                  # ignore terms appearing in <2 docs
            max_df=0.85,               # ignore overly common terms
            strip_accents="unicode",
            analyzer="word",
        )
        self.is_fitted = False

    def fit(self, texts: list):
        logger.info(f"Fitting TF-IDF on {len(texts)} documents …")
        self.tfidf.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts: list):
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        return self.tfidf.transform(texts)

    def fit_transform(self, texts: list):
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> list:
        return self.tfidf.get_feature_names_out().tolist()

    def top_terms_for_vector(self, vector, n: int = 10) -> list:
        """Return top-n terms for a given document vector."""
        feature_names = self.get_feature_names()
        arr = vector.toarray().flatten()
        top_indices = arr.argsort()[-n:][::-1]
        return [(feature_names[i], round(float(arr[i]), 4)) for i in top_indices if arr[i] > 0]


# ---------------------------------------------------------------------------
# Structured Feature Matrix
# ---------------------------------------------------------------------------

STRUCTURED_FEATURE_COLS = [
    "skill_count",
    "education_score",
    "experience_years",
    "projects_count",
    "hackathon_count",
    "cert_count",
    "github_present",
    "hackathon_participated",
]


class StructuredFeatureBuilder:
    """
    Converts the structured dict from ResumeFeatureExtractor into a normalised
    numpy array ready for concatenation with TF-IDF features.
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _to_df(self, features_list: list) -> pd.DataFrame:
        rows = []
        for f in features_list:
            rows.append({
                "skill_count": f.get("skill_count", 0),
                "education_score": f.get("education_score", 0),
                "experience_years": min(f.get("experience_years", 0), 20),  # cap at 20
                "projects_count": min(f.get("projects_count", 0), 20),
                "hackathon_count": min(f.get("hackathon_count", 0), 10),
                "cert_count": min(f.get("cert_count", 0), 10),
                "github_present": int(f.get("github_present", False)),
                "hackathon_participated": int(f.get("hackathon_participated", False)),
            })
        return pd.DataFrame(rows, columns=STRUCTURED_FEATURE_COLS)

    def fit(self, features_list: list):
        df = self._to_df(features_list)
        self.scaler.fit(df)
        self.is_fitted = True
        return self

    def transform(self, features_list: list) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform()")
        df = self._to_df(features_list)
        return self.scaler.transform(df)

    def fit_transform(self, features_list: list) -> np.ndarray:
        self.fit(features_list)
        return self.transform(features_list)


# ---------------------------------------------------------------------------
# Combined Feature Matrix
# ---------------------------------------------------------------------------

class ResumeFeatureMatrix:
    """
    Combines TF-IDF sparse matrix with normalised structured features
    into a single dense or sparse matrix for clustering.
    """

    def __init__(self, tfidf_weight: float = 0.7, structured_weight: float = 0.3):
        self.tfidf_weight = tfidf_weight
        self.structured_weight = structured_weight
        self.tfidf_vectorizer = ResumeTfidfVectorizer()
        self.structured_builder = StructuredFeatureBuilder()

    def fit_transform(self, clean_texts: list, features_list: list):
        """
        Parameters
        ----------
        clean_texts   : list of pre-processed resume strings
        features_list : list of feature dicts from ResumeFeatureExtractor

        Returns
        -------
        numpy array (n_samples, n_features_tfidf + n_structured)
        """
        logger.info("Building combined TF-IDF + structured feature matrix …")

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_texts)
        structured_matrix = self.structured_builder.fit_transform(features_list)

        # Weight and combine
        weighted_tfidf = tfidf_matrix.multiply(self.tfidf_weight)
        weighted_structured = csr_matrix(structured_matrix * self.structured_weight)

        combined = hstack([weighted_tfidf, weighted_structured])
        logger.info(f"Feature matrix shape: {combined.shape}")
        return combined

    def transform(self, clean_texts: list, features_list: list):
        tfidf_matrix = self.tfidf_vectorizer.transform(clean_texts)
        structured_matrix = self.structured_builder.transform(features_list)
        weighted_tfidf = tfidf_matrix.multiply(self.tfidf_weight)
        weighted_structured = csr_matrix(structured_matrix * self.structured_weight)
        return hstack([weighted_tfidf, weighted_structured])

    def get_tfidf_vectorizer(self) -> ResumeTfidfVectorizer:
        return self.tfidf_vectorizer
