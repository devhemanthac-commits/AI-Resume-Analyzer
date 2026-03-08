"""
analyzer/scorer.py
------------------
Scores each resume on 5 axes and returns a composite score (0-100).
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1)
WEIGHTS = {
    "skills_breadth":   0.25,
    "project_depth":    0.20,
    "hackathon":        0.20,
    "education":        0.15,
    "experience":       0.20,
}

MAX_SKILL_COUNT = 30
MAX_PROJECT_COUNT = 10
MAX_HACKATHON_COUNT = 5
MAX_EXPERIENCE_YEARS = 15
MAX_EDUCATION_SCORE = 5


def score_resume(features: dict) -> dict:
    """
    Score a single resume and return a breakdown dict.

    Parameters
    ----------
    features : dict output of ResumeFeatureExtractor.extract()

    Returns
    -------
    dict with individual axis scores and composite score (0-100)
    """
    # --- Skills Breadth (0-100) ---
    skill_score = min(features.get("skill_count", 0) / MAX_SKILL_COUNT, 1.0) * 100

    # --- Project Depth (0-100) ---
    project_score = min(features.get("projects_count", 0) / MAX_PROJECT_COUNT, 1.0) * 100

    # --- Hackathon Activity (0-100) ---
    participated = int(features.get("hackathon_participated", False))
    hack_count = features.get("hackathon_count", 0)
    hackathon_score = min(participated * 30 + hack_count * 14, 100)

    # --- Education (0-100) ---
    edu_score = (features.get("education_score", 0) / MAX_EDUCATION_SCORE) * 100

    # --- Experience (0-100) ---
    exp = min(features.get("experience_years", 0), MAX_EXPERIENCE_YEARS)
    exp_score = (exp / MAX_EXPERIENCE_YEARS) * 100

    # --- Composite ---
    composite = (
        WEIGHTS["skills_breadth"] * skill_score +
        WEIGHTS["project_depth"] * project_score +
        WEIGHTS["hackathon"] * hackathon_score +
        WEIGHTS["education"] * edu_score +
        WEIGHTS["experience"] * exp_score
    )

    return {
        "score_skills_breadth": round(skill_score, 2),
        "score_project_depth": round(project_score, 2),
        "score_hackathon": round(hackathon_score, 2),
        "score_education": round(edu_score, 2),
        "score_experience": round(exp_score, 2),
        "composite_score": round(composite, 2),
    }


def score_all(features_list: list) -> list:
    """Score all resumes and return list of score dicts."""
    logger.info(f"Scoring {len(features_list)} resumes …")
    scores = [score_resume(f) for f in features_list]
    composites = [s["composite_score"] for s in scores]
    logger.info(f"Score stats – Min: {min(composites):.1f} | Max: {max(composites):.1f} | "
                f"Avg: {np.mean(composites):.1f}")
    return scores
