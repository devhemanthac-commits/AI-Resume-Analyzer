"""
tests/test_scorer.py
--------------------
Unit tests for the resume scorer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analyzer.scorer import score_resume, score_all


LOW_FEATURES = {
    "skill_count": 2,
    "education_score": 0,
    "experience_years": 0,
    "projects_count": 0,
    "hackathon_participated": False,
    "hackathon_count": 0,
    "cert_count": 0,
    "github_present": False,
}

HIGH_FEATURES = {
    "skill_count": 25,
    "education_score": 5,
    "experience_years": 10,
    "projects_count": 8,
    "hackathon_participated": True,
    "hackathon_count": 4,
    "cert_count": 3,
    "github_present": True,
}

MID_FEATURES = {
    "skill_count": 10,
    "education_score": 3,
    "experience_years": 3,
    "projects_count": 3,
    "hackathon_participated": True,
    "hackathon_count": 1,
    "cert_count": 1,
    "github_present": True,
}


def test_score_in_valid_range_low():
    result = score_resume(LOW_FEATURES)
    assert 0 <= result["composite_score"] <= 100


def test_score_in_valid_range_high():
    result = score_resume(HIGH_FEATURES)
    assert 0 <= result["composite_score"] <= 100


def test_high_scores_higher_than_low():
    low = score_resume(LOW_FEATURES)["composite_score"]
    high = score_resume(HIGH_FEATURES)["composite_score"]
    assert high > low


def test_all_axes_present():
    result = score_resume(MID_FEATURES)
    expected_keys = {"score_skills_breadth", "score_project_depth", "score_hackathon",
                     "score_education", "score_experience", "composite_score"}
    assert expected_keys == set(result.keys())


def test_all_axis_scores_in_range():
    result = score_resume(HIGH_FEATURES)
    for key, val in result.items():
        assert 0 <= val <= 100, f"{key} out of range: {val}"


def test_score_all_returns_list():
    features_list = [LOW_FEATURES, MID_FEATURES, HIGH_FEATURES]
    results = score_all(features_list)
    assert len(results) == 3
    for r in results:
        assert "composite_score" in r
        assert 0 <= r["composite_score"] <= 100


def test_empty_features_handled():
    result = score_resume({})
    assert 0 <= result["composite_score"] <= 100
