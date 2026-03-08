"""
tests/test_preprocess.py
------------------------
Unit tests for the NLTK preprocessing pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.preprocess import ResumePreprocessor


SAMPLE_RESUMES = [
    "John Doe is a Data Scientist with 5 years of experience in Python, Pandas, and Machine Learning.",
    "Participated in Smart India Hackathon 2023 and won first prize.",
    "Projects: Built a web scraper using Python and BeautifulSoup. Deployed on AWS.",
    "Education: B.Tech in Computer Science from VIT University.",
    "GitHub: https://github.com/johndoe | Certifications: AWS Certified Solutions Architect",
]


@pytest.fixture
def preprocessor():
    return ResumePreprocessor()


def test_preprocess_returns_dict(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[0])
    assert isinstance(result, dict)
    expected_keys = {"clean_text", "tokens", "sentences", "entities",
                     "has_hackathon", "hackathon_count", "has_projects",
                     "github_links", "has_certs", "experience_years"}
    assert expected_keys.issubset(result.keys())


def test_tokens_are_cleaned(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[0])
    # Tokens should not contain stopwords like 'is', 'a', 'in'
    assert "is" not in result["tokens"]
    assert "a" not in result["tokens"]


def test_hackathon_detection(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[1])
    assert result["has_hackathon"] is True
    assert result["hackathon_count"] >= 1


def test_no_hackathon(preprocessor):
    result = preprocessor.preprocess("Software engineer with 3 years experience in Java.")
    assert result["has_hackathon"] is False


def test_github_extraction(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[4])
    assert len(result["github_links"]) >= 1
    assert "github.com" in result["github_links"][0]


def test_cert_detection(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[4])
    assert result["has_certs"] is True


def test_experience_extraction(preprocessor):
    result = preprocessor.preprocess(SAMPLE_RESUMES[0])
    assert result["experience_years"] == 5


def test_clean_text_not_empty(preprocessor):
    for resume in SAMPLE_RESUMES:
        result = preprocessor.preprocess(resume)
        assert len(result["clean_text"]) > 0
