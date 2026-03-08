"""
tests/test_extractor.py
-----------------------
Unit tests for the resume feature extractor.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from features.extractor import ResumeFeatureExtractor


PYTHON_RESUME = """
Jane Smith
M.Tech in Artificial Intelligence from IIT Madras.
5 years of experience in Machine Learning.
Skills: Python, TensorFlow, PyTorch, Keras, Scikit-learn, Pandas, NumPy.
Projects:
  Project 1: Built a sentiment analysis model using BERT achieving 94% accuracy.
  Project 2: Developed a recommendation system deployed on AWS.
Hackathons: Participated in Smart India Hackathon, Google Solution Challenge.
GitHub: https://github.com/janesmith
Certifications: AWS Certified Solutions Architect, Coursera Deep Learning Specialization.
"""

WEB_RESUME = """
Alex Kumar
B.Tech in Computer Science.
3 years experience in web development.
Skills: HTML, CSS, JavaScript, React, Node.js, Django, REST API.
Projects: Built an e-commerce platform using React and Django.
GitHub: https://github.com/alexkumar
"""

NO_HACK_RESUME = """
Raj Patel, Database Administrator. 
8 years of experience. 
Skills: MySQL, PostgreSQL, MongoDB, Redis.
"""


@pytest.fixture
def extractor():
    return ResumeFeatureExtractor()


def test_extract_returns_dict(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert isinstance(result, dict)


def test_skill_extraction(extractor):
    result = extractor.extract(PYTHON_RESUME)
    skills = result["skills"]
    assert isinstance(skills, list)
    assert len(skills) > 3
    assert "python" in skills


def test_skill_count_positive(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["skill_count"] > 0


def test_domain_detection_ml(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["domain"] in ["Machine Learning", "Natural Language Processing",
                                 "Data Science", "Computer Vision"]


def test_domain_detection_web(extractor):
    result = extractor.extract(WEB_RESUME)
    assert result["domain"] == "Web Development"


def test_hackathon_detection(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["hackathon_participated"] is True
    assert result["hackathon_count"] >= 1


def test_no_hackathon(extractor):
    result = extractor.extract(NO_HACK_RESUME)
    assert result["hackathon_participated"] is False


def test_github_detection(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["github_present"] is True
    assert len(result["github_links"]) >= 1


def test_education_extraction(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["education_score"] >= 3  # M.Tech = 4


def test_experience_years(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["experience_years"] == 5


def test_cert_extraction(extractor):
    result = extractor.extract(PYTHON_RESUME)
    assert result["cert_count"] >= 1
