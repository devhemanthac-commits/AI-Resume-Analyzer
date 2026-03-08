# 🤖 AI Resume Analyzer

> **Intelligent resume clustering system** that uses **NLTK** for NLP preprocessing and **KMeans/DBSCAN** clustering algorithms to group candidates by their domain, projects, hackathon participation, skills, and experience level.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)  
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Kaggle API Setup](#-kaggle-api-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How Clustering Works](#-how-clustering-works)
- [Scoring System](#-scoring-system)
- [Outputs](#-outputs)
- [Running Tests](#-running-tests)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

Recruiters spend hours manually reviewing resumes. This project automates that process by:

1. **Downloading** a real-world resume dataset from Kaggle (with a synthetic fallback)
2. **Preprocessing** resume text using NLTK (tokenization, lemmatization, NER, stop-word removal)
3. **Extracting** structured features: domain, skills, projects, hackathons, education, experience, certifications, GitHub links
4. **Vectorizing** resumes into TF-IDF + structured feature matrices
5. **Clustering** resumes into meaningful groups using KMeans and/or DBSCAN
6. **Scoring** each candidate on 5 axes (0–100)
7. **Reporting** results as an interactive HTML report and a CSV

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI RESUME ANALYZER                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐  │
│  │  Kaggle  │    │    NLTK     │    │ Feature Extractor │  │
│  │ Dataset  │───▶│ Preprocessor│───▶│ (domain, skills,  │  │
│  │(500+ CVs)│    │             │    │  hackathons, etc) │  │
│  └──────────┘    └─────────────┘    └────────┬──────────┘  │
│                                               │              │
│  ┌────────────────────────────────────────────▼──────────┐  │
│  │              Feature Matrix Builder                    │  │
│  │     TF-IDF (5000 features, bigrams) × 0.7             │  │
│  │   + Structured Features (normalised) × 0.3            │  │
│  └────────────────────────┬───────────────────────────────┘ │
│                           │                                  │
│              ┌────────────┴────────────┐                    │
│              ▼                         ▼                    │
│   ┌─────────────────┐      ┌─────────────────────┐         │
│   │    KMeans       │      │      DBSCAN          │         │
│   │ (Elbow Method + │      │  (k-distance auto    │         │
│   │  Silhouette)    │      │   eps + outliers)    │         │
│   └────────┬────────┘      └──────────┬──────────┘         │
│            └──────────┬───────────────┘                     │
│                       ▼                                      │
│   ┌────────────────────────────────────────┐               │
│   │              Analyzer                   │               │
│   │  Scorer (Skills/Projects/Hackathon/     │               │
│   │          Education/Experience)          │               │
│   └────────────────────┬───────────────────┘               │
│                        ▼                                     │
│   ┌────────────────────────────────────────┐               │
│   │  Reporter: HTML Report + CSV + Plots   │               │
│   │  PCA | t-SNE | Word Cloud | Bar Charts │               │
│   └────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---------|--------|
| 🧠 **NLTK NLP Pipeline** | Tokenization, Lemmatization, NER, Stop-word removal |
| 🎯 **20-Domain Detection** | Data Science, Web Dev, ML, NLP, CV, DevOps, Blockchain, etc. |
| 🛠️ **500+ Skills Dictionary** | Automatic skill extraction and counting |
| 🏆 **Hackathon Detection** | SIH, MLH, Google Solution Challenge, IEEE, etc. |
| 🎓 **Education Parsing** | PhD > M.Tech > B.Tech > Diploma hierarchy |
| 📊 **KMeans Clustering** | Auto optimal K via Elbow + Silhouette Score |
| 🔵 **DBSCAN Clustering** | Organic density-based grouping + outlier detection |
| 📈 **Visualizations** | PCA, t-SNE, Word Clouds, Bar Charts, Score Boxes |
| 🏅 **Resume Scoring** | 5-axis weighted score (0–100 per resume) |
| 📋 **HTML Report** | Beautiful, interactive cluster analysis report |
| 🧪 **Unit Tests** | pytest coverage for all modules |
| ⚙️ **CLI** | One-command pipeline with flags |

---

## 🔧 Prerequisites

- Python **3.9+**
- pip
- (Optional) [Kaggle account](https://www.kaggle.com/) for the real dataset

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. NLTK data is auto-downloaded on first run, but you can pre-download:
python -c "import nltk; nltk.download('all')"
```

---

## 🔑 Kaggle API Setup

> **Skip this step if you want to use the built-in synthetic dataset (500 auto-generated resumes).**

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Under **API** → click **Create New Token**
3. A `kaggle.json` file downloads automatically
4. Copy your credentials to `.env`:

```bash
cp .env.example .env
# Edit .env and fill in:
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

The dataset used: [Resume Dataset by Sneha Anbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (2484 real resumes across 25 categories).

---

## 🚀 Usage

### Download Dataset

```bash
# Download from Kaggle (or generate synthetic if no credentials)
python data/download_dataset.py
```

### Run the Full Pipeline

```bash
# KMeans only (auto-detect optimal clusters)
python main.py --visualize --report

# KMeans with explicit cluster count
python main.py --clusters 10 --visualize --report

# DBSCAN only
python main.py --algorithm dbscan --visualize --report

# Both KMeans AND DBSCAN
python main.py --algorithm both --clusters 10 --visualize --report

# Custom input file
python main.py --input path/to/your/resumes.csv --clusters 8 --algorithm kmeans --report
```

### CLI Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/raw/UpdatedResumeDataSet.csv` | Path to resume CSV |
| `--clusters` | auto | Number of KMeans clusters |
| `--algorithm` | `kmeans` | `kmeans`, `dbscan`, or `both` |
| `--visualize` | off | Generate all plots |
| `--report` | off | Generate HTML + CSV report |
| `--download` | off | Force re-download dataset |

### Input CSV Format

Your CSV must have at least one text column — any of:

| Column Name | Description |
|-------------|-------------|
| `Resume_str` | Full resume text (default) |
| `resume` / `Resume` | Alternative names |
| `text` / `Text` | Generic text column |

Optional columns: `Name`, `Category` (ground truth domain).

---

## 📁 Project Structure

```
ai-resume-analyzer/
├── data/
│   ├── __init__.py
│   ├── download_dataset.py     # Kaggle downloader + synthetic generator
│   ├── preprocess.py           # NLTK preprocessing pipeline
│   └── raw/                    # Downloaded/generated CSV (git-ignored)
│
├── features/
│   ├── __init__.py
│   ├── extractor.py            # Domain, skills, hackathon, education extraction
│   └── vectorizer.py           # TF-IDF + structured feature matrix builder
│
├── clustering/
│   ├── __init__.py
│   ├── kmeans_cluster.py       # KMeans with Elbow + Silhouette tuning
│   ├── dbscan_cluster.py       # DBSCAN with auto eps estimation
│   └── visualizer.py           # PCA, t-SNE, word clouds, bar charts
│
├── analyzer/
│   ├── __init__.py
│   ├── scorer.py               # 5-axis resume scorer (0-100)
│   └── reporter.py             # HTML + CSV report generator
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py      # NLTK pipeline tests
│   ├── test_extractor.py       # Feature extractor tests
│   ├── test_clustering.py      # KMeans + DBSCAN tests
│   └── test_scorer.py          # Scorer tests
│
├── outputs/                    # Auto-created, git-ignored
│   ├── plots/
│   │   ├── clusters_pca.png
│   │   ├── clusters_tsne.png
│   │   ├── domain_distribution.png
│   │   ├── hackathon_per_cluster.png
│   │   ├── wordclouds_per_cluster.png
│   │   ├── score_distribution.png
│   │   ├── kmeans_elbow_silhouette.png
│   │   └── dbscan_k_distance.png
│   ├── cluster_report.csv
│   ├── summary.html
│   └── analyzer.log
│
├── main.py                     # CLI entry point
├── conftest.py                 # Pytest path setup
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🔬 How Clustering Works

### Step 1 — NLTK Preprocessing
Each resume is processed through:
- **Sentence tokenization** (`sent_tokenize`)
- **Word tokenization** (`word_tokenize`)
- **Stop-word removal** (NLTK English corpus, domain-adjusted)
- **Lemmatization** (`WordNetLemmatizer`)
- **Named Entity Recognition** (`ne_chunk` with POS tagging)
- **Regex extraction** for hackathons, GitHub URLs, certifications, experience years

### Step 2 — Feature Extraction
20 structured features per resume:
- `domain` (20 categories), `domain_confidence`
- `skills` list + `skill_count` (from 500+ skill dictionary)
- `education_level`, `education_score` (PhD=5 > M.Tech=4 > B.Tech=3 …)
- `experience_years`
- `projects_count`
- `hackathon_participated`, `hackathon_count`, `hackathon_mentions`
- `certifications`, `cert_count`
- `github_present`, `github_links`

### Step 3 — Vectorization
```
Feature Matrix = TF-IDF (5000 features, bigrams) × 0.7
               + Normalised Structured Features × 0.3
```

### Step 4 — KMeans Clustering (auto-tuned)
1. **TruncatedSVD** (LSA) reduces to 50 dimensions
2. **Elbow Method**: test K = 2 … 20, measure WCSS (inertia)
3. **Silhouette Score**: pick K that maximises cluster cohesion
4. **Final fit** with optimal K, `n_init=10, max_iter=500`

### Step 5 — DBSCAN Clustering (density-based)
1. **k-distance graph** automatically estimates the optimal `eps` value
2. **Organic clusters** — no need to specify K
3. **Outlier detection** — noise resumes labelled as `-1`

---

## 🏅 Scoring System

Each resume is scored on **5 axes** (all 0–100):

| Axis | Weight | Criteria |
|------|--------|----------|
| 🛠️ Skills Breadth | 25% | `skill_count / 30` |
| 📂 Project Depth | 20% | `projects_count / 10` |
| 🏆 Hackathon Activity | 20% | `participated×30 + count×14` (capped 100) |
| 🎓 Education | 15% | Degree level (PhD→100, B.Tech→60) |
| 💼 Experience | 20% | `years / 15` |

The **Composite Score** is the weighted average of all 5 axes.

---

## 📊 Outputs

After running with `--visualize --report`:

| File | Description |
|------|-------------|
| `outputs/summary.html` | 🌐 Full interactive HTML report — open in browser |
| `outputs/cluster_report.csv` | 📋 All resumes with cluster labels, scores, features |
| `outputs/plots/clusters_pca.png` | PCA 2D scatter coloured by cluster |
| `outputs/plots/clusters_tsne.png` | t-SNE 2D scatter for nonlinear structure |
| `outputs/plots/domain_distribution.png` | Domain breakdown per cluster |
| `outputs/plots/hackathon_per_cluster.png` | Hackathon participation per cluster |
| `outputs/plots/wordclouds_per_cluster.png` | Top skills word cloud per cluster |
| `outputs/plots/score_distribution.png` | Box plot of scores per cluster |
| `outputs/plots/kmeans_elbow_silhouette.png` | Elbow + Silhouette curve |
| `outputs/plots/dbscan_k_distance.png` | k-distance graph for eps estimation |
| `outputs/analyzer.log` | Full run log |

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_preprocess.py -v
python -m pytest tests/test_extractor.py -v
python -m pytest tests/test_clustering.py -v
python -m pytest tests/test_scorer.py -v

# Run with coverage report
pip install pytest-cov
python -m pytest tests/ -v --cov=. --cov-report=html
```

Expected output:
```
tests/test_preprocess.py ........   PASSED
tests/test_extractor.py .........   PASSED
tests/test_clustering.py .......    PASSED
tests/test_scorer.py .........      PASSED
========================= 33 passed =========================
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Ideas for Contribution
- [ ] Add PDF/DOCX resume upload support
- [ ] Integrate Sentence-BERT embeddings for richer representations
- [ ] Build a Streamlit/Flask web UI
- [ ] Add LLM-based resume summarization
- [ ] Support multilingual resumes

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## 🙏 Acknowledgements

- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) by Sneha Anbhawal
- [NLTK](https://www.nltk.org/) – Natural Language Toolkit
- [Scikit-learn](https://scikit-learn.org/) – Machine Learning in Python
- [WordCloud](https://github.com/amueller/word_cloud) – Word cloud generation
- [Plotly](https://plotly.com/python/) – Interactive visualizations

---

<p align="center">
  Built with ❤️ by Hemanth &nbsp;|&nbsp; AI Resume Analyzer
</p>
