"""
analyzer/reporter.py
--------------------
Generates per-cluster summary tables and a full HTML report.
"""

import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from jinja2 import Template

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Jinja2 HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Resume Analyzer – Cluster Report</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
  h1 { color: #1a1a2e; text-align: center; font-size: 2rem; margin-bottom: 5px; }
  .subtitle { text-align: center; color: #555; margin-bottom: 30px; }
  .cluster-card { background: white; border-radius: 12px; padding: 20px;
                  margin-bottom: 24px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); }
  .cluster-header { display: flex; justify-content: space-between; align-items: center;
                    border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; margin-bottom: 15px; }
  .cluster-title { font-size: 1.2rem; font-weight: 700; color: #1a1a2e; }
  .cluster-badge { background: #667eea; color: white; padding: 4px 12px;
                   border-radius: 20px; font-size: 0.85rem; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
  .stat-box { background: #f8f9fa; border-radius: 8px; padding: 10px; text-align: center; }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: #667eea; }
  .stat-label { font-size: 0.75rem; color: #888; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.85rem; }
  th { background: #1a1a2e; color: white; padding: 8px 12px; text-align: left; }
  tr:nth-child(even) { background: #f8f9fa; }
  td { padding: 7px 12px; border-bottom: 1px solid #eee; }
  .score-bar { background: #e0e0e0; border-radius: 4px; height: 8px; width: 100%; }
  .score-fill { background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 4px; height: 8px; }
  .skills-tags { display: flex; flex-wrap: wrap; gap: 5px; }
  .skill-tag { background: #e8f0fe; color: #1a73e8; padding: 2px 8px;
               border-radius: 12px; font-size: 0.75rem; }
  .footer { text-align: center; color: #aaa; margin-top: 30px; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>🤖 AI Resume Analyzer</h1>
<p class="subtitle">Cluster Analysis Report &nbsp;|&nbsp; Algorithm: <strong>{{ algorithm }}</strong>
&nbsp;|&nbsp; Total Resumes: <strong>{{ total }}</strong>
&nbsp;|&nbsp; Clusters Found: <strong>{{ n_clusters }}</strong></p>

{% for cluster in clusters %}
<div class="cluster-card">
  <div class="cluster-header">
    <span class="cluster-title">📂 Cluster {{ cluster.id }} – {{ cluster.dominant_domain }}</span>
    <span class="cluster-badge">{{ cluster.size }} Candidates</span>
  </div>
  <div class="stats-grid">
    <div class="stat-box">
      <div class="stat-value">{{ cluster.avg_score }}</div>
      <div class="stat-label">Avg Score</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ cluster.hackathon_pct }}%</div>
      <div class="stat-label">Hackathon Participants</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ cluster.avg_projects }}</div>
      <div class="stat-label">Avg Projects</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ cluster.avg_experience }}y</div>
      <div class="stat-label">Avg Experience</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ cluster.avg_skills }}</div>
      <div class="stat-label">Avg Skills</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ cluster.github_pct }}%</div>
      <div class="stat-label">GitHub Present</div>
    </div>
  </div>
  <p><strong>Top Skills:</strong></p>
  <div class="skills-tags">
    {% for s in cluster.top_skills %}<span class="skill-tag">{{ s }}</span>{% endfor %}
  </div>
  <table>
    <thead><tr>
      <th>#</th><th>Name / ID</th><th>Domain</th><th>Score</th>
      <th>Skills</th><th>Projects</th><th>Hackathons</th><th>Experience</th>
    </tr></thead>
    <tbody>
    {% for r in cluster.top_candidates %}
    <tr>
      <td>{{ loop.index }}</td>
      <td>{{ r.name }}</td>
      <td>{{ r.domain }}</td>
      <td>
        <div class="score-bar"><div class="score-fill" style="width:{{ r.score }}%"></div></div>
        {{ r.score }}
      </td>
      <td>{{ r.skill_count }}</td>
      <td>{{ r.projects }}</td>
      <td>{{ r.hackathons }}</td>
      <td>{{ r.experience }}y</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endfor %}

<div class="footer">Generated by AI Resume Analyzer &nbsp;•&nbsp; NLTK + Scikit-learn</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class ClusterReporter:
    """Builds per-cluster summaries and exports CSV + HTML reports."""

    def __init__(self, algorithm: str = "kmeans"):
        self.algorithm = algorithm

    def build_report(self,
                     df: pd.DataFrame,
                     labels: list,
                     features_list: list,
                     scores_list: list) -> pd.DataFrame:
        """
        Parameters
        ----------
        df            : original DataFrame with at least 'Resume_str' and optionally 'Category', 'Name'
        labels        : cluster labels (one per row)
        features_list : list of feature dicts from extractor
        scores_list   : list of score dicts from scorer

        Returns
        -------
        full DataFrame with cluster assignments and scores
        """
        # Attach results to dataframe
        result_df = df.copy().reset_index(drop=True)
        result_df["cluster"] = labels
        for key in ["domain", "skill_count", "education_level", "experience_years",
                    "projects_count", "hackathon_participated", "hackathon_count",
                    "cert_count", "github_present"]:
            result_df[key] = [f.get(key, "") for f in features_list]

        for key in ["composite_score", "score_skills_breadth", "score_project_depth",
                    "score_hackathon", "score_education", "score_experience"]:
            result_df[key] = [s.get(key, 0) for s in scores_list]

        # Save CSV
        csv_path = OUTPUTS_DIR / "cluster_report.csv"
        result_df.to_csv(csv_path, index=False)
        logger.info(f"Cluster CSV report saved to {csv_path}")

        # Build HTML
        self._render_html(result_df)

        return result_df

    def _render_html(self, df: pd.DataFrame):
        clusters_data = []
        unique_labels = sorted(l for l in df["cluster"].unique() if l != -1)

        for lbl in unique_labels:
            sub = df[df["cluster"] == lbl]
            top_skills_all = []
            for feat_skills in sub.get("skills", pd.Series()).tolist() if "skills" in sub.columns else []:
                if isinstance(feat_skills, list):
                    top_skills_all.extend(feat_skills)

            common_skills = [s for s, _ in Counter(top_skills_all).most_common(10)]

            top_candidates = []
            for _, row in sub.nlargest(10, "composite_score").iterrows():
                top_candidates.append({
                    "name": row.get("Name", f"Resume-{int(row.name)}"),
                    "domain": row.get("domain", "Unknown"),
                    "score": round(row.get("composite_score", 0), 1),
                    "skill_count": int(row.get("skill_count", 0)),
                    "projects": int(row.get("projects_count", 0)),
                    "hackathons": int(row.get("hackathon_count", 0)),
                    "experience": int(row.get("experience_years", 0)),
                })

            hackathon_pct = round(sub["hackathon_participated"].mean() * 100) if "hackathon_participated" in sub else 0
            github_pct = round(sub["github_present"].mean() * 100) if "github_present" in sub else 0

            clusters_data.append({
                "id": lbl,
                "size": len(sub),
                "dominant_domain": sub["domain"].mode()[0] if "domain" in sub.columns and len(sub) > 0 else "Unknown",
                "avg_score": round(sub["composite_score"].mean(), 1),
                "avg_skills": round(sub["skill_count"].mean(), 1) if "skill_count" in sub else 0,
                "avg_projects": round(sub["projects_count"].mean(), 1) if "projects_count" in sub else 0,
                "avg_experience": round(sub["experience_years"].mean(), 1) if "experience_years" in sub else 0,
                "hackathon_pct": hackathon_pct,
                "github_pct": github_pct,
                "top_skills": common_skills,
                "top_candidates": top_candidates,
            })

        total = len(df[df["cluster"] != -1])
        html = Template(HTML_TEMPLATE).render(
            algorithm=self.algorithm.upper(),
            total=total,
            n_clusters=len(unique_labels),
            clusters=clusters_data,
        )
        html_path = OUTPUTS_DIR / "summary.html"
        html_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved to {html_path}")
