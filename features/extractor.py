"""
features/extractor.py
---------------------
Structured feature extraction from preprocessed resume data.
"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Keyword Dictionary
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "Data Science": ["data science", "eda", "statistics", "data analysis",
                     "data analyst", "business intelligence", "tableau", "power bi",
                     "pandas", "numpy", "matplotlib", "seaborn", "excel"],
    "Machine Learning": ["machine learning", "deep learning", "neural network",
                         "tensorflow", "pytorch", "keras", "xgboost", "scikit-learn",
                         "model training", "feature engineering", "regression", "classification"],
    "Natural Language Processing": ["nlp", "natural language processing", "bert",
                                     "transformers", "text classification", "ner", "spacy",
                                     "nltk", "sentiment analysis", "summarization",
                                     "hugging face", "language model"],
    "Computer Vision": ["computer vision", "opencv", "yolo", "image classification",
                        "object detection", "cnn", "image segmentation", "gan", "mediapipe"],
    "Web Development": ["html", "css", "javascript", "react", "angular", "vue",
                        "node.js", "django", "flask", "rest api", "graphql",
                        "typescript", "next.js", "tailwind"],
    "Mobile Development": ["flutter", "dart", "kotlin", "swift", "android",
                           "ios", "react native", "firebase", "mobile app"],
    "DevOps": ["devops", "docker", "kubernetes", "jenkins", "ci/cd", "terraform",
               "ansible", "linux", "bash", "pipeline", "deployment", "monitoring"],
    "Cloud Computing": ["aws", "gcp", "azure", "serverless", "lambda", "s3",
                        "cloud", "microservices", "api gateway", "cloud formation"],
    "Cybersecurity": ["cybersecurity", "penetration testing", "owasp", "kali linux",
                      "wireshark", "metasploit", "cryptography", "siem", "firewall",
                      "ethical hacking", "vulnerability"],
    "Blockchain": ["blockchain", "solidity", "ethereum", "smart contract",
                   "web3", "defi", "nft", "ipfs", "cryptocurrency"],
    "Embedded Systems": ["embedded", "arduino", "raspberry pi", "rtos", "uart",
                         "i2c", "spi", "microcontroller", "pcb", "stm32", "firmware"],
    "Game Development": ["unity", "unreal engine", "game development", "c#",
                         "blender", "opengl", "shader", "game ai", "2d game", "3d game"],
    "UI/UX Design": ["figma", "adobe xd", "sketch", "wireframe", "prototype",
                     "user research", "usability", "design system", "ux", "ui design"],
    "Database Admin": ["mysql", "postgresql", "mongodb", "redis", "oracle",
                       "database", "sql", "query optimization", "nosql", "cassandra"],
    "Network Engineering": ["cisco", "bgp", "ospf", "vlan", "networking",
                             "tcp/ip", "subnetting", "vpn", "sdn", "routing"],
    "Robotics": ["ros", "robotics", "kinematics", "slam", "sensor fusion",
                 "pid", "gazebo", "autonomous", "actuator"],
    "Bioinformatics": ["bioinformatics", "biopython", "blast", "genome",
                       "proteomics", "ngs", "bioconductor", "sequencing"],
    "FinTech": ["fintech", "algorithmic trading", "quantlib", "risk management",
                "compliance", "payment", "trading", "financial modelling"],
    "AR/VR Development": ["ar", "vr", "augmented reality", "virtual reality",
                          "arkit", "arcore", "oculus", "webxr", "spatial computing"],
    "Quantum Computing": ["quantum", "qiskit", "cirq", "q#", "quantum circuit",
                          "vqe", "quantum cryptography", "superposition", "entanglement"],
}

# ---------------------------------------------------------------------------
# Skills Dictionary (500+ skills condensed into categories)
# ---------------------------------------------------------------------------

ALL_SKILLS = set()
for skills in DOMAIN_KEYWORDS.values():
    ALL_SKILLS.update(skills)

# Add additional curated skills
EXTRA_SKILLS = {
    "python", "java", "c++", "c", "go", "rust", "scala", "r", "matlab",
    "git", "github", "agile", "scrum", "jira", "confluence", "postman",
    "selenium", "pytest", "junit", "maven", "gradle", "npm", "yarn",
    "fastapi", "spring boot", "hibernate", "microservices", "kafka",
    "rabbitmq", "elasticsearch", "spark", "hadoop", "airflow", "dbt",
    "tableau", "power bi", "looker", "grafana", "prometheus",
    "jwt", "oauth", "ldap", "ssl/tls", "https",
    "photoshop", "illustrator", "canva", "after effects",
    "excel", "word", "powerpoint", "latex",
    "linux", "windows server", "macos", "bash", "powershell",
    "tcp/ip", "http/https", "rest", "soap", "grpc", "websocket",
    "jax", "onnx", "mlflow", "kubeflow", "wandb", "dvc",
    "langchain", "openai", "gemini", "llm",
}
ALL_SKILLS.update(EXTRA_SKILLS)

# ---------------------------------------------------------------------------
# Education Keywords
# ---------------------------------------------------------------------------

EDUCATION_DEGREES = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "m.tech": 4, "m.e.": 4, "m.s.": 4, "mtech": 4, "master": 4, "m.sc": 4, "mba": 4,
    "b.tech": 3, "b.e.": 3, "btech": 3, "b.sc": 3, "bachelor": 3, "be": 3, "bs": 3,
    "diploma": 2, "associate": 2,
    "12th": 1, "hsc": 1, "secondary": 1,
}

CERT_KEYWORDS = [
    "aws certified", "google professional", "microsoft certified", "azure",
    "coursera", "nptel", "udemy certificate", "edx", "oracle certified",
    "cisco ccna", "certified", "certification", "nanodegree",
]

HACKATHON_RE = re.compile(
    r"\b(hackathon|ideathon|datathon|code\s?sprint|smart india hackathon|"
    r"mlh|nasa space apps|google solution challenge|flipkart grid|"
    r"ethglobal|icpc|ieee xtreme|devfolio|compete|competition|contest|challenge)\b",
    re.IGNORECASE,
)

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-]+", re.IGNORECASE)

EXPERIENCE_RE = re.compile(
    r"(\d{1,2})\+?\s*year[s]?\s*(of\s*)?(experience|exp)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class ResumeFeatureExtractor:
    """Extract structured features from a resume text."""

    def extract(self, resume_text: str, preprocessed: dict = None) -> dict:
        """
        Parameters
        ----------
        resume_text   : raw or lightly cleaned resume string
        preprocessed  : optional dict from ResumePreprocessor.preprocess()

        Returns
        -------
        dict with all extracted features
        """
        text_lower = resume_text.lower()

        domain, domain_score = self._detect_domain(text_lower)
        skills = self._extract_skills(text_lower)
        edu_level, edu_score = self._extract_education(text_lower)
        exp_years = self._extract_experience(resume_text)
        projects_count = self._count_projects(resume_text)
        hackathon_mentions, hackathon_count = self._extract_hackathons(resume_text)
        certs = self._extract_certs(text_lower)
        github_links = GITHUB_RE.findall(resume_text)

        return {
            "domain": domain,
            "domain_confidence": round(domain_score, 4),
            "skills": skills,
            "skill_count": len(skills),
            "education_level": edu_level,
            "education_score": edu_score,
            "experience_years": exp_years,
            "projects_count": projects_count,
            "hackathon_participated": len(hackathon_mentions) > 0,
            "hackathon_count": hackathon_count,
            "hackathon_mentions": hackathon_mentions[:5],
            "certifications": certs,
            "cert_count": len(certs),
            "github_present": len(github_links) > 0,
            "github_links": github_links,
        }

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _detect_domain(self, text_lower: str):
        scores = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = count / len(keywords)
        if not any(scores.values()):
            return "Unknown", 0.0
        best = max(scores, key=scores.get)
        return best, scores[best]

    def _extract_skills(self, text_lower: str) -> list:
        found = []
        for skill in sorted(ALL_SKILLS):
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found.append(skill)
        return found

    def _extract_education(self, text_lower: str):
        best_deg, best_score = "Not Specified", 0
        for deg, score in EDUCATION_DEGREES.items():
            if deg in text_lower and score > best_score:
                best_deg, best_score = deg.upper(), score
        return best_deg, best_score

    def _extract_experience(self, text: str) -> int:
        matches = EXPERIENCE_RE.findall(text)
        if matches:
            return max(int(m[0]) for m in matches)
        return 0

    def _count_projects(self, text: str) -> int:
        """Count project mentions by looking for Project N: or project headings."""
        numbered = re.findall(r"project\s+\d+", text, re.IGNORECASE)
        sections = re.findall(
            r"\n\s*(project[s]?|portfolio)\s*[:\n]", text, re.IGNORECASE
        )
        # Fallback: count sentences with "built", "developed", "created"
        action_verbs = re.findall(
            r"\b(built|developed|created|designed|implemented|deployed)\b",
            text, re.IGNORECASE
        )
        if numbered:
            return len(numbered)
        return max(len(action_verbs), len(sections) * 2)

    def _extract_hackathons(self, text: str):
        raw_matches = HACKATHON_RE.findall(text)
        # Deduplicate by lower-casing
        unique = list({m.lower() for m in raw_matches})
        return unique, len(raw_matches)

    def _extract_certs(self, text_lower: str) -> list:
        found = []
        for cert in CERT_KEYWORDS:
            if cert in text_lower and cert not in found:
                found.append(cert)
        return found
