"""
data/download_dataset.py
------------------------
Downloads the Kaggle Resume Dataset. If Kaggle credentials are unavailable,
falls back to generating synthetic resume data.
"""

import os
import sys
import csv
import random
import zipfile
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = RAW_DIR / "UpdatedResumeDataSet.csv"


# ---------------------------------------------------------------------------
# Kaggle Download
# ---------------------------------------------------------------------------

def download_from_kaggle():
    """Attempt to download the dataset using the Kaggle API."""
    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")

    if not username or not api_key:
        logger.warning("Kaggle credentials not found in .env – falling back to synthetic data.")
        return False

    # Write kaggle.json temporarily
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(f'{{"username":"{username}","key":"{api_key}"}}')
    kaggle_json.chmod(0o600)

    try:
        import kaggle  # noqa: F401  (requires valid credentials to be present)
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()
        logger.info("Downloading dataset: snehaanbhawal/resume-dataset …")
        api.dataset_download_files(
            "snehaanbhawal/resume-dataset",
            path=str(RAW_DIR),
            unzip=True,
        )
        logger.info(f"Dataset saved to {RAW_DIR}")
        return True
    except Exception as exc:
        logger.error(f"Kaggle download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

DOMAINS = [
    "Data Science", "Web Development", "Mobile Development", "DevOps",
    "Cybersecurity", "Machine Learning", "Cloud Computing", "Blockchain",
    "Embedded Systems", "Game Development", "UI/UX Design", "Database Admin",
    "Network Engineering", "AR/VR Development", "Bioinformatics",
    "Natural Language Processing", "Computer Vision", "Robotics",
    "Quantum Computing", "FinTech",
]

SKILLS_MAP = {
    "Data Science": ["Python", "R", "Pandas", "NumPy", "Matplotlib", "Seaborn",
                     "Scikit-learn", "Statistics", "EDA", "Jupyter"],
    "Web Development": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Django",
                        "Flask", "REST API", "TypeScript", "SQL"],
    "Mobile Development": ["Flutter", "Dart", "Kotlin", "Swift", "React Native",
                           "Android SDK", "iOS", "Firebase", "Retrofit", "MVVM"],
    "DevOps": ["Docker", "Kubernetes", "Jenkins", "CI/CD", "Terraform", "Ansible",
               "Linux", "Bash", "AWS", "Git"],
    "Cybersecurity": ["Penetration Testing", "OWASP", "Wireshark", "Metasploit",
                      "Kali Linux", "Cryptography", "SIEM", "Firewall", "ISO 27001"],
    "Machine Learning": ["TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM",
                         "Feature Engineering", "Model Deployment", "MLflow", "ONNX"],
    "Cloud Computing": ["AWS", "GCP", "Azure", "Serverless", "Lambda", "S3",
                        "VPC", "CloudFormation", "IAM", "Load Balancer"],
    "Blockchain": ["Solidity", "Ethereum", "Smart Contracts", "Web3.js", "Hardhat",
                   "IPFS", "DeFi", "NFT", "Rust", "Consensus Algorithms"],
    "Embedded Systems": ["C", "C++", "Arduino", "Raspberry Pi", "RTOS", "UART",
                         "I2C", "SPI", "PCB Design", "STM32"],
    "Game Development": ["Unity", "Unreal Engine", "C#", "Blender", "OpenGL",
                         "Physics Engine", "Shader Programming", "Game AI"],
    "UI/UX Design": ["Figma", "Adobe XD", "Sketch", "Prototyping", "Wireframing",
                     "User Research", "Accessibility", "Design Systems"],
    "Database Admin": ["MySQL", "PostgreSQL", "MongoDB", "Redis", "Oracle",
                       "Query Optimization", "Replication", "Backup & Recovery"],
    "Network Engineering": ["Cisco IOS", "BGP", "OSPF", "VLAN", "SDN", "Wireshark",
                             "TCP/IP", "Subnetting", "VPN", "QoS"],
    "AR/VR Development": ["Unity", "ARKit", "ARCore", "Oculus SDK", "WebXR",
                          "OpenXR", "3D Modeling", "Spatial Computing"],
    "Bioinformatics": ["Python", "R", "Biopython", "BLAST", "Genome Assembly",
                       "Proteomics", "NGS", "Bioconductor"],
    "Natural Language Processing": ["spaCy", "Transformers", "BERT", "NLTK",
                                     "Text Classification", "NER", "Summarization",
                                     "Sentiment Analysis", "Hugging Face"],
    "Computer Vision": ["OpenCV", "TensorFlow", "YOLO", "CNNs", "Image Segmentation",
                        "Object Detection", "GAN", "MediaPipe"],
    "Robotics": ["ROS", "Python", "C++", "MATLAB", "Kinematics", "SLAM",
                 "Sensor Fusion", "PID Control", "Gazebo"],
    "Quantum Computing": ["Qiskit", "Cirq", "Q#", "Quantum Circuits", "VQE",
                          "Quantum Cryptography", "Linear Algebra"],
    "FinTech": ["Python", "Algorithmic Trading", "Risk Management", "Quantlib",
                "Pandas", "Regulatory Compliance", "Blockchain", "Payment APIs"],
}

HACKATHON_EVENTS = [
    "Smart India Hackathon", "HackerEarth Hackathon", "MLH Local Hack Day",
    "Google Solution Challenge", "NASA Space Apps Challenge",
    "HackWithInfy", "Flipkart GRiD", "Microsoft Imagine Cup",
    "Facebook Hacker Cup", "Devfolio Hackathon", "ETHGlobal",
    "Kaggle Competition", "ICPC", "IEEE Xtreme", "CodeChef Snackdown",
]

EDUCATION_LEVELS = [
    ("B.Tech", "Computer Science", "VIT University"),
    ("B.Tech", "Information Technology", "Anna University"),
    ("M.Tech", "Artificial Intelligence", "IIT Madras"),
    ("M.Sc", "Data Science", "BITS Pilani"),
    ("B.E.", "Electronics", "NIT Trichy"),
    ("MBA", "Technology Management", "IIM Bangalore"),
    ("PhD", "Machine Learning", "IISc Bangalore"),
    ("B.Sc", "Computer Applications", "Madras University"),
]

FIRST_NAMES = ["Arjun", "Priya", "Rahul", "Sneha", "Aditya", "Kavya", "Vikram",
               "Ananya", "Rohan", "Meera", "Karthik", "Divya", "Sanjay", "Pooja",
               "Nikhil", "Lakshmi", "Suresh", "Nithya", "Mohan", "Asha",
               "Alex", "Emma", "James", "Olivia", "Ethan", "Sophia", "Liam", "Mia"]

LAST_NAMES = ["Kumar", "Sharma", "Reddy", "Nair", "Gupta", "Singh", "Patel",
              "Iyer", "Rao", "Verma", "Mehta", "Joshi", "Menon", "Das",
              "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]


def _random_resume(idx: int) -> dict:
    domain = random.choice(DOMAINS)
    domain_skills = SKILLS_MAP[domain]
    k_skills = min(random.randint(4, 8), len(domain_skills))
    skills = random.sample(domain_skills, k=k_skills)
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    edu = random.choice(EDUCATION_LEVELS)
    exp_years = random.randint(0, 12)
    num_projects = random.randint(1, 5)
    hackathons = random.sample(HACKATHON_EVENTS, k=random.randint(0, min(4, len(HACKATHON_EVENTS))))
    num_certs = random.randint(0, 3)

    projects_text = ""
    for p in range(1, num_projects + 1):
        k_proj = min(2, len(skills))
        proj_skills = random.sample(skills, k=k_proj) if k_proj > 0 else skills[:1]
        projects_text += (f"Project {p}: Built a {domain} solution using {', '.join(proj_skills)} "
                          f"achieving significant improvement. Collaborated in a team of {random.randint(2,5)}. ")

    hackathon_text = ""
    if hackathons:
        hackathon_text = f"Participated in: {', '.join(hackathons)}. "

    cert_names = ["AWS Certified Solutions Architect", "Google Professional Data Engineer",
                  "Microsoft Azure Fundamentals", "Coursera Deep Learning Specialization",
                  "NPTEL Programming in Python", "Cisco CCNA", "Oracle Java SE Certification"]
    certs_text = ""
    if num_certs:
        certs_text = "Certifications: " + ", ".join(random.sample(cert_names, k=num_certs)) + ". "

    resume_text = (
        f"{name}\n\n"
        f"Objective: Passionate {domain} professional seeking challenging roles.\n\n"
        f"Education: {edu[0]} in {edu[1]} from {edu[2]}.\n\n"
        f"Experience: {exp_years} years of hands-on experience in {domain}.\n\n"
        f"Skills: {', '.join(skills)}.\n\n"
        f"Projects: {projects_text}\n\n"
        f"Hackathons & Competitions: {hackathon_text if hackathon_text else 'None.'}\n\n"
        f"{certs_text}"
        f"GitHub: https://github.com/{name.lower().replace(' ', '')}"
    )

    return {
        "ID": idx,
        "Resume_str": resume_text,
        "Category": domain,
        "Name": name,
    }


def generate_synthetic_data(n: int = 500):
    """Generate n synthetic resume records and save as CSV."""
    logger.info(f"Generating {n} synthetic resume records …")
    records = [_random_resume(i) for i in range(1, n + 1)]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Resume_str", "Category", "Name"])
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Synthetic dataset saved to {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if OUTPUT_CSV.exists():
        logger.info(f"Dataset already exists at {OUTPUT_CSV} – skipping download.")
        return

    success = download_from_kaggle()
    if not success:
        generate_synthetic_data(500)


if __name__ == "__main__":
    main()
