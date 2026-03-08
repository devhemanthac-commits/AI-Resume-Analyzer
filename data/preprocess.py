"""
data/preprocess.py
------------------
NLTK-based text preprocessing pipeline for resume text.
"""

import re
import string
import logging
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK Resource Bootstrap
# ---------------------------------------------------------------------------

NLTK_RESOURCES = [
    "punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng", "maxent_ne_chunker", "maxent_ne_chunker_tab",
    "words", "omw-1.4",
]


def ensure_nltk_resources():
    """Download required NLTK data if not already present."""
    for resource in NLTK_RESOURCES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.data.find(f"taggers/{resource}")
                except LookupError:
                    try:
                        nltk.data.find(f"chunkers/{resource}")
                    except LookupError:
                        logger.info(f"Downloading NLTK resource: {resource}")
                        nltk.download(resource, quiet=True)


ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Regex Patterns
# ---------------------------------------------------------------------------

HACKATHON_PATTERN = re.compile(
    r"\b(hackathon|ideathon|datathon|code\s?sprint|code\s?jam|thon|makeathon|"
    r"smart india hackathon|mlh|nasa space apps|google solution challenge|"
    r"hacker(rank|earth)|flipkart grid|devfolio|ethglobal|icpc|ieee xtreme|"
    r"competition|contest|challenge)\b",
    re.IGNORECASE,
)

PROJECT_SECTION_PATTERN = re.compile(
    r"\b(project[s]?|portfolio|work\s?done|developed|built|implemented|"
    r"created|designed|deployed)\b",
    re.IGNORECASE,
)

GITHUB_PATTERN = re.compile(r"https?://github\.com/[\w\-]+", re.IGNORECASE)

CERT_PATTERN = re.compile(
    r"\b(aws|gcp|azure|google cloud|coursera|nptel|udemy|edx|oracle|cisco|"
    r"certified|certification|certificate|credential)\b",
    re.IGNORECASE,
)

EXPERIENCE_YEAR_PATTERN = re.compile(
    r"(\d{1,2})\+?\s*year[s]?\s*(of\s*)?(experience|exp|work\s*experience)",
    re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_PATTERN = re.compile(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# ---------------------------------------------------------------------------
# Preprocessor Class
# ---------------------------------------------------------------------------


class ResumePreprocessor:
    """Full NLTK preprocessing pipeline for resume text."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Keep domain-relevant stop words
        self.stop_words -= {"no", "not", "nor", "with", "for", "in", "at", "to"}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def preprocess(self, text: str) -> dict:
        """
        Run the full pipeline on raw resume text.

        Returns a dict with:
            - clean_text      : lowercased, lemmatized, stop-word-free string
            - tokens          : list of clean tokens
            - sentences       : list of original sentences
            - entities        : list of (entity_text, entity_type) tuples
            - has_hackathon   : bool
            - hackathon_count : int
            - has_projects    : bool
            - github_links    : list of GitHub URLs
            - has_certs       : bool
            - experience_years: int (best guess, 0 if not found)
        """
        text = self._sanitize(text)
        sentences = sent_tokenize(text)
        tokens_raw = word_tokenize(text)
        tokens_clean = self._clean_tokens(tokens_raw)
        entities = self._extract_entities(text)

        return {
            "clean_text": " ".join(tokens_clean),
            "tokens": tokens_clean,
            "sentences": sentences,
            "entities": entities,
            "has_hackathon": bool(HACKATHON_PATTERN.search(text)),
            "hackathon_count": len(HACKATHON_PATTERN.findall(text)),
            "has_projects": bool(PROJECT_SECTION_PATTERN.search(text)),
            "github_links": GITHUB_PATTERN.findall(text),
            "has_certs": bool(CERT_PATTERN.search(text)),
            "experience_years": self._extract_experience_years(text),
        }

    # ------------------------------------------------------------------ #
    # Internal Helpers
    # ------------------------------------------------------------------ #

    def _sanitize(self, text: str) -> str:
        """Remove PII noise (emails, phones) but keep structure."""
        text = EMAIL_PATTERN.sub(" EMAIL ", text)
        text = PHONE_PATTERN.sub(" PHONE ", text)
        # Remove non-GitHub URLs
        text = re.sub(r"https?://(?!github\.com)\S+", " URL ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _clean_tokens(self, tokens: list) -> list:
        """Lowercase, remove punctuation/stopwords, lemmatize."""
        clean = []
        for tok in tokens:
            tok_lower = tok.lower()
            if tok_lower in string.punctuation:
                continue
            if tok_lower in self.stop_words:
                continue
            if len(tok_lower) < 2:
                continue
            if re.match(r"^\d+$", tok_lower):
                continue
            lemma = self.lemmatizer.lemmatize(tok_lower)
            clean.append(lemma)
        return clean

    def _extract_entities(self, text: str) -> list:
        """Use NLTK NE chunker to extract named entities."""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            tree = ne_chunk(pos_tags, binary=False)
            entities = []
            for subtree in tree:
                if hasattr(subtree, "label"):
                    entity = " ".join(word for word, _ in subtree.leaves())
                    label = subtree.label()
                    entities.append((entity, label))
            return entities
        except Exception:
            return []

    def _extract_experience_years(self, text: str) -> int:
        """Extract the best estimate of years of experience."""
        matches = EXPERIENCE_YEAR_PATTERN.findall(text)
        if matches:
            return max(int(m[0]) for m in matches)
        return 0
