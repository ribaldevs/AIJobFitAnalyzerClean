"""Core text analysis helpers for the job fit analyzer web application."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

# A compact list of stop words tailored for resume analysis to avoid heavy dependencies.
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "you",
    "your",
}


def _normalize(text: str) -> str:
    """Lowercase the text and collapse whitespace."""

    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def _tokenize(text: str) -> List[str]:
    """Tokenize the text into words while filtering stop words and short tokens."""

    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    filtered = [token for token in tokens if len(token) > 2 and token not in _STOP_WORDS]
    return filtered


def _top_keywords(tokens: Iterable[str], limit: int = 12) -> List[str]:
    """Return the top keywords ordered by frequency descending."""

    counter = Counter(tokens)
    most_common = [keyword for keyword, _ in counter.most_common(limit)]
    return most_common


def _term_frequencies(tokens: List[str]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = float(sum(counts.values())) or 1.0
    return {term: count / total for term, count in counts.items()}


def _inverse_document_frequencies(documents: List[List[str]]) -> Dict[str, float]:
    document_count = len(documents)
    doc_freq: Counter[str] = Counter()
    for tokens in documents:
        doc_freq.update(set(tokens))

    # Apply smoothing to avoid division by zero and keep numbers stable for tiny corpora.
    return {
        term: math.log((document_count + 1) / (freq + 1)) + 1.0
        for term, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = _term_frequencies(tokens)
    return {term: tf_value * idf.get(term, 0.0) for term, tf_value in tf.items()}


def _cosine_similarity(resume_tokens: List[str], job_tokens: List[str]) -> float:
    """Compute cosine similarity between two token lists using a minimal TF-IDF approach."""

    if not resume_tokens or not job_tokens:
        return 0.0

    idf = _inverse_document_frequencies([resume_tokens, job_tokens])
    resume_vector = _tfidf_vector(resume_tokens, idf)
    job_vector = _tfidf_vector(job_tokens, idf)

    numerator = sum(resume_vector.get(term, 0.0) * weight for term, weight in job_vector.items())
    resume_norm = math.sqrt(sum(value * value for value in resume_vector.values()))
    job_norm = math.sqrt(sum(value * value for value in job_vector.values()))

    if resume_norm == 0 or job_norm == 0:
        return 0.0

    return numerator / (resume_norm * job_norm)


@dataclass
class JobFitResult:
    """Result summary of the resume versus job description comparison."""

    similarity: float
    resume_keywords: List[str]
    job_keywords: List[str]
    missing_keywords: List[str]
    extra_keywords: List[str]

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON serializable dictionary representation."""

        return {
            "similarity": self.similarity,
            "resume_keywords": self.resume_keywords,
            "job_keywords": self.job_keywords,
            "missing_keywords": self.missing_keywords,
            "extra_keywords": self.extra_keywords,
        }


def analyze_job_fit(resume_text: str, job_text: str) -> JobFitResult:
    """Analyze how well a resume aligns with a job description."""

    if not resume_text.strip():
        raise ValueError("Resume text must not be empty.")
    if not job_text.strip():
        raise ValueError("Job description text must not be empty.")

    normalized_resume = _normalize(resume_text)
    normalized_job = _normalize(job_text)

    resume_tokens = _tokenize(normalized_resume)
    job_tokens = _tokenize(normalized_job)

    resume_keywords = _top_keywords(resume_tokens)
    job_keywords = _top_keywords(job_tokens)

    missing = [kw for kw in job_keywords if kw not in resume_keywords]
    extra = [kw for kw in resume_keywords if kw not in job_keywords]

    similarity_score = _cosine_similarity(resume_tokens, job_tokens)

    return JobFitResult(
        similarity=round(similarity_score * 100, 1),
        resume_keywords=resume_keywords,
        job_keywords=job_keywords,
        missing_keywords=missing,
        extra_keywords=extra,
    )


__all__ = ["JobFitResult", "analyze_job_fit"]
