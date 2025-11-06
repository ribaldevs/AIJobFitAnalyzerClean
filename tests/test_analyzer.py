"""Tests for the job analyzer module."""

from src.job_analyzer.analyzer import JobFitResult, analyze_job_fit


def test_analyze_job_fit_computes_similarity_and_keywords():
    resume = """Experienced data analyst skilled in Python, SQL, and building dashboards.
    Developed machine learning models and collaborated with product teams."""

    job = """Looking for a data analyst with strong SQL abilities, dashboard design expertise,
    and experience collaborating with product managers."""

    result = analyze_job_fit(resume, job)

    assert isinstance(result, JobFitResult)
    assert 0 <= result.similarity <= 100
    assert "sql" in result.job_keywords
    assert "python" in result.resume_keywords


def test_analyze_job_fit_detects_missing_keywords():
    resume = """Project manager with agile delivery experience and strong communication skills."""
    job = """Senior project manager familiar with agile, risk management, budgeting, and stakeholder communication."""

    result = analyze_job_fit(resume, job)

    assert "budgeting" in result.missing_keywords
    assert "agile" in result.resume_keywords


def test_analyze_job_fit_validates_inputs():
    job = "Job description"

    try:
        analyze_job_fit("", job)
    except ValueError as exc:
        assert "Resume" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty resume")

    try:
        analyze_job_fit("resume", " ")
    except ValueError as exc:
        assert "Job description" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty job description")
