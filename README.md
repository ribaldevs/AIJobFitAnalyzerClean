# AI Job Fit Analyzer

A lightweight web application for comparing a resume with a job description. Paste both texts to receive a similarity score, identify overlapping keywords, and highlight potential gaps to address.

## Features

- **Compatibility score** powered by a lightweight TF-IDF cosine similarity implementation.
- **Keyword extraction** that surfaces the prominent concepts in each document.
- **Gap analysis** highlighting keywords present in the job description but missing from the resume, along with unique resume strengths.
- **Web interface** built with Flask for quick experimentation.

## Getting started

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:

   ```bash
   flask --app app run --debug
   ```

4. Open <http://localhost:5000> and paste your resume and the job description to analyze the fit.

## Testing

Run unit tests with:

```bash
pytest
```

## Project structure

```
├── app.py               # Flask entrypoint
├── requirements.txt     # Python dependencies
├── src/
│   └── job_analyzer/
│       ├── __init__.py
│       └── analyzer.py  # Core text analysis logic
├── templates/           # HTML templates
└── static/              # Stylesheet assets
```
