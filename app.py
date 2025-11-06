"""Flask entrypoint for the AI Job Fit Analyzer application."""

from __future__ import annotations

import os
from typing import Optional

from flask import Flask, flash, redirect, render_template, request, url_for

from src.job_analyzer import analyze_job_fit


def create_app(test_config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get("FLASK_SECRET_KEY", "dev"))

    if test_config:
        app.config.update(test_config)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            resume_text = request.form.get("resume", "")
            job_text = request.form.get("job", "")

            try:
                result = analyze_job_fit(resume_text, job_text)
            except ValueError as exc:  # Input validation failure
                flash(str(exc), "error")
                return redirect(url_for("index"))

            return render_template("result.html", result=result)

        return render_template("index.html")

    @app.route("/health")
    def health() -> str:
        return "ok"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
