FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    MEDICAL_PROJECT_ROOT=/app \
    MEDICAL_STORAGE_ROOT=/app/.runtime \
    MEDICAL_API_URL=http://127.0.0.1:8000

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md requirements-phase*.txt ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY streamlit_app.py ./streamlit_app.py
COPY models/phase2 ./models/phase2
COPY models/phase3/text_cnn.pt ./models/phase3/text_cnn.pt
COPY models/phase4/bert/model.pt ./models/phase4/bert/model.pt
COPY models/phase4/bert/tokenizer ./models/phase4/bert/tokenizer
COPY reports/phase4/bert/model_comparison.csv ./reports/phase4/bert/model_comparison.csv
COPY data/examples ./data/examples

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    && python -m pip install -e ".[ml,dl,nlp,web]" -c requirements-phase5.txt \
    && python scripts/fetch_phase4_models.py

RUN useradd --create-home --uid 1000 appuser && mkdir -p /app/.runtime /app/reports/phase5/logs && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --start-period=60s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4)"
CMD ["python", "scripts/run_app.py", "--frontend-host", "0.0.0.0", "--frontend-port", "7860", "--api-host", "0.0.0.0"]
