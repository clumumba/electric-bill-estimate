FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    MODEL_DIR=/models \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

COPY requirements.txt ./requirements.txt
RUN pip install --only-binary=:all: -r requirements.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir /models && chown app:app /models
COPY --chown=app:app energy_mlops/ ./energy_mlops/
COPY --chown=app:app params.yaml ./params.yaml
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"
CMD ["uvicorn", "energy_mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "30", "--no-proxy-headers"]
