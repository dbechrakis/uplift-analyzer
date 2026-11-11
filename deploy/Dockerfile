FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c /uv /usr/local/bin/uv

WORKDIR /build
COPY requirements.txt requirements-audit.txt ./

RUN uv venv /opt/venv \
 && uv pip sync --python /opt/venv/bin/python --require-hashes requirements.txt

RUN uv venv /opt/audit \
 && uv pip sync --python /opt/audit/bin/python --require-hashes requirements-audit.txt \
 && /opt/audit/bin/pip-audit --requirement requirements.txt --require-hashes --disable-pip --strict

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    UPLIFT_DATA_DIR=/data \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN apt-get update \
 && apt-get install -y --no-install-recommends --only-upgrade \
      apt=3.0.4 libapt-pkg7.0=3.0.4 zlib1g=1:1.3.dfsg+really1.3.1-1+deb13u1 \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
 && mkdir /data \
 && chown app:app /data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=root:root core ./core
COPY --chown=root:root app ./app

USER 10001:10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]

CMD ["python", "-m", "streamlit", "run", "app/main.py"]
