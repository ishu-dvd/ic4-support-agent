# Single-container image: fixture upstream API (port 8081, in-process background) + agent HTTP layer
# (uvicorn, agent.serve:app on $PORT). scripts/start.sh wires the two together.
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    UPSTREAM_BASE_URL=http://127.0.0.1:8081 \
    RUNS_DIR=/data/runs \
    TRACE_STDOUT=true

# Serving deps only (fastapi/uvicorn/pydantic); the agent core is stdlib.
COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code + fixture data + docs the dashboard/README link to.
# (.dockerignore keeps .venv, runs/, .env*, tests and notes out of the build context.)
COPY agent/ ./agent/
COPY server/ ./server/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY docs/API.md ./docs/API.md
COPY specs/ ./specs/
COPY Makefile README.md SOLUTION.md ./
# dashboard/ is optional (static assets may be served from agent/ instead). COPY fails on a missing
# source, so stage the (dockerignore-filtered) context and copy the directory only if present.
COPY . /tmp/src/
RUN if [ -d /tmp/src/dashboard ]; then cp -r /tmp/src/dashboard ./dashboard; fi && rm -rf /tmp/src

# Non-root user; RUNS_DIR must be writable (App Platform has no persistent disk, so this resets on deploy).
RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /data/runs \
    && chmod +x /app/scripts/start.sh \
    && chown -R app:app /app /data

USER app

EXPOSE 8080

# curl is not in python:slim; probe /healthz with stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8080'), timeout=4).status == 200 else 1)"

CMD ["/app/scripts/start.sh"]
