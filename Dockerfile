FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache fastapi uvicorn faiss-cpu numpy ijson uvloop httptools

# Build the Faiss index at image-build time.
# The host has no memory limit, so JSON parsing + training (~500 MB peak) is fine.
# The resulting binary files are ~25 MB total; the 284 MB JSON is removed afterwards.
COPY resources/references.json.gz ./resources/
COPY scripts/build_index.py ./scripts/
RUN python scripts/build_index.py && rm resources/references.json.gz

COPY src/ ./src/

EXPOSE 8080

CMD ["uvicorn", "src.main:main_app", "--host", "0.0.0.0", "--port", "8080", "--loop", "uvloop", "--http", "httptools", "--no-access-log"]
