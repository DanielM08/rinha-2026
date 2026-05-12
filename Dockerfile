FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache fastapi uvicorn

COPY src/ ./src/

EXPOSE 8080

CMD ["uvicorn", "src.main:main_app", "--host", "0.0.0.0", "--port", "8080"]
