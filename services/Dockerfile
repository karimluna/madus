FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir uv && uv sync --no-dev --no-extra databricks --no-extra colpali

COPY . .

CMD ["uv", "run", "uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# here we exclude databricks for the github repo
