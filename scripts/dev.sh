#!/usr/bin/env bash
set -e
docker compose up -d chroma redis
uv run uvicorn services.api.app:app --reload --port 8000

