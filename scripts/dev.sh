#!/usr/bin/env bash
set -e
docker compose up -d redis chroma
uv run uvicorn services.api.app:app --reload --port 8000

