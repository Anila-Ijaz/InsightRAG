# Docker images

Supplementary Dockerfiles beyond the root `Dockerfile` (which is the production API image).

| File | Purpose |
|---|---|
| `Dockerfile.worker` | Celery worker for async ingestion. Same dependencies as the API but different entrypoint. |
| `Dockerfile.dev` | Development image with `ipdb`, `ipython`, hot-reload via `watchfiles`. Used with the override below. |
| `docker-compose.override.yml` | Activates hot-reload dev mode when copied to the project root. |

## Dev workflow

```bash
# Enable hot-reload dev mode
cp docker/docker-compose.override.yml docker-compose.override.yml

# `docker compose up` now uses Dockerfile.dev with src/ bind-mounted
docker compose up

# Edit any file in src/ — uvicorn reloads automatically
```

## Production workflow

Plain `docker compose up` (without the override) builds the root `Dockerfile` — the
multi-stage, non-root, healthchecked production image.

## Worker workflow

Once you have async ingestion wired through Celery (the API endpoint
currently runs ingestion synchronously — the worker image is ready for when you
switch it over):

```bash
docker build -f docker/Dockerfile.worker -t insightrag-worker .
docker run --env-file .env --network insightrag_default insightrag-worker
```
