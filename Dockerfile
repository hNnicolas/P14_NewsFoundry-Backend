FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

# Rendre le dossier /app importable
ENV PYTHONPATH="/app"

# Installer les dépendances
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copier le code source
COPY src/ ./src/

EXPOSE 8000

# Lancer Uvicorn proprement
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
