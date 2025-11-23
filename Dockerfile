FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copier le code source
COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "python", "src/main.py"]
