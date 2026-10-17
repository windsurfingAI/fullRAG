# ==========================================
# ÉTAPE 1 : BASE (Commune à la Prod et au Dev)
# ==========================================
FROM python:3.11-slim AS base
WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python (Production)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download fr_core_news_md

# Pré-téléchargement du modèle de reranking dans le cache de l'image
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"


# ==========================================
# ÉTAPE 2 : PRODUCTION (L'image pour le serveur)
# ==========================================
FROM base AS prod
# On copie uniquement le code source
COPY src/ .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# ==========================================
# ÉTAPE 3 : DÉVELOPPEMENT (Pour coder en local)
# ==========================================
FROM base AS dev
# On installe pytest UNIQUEMENT pour cette version
RUN pip install --no-cache-dir pytest

# On copie le code et les tests (sécurité au cas où on n'utilise pas de volumes)
COPY src/ .
COPY tests/ ./tests/

EXPOSE 8000
# Ajout de --reload pour recharger automatiquement Uvicorn à chaque sauvegarde
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]