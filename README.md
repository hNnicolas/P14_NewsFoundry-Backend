# 📰 NewsFoundry — Documentation Technique Backend

## 🎯 Objectif

Cette documentation décrit l’architecture, les choix techniques et le fonctionnement
du backend **NewsFoundry**.  
Son objectif est de **réduire la perte de contexte** lorsqu’un nouveau développeur
rejoint le projet.

---

## 🏗️ Architecture Générale

### Technologies principales

- **FastAPI** — API REST
- **SQLModel** — ORM + Pydantic
- **JWT** — Authentification stateless
- **PydanticAI** — Agent LLM avec tools
- **World News API** — Source d’actualités
- **Pytest** — Tests automatisés

---

## 🧠 Modèles de données

User
email (unique)
hashed_password
relation chats
Chat
user_id
title
messages (JSON mutable)
top_news_articles (JSON)
press_review_title / summary / articles
timestamps

👉 Les messages sont stockés en JSON pour faciliter l’injection directe
dans les prompts IA.

SystemPrompt

Stocke dynamiquement le prompt système utilisé par l’agent IA.

## 🔐 Authentification

L’authentification repose sur JWT.
Fonction clé

```bash
get_current_user()
```

Vérifie le header Authorization: Bearer <token>
Décode le JWT
Récupère l’utilisateur en base
Injecte user et token dans les routes et tools

Toutes les routes sensibles utilisent :

```bash
Depends(get_current_user)
```

## 💬 Gestion des conversations

Créer un chat

```bash
POST /chats
```

Crée une discussion
Appelle l’agent IA
Sauvegarde l’historique immédiatement

Lister les chats

```bash
GET /chats
```

Liste toutes les discussions de l’utilisateur
Tri par updated_at DESC

Récupérer un chat

```bash
GET /chats/{chat_id}
```

Sécurisé par utilisateur
Retourne l’historique complet
Ajouter un message

```bash
POST /chats/{chat_id}/messages
```

Ajoute le message utilisateur
L’agent décide automatiquement s’il doit appeler un tool
Sauvegarde la réponse IA

## 🤖 Agents IA & Tools

Agent principal
Basé sur PydanticAI

Modèle :
OpenAI (gpt-4o-mini) ou
HuggingFace (fallback)

Tool search_news_tool
Appelle l’endpoint interne /search-news
Utilise le token JWT de l’utilisateur
Retourne des articles structurés

👉 Le LLM décide quand appeler le tool, pas le backend.

## 🗞️ Actualités & Revue de presse

Top news

```bash
POST /top-news
```

Charge les actualités du jour
Crée automatiquement un chat
Stocke les articles bruts

Recherche d’articles

```bash
POST /search-news
```

Proxy sécurisé vers World News API
Simplifie le format pour le LLM

Génération de revue de presse

```bash
POST /chats/{chat_id}/generate-press-review
```

Utilise les articles déjà chargés
Génère :
titre
synthèse
liste d’articles
Persistance en base

Récupération

```bash
GET /chats/{chat_id}/press-review
```

1. Copier le fichier `.env.example` dans `.env`

2. Installer les dépendances:

```bash
uv sync
```

2. Démarrer la base de données:

```bash
docker run \
  --name newsfoundry_db \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=newsfoundry \
  -p 5432:5432 \
  postgres:17
```

## ▶️ Lancement du backend

1. Variables d’environnement

```bash
SECRET_KEY=...
OPENAI_API_KEY=...
WORLD_NEWS_API_KEY=...
```

2. Lancer le serveur

```bash
uvicorn src.main:app --reload
```
