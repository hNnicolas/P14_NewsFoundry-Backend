import re
import unicodedata
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database import init_db, engine
from src.models import User, Chat, SystemPrompt
import jwt
import bcrypt
import uvicorn
import os
import json
from datetime import datetime
from pydantic import BaseModel, Field  
from pydantic_ai import Agent, RunContext, output
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
import inspect

# Charger dotenv uniquement si le fichier existe (local dev)
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

app = FastAPI()

# -------------------
# CORS
# -------------------
origins = [
    "http://localhost:3000",
    "https://p14-news-foundry-frontend.vercel.app",
    "https://p14-news-foundry-frontend.vercel.app/login",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# JWT config
# -------------------
SECRET_KEY = os.getenv("SECRET_KEY", "TEST_SECRET")
ALGORITHM = "HS256"

# -------------------
# DB session helper
# -------------------
def get_db():
    with Session(engine) as session:
        yield session

# -------------------
# Auth helper
# -------------------
def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant ou invalide")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Token invalide")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalide")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")

    return {
        "user": user,
        "token": token
    }


# -------------------
# Helpers: normalisation & détection affirmative
# -------------------
YES_WORDS = {
    "oui", "ouais", "si", "yes", "yep", "d'accord", "ok", "okey", "okay", "je veux", "oui s'il vous plaît", "oui stp", "oui svp", "bien sûr"
}

def normalize_text(s: str) -> str:
    """Normalise accents, casse, espaces et supprime ponctuation superflue."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    # replace multiple spaces
    s = re.sub(r"\s+", " ", s)
    # remove surrounding punctuation
    s = s.strip(" .!?;:,")
    return s

def is_affirmative(s: str) -> bool:
    n = normalize_text(s)
    # phrase contains explicit yes words
    for w in YES_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", n):
            return True
    # catch short forms: just "oui", "oui."
    if n in YES_WORDS:
        return True
    return False

def extract_article_index(s: str, max_index: int) -> Optional[int]:
    """
    Tente d'extraire un numéro d'article à partir du texte (ex. "le 2", "article 1", "n°3").
    Retourne un index (0-based) si trouvé et valide, sinon None.
    """
    n = normalize_text(s)
    # rechercher un nombre
    m = re.search(r"\b(?:n°|numero|numéro|article|le|la|l'|n|#)\s*([1-9][0-9]?)\b", n)
    if not m:
        # aussi détecter simple "2"
        m2 = re.search(r"\b([1-9][0-9]?)\b", n)
        if m2:
            val = int(m2.group(1))
        else:
            return None
    else:
        val = int(m.group(1))

    if 1 <= val <= max_index:
        return val - 1
    return None

# -------------------
# PydanticAI Agent Configuration
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "").strip()

if not MODEL_NAME:
    if OPENAI_API_KEY:
        MODEL_NAME = "openai:gpt-4o-mini"
    elif HF_TOKEN:
        MODEL_NAME = "huggingface:HuggingFaceH4/zephyr-7b-beta"
    else:
        raise RuntimeError(
            "❌ Aucune clé API configurée! Ajoutez OPENAI_API_KEY ou HF_TOKEN dans l'environnement Railway"
        )

# System prompt flexible et intelligent pour détecter toute demande de revue de presse
agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
        "Tu es l'assistant NewsFoundry, expert en actualités et revues de presse. "
        "Chaque fois que l'utilisateur demande des informations détaillées sur un sujet, "
        "une revue de presse ou des articles récents, tu dois automatiquement appeler le tool "
        "'advanced_search_news' avec le sujet exact. "
        "Sois flexible : l'utilisateur peut formuler sa demande de manière naturelle. "
        "Retourne toujours un résumé structuré : titre de la revue, synthèse générale, "
        "liste d'articles avec titre et résumé, éventuellement le lien vers la source. "
        "Si aucun sujet n'est clairement demandé, pose une question polie pour clarifier le sujet."
    )
)

# -------------------
# Clé API World News
# -------------------
WORLD_NEWS_API_KEY = os.getenv("WORLD_NEWS_API_KEY")
WORLD_NEWS_URL = "https://api.worldnewsapi.com/top-news"

# -------------------
# Agent Revue de Presse
# -------------------
class PressReviewOutputModel(BaseModel):
    title: str = Field(description="Titre de la revue de presse")
    summary: str = Field(description="Synthèse générale de la revue de presse")
    articles: list = Field(description="Liste des articles synthétisés, chaque article a title et summary")

press_review_agent = Agent(
    model=MODEL_NAME,
    system_prompt=(
        "Tu es un assistant qui génère une revue de presse en français à partir "
        "de l'historique de discussion. Structure la sortie comme un titre, une "
        "synthèse générale, et des synthèses pour chaque article."
    )
)

# -------------------
# Routes simples
# -------------------
@app.get("/")
def hello():
    return {"message": "👋 API NewsFoundry fonctionne !", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

# -------------------
# Login
# -------------------
@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email et mot de passe requis")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user or not bcrypt.checkpw(password.encode(), user.hashed_password.encode()):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    token = jwt.encode({"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token, "user": {"id": user.id, "email": user.email}}

# -------------------
# Créer un chat
# -------------------
@app.post("/chats")
def create_chat(
    payload: dict = {},
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):    
    user_message = payload.get("message", "").strip()
    assistant_message = payload.get("assistant_message", "").strip()

    if not user_message and not assistant_message:
        raise HTTPException(status_code=400, detail="Aucun message fourni")

    messages = []

    if user_message:
        messages.append({"role": "user", "content": user_message})

    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})

    chat = Chat(
        user_id=current_user.id,
        title=payload.get("title", "Nouvelle conversation"),
        messages=messages,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    # Récupérer le system prompt
    system_prompt_obj = db.exec(select(SystemPrompt)).first()
    system_prompt_text = system_prompt_obj.prompt_text if system_prompt_obj else "Tu es l'assistant NewsFoundry."

    # Construire le prompt pour l'agent
    combined_prompt = f"{system_prompt_text}\n\n{user_message}" if user_message else assistant_message

    # Appel agent
    try:
        result = agent.run_sync(user_prompt=combined_prompt)
        assistant_response = getattr(result, "data", getattr(result, "output", str(result)))
        if not isinstance(assistant_response, str):
            assistant_response = str(assistant_response)
    except Exception as e:
        assistant_response = (
            "⚠️ Je n'ai pas pu récupérer les actualités pour le moment, "
            "mais la conversation a bien été créée. Tu peux continuer à discuter normalement."
        )
        print("❌ ERREUR IA lors de la création du chat :", str(e))

    # Ajouter la réponse de l'assistant au chat
    if assistant_response:
        chat.messages.append({"role": "assistant", "content": assistant_response})
        chat.updated_at = datetime.utcnow()

        try:
            db.add(chat)
            db.commit()
            db.refresh(chat)
        except Exception as e_db:
            db.rollback()
            print("❌ ERREUR DB lors de la sauvegarde chat :", str(e_db))
            raise HTTPException(status_code=500, detail=f"Erreur DB: {str(e_db)}")

    return {
        "chat_id": chat.id,
        "assistant_response": assistant_response,
        "messages": chat.messages
    }

# -------------------
# Ajouter un message via l'agent (avec tools)
# -------------------
@app.post("/chats/{chat_id}/messages")
def add_message(
    chat_id: int,
    payload: dict,
    auth = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ajoute un message utilisateur dans un chat existant.
    L'agent décide seul s'il doit appeler un tool (ex: search_news_tool).
    """

    # ============================
    # Auth
    # ============================
    user = auth["user"]
    token = auth["token"]

    print("🔐 [AUTH] User :", user.email)
    print("🔐 [AUTH] Token présent :", bool(token))

    # ============================
    # Message utilisateur
    # ============================
    message_content = payload.get("message", "").strip()
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")

    print("💬 [USER MESSAGE] :", message_content)

    # ============================
    # Récupérer le chat
    # ============================
    chat = db.exec(
        select(Chat).where(
            (Chat.id == chat_id) &
            (Chat.user_id == user.id)
        )
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    messages = chat.messages or []

    # Ajouter message utilisateur
    messages.append({
        "role": "user",
        "content": message_content
    })

    # ============================
    # Charger le system prompt
    # ============================
    system_prompt_obj = db.exec(select(SystemPrompt)).first()
    system_prompt_text = system_prompt_obj.prompt_text if system_prompt_obj else (
        "Tu es l’assistant NewsFoundry.\n\n"
        "Quand l’utilisateur demande :\n"
        "- plus d’informations\n"
        "- des détails supplémentaires\n"
        "- approfondir un sujet\n"
        "- des articles récents\n\n"
        "Tu DOIS appeler le tool `search_news_tool` avec le sujet précis.\n\n"
        "Après l’appel du tool :\n"
        "- base ta réponse uniquement sur les articles retournés\n"
        "- fais une synthèse claire et structurée\n"
        "- cite les titres des articles\n"
        "- réponds en français\n"
    )

    # ============================
    # Construire le prompt
    # ============================
    conversation = [
        {"role": "system", "content": system_prompt_text},
        *messages
    ]

    raw_prompt = "\n".join(
        f"{m['role']}: {m['content']}" for m in conversation
    )

    print("🧠 [AGENT] Prompt envoyé au LLM ↓↓↓")
    print(raw_prompt)

    # ============================
    # Appel de l'agent (AVEC TOOLS)
    # ============================
    try:
        result = agent.run_sync(
            user_prompt=raw_prompt,
            deps={
                "token": token  # ✅ JWT réel
            }
        )

        print("🧠 [AGENT] Result type :", type(result))
        print("🧠 [AGENT] Result brut :", result)

        assistant_content = (
            result.data
            if hasattr(result, "data")
            else result.output
            if hasattr(result, "output")
            else str(result)
        )

        print("🤖 [AGENT] Réponse assistant générée")

    except Exception as e:
        print("❌ [AGENT ERROR] :", e)
        assistant_content = (
            "⚠️ Je n’ai pas pu traiter votre demande pour le moment. "
            "Merci de réessayer."
        )

    # ============================
    # Sauvegarde réponse assistant
    # ============================
    messages.append({
        "role": "assistant",
        "content": assistant_content
    })

    try:
        chat.messages = json.loads(json.dumps(messages))
        chat.updated_at = datetime.utcnow()
        db.add(chat)
        db.commit()
        db.refresh(chat)

        print("💾 [DB] Message sauvegardé")

    except Exception as e_db:
        db.rollback()
        print("❌ [DB ERROR] :", e_db)
        raise HTTPException(
            status_code=500,
            detail="Erreur lors de la sauvegarde du message"
        )

    return {
        "assistant_response": assistant_content,
        "messages": chat.messages
    }


@app.post("/search-news")
def search_news(
    payload: dict = Body(...),
    current_user: User = Depends(get_current_user)
):
    query = payload.get("query", "").strip()
    print("🔎 [SEARCH-NEWS] Query reçue :", query)
    if not query:
        raise HTTPException(400, "Query manquante")

    params = {
        "text": query,
        "language": "fr",
        "number": 5
    }

    try:
        print("🌍 [SEARCH-NEWS] Appel World News API")
        resp = requests.get(
            "https://api.worldnewsapi.com/search-news",
            headers={"x-api-key": WORLD_NEWS_API_KEY},
            params=params,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        print("✅ [SEARCH-NEWS] Réponse API OK")
    except Exception as e:
        raise HTTPException(502, f"Erreur World News API: {str(e)}")

    articles = [
        {
            "title": a.get("title"),
            "summary": a.get("summary") or a.get("text", "")[:400],
            "url": a.get("url"),
            "date": a.get("publish_date")
        }
        for a in data.get("news", [])
    ]

    return {
        "query": query,
        "count": len(articles),
        "articles": articles
    }

@agent.tool
def search_news_tool(ctx: RunContext, query: str) -> dict:
    """
    Recherche des articles récents via l’API interne /search-news
    """
    print("🔧 [TOOL] search_news_tool CALLED")
    print("🔍 [TOOL] Query :", query)
    print("🔐 [TOOL] Token présent :", bool(ctx.deps.get("token")))
    try:
        resp = requests.post(
            "http://localhost:8000/search-news",
            headers={
                "Authorization": f"Bearer {ctx.deps['token']}"
            },
            json={"query": query},
            timeout=10
        )
        print("📡 [TOOL] Status code :", resp.status_code)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {
            "error": str(e),
            "articles": [],
            "count": 0
        }


@app.post("/chats/{chat_id}/generate-press-review")
def generate_press_review_no_tool(
    chat_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    theme = payload.get("theme")
    if not theme:
        raise HTTPException(
            status_code=400,
            detail="Un thème est requis pour générer la revue de presse."
        )

    chat = db.exec(
        select(Chat).where(
            (Chat.id == chat_id) &
            (Chat.user_id == current_user.id)
        )
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat introuvable")

    # --- Articles récupérés (RAG / API news) ---
    articles_data = chat.top_news_articles or []

    # Séparation articles
    main_articles = articles_data[:10]
    additional_articles = articles_data[10:]

    def normalize_article(a: dict) -> dict:
        return {
            "title": a.get("title", "Titre indisponible"),
            "summary": a.get("summary") or a.get("content") or "",
            "url": a.get("url", "")
        }

    main_articles_formatted = [normalize_article(a) for a in main_articles]
    additional_articles_formatted = [
        normalize_article(a) for a in additional_articles
    ]

    # Résumé global
    global_summary = (
        f"Cette revue de presse présente une synthèse des actualités "
        f"liées au thème « {theme} ». "
        f"{len(main_articles_formatted)} articles principaux ont été analysés."
    )

    review_result = {
        "title": f"Revue de presse — {theme}",
        "summary": global_summary,
        "articles": main_articles_formatted,
        "additional_articles": additional_articles_formatted
    }

    # --- Sauvegarde ---
    chat.press_review_title = review_result["title"]
    chat.press_review_summary = review_result["summary"]
    chat.press_review_articles = review_result["articles"]
    chat.updated_at = datetime.utcnow()

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "message": "Revue de presse générée avec succès.",
        "review": review_result
    }

# -------------------
# Endpoint pour récupérer les actualités et mettre à jour le prompt système
# -------------------

@app.post("/top-news")
def get_top_news(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    print("🔎 [top-news] Requête reçue :", payload)

    # ============================
    # Vérification de la clé API
    # ============================
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(500, "Clé API World News non configurée")

    user_message = payload.get("user_message", "").strip()
    if not user_message:
        raise HTTPException(400, "Aucun message utilisateur fourni")
    print(f"👤 [top-news] Message utilisateur : {user_message}")

    # ============================
    # Fonction utilitaire pour récupérer les articles
    # ============================
    def fetch_top_news(country="fr", language="fr", date=None):
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        params = {
            "source-country": country,
            "language": language,
            "date": date,
            "max-news-per-cluster": 5
        }
        print(f"🌍 [top-news] Appel API World News: country={country}, date={date}")
        try:
            resp = requests.get(
                WORLD_NEWS_URL,
                headers={"x-api-key": WORLD_NEWS_API_KEY},
                params=params,
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print("[top-news] Erreur API :", e)
            return []

        clusters = data.get("top_news", [])
        articles = []
        for cluster in clusters:
            for a in cluster.get("news", []):
                if not a.get("title"):
                    continue
                articles.append({
                    "title": a.get("title", "").strip(),
                    "summary": a.get("summary") or a.get("text") or "",
                    "url": a.get("url", ""),
                    "publish_date": a.get("publish_date", "")
                })
        return articles

    # ============================
    # Récupération des articles
    # ============================
    articles = fetch_top_news("fr", "fr")
    if not articles:
        # fallback US
        print("⚠️ Aucun article FR — fallback US")
        articles = fetch_top_news("us", "en")

    if not articles:
        # fallback global fictif si rien trouvé
        articles = [{
            "title": "Aucune actualité disponible",
            "summary": "Impossible de récupérer des articles à cette date.",
            "url": "",
            "publish_date": datetime.utcnow().strftime("%Y-%m-%d")
        }]

    # ============================
    # Génération du message assistant
    # ============================
    list_preview = "\n".join([f"- {a['title']}" for a in articles[:5]])
    assistant_message = (
        f"Voici les dernières actualités du jour :\n\n"
        f"{list_preview}\n\n"
        "Souhaitez-vous une revue de presse détaillée sur un sujet précis ?"
    )
    print("🤖 [top-news] Assistant message généré.")
    
    # ============================
    # CRÉATION DU CHAT ICI
    # ============================
    now = datetime.utcnow()

    chat = Chat(
        user_id=current_user.id,
        title="Discussion du",
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ],
        top_news_articles=articles,
        created_at=now,
        updated_at=now
    )

    db.add(chat)
    db.commit()
    db.refresh(chat)

    # ============================
    # Réponse API
    # ============================
    return {
        "status": "success",
        "assistant_message": assistant_message,
        "articles_count": len(articles),
        "articles": articles,
        "chat_id": chat.id if chat else None,
        "updated_at": now.isoformat()
    }
    
@app.get("/chats/{chat_id}/press-review")
def get_press_review(
    chat_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    chat = db.exec(
        select(Chat).where(
            (Chat.id == chat_id) &
            (Chat.user_id == current_user.id)
        )
    ).first()

    if not chat or not chat.press_review_articles:
        raise HTTPException(status_code=404, detail="Revue introuvable")

    return {
        "title": chat.press_review_title,
        "summary": chat.press_review_summary,
        "articles": chat.press_review_articles
    }
    
# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
