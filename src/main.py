import re
import unicodedata
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database import init_db, engine
from src.models import User, Chat, SystemPrompt
import jwt
import bcrypt
import uvicorn
import os
from datetime import datetime
from pydantic import BaseModel, Field  
from pydantic_ai import Agent, RunContext, output

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
    return user

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

agent = Agent(
    model=MODEL_NAME,
    system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
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
# Endpoint FastAPI pour générer la revue de presse
# -------------------
@app.post("/press-review", response_model=PressReviewOutputModel)
def generate_press_review(prompt: str, user=Depends(get_current_user)):
    # On utilise agent.call() avec parse_with pour obtenir un objet Pydantic
    response = press_review_agent.call(
        prompt=prompt,
        parse_with=PressReviewOutputModel
    )
    return response

# -------------------
# Tool : recherche d’articles
# -------------------
@agent.tool
def search_news(context: RunContext = None, query: str = "") -> str:
    """Retourne les articles récents. Filtre par mot-clé si query fourni."""
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News manquante")

    try:
        r = requests.get(
            WORLD_NEWS_URL,
            params={
                "apiKey": WORLD_NEWS_API_KEY,
                "lang": "en",
                "pageSize": 50
            }
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur World News API: {e}")

    articles = data.get("articles", [])
    if query:
        query_lower = query.lower()
        articles = [
            a for a in articles
            if query_lower in (a.get("title") or "").lower() or
               query_lower in (a.get("description") or "").lower()
        ]

    if not articles:
        return "Aucun article trouvé."

    return "\n".join(
        f"- {a.get('title','')} : {a.get('description','')}"
        for a in articles[:10]
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
def create_chat(payload: dict = {}, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    user_message = payload.get("message", "")
    if not user_message:
        raise HTTPException(status_code=400, detail="Message requis")

    chat = Chat(
        user_id=current_user.id,
        title="Nouvelle conversation",
        messages=[{"role": "user", "content": user_message}],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    system_prompt = db.exec(select(SystemPrompt)).first()
    full_prompt = system_prompt.prompt_text if system_prompt else "Tu es l'assistant NewsFoundry."

    try:
        result = agent.run_sync(user_message, system_prompt=full_prompt)
        assistant_response = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

    chat.messages.append({"role": "assistant", "content": assistant_response})

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {"chat_id": chat.id, "assistant_response": assistant_response, "messages": chat.messages}

# -------------------
# Lister tous les chats
# -------------------
@app.get("/chats")
def list_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = db.exec(
        select(Chat).where(Chat.user_id == current_user.id).order_by(Chat.updated_at.desc())
    ).all()
    return [
        {
            "chat_id": chat.id,
            "title": chat.title,
            "messages": chat.messages,
            "created_at": chat.created_at.isoformat() if chat.created_at else None,
            "updated_at": chat.updated_at.isoformat() if chat.updated_at else None
        }
        for chat in chats
    ]

# -------------------
# Récupérer un chat
# -------------------
@app.get("/chats/{chat_id}")
def get_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.exec(select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")
    return {
        "chat_id": chat.id,
        "title": chat.title,
        "messages": chat.messages,
        "created_at": chat.created_at.isoformat() if chat.created_at else None,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None
    }


# -------------------
# Ajouter un message
# -------------------
@app.post("/chats/{chat_id}/messages")
def add_message(
    chat_id: int,
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    message_content = payload.get("message", "")
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")

    chat = db.exec(
        select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    chat.messages.append({"role": "user", "content": message_content})

    # Récupérer le prompt système actualisé
    system_prompt = db.exec(select(SystemPrompt)).first()
    system_prompt_text = (
        system_prompt.prompt_text if system_prompt
        else "Tu es l'assistant NewsFoundry."
    )

    # ---------------------------
    # Détection du mot-clé 
    # ---------------------------
    if is_affirmative(message_content):
        idx = extract_article_index(message_content, 3)
        return generate_detailed_press_review(chat, db, article_index=idx)

    # ---------------------------
    # Réponse normale du chat
    # ---------------------------
    try:
        result = agent.run_sync(
            message_content,
            system_prompt=system_prompt_text
        )
        assistant_response = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

    chat.messages.append({"role": "assistant", "content": assistant_response})
    chat.updated_at = datetime.utcnow()
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "assistant_response": assistant_response,
        "messages": chat.messages
    }

# -------------------
# Génération revue de presse détaillée
# -------------------
def generate_detailed_press_review(chat: Chat, db: Session, article_index: Optional[int] = None):
    """
    Utilise le SystemPrompt sauvegardé pour produire une revue de presse détaillée.
    Si article_index est fourni (0-based), on demande le détail pour cet article.
    """
    system_prompt = db.exec(select(SystemPrompt)).first()
    if not system_prompt or not system_prompt.prompt_text:
        raise HTTPException(status_code=400, detail="Aucune actualité disponible pour générer la revue de presse.")

    # Si l'utilisateur demande un article précis, on construit une instruction qui cible cet article.
    if article_index is not None:
        instruction = (
            f"Donne le détail complet de l'article n°{article_index + 1} parmi les sujets listés "
            "dans ce prompt (titre, résumé détaillé, source si disponible, et 3 points clés)."
        )
    else:
        instruction = (
            "Génère une revue de presse détaillée : pour chaque sujet listé dans le prompt, fournis "
            "un titre, un court chapeau, puis 3 éléments clés et un lien ou source si disponible."
        )

    try:
        result = agent.run_sync(
            instruction,
            system_prompt=system_prompt.prompt_text
        )
        detailed_review = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA lors de la revue de presse: {str(e)}")

    # Sauvegarde dans le chat (on ajoute un message assistant)
    chat.messages.append({"role": "assistant", "content": detailed_review})
    chat.updated_at = datetime.utcnow()

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "assistant_response": detailed_review,
        "messages": chat.messages
    }


# -------------------
# Endpoint pour récupérer les actualités et mettre à jour le prompt système
# -------------------

@app.get("/top-news")
def get_top_news(db: Session = Depends(get_db)):
    print("[INFO] Début de la récupération des actualités")
    
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News non configurée")

    params = {
        "source-country": "fr",        
        "language": "fr",
        "date": "2025-12-07"  # optionnel, tu peux utiliser la date du jour si nécessaire
    }
    print(f"[DEBUG] Params envoyés à l'API: {params}")

    try:
        response = requests.get(
            WORLD_NEWS_URL,
            headers={"x-api-key": WORLD_NEWS_API_KEY},  # ✅ header obligatoire
            params=params
        )
        print(f"[DEBUG] URL finale appelée: {response.url}")
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])[:10]  
        if not articles:
            raise HTTPException(status_code=404, detail="Aucun article trouvé")
        print(f"[INFO] Nombre d'articles reçus: {len(articles)}")
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Erreur HTTP: {e}")
        print(f"[DEBUG] Contenu de la réponse: {e.response.text if e.response else 'No response'}")
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")

    text_to_summarize = "\n".join([f"- {a.get('title','')} : {a.get('description','')}" for a in articles])
    
    try:
        result = agent.run_sync(
            f"Résume ces articles politiques :\n{text_to_summarize}",
            system_prompt="Résume chaque article en une phrase courte en français."
        )
        summarized_articles = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur IA: {str(e_agent)}")

    final_prompt = (
        "Voici un résumé des dernières nouvelles politiques :\n\n"
        f"{summarized_articles}\n\n"
        "Souhaitez-vous que je génère une revue de presse détaillée sur l'un des sujets ?"
    )

    now = datetime.utcnow()
    system_prompt = db.exec(select(SystemPrompt)).first()
    if system_prompt:
        system_prompt.prompt_text = final_prompt
        system_prompt.updated_at = now
    else:
        system_prompt = SystemPrompt(prompt_text=final_prompt, updated_at=now)
        db.add(system_prompt)
    db.commit()
    db.refresh(system_prompt)

    return {
        "message": "Actualités politiques mises à jour",
        "system_prompt_preview": final_prompt,
        "updated_at": now.isoformat()
    }


@app.get("/search-news")
def search_news(query: str, db: Session = Depends(get_db)):
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News non configurée")
    
    try:
        response = requests.get(
            "https://api.worldnewsapi.com/search-news",
            params={
                "apiKey": WORLD_NEWS_API_KEY,
                "q": query,
                "lang": "en", 
                "sortBy": "relevance",
                "pageSize": 10
            }
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")
    
    # Extraire titre + résumé pour faciliter lecture par le LLM
    articles = data.get("articles", [])[:10]
    news_summary = [
        {"title": a.get("title", ""), "description": a.get("description", ""), "url": a.get("url", "")}
        for a in articles
    ]
    
    return {"query": query, "articles": news_summary}

@app.post("/chats/{chat_id}/press-review")
def generate_press_review(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Récupérer le chat
    chat = db.exec(select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    # Construire le texte à synthétiser à partir des messages
    chat_history_text = "\n".join(f"{m['role']}: {m['content']}" for m in chat.messages)

    # Générer la revue de presse
    try:
        result = press_review_agent.run_sync(chat_history_text)
        review: PressReviewOutput = result.data  # correspond à notre Output model
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur génération revue de presse: {str(e_agent)}")

    # Stocker dans le chat
    chat.press_review_title = review.title
    chat.press_review_summary = review.summary
    chat.press_review_articles = review.articles
    chat.updated_at = datetime.utcnow()

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "chat_id": chat.id,
        "press_review_title": chat.press_review_title,
        "press_review_summary": chat.press_review_summary,
        "press_review_articles": chat.press_review_articles
    }


# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
