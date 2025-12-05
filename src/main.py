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

# Charger dotenv uniquement si le fichier existe (local dev)
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

from pydantic_ai import Agent, Tool

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

try:
    agent = Agent(
        model=MODEL_NAME,
        system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
    )
except Exception as e:
    print(f"❌ ERREUR d'initialisation de l'agent: {e}")
    raise

# -------------------
# Clé API World News
# -------------------
WORLD_NEWS_API_KEY = os.getenv("WORLD_NEWS_API_KEY")
WORLD_NEWS_URL = "https://api.worldnewsapi.com/top-news"

# -------------------
# Tools : recherche d'articles
# -------------------
def search_news_tool(query: str) -> str:
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
                "pageSize": 5
            }
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")

    articles = data.get("articles", [])
    if not articles:
        return "Aucun article trouvé pour ce sujet."
    
    return "\n".join([f"- {a.get('title','')} : {a.get('description','')}" for a in articles])

search_tool = Tool.from_function(
    search_news_tool,
    name="search_news",
    description="Permet de rechercher des articles sur un sujet spécifique."
)

agent.add_tools([search_tool])

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

    # Inclure prompt système actualités
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

    # Inclure prompt système actualités
    system_prompt = db.exec(select(SystemPrompt)).first()
    full_prompt = system_prompt.prompt_text if system_prompt else "Tu es l'assistant NewsFoundry."

    try:
        result = agent.run_sync(message_content, system_prompt=full_prompt)
        assistant_response = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

    chat.messages.append({"role": "assistant", "content": assistant_response})
    chat.updated_at = datetime.utcnow()
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {"assistant_response": assistant_response, "messages": chat.messages}

# -------------------
# Endpoint pour récupérer les actualités et mettre à jour le prompt système
# -------------------

@app.get("/top-news")
def get_top_news(db: Session = Depends(get_db), lang: str = "en", country: str = "us"):
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News non configurée")
    
    params = {
        "api-key": WORLD_NEWS_API_KEY,
        "category": "general",
        "language": lang,
        "source-country": country
    }

    try:
        response = requests.get(WORLD_NEWS_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")

    articles = data.get("articles", [])[:10] 
    if not articles:
        raise HTTPException(status_code=404, detail="Aucun article trouvé")

    raw_news = "\n".join([
        f"- {a.get('title', '')}: {a.get('text', a.get('description',''))}" 
        for a in articles
    ])

    # -----------------------------
    # Synthèse LLM
    # -----------------------------
    system_prompt_text = "Tu es un assistant qui résume les actualités en français de manière concise."
    try:
        result = agent.run_sync(
            f"Résume ces actualités en 5-6 phrases concises :\n{raw_news}",
            system_prompt=system_prompt_text
        )
        # Récupération de la sortie texte
        news_summary = getattr(result, "data", getattr(result, "output", str(result)))
    except Exception as e_agent:
        raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

    # -----------------------------
    # Sauvegarde dans la DB
    # -----------------------------
    system_prompt = db.exec(select(SystemPrompt)).first()
    prompt_text = f"Tu es l'assistant NewsFoundry. Voici les dernières actualités :\n{news_summary}"
    now = datetime.utcnow()
    if system_prompt:
        system_prompt.prompt_text = prompt_text
        system_prompt.updated_at = now
        db.add(system_prompt)
    else:
        system_prompt = SystemPrompt(prompt_text=prompt_text, created_at=now, updated_at=now)
        db.add(system_prompt)
    db.commit()
    db.refresh(system_prompt)

    return {"top_news_summary": news_summary, "updated_at": system_prompt.updated_at.isoformat()}

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

# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
