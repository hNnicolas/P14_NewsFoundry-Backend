from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database import init_db, engine
from src.models import User, Chat
import jwt
import bcrypt
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv
from pydantic_ai import Agent
import traceback

# -------------------
# Charger .env
# -------------------
load_dotenv()

app = FastAPI()

# -------------------
# CORS
# -------------------
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://p14newsfoundry-frontend-production.up.railway.app",
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
# Variables Agent / API keys
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL")

if not MODEL_NAME:
    if OPENAI_API_KEY:
        MODEL_NAME = "openai:gpt-4o-mini"
        print("✅ Utilisation d'OpenAI par défaut")
    elif HF_TOKEN:
        MODEL_NAME = "huggingface:HuggingFaceH4/zephyr-7b-beta"
        print("⚠️  Utilisation HuggingFace (peut être instable)")
    else:
        raise RuntimeError("❌ Aucune clé API configurée! Ajoutez OPENAI_API_KEY ou HF_TOKEN")

if MODEL_NAME.startswith("openai:") and not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY requis pour OpenAI")
if MODEL_NAME.startswith("huggingface:") and not HF_TOKEN:
    raise RuntimeError("❌ HF_TOKEN requis pour HuggingFace")

# -------------------
# Agent "lazy init" (safe prod)
# -------------------
agent = None

def get_agent():
    global agent
    if agent is None:
        try:
            agent = Agent(
                model=MODEL_NAME,
                system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
            )
            print("✅ Agent initialisé")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[ERREUR AGENT INIT]\n{tb}")
            raise RuntimeError(f"Impossible d'initialiser l'agent: {e}")
    return agent

# -------------------
# Routes
# -------------------
@app.get("/")
def hello():
    return {"message": "👋 API NewsFoundry fonctionne !", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email et mot de passe requis")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = jwt.encode({"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token}

# -------------------
# Créer un chat
# -------------------
@app.post("/chats")
def create_chat(
    payload: dict = {},
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
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

        # Agent
        try:
            assistant_response = get_agent().run_sync(user_message)
            if hasattr(assistant_response, 'data'):
                assistant_response = assistant_response.data
            elif hasattr(assistant_response, 'output'):
                assistant_response = assistant_response.output
            else:
                assistant_response = str(assistant_response)
        except Exception as e_agent:
            tb = traceback.format_exc()
            print(f"[ERREUR AGENT]\n{tb}")
            assistant_response = "Désolé, l'assistant n'a pas pu répondre"

        chat.messages.append({"role": "assistant", "content": assistant_response})

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return {
            "chat_id": chat.id,
            "assistant_response": assistant_response,
            "messages": chat.messages
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        tb = traceback.format_exc()
        print(f"[ERREUR CREATE CHAT]\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))

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
    chat = db.exec(
        select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))
    ).first()
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
def add_message(chat_id: int, payload: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    message_content = payload.get("message", "")
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")

    chat = db.exec(
        select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    if chat.messages is None:
        chat.messages = []
    chat.messages.append({"role": "user", "content": message_content})
    chat.updated_at = datetime.utcnow()

    try:
        db.add(chat)
        db.commit()
        db.refresh(chat)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur ajout message utilisateur")

    # Agent
    try:
        assistant_response = get_agent().run_sync(message_content)
        if hasattr(assistant_response, 'data'):
            assistant_response = assistant_response.data
        elif hasattr(assistant_response, 'output'):
            assistant_response = assistant_response.output
        else:
            assistant_response = str(assistant_response)
    except Exception as e_agent:
        tb = traceback.format_exc()
        print(f"[ERREUR AGENT]\n{tb}")
        assistant_response = "Désolé, l'assistant n'a pas pu répondre"

    chat.messages.append({"role": "assistant", "content": assistant_response})
    chat.updated_at = datetime.utcnow()

    try:
        db.add(chat)
        db.commit()
        db.refresh(chat)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur ajout réponse assistant")

    return {"assistant_response": assistant_response, "messages": chat.messages}

# -------------------
# Lancement serveur prod
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port)
