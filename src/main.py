# src/main.py
from typing import Optional, List
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
import traceback
from dotenv import load_dotenv

# Charger .env
load_dotenv()

app = FastAPI(title="NewsFoundry API")

# -------------------
# CORS
# -------------------
origins = [
    "http://localhost:3000",
    "https://p14newsfoundry-frontend.vercel.app",
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
# Agent IA (lazy init)
# -------------------
agent = None

@app.on_event("startup")
def startup_event():
    global agent
    from pydantic_ai import Agent

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL")

    if not MODEL_NAME:
        if OPENAI_API_KEY:
            MODEL_NAME = "openai:gpt-4o-mini"
            print("✅ Utilisation d'OpenAI par défaut")
        elif HF_TOKEN:
            MODEL_NAME = "huggingface:HuggingFaceH4/zephyr-7b-beta"
            print("⚠️ Utilisation HuggingFace")
        else:
            raise RuntimeError("❌ Aucune clé API configurée!")

    agent = Agent(
        model=MODEL_NAME,
        system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
    )
    print("✅ Agent IA initialisé")

# -------------------
# Routes
# -------------------
@app.get("/")
def hello():
    return {"message": "👋 API NewsFoundry fonctionne !"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    password = payload.get("password")
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email et mot de passe requis")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    if not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
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

        # Appeler l'agent
        try:
            result = agent.run_sync(user_message)
            if hasattr(result, 'data'):
                assistant_response = result.data
            elif hasattr(result, 'output'):
                assistant_response = result.output
            else:
                assistant_response = str(result)
        except Exception as e_agent:
            tb = traceback.format_exc()
            print(f"[ERREUR AGENT]\n{tb}")
            raise HTTPException(
                status_code=500,
                detail=f"Erreur du modèle IA: {str(e_agent)}"
            )

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
def list_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
def get_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
        raise HTTPException(status_code=500, detail="Erreur lors de l'ajout du message utilisateur")

    try:
        result = agent.run_sync(message_content)
        if hasattr(result, 'data'):
            assistant_response = result.data
        elif hasattr(result, 'output'):
            assistant_response = result.output
        else:
            assistant_response = str(result)
    except Exception:
        assistant_response = "Désolé, l'assistant n'a pas pu répondre pour le moment."

    chat.messages.append({"role": "assistant", "content": assistant_response})
    chat.updated_at = datetime.utcnow()

    try:
        db.add(chat)
        db.commit()
        db.refresh(chat)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Erreur lors de l'ajout de la réponse assistant")

    return {"assistant_response": assistant_response, "messages": chat.messages}

# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port)
