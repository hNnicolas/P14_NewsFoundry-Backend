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
from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

app = FastAPI()

# -------------------
# CORS
# -------------------
origins = [
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "https://p14newsfoundry.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
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
# DB session
# -------------------
def get_db():
    with Session(engine) as session:
        yield session

# -------------------
# Auth helper
# -------------------
def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
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
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide")

    user = db.exec(select(User).where(User.email == email)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Utilisateur introuvable")
    return user

# -------------------
# LLM
# -------------------
agent = Agent(
    os.getenv("PYDANTIC_AI_MODEL", "openai:gpt-4"),
    instructions="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative."
)
# -------------------
# Routes
# -------------------

@app.get("/")
def hello():
    return {"message": "👋 API fonctionne !"}

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

    # Vérification du mot de passe
    if not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = jwt.encode({"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM)
    return {"token": token}

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
        messages=[{"role": "user", "content": user_message}],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    response = llm.generate(messages=chat.messages)
    chat.messages.append({"role": "assistant", "content": response})

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {"chat_id": chat.id, "assistant_response": response, "messages": chat.messages}

# -------------------
# Récupérer un chat
# -------------------
@app.get("/chats/{chat_id}")
def get_chat(chat_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = db.exec(select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion non trouvée")
    return {"chat_id": chat.id, "title": chat.title, "messages": chat.messages}

# -------------------
# Ajouter un message
# -------------------
@app.post("/chats/{chat_id}/messages")
def add_message(chat_id: int, payload: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    message_content = payload.get("message", "")
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")

    chat = db.exec(select(Chat).where(Chat.id == chat_id, Chat.user_id == current_user.id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    if chat.messages is None:
        chat.messages = []

    chat.messages.append({"role": "user", "content": message_content})
    llm_response = llm.generate(messages=chat.messages)
    chat.messages.append({"role": "assistant", "content": llm_response})

    chat.updated_at = datetime.utcnow()
    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {"assistant_response": llm_response, "messages": chat.messages}

# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
