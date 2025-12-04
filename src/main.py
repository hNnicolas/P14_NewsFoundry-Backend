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
import traceback

# Charger dotenv uniquement si le fichier existe (local dev)
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

from pydantic_ai import Agent

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

@app.get("/env")
def show_env():
    import os
    return {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "HF_TOKEN": bool(os.getenv("HF_TOKEN")),
        "PYDANTIC_AI_MODEL": os.getenv("PYDANTIC_AI_MODEL"),
        "SECRET_KEY": bool(os.getenv("SECRET_KEY"))
    }

# -------------------
# PydanticAI Agent Configuration
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL")


print("===============================")

# Vérification des clés
if not MODEL_NAME:
    if OPENAI_API_KEY:
        MODEL_NAME = "openai:gpt-4o-mini"
        print("✅ Utilisation d'OpenAI par défaut")
    elif HF_TOKEN:
        MODEL_NAME = "huggingface:HuggingFaceH4/zephyr-7b-beta"
        print("⚠️ Utilisation de HuggingFace (peut être instable)")
    else:
        raise RuntimeError(
            "❌ Aucune clé API configurée! Ajoutez OPENAI_API_KEY ou HF_TOKEN dans l'environnement Railway"
        )


# Initialisation de l'agent
try:
    agent = Agent(
        model=MODEL_NAME,
        system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
    )
except Exception as e:
    print(f"❌ ERREUR d'initialisation de l'agent: {e}")
    raise

print("===========================\n")

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
# -------------------
# PydanticAI Agent Configuration
# -------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "").strip()

# Vérification des variables
print("===============================")
print("OPENAI_API_KEY:", repr(OPENAI_API_KEY))
print("HF_TOKEN:", repr(HF_TOKEN))
print("PYDANTIC_AI_MODEL:", repr(MODEL_NAME))
print("===============================")

# Vérification et fallback
if not MODEL_NAME:
    if OPENAI_API_KEY:
        MODEL_NAME = "openai:gpt-4o-mini"
        print("Utilisation d'OpenAI par défaut")
    elif HF_TOKEN:
        MODEL_NAME = "huggingface:HuggingFaceH4/zephyr-7b-beta"
        print("Utilisation de HuggingFace (peut être instable)")
    else:
        raise RuntimeError(
            "❌ Aucune clé API configurée! Ajoutez OPENAI_API_KEY ou HF_TOKEN dans l'environnement Railway"
        )

# Initialisation de l'agent
try:
    agent = Agent(
        model=MODEL_NAME,
        system_prompt="Tu es l'assistant NewsFoundry. Réponds de manière concise et informative en français."
    )
except Exception as e:
    print(f"❌ ERREUR d'initialisation de l'agent: {e}")
    raise

print("===========================\n")

# -------------------
# Créer un chat
# -------------------
@app.post("/chats")
def create_chat(payload: dict = {}, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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

        try:
            result = agent.run_sync(user_message)
            if hasattr(result, 'data'):
                assistant_response = result.data
            elif hasattr(result, 'output'):
                assistant_response = result.output
            else:
                assistant_response = str(result)
        except Exception as e_agent:
            raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

        chat.messages.append({"role": "assistant", "content": assistant_response})

        db.add(chat)
        db.commit()
        db.refresh(chat)

        return {"chat_id": chat.id, "assistant_response": assistant_response, "messages": chat.messages}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
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
def add_message(chat_id: int, payload: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        message_content = payload.get("message", "")
        if not message_content:
            raise HTTPException(status_code=400, detail="Message requis")

        chat = db.exec(select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Discussion introuvable")

        if chat.messages is None:
            chat.messages = []

        chat.messages.append({"role": "user", "content": message_content})

        try:
            result = agent.run_sync(message_content)
            if hasattr(result, 'data'):
                assistant_response = result.data
            elif hasattr(result, 'output'):
                assistant_response = result.output
            else:
                assistant_response = str(result)
        except Exception as e_agent:
            raise HTTPException(status_code=500, detail=f"Erreur du modèle IA: {str(e_agent)}")

        chat.messages.append({"role": "assistant", "content": assistant_response})
        chat.updated_at = datetime.utcnow()
        db.add(chat)
        db.commit()
        db.refresh(chat)

        return {"assistant_response": assistant_response, "messages": chat.messages}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)