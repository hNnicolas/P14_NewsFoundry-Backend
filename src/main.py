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
import json
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
# Tool : recherche avancée d’articles 
# -------------------
@agent.tool
def advanced_search_news(context: RunContext, query: str) -> dict:
    """
    Recherche des articles sur un sujet spécifique via l'API /search-news.
    Retourne une liste d’articles avec title, summary, url.
    """
    # Log pour debug
    print("DEBUG: Tool 'advanced_search_news' appelé avec query =", query)

    # Vérification clé API
    if not WORLD_NEWS_API_KEY:
        return {"error": "Clé API World News manquante."}

    # Vérification de la requête
    if not query or len(query.strip()) < 3:
        return {"error": "La requête doit contenir au moins 3 caractères."}

    try:
        # Requête vers l'API
        response = requests.get(
            "https://api.worldnewsapi.com/search-news",
            headers={"x-api-key": WORLD_NEWS_API_KEY},
            params={
                "text": query,
                "language": "fr",
                "number": 10,
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print("ERROR: API World News échouée", e)
        return {"error": f"Erreur API World News : {str(e)}"}

    news = data.get("news", [])
    results = []

    for n in news:
        results.append({
            "title": n.get("title", "Titre indisponible"),
            "summary": (n.get("summary") or "")[:300],
            "url": n.get("url", "")
        })

    if not results:
        return {"query": query, "count": 0, "articles": [], "message": "Aucun article trouvé"}

    return {
        "query": query,
        "count": len(results),
        "articles": results
    }



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
    print("DEBUG: Début création chat avec payload =", payload)
    
    user_message = payload.get("message", "").strip()
    assistant_message = payload.get("assistant_message", "").strip()

    if not user_message and not assistant_message:
        raise HTTPException(status_code=400, detail="Aucun message fourni")

    messages = []

    if user_message:
        messages.append({"role": "user", "content": user_message})
        print("DEBUG: message utilisateur ajouté =", user_message)

    if assistant_message:
        messages.append({"role": "assistant", "content": assistant_message})
        print("DEBUG: message assistant initial ajouté =", assistant_message)

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
    print("DEBUG: chat créé en DB avec id =", chat.id)

    # Récupérer le system prompt
    system_prompt_obj = db.exec(select(SystemPrompt)).first()
    system_prompt_text = system_prompt_obj.prompt_text if system_prompt_obj else "Tu es l'assistant NewsFoundry."
    print("DEBUG: system_prompt_text =", system_prompt_text)

    # Construire le prompt pour l'agent
    combined_prompt = f"{system_prompt_text}\n\n{user_message}" if user_message else assistant_message
    print("DEBUG: prompt combiné envoyé à l'agent =", combined_prompt)

    # Appel agent
    try:
        result = agent.run_sync(user_prompt=combined_prompt)
        assistant_response = getattr(result, "data", getattr(result, "output", str(result)))
        if not isinstance(assistant_response, str):
            assistant_response = str(assistant_response)
        print("DEBUG: réponse agent =", assistant_response)
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
        print("DEBUG: message assistant ajouté au chat")

        try:
            db.add(chat)
            db.commit()
            db.refresh(chat)
            print("DEBUG: chat mis à jour avec réponse assistant")
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
    """
    Ajoute un message utilisateur dans un chat existant et retourne la réponse du LLM,
    en utilisant les tools définis (ex: advanced_search_news).
    """

    print("DEBUG: Payload reçu =", payload)
    print("DEBUG: Utilisateur courant =", current_user.email)

    # --- Récupérer le contenu utilisateur ---
    message_content = payload.get("message", "").strip()
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")
    print("DEBUG: message_content =", message_content)

    # --- Vérifier que le chat existe et appartient à l'utilisateur ---
    chat = db.exec(
        select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))
    ).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")
    print("DEBUG: chat récupéré =", chat)

    # --- Ajouter le message utilisateur à l'historique ---
    messages = chat.messages or []
    messages.append({"role": "user", "content": message_content})
    print("DEBUG: messages après ajout utilisateur =", messages)

    # --- Charger le prompt système ---
    system_prompt_obj = db.exec(select(SystemPrompt)).first()
    system_prompt_text = system_prompt_obj.prompt_text if system_prompt_obj else (
        "Tu es l'assistant NewsFoundry. "
        "Si l'utilisateur demande une revue de presse ou des actualités détaillées, "
        "tu dois appeler le tool 'advanced_search_news' avec le sujet exact et présenter les articles de façon claire. "
        "Sinon, répond normalement."
    )
    system_prompt_text = system_prompt_text[:3000] + "..." if len(system_prompt_text) > 3000 else system_prompt_text
    print("DEBUG: system_prompt_text =", system_prompt_text)

    # --- Construire le prompt complet pour le LLM ---
    conversation = [{"role": "system", "content": system_prompt_text}] + messages
    print("DEBUG: conversation envoyée au LLM =", conversation)

    # --- Appel du LLM via PydanticAI ---
    try:
        result = agent.run_sync(
            user_prompt=conversation
        )

        # Extraire le contenu
        if isinstance(result, dict) and "content" in result:
            assistant_content = result["content"]
        else:
            assistant_content = str(result)

        assistant_content = assistant_content.replace("\x00", "")
        print("DEBUG: assistant_content =", assistant_content)

    except Exception as e_agent:
        print("ERROR: appel LLM échoué", e_agent)
        assistant_content = (
            "⚠️ Je n'ai pas pu traiter votre demande pour le moment, "
            "mais la conversation a bien été enregistrée."
        )

    # --- Ajouter la réponse assistant à l'historique ---
    messages.append({"role": "assistant", "content": assistant_content})
    print("DEBUG: messages après ajout assistant =", messages)

    # --- Sérialiser correctement le JSON avant sauvegarde ---
    try:
        chat.messages = json.loads(json.dumps(messages))
    except Exception as e_json:
        print("ERROR: messages non sérialisables", e_json)
        raise HTTPException(status_code=500, detail=f"Messages non sérialisables: {str(e_json)}")

    chat.updated_at = datetime.utcnow()

    # --- Sauvegarder dans la DB ---
    try:
        db.add(chat)
        db.commit()
        db.refresh(chat)
        print("DEBUG: chat sauvegardé avec succès")
    except Exception as e_db:
        print("ERROR: problème lors du commit DB", e_db)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erreur DB: {str(e_db)}")

    # --- Retour JSON ---
    return {
        "assistant_response": assistant_content,
        "messages": chat.messages,
        "system_prompt_used": system_prompt_text,
        "user_prompt_received": message_content
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

    # Sauvegarde dans le chat 
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
def get_top_news(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News non configurée")

    params = {
        "source-country": "fr",
        "language": "fr",
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "headlines-only": "false",
        "max-news-per-cluster": 1
    }

    try:
        response = requests.get(
            WORLD_NEWS_URL,
            headers={"x-api-key": WORLD_NEWS_API_KEY},
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Extraire les articles
        articles = []
        for cluster in data.get("top_news", []):
            for a in cluster.get("news", []):
                articles.append({
                    "title": a.get("title", ""),
                    "summary": a.get("summary") or a.get("text", ""),
                    "url": a.get("url", "")
                })
        articles = articles[:10]

        if not articles:
            raise HTTPException(status_code=404, detail="Aucun article trouvé")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")

    # --- Créer le message structuré pour l'assistant ---
    top_title = articles[0]["title"] if articles else "Actualités"
    top_articles_list = "\n".join(
        [f"- {a['title']}" for a in articles[:3]]
    )
    assistant_message = (
        f"Voici un résumé des dernières {top_title} :\n\n"
        f"{top_articles_list}\n\n"
        "Souhaitez-vous que je génère une revue de presse détaillée sur l'un de ces sujets ?"
    )

    # --- Mettre à jour le SystemPrompt ---
    system_prompt_text = assistant_message 
    now = datetime.utcnow()
    system_prompt = db.exec(select(SystemPrompt)).first()
    if system_prompt:
        system_prompt.prompt_text = system_prompt_text
        system_prompt.updated_at = now
    else:
        system_prompt = SystemPrompt(prompt_text=system_prompt_text, updated_at=now)
        db.add(system_prompt)
    db.commit()
    db.refresh(system_prompt)

    # --- Ajouter le message dans le chat ---
    chat = db.exec(
        select(Chat).where(Chat.user_id == current_user.id).order_by(Chat.updated_at.desc())
    ).first()
    if chat:
        chat.messages.append({"role": "assistant", "content": assistant_message})
        chat.updated_at = now
        db.add(chat)
        db.commit()
        db.refresh(chat)

    return {
        "message": "Actualités mises à jour et ajoutées dans le chat",
        "assistant_message": assistant_message,
        "system_prompt_preview": system_prompt_text,
        "chat_id": chat.id if chat else None,
        "updated_at": now.isoformat()
    }



@app.get("/search-news")
def search_news_full(
    text: str = None,
    text_match_indexes: str = None,
    source_country: str = None,
    language: str = None,
    min_sentiment: float = None,
    max_sentiment: float = None,
    earliest_publish_date: str = None,
    latest_publish_date: str = None,
    news_sources: str = None,
    authors: str = None,
    categories: str = None,
    entities: str = None,
    location_filter: str = None,
    sort: str = None,
    sort_direction: str = None,
    offset: int = None,
    number: int = None,
):
    if not WORLD_NEWS_API_KEY:
        raise HTTPException(status_code=500, detail="Clé API World News non configurée")

    params = {"api-key": WORLD_NEWS_API_KEY}

    locals_dict = locals()
    for key, value in locals_dict.items():
        if key == "WORLD_NEWS_API_KEY":
            continue
        if value is not None:
            params[key.replace("_", "-")] = value

    try:
        response = requests.get(
            "https://api.worldnewsapi.com/search-news",
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur API World News: {str(e)}")

    return {
        "params_used": params,
        "response": data
    }


# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
