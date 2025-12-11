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
# Ajouter un message intelligent avec détection automatique de revue de presse
# -------------------
@app.post("/chats/{chat_id}/messages")
def add_message_smart(
    chat_id: int,
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Ajoute un message utilisateur dans un chat existant et retourne la réponse de l'agent.
    L'agent détecte automatiquement toute demande de revue de presse et appelle le tool
    'advanced_search_news' si nécessaire. Version avec debug étendu et fallback direct.
    """
    message_content = payload.get("message", "").strip()
    if not message_content:
        raise HTTPException(status_code=400, detail="Message requis")

    # --- Récupérer le chat ---
    chat = db.exec(select(Chat).where((Chat.id == chat_id) & (Chat.user_id == current_user.id))).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Discussion introuvable")

    messages = chat.messages or []
    messages.append({"role": "user", "content": message_content})

    # --- Charger ou créer system prompt intelligent ---
    system_prompt_obj = db.exec(select(SystemPrompt)).first()
    system_prompt_text = system_prompt_obj.prompt_text if system_prompt_obj else (
        "Tu es l'assistant NewsFoundry, expert en actualités et revues de presse. "
        "Chaque fois que l'utilisateur demande des informations détaillées sur un sujet, "
        "une revue de presse ou des articles récents, tu dois automatiquement appeler le tool "
        "'advanced_search_news' avec le sujet exact. "
        "Sois flexible : l'utilisateur peut formuler sa demande de manière naturelle. "
        "Retourne toujours un résumé structuré : titre de la revue, synthèse générale, "
        "liste d'articles avec titre et résumé, éventuellement le lien vers la source. "
        "Si aucun sujet n'est clairement demandé, pose une question polie pour clarifier le sujet."
    )

    # --- Conversion conversation en texte brut pour l'agent ---
    raw_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in [{"role": "system", "content": system_prompt_text}] + messages])

    # --- Détection simple d'intention 'revue de presse' et extraction du sujet ---
    lowered = message_content.lower()
    wants_review = bool(re.search(r"\b(revue de presse|revue|revue détaill|revue détaillée|revu[eé] de presse|détaill|détail)\b", lowered))

    topic = None
    if wants_review:
        # try to extract topic after phrases like 'revue de presse', 'revue de', 'revue sur', 'revue:'
        # fallback: take remaining words (maximum safe length)
        # Examples user messages: "revue de presse miss provence", "je veux une revue sur miss provence", "revue: miss provence"
        # We remove keywords then strip
        topic = re.sub(r"(?i)\b(revue de presse|revue de|revue sur|revue:|revue|revue détaillée|revue détaill|détaillé|détaillée)\b", "", message_content)
        topic = topic.strip(" .:,-\n\t")
        # if topic too short or empty, try to use system_prompt or previous assistant suggestion
        if not topic or len(topic) < 3:
            # try to infer from system_prompt_text (which contains the top headlines list)
            # take first "ligne" after dash
            m = re.search(r"-\s*(.+)", system_prompt_text)
            if m:
                topic = m.group(1).split("\n")[0].strip()
        # final fallback: the entire user message
        if not topic or len(topic) < 3:
            topic = message_content

    assistant_content = None

    # --- Si on détecte une demande de revue: appeler directement le tool (fallback fiable) ---
    if wants_review and topic:
        try:
            # Appel direct au tool Python (le decorator @agent.tool l'a juste enregistré, mais on peut l'appeler directement)
            # advanced_search_news prend (context, query, ...). Le context n'est pas nécessaire pour l'appel direct ici.
            tool_result = advanced_search_news(None, topic)

            # Si le tool renvoie une erreur structurée
            if isinstance(tool_result, dict) and tool_result.get("error"):
                assistant_content = f"⚠️ Le moteur de recherche d'articles a retourné une erreur : {tool_result.get('error')}"
            else:
                # Formatter la réponse: titre + synthèse + liste d'articles
                articles = tool_result.get("articles", []) if isinstance(tool_result, dict) else []
                count = tool_result.get("count", len(articles)) if isinstance(tool_result, dict) else len(articles)
                available = tool_result.get("available", count) if isinstance(tool_result, dict) else count

                title = f"Revue de presse — {topic}"
                summary = f"Résultats trouvés : {count} (available: {available}). Voici une synthèse des premiers articles."
                body_lines = [f"**{title}**", summary, ""]

                # limiter pour affichage
                for i, a in enumerate(articles[:10]):
                    a_title = a.get("title") or a.get("headline") or "Titre indisponible"
                    a_summary = (a.get("summary") or a.get("content") or "")[:400]
                    a_url = a.get("url") or ""
                    body_lines.append(f"{i+1}. {a_title}")
                    if a_summary:
                        body_lines.append(f"   Résumé: {a_summary}")
                    if a_url:
                        body_lines.append(f"   Source: {a_url}")
                    body_lines.append("")

                assistant_content = "\n".join(body_lines)

        except Exception as e_tool:
            print("ERROR: appel direct tool failed:", e_tool)
            assistant_content = (
                "⚠️ Impossible d'interroger le service de recherche d'articles pour le moment. "
                "Je peux néanmoins essayer de répondre autrement."
            )

    else:
        try:
            result = agent.run_sync(user_prompt=raw_prompt, max_iterations=2)
            # introspecter result pour debug
            try:
                print("DEBUG: result dir():", dir(result)[:50])
            except Exception:
                pass

            # Extraction du contenu selon formes possibles
            if isinstance(result, dict) and "content" in result:
                assistant_content = result["content"]
            else:
                # agent.run_sync renvoie parfois un objet avec .data ou .output
                assistant_content = getattr(result, "data", None) or getattr(result, "output", None) or str(result)

        except Exception as e_agent:
            # fallback clair pour l'utilisateur
            assistant_content = (
                "⚠️ Je n'ai pas pu traiter votre demande pour le moment, "
                "mais la conversation a bien été enregistrée."
            )

    # --- Ajouter réponse assistant à l'historique ---
    messages.append({"role": "assistant", "content": assistant_content})
    try:
        chat.messages = json.loads(json.dumps(messages))
        chat.updated_at = datetime.utcnow()
        db.add(chat)
        db.commit()
        db.refresh(chat)
    except Exception as e_db:
        db.rollback()
        print("ERROR DB lors de la sauvegarde du chat:", e_db)
        raise HTTPException(status_code=500, detail=f"Erreur DB: {str(e_db)}")

    return {
        "assistant_response": assistant_content,
        "messages": chat.messages,
        "system_prompt_used": system_prompt_text,
        "user_prompt_received": message_content
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
    
# -------------------
# Tool : recherche avancée d’articles 
# -------------------
@agent.tool
def advanced_search_news(context: RunContext, query: str, language: str = "fr", number: int = 10) -> dict:
    if not WORLD_NEWS_API_KEY:
        return {"error": "Clé API World News manquante."}

    if not query or len(query.strip()) < 3:
        return {"error": "La requête doit contenir au moins 3 caractères."}

    try:
        response = requests.get(
            "https://api.worldnewsapi.com/search-news",
            headers={"x-api-key": WORLD_NEWS_API_KEY},
            params={
                "text": query,
                "language": language,
                "number": number,
                "sort": "publish-time",
                "sort-direction": "DESC"
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        return {"error": f"Erreur API World News : {str(e)}"}

    articles = []
    for n in data.get("news", []):
        articles.append({
            "title": n.get("title"),
            "summary": n.get("summary"),
            "content": n.get("text"),
            "url": n.get("url"),
            "image": n.get("image"),
            "video": n.get("video"),
            "publish_date": n.get("publish_date"),
            "authors": n.get("authors", []),
            "category": n.get("category"),
            "language": n.get("language"),
            "source_country": n.get("source_country"),
            "sentiment": n.get("sentiment")
        })

    return {
        "query": query,
        "count": len(articles),
        "available": data.get("available", len(articles)),
        "articles": articles
    }
    
# -------------------
# Tool : final_result (résultat structuré pour la revue de presse)
# -------------------
@agent.tool
def final_result(context: RunContext, title: str, summary: str, articles: list) -> dict:
    """
    Retourne la revue de presse structurée.
    """
    return {
        "title": title,
        "summary": summary,
        "articles": articles
    }

# IMPORTANT : fournir un JSON Schema valide
final_result.__doc__ = """
{
    "name": "final_result",
    "description": "Retour structuré de la revue de presse",
    "parameters": {
        "type": "object",
        "properties": {
            "title": { "type": "string" },
            "summary": { "type": "string" },
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "summary": { "type": "string" },
                        "url": { "type": "string" }
                    },
                    "required": ["title"]
                }
            }
        },
        "required": ["title", "summary", "articles"]
    }
}
"""

# -------------------
# Générer une revue de presse à partir du thème
# -------------------
@app.post("/chats/{chat_id}/generate-press-review")
def generate_press_review(
    chat_id: int,
    payload: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    theme = payload.get("theme")
    if not theme:
        raise HTTPException(status_code=400, detail="Un thème est requis pour générer la revue de presse.")

    # --- Vérification du chat ---
    chat = db.exec(
        select(Chat).where(
            (Chat.id == chat_id) & (Chat.user_id == current_user.id)
        )
    ).first()

    if not chat:
        raise HTTPException(status_code=404, detail="Chat introuvable")


    # ====================================================
    # PHASE 1 : RAG articles chargés
    # ====================================================
    article_urls = chat.loaded_articles or []
    documents = []

    for url in article_urls:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                documents.append(Document(text=resp.text, metadata={"url": url}))
        except:
            pass

    rag_context = ""
    if documents:
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)

        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        )

        rag_results = index.as_query_engine().query(f"Articles pertinents pour : {theme}")
        rag_context = f"\n\n=== ARTICLES RÉCUPÉRÉS VIA RAG ===\n{rag_results}\n\n"


    # ====================================================
    # PHASE 2 : Reconstruction du chat
    # ====================================================
    conversation_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in chat.messages]
    )


    # ====================================================
    # PHASE 3 : Prompt final pour l’agent
    # ====================================================
    prompt = (
        f"Génère une revue de presse complète sur le thème '{theme}'.\n"
        f"Utilise OBLIGATOIREMENT le tool final_result pour renvoyer le résultat.\n\n"
        f"=== CONVERSATION ===\n{conversation_text}\n\n"
        f"{rag_context}"
    )


    # ====================================================
    # PHASE 4 : Appel LLM → l’agent va appeler final_result()
    # ====================================================
    try:
        result = press_review_agent.run_sync(
            user_prompt=prompt,
            output_type=PressReviewOutputModel
        )

        # --- Extraction correcte du tool_call ---
        tool_call = result.data["tool_calls"][0]
        review = tool_call["arguments"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA lors de la revue de presse: {str(e)}")


    # ====================================================
    # PHASE 5 : Sauvegarde BDD
    # ====================================================
    chat.press_review_title = review["title"]
    chat.press_review_summary = review["summary"]
    chat.press_review_articles = review["articles"]
    chat.updated_at = datetime.utcnow()

    db.add(chat)
    db.commit()
    db.refresh(chat)

    return {
        "message": "Revue de presse générée",
        "review": review
    }


# -------------------
# Lancement du serveur
# -------------------
if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=True)
