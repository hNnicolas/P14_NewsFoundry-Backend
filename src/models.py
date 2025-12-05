# src/models.py
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import JSON
from sqlalchemy.ext.mutable import MutableList
from datetime import datetime

# --------------------------
# Modèle Utilisateur
# --------------------------
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str = Field()

    # Relation vers les chats
    chats: List["Chat"] = Relationship(back_populates="user")

# --------------------------
# Modèle Chat
# --------------------------
class Chat(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False)
    title: Optional[str] = Field(default="Nouvelle discussion")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Historique des messages suivi par SQLAlchemy
    messages: List[dict] = Field(
        default_factory=list,
        sa_column=Column(MutableList.as_mutable(JSON))
    )

    # Champs pour la revue de presse
    press_review_title: Optional[str] = None
    press_review_summary: Optional[str] = None
    press_review_articles: Optional[List[dict]] = Field(
        default_factory=list,
        sa_column=Column(MutableList.as_mutable(JSON))
    )

    # Relation avec l'utilisateur
    user: Optional[User] = Relationship(back_populates="chats")
    
# --------------------------
# Modèle SystemPrompt
# --------------------------
class SystemPrompt(SQLModel, table=True):
    """
    Stocke le prompt système à jour incluant les dernières actualités
    afin que le LLM réponde avec des informations récentes.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_text: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
