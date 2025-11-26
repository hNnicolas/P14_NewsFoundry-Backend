from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import JSON
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

    # Historique des messages stocké en JSON
    # Exemple : [{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Salut !"}]
    messages: List[dict] = Field( sa_column=Column(JSON, default=[]) )
    
    # Relation avec l'utilisateur
    user: Optional[User] = Relationship(back_populates="chats")
