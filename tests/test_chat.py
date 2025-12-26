import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select
from src.main import app
from src.models import User
from src.database import init_db, engine
import bcrypt
import jwt
import os

# =========================
# Configuration JWT
# =========================
SECRET_KEY = os.getenv("SECRET_KEY", "TEST_SECRET")
ALGORITHM = "HS256"


# =========================
# Client FastAPI
# =========================
@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


# =========================
# Utilisateur principal
# =========================
@pytest.fixture
def test_user():
    email = "test@test.com"
    password = "test"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")

    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.email == email)
        ).first()

        if not user:
            user = User(email=email, hashed_password=hashed)
            session.add(user)
            session.commit()
            session.refresh(user)

        yield user


# =========================
# Second utilisateur
# =========================
@pytest.fixture
def other_user():
    email = "other@test.com"
    password = "test"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")

    with Session(engine) as session:
        user = session.exec(
            select(User).where(User.email == email)
        ).first()

        if not user:
            user = User(email=email, hashed_password=hashed)
            session.add(user)
            session.commit()
            session.refresh(user)

        yield user


# =========================
# Tokens JWT
# =========================
@pytest.fixture
def auth_token(test_user):
    return jwt.encode(
        {"sub": test_user.email},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


@pytest.fixture
def other_auth_token(other_user):
    return jwt.encode(
        {"sub": other_user.email},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


# =========================
# Tests
# =========================

def test_create_chat(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    payload = {"message": "Bonjour"}

    response = client.post("/chats", json=payload, headers=headers)

    assert response.status_code == 200

    data = response.json()
    assert "chat_id" in data
    assert "assistant_response" in data
    assert "messages" in data
    assert len(data["messages"]) == 2


def test_get_chat_nominal(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Création du chat
    res = client.post("/chats", json={"message": "Hello"}, headers=headers)
    chat_id = res.json()["chat_id"]

    # Accès au chat
    response = client.get(f"/chats/{chat_id}", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["chat_id"] == chat_id
    assert isinstance(data["messages"], list)


def test_add_message_nominal(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    res = client.post("/chats", json={"message": "Salut"}, headers=headers)
    chat_id = res.json()["chat_id"]

    response = client.post(
        f"/chats/{chat_id}/messages",
        json={"message": "Comment ça va ?"},
        headers=headers
    )

    assert response.status_code == 200
    assert "messages" in response.json()
    assert len(response.json()["messages"]) >= 3


def test_user_cannot_access_other_user_chat(
    client,
    auth_token,
    other_auth_token
):
    """
    Vérifie qu’un utilisateur ne peut pas accéder
    au chat d’un autre utilisateur
    """

    # User A crée un chat
    headers_user_a = {"Authorization": f"Bearer {auth_token}"}
    res = client.post(
        "/chats",
        json={"message": "Chat privé"},
        headers=headers_user_a
    )
    chat_id = res.json()["chat_id"]

    # User B tente d’y accéder
    headers_user_b = {"Authorization": f"Bearer {other_auth_token}"}
    response = client.get(
        f"/chats/{chat_id}",
        headers=headers_user_b
    )

    # Selon l’implémentation : 404 (sécurité) ou 403
    assert response.status_code in (403, 404)
