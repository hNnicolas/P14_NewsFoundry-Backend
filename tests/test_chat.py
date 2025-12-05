import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from src.main import app, get_db
from src.models import User, Chat
from src.database import init_db, engine
import bcrypt
import jwt
import os

SECRET_KEY = os.getenv("SECRET_KEY", "TEST_SECRET")
ALGORITHM = "HS256"

@pytest.fixture(scope="module")
def client():
    # Initialisation DB
    init_db()
    with TestClient(app) as c:
        yield c

@pytest.fixture
def test_user():
    email = "test@test.com"
    password = "test"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")
    with Session(engine) as session:
        user = session.exec(User.select().where(User.email == email)).first()
        if not user:
            user = User(email=email, hashed_password=hashed)
            session.add(user)
            session.commit()
            session.refresh(user)
        yield user

@pytest.fixture
def auth_token(test_user):
    return jwt.encode({"sub": test_user.email}, SECRET_KEY, algorithm=ALGORITHM)

def test_create_chat(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    payload = {"message": "Bonjour"}
    response = client.post("/chats", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "chat_id" in data
    assert "assistant_response" in data
    assert len(data["messages"]) == 2  

def test_get_chat(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    # Créer un chat
    payload = {"message": "Hello"}
    res = client.post("/chats", json=payload, headers=headers)
    chat_id = res.json()["chat_id"]

    # Récupérer le chat
    response = client.get(f"/chats/{chat_id}", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["chat_id"] == chat_id
    assert len(data["messages"]) >= 2

def test_add_message(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    # Créer un chat
    res = client.post("/chats", json={"message": "Salut"}, headers=headers)
    chat_id = res.json()["chat_id"]

    # Ajouter un message
    payload = {"message": "Comment ça va ?"}
    response = client.post(f"/chats/{chat_id}/messages", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) >= 3  
