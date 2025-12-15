import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from src.main import app
from src.models import User, Chat
from src.database import init_db, engine
import bcrypt
import jwt
import os

SECRET_KEY = os.getenv("SECRET_KEY", "TEST_SECRET")
ALGORITHM = "HS256"


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def test_user():
    email = "test@test.com"
    password = "test"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode("utf-8")

    with Session(engine) as session:
        user = session.exec(
            User.select().where(User.email == email)
        ).first()

        if not user:
            user = User(email=email, hashed_password=hashed)
            session.add(user)
            session.commit()
            session.refresh(user)

        yield user


@pytest.fixture
def auth_token(test_user):
    return jwt.encode(
        {"sub": test_user.email},
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def test_top_news_creates_chat(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    payload = {
        "user_message": "résumé de l'actualité économique"
    }

    response = client.post(
        "/top-news",
        json=payload,
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()
    assert "assistant_message" in data
    assert "articles" in data
    assert "chat_id" in data
    assert isinstance(data["articles"], list)


def test_search_news(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    payload = {
        "query": "économie",
        "language": "fr"
    }

    response = client.post(
        "/search-news",
        json=payload,
        headers=headers
    )

    # 200 si API OK, 502 si World News indisponible
    assert response.status_code in (200, 502)

    if response.status_code == 200:
        data = response.json()
        assert "articles" in data
        assert isinstance(data["articles"], list)


def test_generate_press_review(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    # 1. Création d’un chat avec top-news
    chat_res = client.post(
        "/top-news",
        json={"user_message": "actualité politique"},
        headers=headers
    )
    chat_id = chat_res.json()["chat_id"]

    # 2. Génération de la revue de presse
    response = client.post(
        f"/chats/{chat_id}/generate-press-review",
        json={"theme": "Politique"},
        headers=headers
    )

    assert response.status_code == 200

    data = response.json()
    assert "review" in data
    assert "title" in data["review"]
    assert "summary" in data["review"]
    assert "articles" in data["review"]
