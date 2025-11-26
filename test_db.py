# test_db.py
import os
from sqlmodel import SQLModel, create_engine, Session, text
from dotenv import load_dotenv

load_dotenv()  # charge ton .env.local

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL non défini")

engine = create_engine(DATABASE_URL, echo=True)

try:
    with Session(engine) as session:
        result = session.execute(text("SELECT 1"))
        print("✅ Connexion DB réussie :", result.fetchone())
except Exception as e:
    print("❌ Erreur de connexion :", e)
