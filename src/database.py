import os
from dotenv import load_dotenv 
from sqlmodel import SQLModel, create_engine, Session, select
from src.models import User
import bcrypt

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not defined.")

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    """Creates tables and ensures default user exists (run once)."""
    SQLModel.metadata.create_all(engine)
    print("Database initialized successfully")

def create_default_user():

    default_email = "test@test.com"
    default_password = "test"

    with Session(engine) as session:
        statement = select(User).where(User.email == default_email)
        user = session.exec(statement).first()

        if not user:
            hashed = bcrypt.hashpw(default_password.encode("utf-8"), bcrypt.gensalt())
            hashed = hashed.decode("utf-8")  # IMPORTANT — store as string

            session.add(
                User(
                    email=default_email,
                    hashed_password=hashed,
                )
            )
            session.commit()
            print(f"Default user {default_email} created.")