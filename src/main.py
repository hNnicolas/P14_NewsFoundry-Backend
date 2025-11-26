from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select
from src.database import init_db, engine
from src.models import User
import jwt
import bcrypt
import uvicorn
import os

app = FastAPI()

origins = [
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "https://p14newsfoundry.vercel.app"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("SECRET_KEY", "TEST_SECRET")
ALGORITHM = "HS256"

def get_db():
    with Session(engine) as session:
        yield session

@app.get("/")
def hello():
    return {"message": "👋 API fonctionne !"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/login")
def login(payload: dict, db: Session = Depends(get_db)):
    email = payload.get("email")
    password = payload.get("password")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email et mot de passe requis")

    statement = select(User).where(User.email == email)
    user = db.exec(statement).first()

    if not user:
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    # Always stored as string
    hashed_bytes = user.hashed_password.encode("utf-8")

    if not bcrypt.checkpw(password.encode("utf-8"), hashed_bytes):
        raise HTTPException(status_code=401, detail="Identifiants invalides")

    token = jwt.encode({"sub": user.email}, SECRET_KEY, algorithm=ALGORITHM)

    return {"token": token}

if __name__ == "__main__":
    init_db()
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
