import os
import asyncpg
import bcrypt
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from api.config import get_settings

settings = get_settings()
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

SECRET_KEY = os.getenv("JWT_SECRET", "supersecretkey_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"user_id": user_id, "username": payload.get("username")}

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    # Direct bcrypt hashing
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), salt).decode('utf-8')
    
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        await conn.execute("INSERT INTO users (username, email, hashed_password) VALUES ($1, $2, $3)", user.username, user.email, hashed_password)
        return {"message": "User created successfully"}
    except asyncpg.exceptions.UniqueViolationError:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    finally:
        await conn.close()

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = await asyncpg.connect(settings.DATABASE_URL)
    try:
        user = await conn.fetchrow("SELECT id, username, hashed_password FROM users WHERE username = $1", form_data.username)
        
        # Direct bcrypt verification
        if not user or not bcrypt.checkpw(form_data.password.encode('utf-8'), user["hashed_password"].encode('utf-8')):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
            
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token = jwt.encode({"sub": str(user["id"]), "username": user["username"], "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}
    finally:
        await conn.close()

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@router.post("/refresh")
async def refresh_token(current_user: dict = Depends(get_current_user)):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode({"sub": current_user["user_id"], "username": current_user["username"], "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60}