import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
import os
import requests
import traceback
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- CONFIG ---
SECRET_KEY = "signbridge_secret"
ALGORITHM = "HS256"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- DATABASE ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db")
if DATABASE_URL.startswith("postgres://"): DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password = Column(String)

class HistoryDB(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String)
    sentence = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- AI ENGINE GLOBALS ---
interpreter = None
input_details = None
output_details = None
labels = []

# Translation State
sentence = ""
current_word = ""
last_char = ""
counter = 0
no_hand_count = 0
last_completed_word = ""
last_meaning = ""

def load_model():
    global interpreter, input_details, output_details, labels
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path="gesture_model_optimized.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        labels = np.load("labels.npy", allow_pickle=True).tolist()

def get_meaning(word):
    try:
        r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=1)
        return r.json()[0]['meanings'][0]['definitions'][0]['definition']
    except: return "Definition not found"

@app.on_event("startup")
async def startup(): load_model()

@app.post("/signup")
async def signup(u: dict):
    db = SessionLocal()
    db.add(UserDB(username=u['username'], password=u['password']))
    db.commit()
    return {"status": "ok"}

@app.post("/login")
async def login(u: dict):
    db = SessionLocal()
    user = db.query(UserDB).filter(UserDB.username == u['username'], UserDB.password == u['password']).first()
    if not user: raise HTTPException(401)
    token = jwt.encode({"sub": u['username'], "exp": datetime.utcnow()+timedelta(hours=2)}, SECRET_KEY, ALGORITHM)
    return {"access_token": token}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sentence, current_word, last_char, counter, no_hand_count, last_completed_word, last_meaning
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "clear":
                sentence = ""; current_word = ""; continue
            
            landmarks = data.get("landmarks")
            conf_threshold = data.get("config", {}).get("threshold", 0.85)
            buffer_val = data.get("config", {}).get("buffer", 5)

            resp = {"text": f"{sentence} {current_word}".strip(), "speak": None, "current_letter": "-"}

            if landmarks:
                no_hand_count = 0
                coords = []
                base_x, base_y = landmarks[0][0], landmarks[0][1]
                for lm in landmarks: coords.extend([lm[0]-base_x, lm[1]-base_y, lm[2]])
                
                interpreter.set_tensor(input_details[0]['index'], np.array([coords], dtype=np.float32))
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])[0]
                
                idx = np.argmax(output)
                conf = float(output[idx])
                
                if conf > conf_threshold:
                    char = labels[idx]
                    resp["current_letter"] = char
                    if char == last_char: counter += 1
                    else: last_char = char; counter = 0
                    
                    if counter >= buffer_val:
                        if char not in ['nothing', 'space', 'del']:
                            current_word += char
                            resp["speak"] = char
                            counter = 0 # RESET: Prevents runaway letters
                        elif char == 'space':
                            last_completed_word = current_word
                            sentence += current_word + " "
                            current_word = ""; counter = 0
            else:
                # FIX: AUTO-SPEAK ON HAND REMOVAL
                no_hand_count += 1
                if no_hand_count == 15 and current_word != "":
                    last_completed_word = current_word
                    last_meaning = get_meaning(last_completed_word)
                    resp["speak"] = last_completed_word
                    resp["completed_word"] = last_completed_word
                    resp["meaning"] = last_meaning
                    sentence += current_word + " "
                    current_word = ""; last_char = ""; no_hand_count = 0

            resp["text"] = f"{sentence} {current_word}".strip()
            await websocket.send_json(resp)
    except: pass
