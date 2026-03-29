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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)

class HistoryDB(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, index=True)
    sentence = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class VocabDB(Base):
    __tablename__ = "vocabulary"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, index=True)
    word = Column(String)
    meaning = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal(); 
    try: yield db
    finally: db.close()

# --- AI ENGINE ---
interpreter = None
input_details = None
output_details = None
labels = []

# Global Translation State
sentence = ""
current_word = ""
last_char = ""
last_completed_word = "" 
last_meaning = ""
counter = 0
no_hand_count = 0

def load_model_globally():
    global interpreter, input_details, output_details, labels
    if interpreter is None:
        try:
            interpreter = tflite.Interpreter(model_path="gesture_model_optimized.tflite")
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            labels = np.load("labels.npy", allow_pickle=True).tolist()
        except Exception as e: print(f"Model Load Error: {e}")

def get_meaning(word):
    if not word: return ""
    try:
        r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=2)
        if r.status_code == 200:
            return r.json()[0]['meanings'][0]['definitions'][0]['definition']
    except: pass
    return "Meaning not found"

@app.on_event("startup")
async def startup_event(): load_model_globally()

@app.post("/signup")
async def signup(data: dict, db=Depends(get_db)):
    new_user = UserDB(username=data['username'], password=data['password'])
    db.add(new_user); db.commit(); return {"status": "success"}

@app.post("/login")
async def login(data: dict, db=Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data['username'], UserDB.password == data['password']).first()
    if user:
        token = jwt.encode({"sub": data['username'], "exp": datetime.utcnow() + timedelta(hours=2)}, SECRET_KEY, ALGORITHM)
        return {"access_token": token}
    raise HTTPException(status_code=401)

@app.get("/api/history")
async def get_history(db=Depends(get_db)):
    h = db.query(HistoryDB).order_by(HistoryDB.timestamp.desc()).limit(10).all()
    return [{"sentence": i.sentence, "time": i.timestamp.strftime("%H:%M")} for i in h]

@app.post("/api/vocab")
async def add_vocab(data: dict, db=Depends(get_db)):
    db.add(VocabDB(username=data['username'], word=data['word'], meaning=data['meaning']))
    db.commit(); return {"status": "success"}

@app.get("/api/vocab")
async def get_vocab(db=Depends(get_db)):
    v = db.query(VocabDB).all()
    return [{"id": i.id, "word": i.word, "meaning": i.meaning} for i in v]

@app.delete("/api/vocab/{v_id}")
async def del_vocab(v_id: int, db=Depends(get_db)):
    db.query(VocabDB).filter(VocabDB.id == v_id).delete()
    db.commit(); return {"status": "success"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sentence, current_word, last_char, counter, no_hand_count, last_completed_word, last_meaning
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "clear":
                sentence = ""; current_word = ""; continue
            elif data.get("command") == "backspace":
                current_word = current_word[:-1]; continue

            landmarks = data.get("landmarks")
            config = data.get("config", {"threshold": 0.85, "buffer": 5})
            action = {"speak": None, "text": f"{sentence} {current_word}".strip(), "current_letter": "-"}

            if landmarks and interpreter:
                no_hand_count = 0
                coords = []
                bx, by = landmarks[0][0], landmarks[0][1]
                for lm in landmarks: coords.extend([lm[0]-bx, lm[1]-by, lm[2]])
                
                interpreter.set_tensor(input_details[0]['index'], np.array([coords], dtype=np.float32))
                interpreter.invoke()
                p = interpreter.get_tensor(output_details[0]['index'])[0]
                conf = float(np.max(p))
                
                if conf > config["threshold"]:
                    char = labels[np.argmax(p)]
                    action["current_letter"] = char
                    if char == last_char: counter += 1
                    else: last_char = char; counter = 0
                    
                    if counter >= config["buffer"]:
                        if char not in ['nothing', 'space', 'del']:
                            current_word += char
                            action["speak"] = char # Spell letter
                            counter = 0 # FIX: RESET COUNTER TO STOP CONTINUOUS TYPING
                        elif char == 'space':
                            sentence += current_word + " "; current_word = ""; counter = 0
            else:
                # FIX: SPEAK WORD WHEN HAND IS REMOVED
                no_hand_count += 1
                if no_hand_count >= 15 and current_word != "":
                    last_completed_word = current_word
                    last_meaning = get_meaning(last_completed_word)
                    action["speak"] = last_completed_word # Triggers TTS for full word
                    action["completed_word"] = last_completed_word
                    action["meaning"] = last_meaning
                    sentence += current_word + " "
                    current_word = ""; last_char = ""; no_hand_count = 0

            action["text"] = f"{sentence} {current_word}".strip()
            await websocket.send_json(action)
    except: pass
