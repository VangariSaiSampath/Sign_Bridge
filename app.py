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

# --- SQLAlchemy Imports ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- CONFIG ---
SECRET_KEY = "signbridge_secret"
ALGORITHM = "HS256"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DATABASE SETUP
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# AI ENGINE GLOBALS & INITIALIZATION
# ==========================================
interpreter = None
input_details = None
output_details = None
labels = []

# Global State for Translation
sentence = ""
current_word = ""
last_char = ""
last_completed_word = "" 
last_meaning = ""
counter = 0
no_hand_count = 0

def load_model_globally():
    """Ensures the TFLite model is loaded into memory correctly."""
    global interpreter, input_details, output_details, labels
    if interpreter is None:
        try:
            model_path = "gesture_model_optimized.tflite"
            labels_path = "labels.npy"
            
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                print(f"❌ Missing files: {model_path} or {labels_path}")
                return

            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            labels = np.load(labels_path, allow_pickle=True).tolist()
            print("✅ TFLite Engine Initialized")
        except Exception as e:
            print(f"❌ Model load error: {e}")

@app.on_event("startup")
async def startup_event():
    load_model_globally()

def get_meaning(word):
    if not word: return ""
    try:
        r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=2)
        if r.status_code == 200:
            return r.json()[0]['meanings'][0]['definitions'][0]['definition']
    except: pass
    return "No word exists"

# ==========================================
# AUTH & APIS
# ==========================================
class UserAuth(BaseModel):
    username: str
    password: str

@app.post("/signup")
async def signup(data: UserAuth, db = Depends(get_db)):
    existing_user = db.query(UserDB).filter(UserDB.username == data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = UserDB(username=data.username, password=data.password)
    db.add(new_user)
    db.commit()
    return {"status": "success"}

@app.post("/login")
async def login(data: UserAuth, db = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.username, UserDB.password == data.password).first()
    if user:
        token = jwt.encode({"sub": data.username, "exp": datetime.utcnow() + timedelta(hours=2)}, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/history")
async def get_history(db = Depends(get_db)):
    history = db.query(HistoryDB).order_by(HistoryDB.timestamp.desc()).limit(20).all()
    return [{"sentence": r.sentence, "time": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for r in history]

@app.post("/api/vocab")
async def add_vocab(item: dict, db = Depends(get_db)):
    new_vocab = VocabDB(username=item.get('username'), word=item.get('word'), meaning=item.get('meaning'))
    db.add(new_vocab)
    db.commit()
    return {"status": "success"}

@app.get("/api/vocab")
async def get_vocab(db = Depends(get_db)):
    vocab = db.query(VocabDB).order_by(VocabDB.timestamp.desc()).all()
    return [{"id": r.id, "word": r.word, "meaning": r.meaning, "time": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for r in vocab]

@app.delete("/api/vocab/{vocab_id}")
async def delete_vocab(vocab_id: int, db = Depends(get_db)):
    item = db.query(VocabDB).filter(VocabDB.id == vocab_id).first()
    if item:
        db.delete(item)
        db.commit()
    return {"status": "success"}

@app.get("/")
async def get():
    with open("index.html", "r") as f: return HTMLResponse(content=f.read())

# ==========================================
# WEBSOCKET ENGINE
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sentence, current_word, last_char, counter, no_hand_count, last_completed_word, last_meaning
    
    load_model_globally()
    token = websocket.query_params.get("token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            cmd = data.get("command")
            if cmd == "clear":
                if sentence.strip():
                    db = SessionLocal()
                    new_hist = HistoryDB(username=username, sentence=sentence.strip())
                    db.add(new_hist)
                    db.commit()
                    db.close()
                sentence = ""; current_word = ""; last_completed_word = ""; last_meaning = ""; continue
            elif cmd == "backspace":
                if current_word: current_word = current_word[:-1]
                elif sentence: sentence = sentence.strip()[:-1]
                continue

            landmarks = data.get("landmarks")
            config = data.get("config", {"threshold": 0.85, "buffer": 5})
            
            action = {
                "speak": None, "text": f"{sentence} {current_word}".strip(),
                "confidence": 0.0, "meaning": last_meaning, 
                "completed_word": last_completed_word, "current_word": current_word,
                "current_letter": "-" 
            }

            if landmarks and interpreter:
                no_hand_count = 0
                coords = []
                bx, by = landmarks[0][0], landmarks[0][1]
                for lm in landmarks:
                    coords.extend([lm[0] - bx, lm[1] - by, lm[2]])
                
                input_data = np.array([coords], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                output_data = interpreter.get_tensor(output_details[0]['index'])
                p = output_data[0]
                confidence = float(np.max(p))
                action["confidence"] = confidence
                
                if confidence > config["threshold"]:
                    char = labels[np.argmax(p)]
                    action["current_letter"] = char 
                    
                    if char == last_char:
                        counter += 1
                    else:
                        last_char = char
                        counter = 0
                    
                    # FIXED: Reset counter to 0 immediately after confirming letter
                    if counter >= config["buffer"]:
                        if char not in ['nothing', 'space', 'del']:
                            current_word += char
                            action["current_word"] = current_word
                            action["speak"] = char 
                            counter = 0 
                            
                        elif char == 'space' and current_word != "":
                            last_completed_word = current_word
                            last_meaning = get_meaning(last_completed_word)
                            action["speak"] = last_completed_word
                            sentence += current_word + " "
                            current_word = ""
                            counter = 0
            else:
                no_hand_count += 1
                if no_hand_count > 15 and current_word != "":
                    last_completed_word = current_word
                    last_meaning = get_meaning(last_completed_word)
                    action["speak"] = last_completed_word
                    sentence += current_word + " "
                    current_word = ""
                    last_char = ""
                    no_hand_count = 0

            action["text"] = f"{sentence} {current_word}".strip()
            await websocket.send_json(action)

    except Exception as e:
        print(f"❌ WebSocket error: {e}")
