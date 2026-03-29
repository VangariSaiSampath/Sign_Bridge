import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from datetime import datetime, timedelta
import os, requests, sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- DB & AUTH ---
SECRET_KEY = "signbridge_secret"
ALGORITHM = "HS256"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db").replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class HistoryDB(Base):
    __tablename__ = "history"; id = Column(Integer, primary_key=True); username = Column(String); sentence = Column(String); timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- AI STATE ---
interpreter = None; labels = []; sentence = ""; current_word = ""; last_char = ""; counter = 0; no_hand_count = 0

def load_model():
    global interpreter, labels
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path="gesture_model_optimized.tflite")
        interpreter.allocate_tensors()
        labels = np.load("labels.npy", allow_pickle=True).tolist()

def get_meaning(word):
    try:
        r = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=1.5)
        return r.json()[0]['meanings'][0]['definitions'][0]['definition']
    except: return "Common sign detected (No dictionary definition)"

@app.on_event("startup")
async def startup(): load_model()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sentence, current_word, last_char, counter, no_hand_count
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("command") == "clear":
                sentence = ""; current_word = ""; continue
            
            landmarks = data.get("landmarks")
            config = data.get("config", {"threshold": 0.85, "buffer": 5})
            resp = {"text": f"{sentence} {current_word}".strip(), "current_letter": "-"}

            if landmarks:
                no_hand_count = 0
                input_data = []
                base = landmarks[0]
                for lm in landmarks: input_data.extend([lm[0]-base[0], lm[1]-base[1], lm[2]])
                
                interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.array([input_data], dtype=np.float32))
                interpreter.invoke()
                out = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
                
                idx = np.argmax(out)
                if out[idx] > config["threshold"]:
                    char = labels[idx]
                    resp["current_letter"] = char
                    if char == last_char: counter += 1
                    else: last_char = char; counter = 0
                    
                    if counter >= config["buffer"]:
                        if char not in ['nothing', 'space', 'del']:
                            current_word += char
                            resp["speak"] = char
                            counter = 0 # FIX: Stop runaway letters
                        elif char == 'space':
                            sentence += current_word + " "; current_word = ""; counter = 0
            else:
                no_hand_count += 1
                if no_hand_count == 15 and current_word:
                    word = current_word
                    resp["completed_word"] = word
                    resp["meaning"] = get_meaning(word)
                    resp["speak"] = word
                    sentence += current_word + " "; current_word = ""; no_hand_count = 0

            await websocket.send_json(resp)
    except: pass
