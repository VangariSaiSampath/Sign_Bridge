import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
import os
import traceback

# --- DB ---
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

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

# DB
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./signbridge.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password = Column(String)

class HistoryDB(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    sentence = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ✅ LOAD MODEL SAFELY
try:
    interpreter = tflite.Interpreter(model_path="gesture_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    labels = np.load("labels.npy", allow_pickle=True).tolist()

    print("MODEL INPUT:", input_details)
    print("MODEL OUTPUT:", output_details)
    print("LABELS:", len(labels))

    assert output_details[0]['shape'][1] == len(labels), "❌ LABEL MISMATCH"

except Exception as e:
    print("❌ MODEL LOAD FAILED:", e)
    interpreter = None

# STATE
sentence = ""
current_word = ""
last_char = ""
counter = 0
no_hand_count = 0

# AUTH
class UserAuth(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login(data: UserAuth, db=Depends(lambda: SessionLocal())):
    user = db.query(UserDB).filter_by(username=data.username, password=data.password).first()
    if not user:
        raise HTTPException(status_code=401)

    token = jwt.encode(
        {"sub": data.username, "exp": datetime.utcnow() + timedelta(hours=5)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return {"access_token": token}

@app.get("/")
async def home():
    return HTMLResponse(open("index.html").read())

# ✅ WEBSOCKET (FIXED)
@app.websocket("/ws")
async def ws(ws: WebSocket):
    global sentence, current_word, last_char, counter, no_hand_count

    token = ws.query_params.get("token")
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except:
        await ws.close()
        return

    await ws.accept()

    while True:
        try:
            data = await ws.receive_json()

            landmarks = data.get("landmarks")

            response = {
                "text": f"{sentence} {current_word}",
                "confidence": 0.0,
                "current_letter": "-",
                "current_word": current_word,
            }

            # ❌ MODEL NOT LOADED
            if interpreter is None:
                await ws.send_json(response)
                continue

            if landmarks:
                no_hand_count = 0

                # ✅ optimized coords
                coords = []
                base_x, base_y = landmarks[0][0], landmarks[0][1]

                for lm in landmarks:
                    coords.extend([
                        lm[0] - base_x,
                        lm[1] - base_y,
                        lm[2]
                    ])

                input_data = np.array([coords], dtype=np.float32)

                # ✅ SHAPE CHECK
                if input_data.shape[1] != input_details[0]['shape'][1]:
                    print("❌ SHAPE MISMATCH:", input_data.shape)
                    await ws.send_json(response)
                    continue

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output = interpreter.get_tensor(output_details[0]['index'])[0]

                confidence = float(np.max(output))
                response["confidence"] = confidence

                if confidence > 0.85:
                    char = labels[np.argmax(output)]
                    response["current_letter"] = char

                    if char == last_char:
                        counter += 1
                    else:
                        last_char = char
                        counter = 0

                    if counter > 3:
                        current_word += char
                        response["current_word"] = current_word

            else:
                # ✅ RESET FIX
                no_hand_count += 1
                response["confidence"] = 0.0
                response["current_letter"] = "-"

                if no_hand_count > 5:
                    last_char = ""

            response["text"] = f"{sentence} {current_word}"

            await ws.send_json(response)

        except Exception as e:
            print("WS ERROR:", e)
            traceback.print_exc()
            break
