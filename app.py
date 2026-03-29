import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import cv2
import threading
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "SignBridge running"}

MODEL_PATH = "gesture_model_optimized.tflite"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model file not found: {MODEL_PATH}")

app = FastAPI()

# --- LOAD ASL MODEL ---
import tflite_runtime.interpreter as tflite
try:
    interpreter = tf.lite.Interpreter(model_path="gesture_model_optimized.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    labels = np.load("labels.npy", allow_pickle=True)

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model loading failed:", e)
labels = sorted(os.listdir(DATA_DIR))

# --- GLOBALS ---
sentence = ""
current_word = ""
last_char = ""
counter = 0
current_emotion = "neutral"
no_hand_count = 0


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sentence, current_word, last_char, counter, current_emotion, no_hand_count
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # --- UI COMMANDS ---
            cmd = data.get("command")
            if cmd == "clear":
                sentence = ""; current_word = ""; continue
            elif cmd == "backspace":
                if current_word: current_word = current_word[:-1]
                elif sentence: sentence = sentence.strip()[:-1]
                continue

            landmarks = data.get("landmarks")
            img_data = data.get("image")
            config = data.get("config", {"threshold": 0.85, "buffer": 5, "emo_threshold": 0.5})
            
            action = {"speak": None, "text": "", "emotion": current_emotion, "confidence": 0}

            # 1. EMOTION (Threaded)
            if img_data:
                encoded = img_data.split(",", 1)[1]
                nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                threading.Thread(target=run_emotion_async, args=(img, config["emo_threshold"]), daemon=True).start()

            # 2. GESTURE LOGIC
            if landmarks:
                no_hand_count = 0
                coords = []
                bx, by = landmarks[0]['x'], landmarks[0]['y']
                for lm in landmarks:
                    coords.extend([lm['x'] - bx, lm['y'] - by, lm['z']])
                
                p = model(np.array([coords]), training=False).numpy()
                confidence = float(np.max(p))
                action["confidence"] = confidence
                input_data = np.array([coords], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                p = interpreter.get_tensor(output_details[0]['index'])[0]
                
                if confidence > config["threshold"]:
                    char = labels[np.argmax(p)]
                    if char == last_char:
                        counter += 1
                    else:
                        last_char = char; counter = 0
                    
                    if counter == config["buffer"]:
                        if char not in ['nothing', 'space', 'del']:
                            current_word += char
                            action["speak"] = char
                        elif char == 'space':
                            action["speak"] = current_word
                            sentence += current_word + " "
                            current_word = ""
            else:
                no_hand_count += 1
                if no_hand_count == 15 and current_word != "":
                    action["speak"] = current_word
                    sentence += current_word + " "
                    current_word = ""
                    last_char = ""

            action["text"] = f"{sentence} {current_word}"
            action["emotion"] = current_emotion
            await websocket.send_json(action)

    except Exception: pass
    
