import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from deepface import DeepFace
import base64
import cv2
import threading
import os

app = FastAPI()

# --- LOAD ASL MODEL ---
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="gesture_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
DATA_DIR = "data/asl_alphabet_train/asl_alphabet_train"
labels = sorted(os.listdir(DATA_DIR))

# --- GLOBALS ---
sentence = ""
current_word = ""
last_char = ""
counter = 0
current_emotion = "neutral"
no_hand_count = 0

def run_emotion_async(img, threshold):
    global current_emotion
    try:
        res = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
        # Only update if the dominant emotion is above the user's slider threshold
        score = res[0]['emotion'][res[0]['dominant_emotion']]
        if score >= (threshold * 100):
            current_emotion = res[0]['dominant_emotion']
    except: pass

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
    
