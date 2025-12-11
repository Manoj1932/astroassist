from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import random
import asyncio
import os

from predict_intent import predict_intent

app = FastAPI(title="AstroAssist API")

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Static Web Files
# ----------------------------------------------------
app.mount("/web", StaticFiles(directory="web"), name="web")

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return FileResponse("web/index.html")

@app.get("/mobile", response_class=HTMLResponse)
def serve_mobile_ui():
    return FileResponse("web/mobile.html")

# ----------------------------------------------------
# Home Route
# ----------------------------------------------------
@app.get("/")
def home():
    return {"message": "AstroAssist API is running ✅"}

# ----------------------------------------------------
# Dashboard Status
# ----------------------------------------------------
@app.get("/dashboard")
def dashboard():
    return {
        "status": "online",
        "service": "AstroAssist Control System",
        "model": "Intent + Emergency Detection Active"
    }

# ----------------------------------------------------
# Prediction API FIXED ⚠️
# Must send embeddings → ONNX expects a vector, not raw text
# ----------------------------------------------------
class Command(BaseModel):
    text: str

@app.post("/predict")
def predict(command: Command):
    text = command.text

    # TEMP: random embeddings (replace with real text embedding later)
    embedding = np.random.rand(1, 512).astype(np.float32)

    intent = predict_intent(embedding)

    return {
        "command": text,
        "predicted_intent": intent
    }

# ----------------------------------------------------
# WebSocket Sensor Stream
# ----------------------------------------------------
@app.websocket("/ws/sensors")
async def sensor_stream(websocket: WebSocket):
    await websocket.accept()

    oxygen = 95.0
    pressure = 101.0
    power = 85.0

    while True:

        oxygen += random.uniform(-1.2, 0.6)
        pressure += random.uniform(-0.8, 0.8)
        power += random.uniform(-0.5, 0.3)

        oxygen = max(10, min(100, oxygen))
        pressure = max(40, min(110, pressure))
        power = max(5, min(100, power))

        await websocket.send_json({
            "oxygen": round(oxygen, 1),
            "pressure": round(pressure, 1),
            "power": round(power, 1)
        })

        await asyncio.sleep(0.5)

# ----------------------------------------------------
# Run App
# ----------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn

    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
