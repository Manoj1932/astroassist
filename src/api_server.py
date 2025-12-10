from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.staticfiles import StaticFiles

from src.predict_intent import predict_intent

app = FastAPI(title="AstroAssist API")

# -------------------------------
# ✅ CORS (So browser can call API)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# ✅ Serve Web Dashboard Files
# -------------------------------
app.mount("/web", StaticFiles(directory="web"), name="web")

# -------------------------------
# ✅ Load UI at /ui
# -------------------------------
@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return FileResponse("web/index.html")

# -------------------------------
# ✅ Home Route
# -------------------------------
@app.get("/")
def home():
    return {"message": "AstroAssist API is running ✅"}

# -------------------------------
# ✅ Dashboard Status API
# -------------------------------
@app.get("/dashboard")
def dashboard():
    return {
        "status": "online",
        "service": "AstroAssist Control System",
        "model": "Intent + Emergency Detection Active"
    }

# -------------------------------
# ✅ Prediction API
# -------------------------------
class Command(BaseModel):
    text: str

@app.post("/predict")
def predict(command: Command):
    intent = predict_intent(command.text)
    return {
        "command": command.text,
        "predicted_intent": intent
    }
from fastapi import WebSocket
import random
import asyncio

@app.websocket("/ws/sensors")
async def sensor_stream(websocket: WebSocket):
    await websocket.accept()

    oxygen = 95.0
    pressure = 101.0
    power = 85.0

    while True:
        # Simulated natural drift
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

# ✅ Mobile UI route
@app.get("/mobile", response_class=HTMLResponse)
def serve_mobile_ui():
    return FileResponse("web/mobile.html")
