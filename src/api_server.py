from fastapi import FastAPI, WebSocket
import asyncio
import random
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.predict_intent import predict_intent

app = FastAPI(title="AstroAssist API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve UI
app.mount("/web", StaticFiles(directory="web"), name="web")


@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    return FileResponse("web/index.html")


@app.get("/")
def home():
    return {"message": "AstroAssist API is running 🚀"}


# -------------------
# Prediction API
# -------------------
class Command(BaseModel):
    text: str

@app.post("/predict")
def predict(command: Command):
    intent = predict_intent(command.text)
    return {"command": command.text, "intent": intent}


# -------------------
# Fake Sensor WebSocket
# -------------------
@app.websocket("/ws/sensors")
async def sensor_stream(ws: WebSocket):
    await ws.accept()

    oxygen = 95
    pressure = 101
    power = 85

    while True:
        oxygen += random.uniform(-1, 1)
        pressure += random.uniform(-1, 1)
        power += random.uniform(-1, 1)

        oxygen = max(10, min(100, oxygen))
        pressure = max(40, min(110, pressure))
        power = max(0, min(100, power))

        await ws.send_json({
            "oxygen": round(oxygen, 1),
            "pressure": round(pressure, 1),
            "power": round(power, 1)
        })

        await asyncio.sleep(0.5)


# Local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
