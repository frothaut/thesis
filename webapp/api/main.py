# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from contextlib import asynccontextmanager

# 1. Datei-Pfad
DATA_FILE = os.path.join(os.path.dirname(__file__), 'polygons.json')

# 2. Stelle sicher, dass die JSON-Datei existiert
def ensure_data_file():
    if not os.path.exists(DATA_FILE):
        empty = {
            "type": "FeatureCollection",
            "features": []
        }
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(empty, f, indent=2)

# 3. Lifespan-Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_file()
    yield

# 4. FastAPI-Instanz mit Lifespan
app = FastAPI(lifespan=lifespan)

# 5. CORS-Middleware: erlaube dein Vite-/React-Devserver-Origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],    # oder ["*"] für alle Origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Test-Endpoint
@app.get("/api/ping")
async def ping():
    return {"status": "ok"}

# 7. Polygone laden
@app.get("/api/polygons")
async def get_polygons():
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        # Fehler trotzdem im JSON-Format zurückgeben
        return {"error": str(e)}

# 8. Polygone speichern
@app.post("/api/polygons")
async def save_polygons(request: Request):
    try:
        collection = await request.json()
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(collection, f, indent=2)
        return {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}
    #uvicorn main:app --reload --host 0.0.0.0 --port 8000