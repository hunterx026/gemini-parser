import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import google.generativeai as genai

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Falta GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

PROMPT = """
Eres un extractor de tablas para un videojuego de liga.

La imagen mostrará 1 o más jugadores con sus estadísticas.
Debes devolver SOLO JSON válido (un array) con ESTE formato:

[
  {
    "game": 1,
    "jugador": "Aero",
    "goles": 2,
    "asistencias": 1,
    "pases": 15,
    "intercepciones": 3,
    "salvadas": 1,
    "puntaje": 8
  }
]

Reglas:
- Si no ves el número de game, usa 1.
- Si falta alguna estadística, pon 0.
- No agregues texto fuera del JSON.
- Los nombres de las claves deben ser EXACTAMENTE esos.
"""

def call_gemini_with_image_bytes(img_bytes: bytes):
    model = genai.GenerativeModel(MODEL_NAME)
    resp = model.generate_content(
        [
            PROMPT,
            {"mime_type": "image/png", "data": img_bytes}
        ]
    )
    return resp.text

@app.post("/parse-stats")
async def parse_stats(file: UploadFile = File(...)):
    img_bytes = await file.read()
    try:
        raw = call_gemini_with_image_bytes(img_bytes)
        return JSONResponse(content={"ok": True, "data": raw})
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)
