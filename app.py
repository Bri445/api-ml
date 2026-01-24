import os
import re
import json
import shutil
import anyio
import joblib
import numpy as np
import requests

from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from scipy.sparse import hstack, csr_matrix

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from emoji_sentiment import EmojiSentiment

# =========================
# Environment / Gemini
# =========================

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_PATH = os.getenv("MODEL_PATH", "streaming_model.joblib")
MODEL_URL = os.getenv("MODEL_URL")

# =========================
# Globals
# =========================

_artifact = None
_lexicon_es = None

# =========================
# Lifespan (startup)
# =========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifact, _lexicon_es

    def startup():
        artifact = joblib.load(MODEL_PATH)
        lex = None
        lex_path = "Emoji_Sentiment_Data_v1.0.csv"
        if os.path.exists(lex_path):
            try:
                lex = EmojiSentiment(lex_path)
            except Exception:
                pass
        return artifact, lex

    _artifact, _lexicon_es = await anyio.to_thread.run_sync(startup)
    yield

# =========================
# FastAPI App (ONE INSTANCE)
# =========================

app = FastAPI(
    title="EmojiTextSentimentAPI",
    lifespan=lifespan
)

# =========================
# CORS (FIXES OPTIONS 405)
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],  # POST, OPTIONS, etc
    allow_headers=["*"],  # Content-Type, Authorization
)

# =========================
# Models
# =========================

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    sentiment: str
    score: float

class GeminiAnalysisRequest(BaseModel):
    text: str

class GeminiAnalysisResponse(BaseModel):
    sentiment: str
    score: float
    keywords: List[str]
    top_keyword: str

# =========================
# Gemini Analysis
# =========================

def analyze_with_gemini(text: str) -> Dict:
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GOOGLE_API_KEY not set")

    model = genai.GenerativeModel("gemini-flash-latest")

    prompt = f"""
Analyze the following text and return ONLY valid JSON.

Text: "{text}"

Return exactly this JSON schema:
{{
  "sentiment": "positive | negative | neutral",
  "score": number between -1 and 1,
  "keywords": ["list", "of", "keywords"],
  "top_keyword": "single keyword"
}}
"""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # --- CLEANUP ---
    raw = raw.replace("```json", "").replace("```", "").strip()

    # extract first JSON object defensively
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise HTTPException(500, f"Invalid Gemini response: {raw}")

    cleaned = raw[start:end + 1]

    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise HTTPException(500, f"JSON parse failed: {cleaned}")

    return data

# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(400, "text required")

    vec = _artifact["vectorizer"]
    clf = _artifact["clf"]
    le = _artifact["label_encoder"]
    emoji_ext = _artifact["emoji_extractor"]

    X_text = vec.transform([req.text])
    X_emoji = csr_matrix(np.array(emoji_ext.transform([req.text])))
    X = hstack([X_text, X_emoji])

    probs = clf.predict_proba(X)[0]
    idx = np.argmax(probs)

    sentiment = le.inverse_transform([idx])[0]
    score = float(probs[idx])

    return {"sentiment": sentiment, "score": score}

@app.post("/analyze_request", response_model=GeminiAnalysisResponse)
async def analyze_request(req: GeminiAnalysisRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is required")

    result = await anyio.to_thread.run_sync(
        lambda: analyze_with_gemini(req.text)
    )

    return result

