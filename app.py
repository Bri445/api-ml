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

from openai import OpenAI
from emoji_sentiment import EmojiSentiment

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_PATH = os.getenv("MODEL_PATH", "streaming_model.joblib")
MODEL_URL = os.getenv("MODEL_URL")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

_artifact = None
_lexicon_es = None

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

app = FastAPI(
    title="EmojiTextSentimentAPI",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def analyze_with_gemini(text: str) -> Dict:
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

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

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, f"OpenRouter error: {str(e)}")

    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise HTTPException(500, f"Invalid LLM response: {raw}")

    cleaned = raw[start:end + 1]

    try:
        data = json.loads(cleaned)
    except Exception:
        raise HTTPException(500, f"JSON parse failed: {cleaned}")

    return data

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
