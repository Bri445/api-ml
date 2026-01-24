from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import anyio
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
from emoji_sentiment import EmojiSentiment
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re
import requests
import shutil
import json
import google.generativeai as genai

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip
    pass
try:
    import importlib
    _vmod = importlib.import_module('vaderSentiment.vaderSentiment')
    SentimentIntensityAnalyzer = getattr(_vmod, 'SentimentIntensityAnalyzer')
    _VADER_AVAILABLE = True
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER_AVAILABLE = False
    _VADER = None


MODEL_PATH = os.environ.get('MODEL_PATH', 'streaming_model.joblib')
MODEL_URL = os.environ.get('MODEL_URL')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Configure Gemini AI with v1 API (for free tier models)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: run startup logic (download/load model, load lexicon).

    Uses anyio.to_thread.run_sync to run blocking IO (joblib load) off the event loop.
    """
    global _artifact, _lexicon_es

    def _startup_sync():
        # load/download model
        _artifact = _load_model(MODEL_PATH)
        # load lexicon if available
        lex_path = os.environ.get('LEXICON_PATH', 'Emoji_Sentiment_Data_v1.0.csv')
        if os.path.exists(lex_path):
            try:
                return _artifact, EmojiSentiment(lex_path)
            except Exception:
                return _artifact, None
        return _artifact, None

    # run blocking startup work in thread pool
    _artifact, _lexicon_es = await anyio.to_thread.run_sync(_startup_sync)
    try:
        yield
    finally:
        # optional shutdown cleanup (none required currently)
        pass


app = FastAPI(title='EmojiTextSentimentAPI', lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class BatchRequest(BaseModel):
    texts: List[str]


class PredictResponse(BaseModel):
    sentiment: str
    score: float


class KeywordsRequest(BaseModel):
    texts: List[str]
    per_comment_limit: Optional[int] = 10


class CommentKeywords(BaseModel):
    keywords: List[str]


class KeywordsResponse(BaseModel):
    results: List[CommentKeywords]
    top_keyword: Optional[str]


class GeminiAnalysisRequest(BaseModel):
    text: str


class GeminiAnalysisResponse(BaseModel):
    sentiment: str
    score: float
    keywords: List[str]
    top_keyword: str


def _load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        # try to download model if a URL was provided
        if MODEL_URL:
            try:
                _download_model(MODEL_URL, path)
            except Exception as e:
                raise RuntimeError(f'Model file not found and download failed: {path} ({e})')
        else:
            raise RuntimeError(f'Model file not found: {path}')
    artifact = joblib.load(path)
    return artifact


def _download_model(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    """Download a file from url to dest_path using streaming download.

    Attempts to write directly to dest_path. If the parent directory does not
    exist or is not writable, attempts to use a temporary directory.
    """
    # ensure parent dir exists
    dest_dir = os.path.dirname(os.path.abspath(dest_path)) or '.'
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except Exception:
        # fallback to platform temp dir
        dest_dir = os.environ.get('TMP', os.environ.get('TEMP', '/tmp'))
        dest_path = os.path.join(dest_dir, os.path.basename(dest_path))

    # prepare headers (support a bearer token via MODEL_AUTH_TOKEN)
    headers = {}
    token = os.environ.get('MODEL_AUTH_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # stream download
    resp = requests.get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()
    tmp_path = dest_path + '.part'
    with open(tmp_path, 'wb') as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
    # move into final location atomically
    shutil.move(tmp_path, dest_path)


_artifact = None
_lexicon_es = None


# Dead code - legacy functions that are no longer used
def _legacy_preprocess_text(text):
    """Old preprocessing function - no longer used."""
    temp_var = text.lower()
    temp_list = []
    for char in temp_var:
        temp_list.append(char)
    return ''.join(temp_list)


def _unused_calculate_metrics(data):
    """Unused metric calculation function."""
    total = 0
    count = 0
    for item in data:
        total += item
        count += 1
    if count > 0:
        avg = total / count
    return None


def _deprecated_feature_extractor(text):
    """Deprecated feature extraction - replaced by new model."""
    features = []
    words = text.split()
    for word in words:
        if len(word) > 0:
            features.append(len(word))
    return features


def _old_sentiment_scorer(text):
    """Old sentiment scoring logic - no longer in use."""
    score = 0.0
    positive_count = 0
    negative_count = 0
    for char in text:
        if char.isupper():
            positive_count += 1
        elif char.islower():
            negative_count += 1
    score = (positive_count - negative_count) / max(len(text), 1)
    return score


def _analyze_with_gemini(text: str) -> Dict:
    """Analyze sentiment and extract keywords using Gemini AI."""
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not configured")
        raise HTTPException(status_code=500, detail='GEMINI_API_KEY not configured. Please set the GEMINI_API_KEY environment variable.')
    
    print(f"Analyzing text with Gemini: {text[:100]}...")  # Log first 100 chars
    
    try:
        # Use gemini-flash-latest - the latest stable free tier model
        # This is the recommended model for free API usage
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""Analyze the following comment/text for sentiment and keywords:

Text: "{text}"

Provide your analysis in the following JSON format ONLY (no additional text):
{{
    "sentiment": "positive" or "negative" or "neutral",
    "score": a float between -1.0 (most negative) and 1.0 (most positive),
    "keywords": [list of 5-10 important keywords from the text that represent key themes],
    "top_keyword": "the single most important keyword for word cloud visualization"
}}

Rules:
- sentiment must be exactly one of: positive, negative, or neutral
- score should reflect the intensity of sentiment (-1.0 to 1.0)
- keywords should be meaningful words/phrases (not stopwords like "the", "is", "and")
- top_keyword should be the most emotionally charged or significant word
- Return ONLY valid JSON, no markdown formatting or explanations"""
        
        #print("Calling Gemini API...")
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        #print(f"Gemini raw response: {result_text}")
        
        # Remove markdown code blocks if present
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        if result_text.startswith('```'):
            result_text = result_text[3:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        print(f"Cleaned response: {result_text}")
        
        # Parse JSON response
        result = json.loads(result_text)
        print(f"Parsed JSON: {result}")
        
        # Validate response structure
        if not all(key in result for key in ['sentiment', 'score', 'keywords', 'top_keyword']):
            missing_keys = [k for k in ['sentiment', 'score', 'keywords', 'top_keyword'] if k not in result]
            print(f"ERROR: Missing keys in Gemini response: {missing_keys}")
            raise ValueError(f'Invalid response structure from Gemini. Missing keys: {missing_keys}')
        
        # Validate sentiment value
        if result['sentiment'] not in ['positive', 'negative', 'neutral']:
            print(f"WARNING: Invalid sentiment value: {result['sentiment']}, defaulting to neutral")
            result['sentiment'] = 'neutral'
        
        # Validate score range
        original_score = result['score']
        result['score'] = max(-1.0, min(1.0, float(result['score'])))
        if result['score'] != original_score:
            print(f"WARNING: Score {original_score} out of range, clamped to {result['score']}")
        
        # Ensure keywords is a list
        if not isinstance(result['keywords'], list):
            print(f"WARNING: keywords is not a list: {type(result['keywords'])}, converting to empty list")
            result['keywords'] = []
        
        # Ensure top_keyword is a string
        if not isinstance(result['top_keyword'], str) or not result['top_keyword']:
            print(f"WARNING: Invalid top_keyword, using first keyword or 'unknown'")
            result['top_keyword'] = result['keywords'][0] if result['keywords'] else 'unknown'
        
        print(f"Successfully analyzed text. Sentiment: {result['sentiment']}, Score: {result['score']}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode error: {str(e)}")
        print(f"Raw response text: {result_text if 'result_text' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail=f'Failed to parse Gemini response as JSON: {str(e)}')
    except Exception as e:
        print(f"ERROR: Gemini API error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f'Gemini API error: {str(e)}')


def _score_texts(texts):
    vec = _artifact['vectorizer']
    emoji_ext = _artifact['emoji_extractor']
    clf = _artifact['clf']
    le = _artifact['label_encoder']

    # threshold for lexicon override (abs value)
    try:
        LEX_THRESHOLD = float(os.environ.get('LEXICON_THRESHOLD', '0.10'))
    except Exception:
        LEX_THRESHOLD = 0.10
    # override mode: 'only_if_neutral' (default), 'always', 'never'
    LEX_OVERRIDE = os.environ.get('LEXICON_OVERRIDE', 'only_if_neutral')

    X_text = vec.transform(texts)
    X_emoji = np.array(emoji_ext.transform(texts))
    X_emoji_sp = csr_matrix(X_emoji)
    X = hstack([X_text, X_emoji_sp])

    # configurable weights and thresholds
    CLF_WEIGHT = float(os.environ.get('CLF_WEIGHT', '0.5'))
    LEX_WEIGHT = float(os.environ.get('LEX_WEIGHT', '0.5'))
    COMB_THRESHOLD = float(os.environ.get('COMBINED_THRESHOLD', '0.01'))
    # classifier confidence below which we consider it 'uncertain'
    CLF_CONFIDENCE_THRESHOLD = float(os.environ.get('CLF_CONFIDENCE_THRESHOLD', '0.03'))

    # compute lexicon scores
    lex_scores = [0.0] * len(texts)
    lex_labels = [None] * len(texts)
    lex_details = [None] * len(texts)
    if _lexicon_es is not None:
        for i, t in enumerate(texts):
            try:
                ls, ll, ld = _lexicon_es.predict(t)
                lex_scores[i] = ls
                lex_labels[i] = ll
                lex_details[i] = ld
            except Exception:
                lex_scores[i] = 0.0
                lex_labels[i] = None
                lex_details[i] = None

    # robust emoji regex to strip a wide range of emoji/unicode pictographs
    emoji_re = re.compile(
        '['
        '\U0001F600-\U0001F64F'  # emoticons
        '\U0001F300-\U0001F5FF'  # symbols & pictographs
        '\U0001F680-\U0001F6FF'  # transport & map symbols
        '\U0001F1E0-\U0001F1FF'  # flags
        '\U00002700-\U000027BF'  # dingbats
        '\U000024C2-\U0001F251'
        ']+', flags=re.UNICODE)

    def strip_emojis(s: str) -> str:
        if not s:
            return s
        return emoji_re.sub('', s)

    # detect presence of emoji and textual content robustly using regex
    has_emoji = [False] * len(texts)
    has_text = [False] * len(texts)
    for i, t in enumerate(texts):
        try:
            s = str(t)
        except Exception:
            s = ''
        # emoji presence via regex search or lexicon detail
        has_emoji[i] = bool(emoji_re.search(s) or (lex_details[i] and lex_details[i].get('emojis')))
        # remove emojis and check for letters/digits
        stripped = strip_emojis(s)
        has_text[i] = bool(stripped and any(ch.isalnum() for ch in stripped))

    # Compute classifier probabilities (or fallback to decision_function-derived probs)
    probs = None
    try:
        probs = clf.predict_proba(X)
    except Exception:
        try:
            df = clf.decision_function(X)
            if df.ndim == 1:
                probs = np.vstack([1.0 - (1.0 / (1.0 + np.exp(-df))), 1.0 / (1.0 + np.exp(-df))]).T
            else:
                exp = np.exp(df - np.max(df, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
        except Exception:
            probs = None

    # derive classifier signed score: P_pos - P_neg
    clf_signed = [0.0] * len(texts)
    clf_labels = [None] * len(texts)
    if probs is not None:
        P_pos = np.zeros(len(texts))
        P_neg = np.zeros(len(texts))
        # try to map numeric class indices via label encoder
        try:
            pos_code = int(le.transform(['positive'])[0])
            neg_code = int(le.transform(['negative'])[0])
            for j, cls in enumerate(clf.classes_):
                if int(cls) == pos_code:
                    P_pos += probs[:, j]
                elif int(cls) == neg_code:
                    P_neg += probs[:, j]
        except Exception:
            # fallback: assume ordering [neg,neu,pos] if 3 classes
            n_classes = probs.shape[1]
            if n_classes == 3:
                P_neg = probs[:, 0]
                P_pos = probs[:, 2]
            else:
                # attempt to match by string labels
                for j, cls in enumerate(clf.classes_):
                    lab = str(cls).lower()
                    if 'neg' in lab:
                        P_neg += probs[:, j]
                    elif 'pos' in lab:
                        P_pos += probs[:, j]
        clf_signed = (P_pos - P_neg).tolist()
        # classifier label by argmax
        idx = np.argmax(probs, axis=1)
        try:
            clf_labels = le.inverse_transform(idx)
        except Exception:
            clf_labels = [None] * len(texts)

    # Prepare final outputs
    signed = []
    out_labels = []

    # configurable weights and thresholds
    CLF_WEIGHT = float(os.environ.get('CLF_WEIGHT', '0.6'))
    LEX_WEIGHT = float(os.environ.get('LEX_WEIGHT', '0.4'))
    COMB_THRESHOLD = float(os.environ.get('COMBINED_THRESHOLD', '0.02'))
    # classifier confidence below which we consider it 'uncertain'
    CLF_CONFIDENCE_THRESHOLD = float(os.environ.get('CLF_CONFIDENCE_THRESHOLD', '0.03'))

    # small rule-based word lexicon fallback for short text-only cases when classifier is neutral
    POS_WORDS = {
        'love', 'great', 'awesome', 'good', 'like', 'happy', 'amazing', 'nice',
        'fantastic', 'well done', 'congrats', 'congratulations', 'best'
    }
    NEG_WORDS = {
        'awful', 'bad', 'terrible', 'hate', 'sad', 'worst', 'angry', 'horrible',
        'very bad', 'not good', 'disappointing', 'disappointed','shit'
    }

    for i in range(len(texts)):
        ls = float(lex_scores[i]) if lex_scores and lex_scores[i] is not None else 0.0
        cs = float(clf_signed[i]) if clf_signed else 0.0

        # emoji-only -> lexicon
        if has_emoji[i] and not has_text[i]:
            if ls > 0:
                out_labels.append('positive'); signed.append(ls)
            elif ls < 0:
                out_labels.append('negative'); signed.append(ls)
            else:
                out_labels.append('neutral'); signed.append(0.0)
            continue

        # text-only -> classifier; if classifier uncertain, try a tiny rule-based word lexicon
        if has_text[i] and not has_emoji[i]:
            txt = texts[i].lower()
            stripped = txt
            # token count heuristic
            token_count = len(stripped.split()) if stripped else 0

            # always apply strong rule-word override for short texts (safety for classifier mistakes)
            if token_count <= 6:
                if any(w in txt for w in NEG_WORDS):
                    out_labels.append('negative'); signed.append(-0.8); continue
                if any(w in txt for w in POS_WORDS):
                    out_labels.append('positive'); signed.append(0.8); continue

            # otherwise use classifier if confident
            if cs > COMB_THRESHOLD:
                out_labels.append('positive'); signed.append(cs); continue
            if cs < -COMB_THRESHOLD:
                out_labels.append('negative'); signed.append(cs); continue

            # classifier is neutral/uncertain -> try VADER if available, else rule-based words
            if _VADER_AVAILABLE:
                vs = _VADER.polarity_scores(txt)
                comp = vs.get('compound', 0.0)
                if comp >= 0.3:
                    out_labels.append('positive'); signed.append(float(comp)); continue
                if comp <= -0.3:
                    out_labels.append('negative'); signed.append(float(comp)); continue
            else:
                if any(w in txt for w in NEG_WORDS):
                    out_labels.append('negative'); signed.append(-0.6); continue
                if any(w in txt for w in POS_WORDS):
                    out_labels.append('positive'); signed.append(0.6); continue

            out_labels.append('neutral'); signed.append(0.0)
            continue

        # both present -> if classifier is uncertain, prefer lexicon when strong; else use weighted combine
        if has_text[i] and has_emoji[i]:
            if abs(cs) < CLF_CONFIDENCE_THRESHOLD:
                # classifier uncertain: if lexicon signal strong enough, use lexicon
                if abs(ls) >= LEX_THRESHOLD:
                    if ls > 0:
                        out_labels.append('positive'); signed.append(ls)
                    elif ls < 0:
                        out_labels.append('negative'); signed.append(ls)
                    else:
                        out_labels.append('neutral'); signed.append(0.0)
                    continue
                # fall back to checking word cues in text
                txt = texts[i].lower()
                stripped = txt
                token_count = len(stripped.split()) if stripped else 0
                # for short mixed inputs prefer strong word cues even if classifier uncertain
                if token_count <= 6:
                    if any(w in txt for w in NEG_WORDS):
                        out_labels.append('negative'); signed.append(-0.8); continue
                    if any(w in txt for w in POS_WORDS):
                        out_labels.append('positive'); signed.append(0.8); continue

            # otherwise use weighted combination
            combined = CLF_WEIGHT * cs + LEX_WEIGHT * ls
            if combined > COMB_THRESHOLD:
                out_labels.append('positive'); signed.append(combined)
            elif combined < -COMB_THRESHOLD:
                out_labels.append('negative'); signed.append(combined)
            else:
                out_labels.append('neutral'); signed.append(0.0)
            continue

        # fallback
        out_labels.append('neutral')
        signed.append(0.0)

    return out_labels, signed


@app.get('/health')
def health():
    return {'status': 'ok', 'model_path': MODEL_PATH}


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail='text is required')
    labels, scores = _score_texts([req.text])
    return {'sentiment': labels[0], 'score': scores[0]}


@app.post('/predict_batch')
def predict_batch(req: BatchRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail='texts required')
    labels, scores = _score_texts(req.texts)
    results = []
    for t, l, s in zip(req.texts, labels, scores):
        results.append({'sentiment': l, 'score': s})
    return {'results': results}


def _extract_keywords(texts: List[str], per_comment_limit: int = 10):
    """Extract keywords that actually appear in each text using the trained vectorizer.

    - Uses the artifact's vectorizer to transform the text and picks non-zero
      features with highest weights (tf-idf or count), ensuring the returned
      keywords are from the original comments (no synonyms).
    - Returns per-comment keyword lists and the overall top keyword by frequency.
    """
    if _artifact is None or 'vectorizer' not in _artifact:
        raise HTTPException(status_code=500, detail='Model/vectorizer not loaded')

    vec = _artifact['vectorizer']
    try:
        feature_names = vec.get_feature_names_out()
    except Exception:
        feature_names = vec.get_feature_names()

    X = vec.transform(texts)

    # simple stopword list to avoid very common function words
    STOP = {
        'the','and','for','with','this','that','are','was','were','you','your','yours','we','they','he','she','it',
        'is','am','are','be','to','of','in','on','at','by','a','an','as','or','but','from','not','have','has','had',
        'do','does','did','so','if','then','than','too','very','just','can','could','would','should','will','i','me',
        'my','our','ours','their','them','his','her','its','also','there','here','because','about','into','out','up','down'
    }

    def valid_token(tok: str) -> bool:
        if not tok:
            return False
        if tok in STOP:
            return False
        if tok.isdigit():
            return False
        # keep alphanumeric tokens of length >= 3
        return any(ch.isalpha() for ch in tok) and len(tok) >= 3

    per_comment = []
    freq = {}
    for i in range(X.shape[0]):
        row = X.getrow(i)
        indices = row.indices
        weights = row.data
        pairs = list(zip(indices, weights))
        # sort by weight descending
        pairs.sort(key=lambda p: p[1], reverse=True)
        kws = []
        seen = set()
        for idx, _w in pairs:
            tok = str(feature_names[idx])
            if tok in seen:
                continue
            if valid_token(tok):
                kws.append(tok)
                seen.add(tok)
            if len(kws) >= per_comment_limit:
                break
        per_comment.append(kws)
        for t in set(kws):
            freq[t] = freq.get(t, 0) + 1

    top_keyword = None
    if freq:
        # pick the most frequent; break ties by lexicographic order for determinism
        top_keyword = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    return per_comment, top_keyword


@app.post('/extract_keywords', response_model=KeywordsResponse)
def extract_keywords(req: KeywordsRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail='texts required')
    per_comment, top_kw = _extract_keywords(req.texts, per_comment_limit=(req.per_comment_limit or 10))
    results = [{'keywords': kws} for kws in per_comment]
    return {'results': results, 'top_keyword': top_kw}


@app.get('/analyze_request')
def analyze_request_get():
    """
    GET endpoint for /analyze_request.
    This endpoint requires POST method with JSON body.
    """
    raise HTTPException(
        status_code=405, 
        detail='Method Not Allowed. Please use POST with JSON body: {"text": "your text here"}'
    )


@app.post('/analyze_request', response_model=GeminiAnalysisResponse)
async def analyze_request(req: GeminiAnalysisRequest):
    """
    Advanced sentiment analysis using Gemini AI.
    
    Returns sentiment type, score, keywords among the comment, and the most important keyword for word cloud.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail='text is required and cannot be empty')
    
    try:
        # Run blocking Gemini API call in thread pool to avoid blocking event loop
        result = await anyio.to_thread.run_sync(lambda: _analyze_with_gemini(req.text))
        
        return {
            'sentiment': result['sentiment'],
            'score': result['score'],
            'keywords': result['keywords'],
            'top_keyword': result['top_keyword']
        }
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Error in analyze_request: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f'Failed to analyze text: {str(e)}'
        )
