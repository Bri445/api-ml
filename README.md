This folder contains the files needed to run the EmojiTextSentiment API.

Required files to deploy (copy or move these into `ml-api`):
- app.py                 # FastAPI application (entrypoint)
- streaming_model.joblib # trained model artifact (vectorizer, emoji_extractor, clf, label_encoder)
- Emoji_Sentiment_Data_v1.0.csv  # emoji lexicon used by EmojiSentiment
- emoji_sentiment.py     # lexicon loader and emoji scoring helpers
- features.py            # EmojiFeatureExtractor used at training and transform time
- requirements.txt       # Python dependencies

Optional useful files (include if you want tests or debugging):
- smoke_test_app.py      # quick smoke test harness
- compute_debug.py       # diagnostic script to inspect probs and scores
- debug_runner.py        # detailed per-sample debug runner

How to deploy
1. Make sure Python 3.10+ is installed and create a virtualenv.
2. Install dependencies:
   pip install -r requirements.txt
3. Start the API (from inside ml-api):
   uvicorn app:app --host 0.0.0.0 --port 8000

Render deployment (recommended)

You can deploy this folder as a Python web service on Render without Docker. Render will run your start command and provide a $PORT environment variable.

Minimal steps on Render:
1. Create a new Web Service in the Render dashboard and connect your repository (or use the folder as the repo root).
2. Set the Build Command to:
    pip install -r requirements.txt
3. Set the Start Command to:
    uvicorn app:app --host 0.0.0.0 --port $PORT
4. Add any environment variables under the service settings (for example `MODEL_PATH=streaming_model.joblib`, `LEXICON_PATH=Emoji_Sentiment_Data_v1.0.csv`).

Sample `render.yaml` (optional, place at repo root):

```yaml
services:
   - type: web
      name: emoji-sentiment
      env: python
      plan: free
      buildCommand: pip install -r requirements.txt
      startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
      envVars:
         - key: MODEL_PATH
            value: streaming_model.joblib
         - key: LEXICON_PATH
            value: Emoji_Sentiment_Data_v1.0.csv
```

Environment variables you can tune:
- MODEL_PATH (default: streaming_model.joblib)
- LEXICON_PATH (default: Emoji_Sentiment_Data_v1.0.csv)
- CLF_WEIGHT, LEX_WEIGHT, COMBINED_THRESHOLD, CLF_CONFIDENCE_THRESHOLD

Files to include in the `ml-api` folder before deploying
- `app.py`                 # FastAPI application (entrypoint)
- `streaming_model.joblib` # trained model artifact
- `Emoji_Sentiment_Data_v1.0.csv`  # emoji lexicon
- `emoji_sentiment.py`     # lexicon loader and helpers
- `features.py`            # EmojiFeatureExtractor
- `requirements.txt`      # dependencies

Optional debug/test files (include only if desired):
- `smoke_test_app.py`, `compute_debug.py`, `debug_runner.py`, `inspect_model.py`

PowerShell commands to move recommended files into `ml-api` (run from the repo root):

```powershell
Move-Item .\app.py .\ml-api\
Move-Item .\streaming_model.joblib .\ml-api\
Move-Item .\Emoji_Sentiment_Data_v1.0.csv .\ml-api\
Move-Item .\emoji_sentiment.py .\ml-api\
Move-Item .\features.py .\ml-api\
Move-Item .\requirements.txt .\ml-api\
```

If you want, I can move these files into `ml-api` for you now. Or I can leave them and you can deploy directly from the repo root — Render can use a subfolder as the service root when connecting the repo.
