# config.py
import os
from pathlib import Path

# ---------- Project Paths ----------
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

RESUME_CACHE_FILE = BASE_DIR / "resume_llm_cache.json"
JD_CACHE_FILE = BASE_DIR / "jd_llm_cache.json"

# ---------- MongoDB ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "candidate_ranker_db"

RESUME_RAW_COLLECTION = "resumes_raw"
CANDIDATE_COLLECTION = "candidates_processed"
JOB_COLLECTION = "jobs"
RANKING_COLLECTION = "rankings"
LOG_COLLECTION = "logs"

# ---------- LLM ----------
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "models/ggml-gpt4all-j-v1.3-groovy.bin"
)

LLM_RATE_LIMIT_SECONDS = 1.5  # simulate heavy local model usage

# ---------- Scoring Weights ----------
SKILL_WEIGHT = 0.6
EXPERIENCE_WEIGHT = 0.3
LOCATION_WEIGHT = 0.1
